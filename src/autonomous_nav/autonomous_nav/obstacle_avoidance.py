#!/usr/bin/env python3
"""
obstacle_avoidance.py — Bug2 reactive obstacle avoidance module (ROS2 Jazzy).

Implements the Bug2 algorithm:

  FREE state  (navigation module controls):
    · Monitor LiDAR sectors each tick.
    · If FRONT / FRONT_RIGHT / FRONT_LEFT < WARNING_DIST:
        - Record hit point  H(x, y)
        - Define  m-line  = segment from H to the active goal waypoint
        - Switch to WALL_FOLLOW

  WALL_FOLLOW state  (this module controls):
    · Follow the wall on the RIGHT side (proportional lateral control).
    · Turn LEFT in-place when FRONT is blocked.
    · Bug2 exit conditions (both required):
        1. Perpendicular distance to m-line  < M_LINE_THRESHOLD
        2. dist(robot, goal) < dist(H, goal) − DISTANCE_PROGRESS_MIN
    · Anti-stuck: if accumulated rotation is too low over STUCK_CHECK_TICKS,
      apply a forced left rotation for STUCK_TURN_TICKS.

  DANGER (watchdog, caller fires at 50 Hz independently):
    · is_front_danger() → True when FRONT min < DANGER_DIST.
    · The caller (debug_nav_node) publishes a zero-velocity command directly.

API (matches debug_nav_node.py / mission_node.py):
    avoider = ObstacleAvoidance(logger=node.get_logger())
    avoider.update_scan(scan_msg)          # /scan callback (BEST_EFFORT)
    avoider.update_force_rotate(delta_yaw) # each control tick (|Δyaw| since last)
    if avoider.is_front_danger(): ...      # 50 Hz watchdog
    cmd, active = avoider.compute(x, y, yaw, wp_x, wp_y)
    state_name  = avoider.get_state().name

Compatibility:
    · Pure-logic module — no ROS2 Node inheritance.
    · Returns VelocityCommand (linear_x, angular_z); clamps to hard speed limits.
    · Tested against ROS2 Jazzy + TB3 Burger LiDAR (LDS-01/02, 360 pts,
      index 0 = front, CCW positive).
"""

# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    """All tuneable parameters in one place — no magic numbers elsewhere."""

    # --- Distance thresholds ---
    DANGER_DIST           = 0.14   # m   — watchdog emergency threshold
    WARNING_DIST          = 0.25   # m   — obstacle triggers avoidance entry
    SAFE_DIST             = 0.32   # m   — considered open space

    # --- Wall following ---
    WALL_FOLLOW_DIST      = 0.16   # m   — desired lateral distance from wall
    WALL_FOLLOW_SPEED     = 0.05   # m/s — forward speed during wall follow
    KP_WALL               = 2.8    # —   — proportional gain for lateral error
    TURN_SPEED            = 0.35   # rad/s — turn speed when front is blocked
    CORNER_TURN_FACTOR    = 0.8    # —   — factor applied at corners
    LOST_WALL_TURN_SPEED  = 0.80   # rad/s — sharp turn when following wall is lost

    # --- Bug2 exit conditions ---
    M_LINE_THRESHOLD      = 0.15   # m   — perpendicular dist to m-line for "on line"
    DISTANCE_PROGRESS_MIN = 0.15   # m   — min extra progress toward goal to exit
    MIN_TRAVEL_FROM_HIT   = 0.20   # m   — min distance from hit point before checking exit

    # --- Anti-stuck ---
    STUCK_CHECK_TICKS     = 40     # ticks (2 s @ 20 Hz) between stuck evaluations
    STUCK_ROTATION_MIN    = 0.10   # rad  — min accumulated rotation to not be stuck
    STUCK_TURN_TICKS      = 20     # ticks — forced left rotation when stuck detected

    # --- Exit cooldown (avoid immediately re-entering WALL_FOLLOW) ---
    EXIT_COOLDOWN_TICKS   = 20     # ticks (~1 s @ 20 Hz)

    # --- Hard speed caps (safety — never exceed these) ---
    LINEAR_MAX            = 0.20   # m/s
    ANGULAR_MAX           = 1.00   # rad/s

    # --- LiDAR sector boundaries [degrees, signed, TB3: 0=front, CCW+] ---
    #   Tuple format: (start_deg, end_deg)  inclusive
    FRONT_SECTOR          = (-20,  20)   # ±20° around front
    FRONT_RIGHT_SECTOR    = (-60, -20)   # 20°–60° to the right of front
    FRONT_LEFT_SECTOR     = ( 20,  60)   # 20°–60° to the left of front
    RIGHT_SECTOR          = (-90, -60)   # pure right side
    LEFT_SECTOR           = ( 60,  90)   # pure left side


# ============================================================
# IMPORTS
# ============================================================
import math
from enum import Enum, auto
from typing import Optional, Tuple

try:
    from sensor_msgs.msg import LaserScan          # available when ROS2 is sourced
except ImportError:
    LaserScan = None                               # standalone / unit-test fallback


# ============================================================
# ENUMERATIONS
# ============================================================
class AvoidState(Enum):
    """States of the Bug2 avoidance state machine."""
    FREE        = auto()   # no active avoidance; navigation module drives
    WALL_FOLLOW = auto()   # Bug2 wall-following active; this module drives
    DANGER      = auto()   # emergency stop (fired by external watchdog)


# ============================================================
# VELOCITY COMMAND
# ============================================================
class VelocityCommand:
    """
    Minimal container for a (linear_x, angular_z) velocity pair.

    Identical interface to the one in navigation.py so that
    mission_node / debug_nav_node can use both interchangeably.
    """
    __slots__ = ('linear_x', 'angular_z')

    def __init__(self, linear_x: float = 0.0, angular_z: float = 0.0) -> None:
        self.linear_x  = linear_x
        self.angular_z = angular_z

    def __repr__(self) -> str:
        return f'VelocityCommand(lin={self.linear_x:.3f}, ang={self.angular_z:.3f})'


# ============================================================
# UTILITIES
# ============================================================
def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp *value* to [lo, hi]."""
    return max(lo, min(hi, value))


def _valid_range(r: float) -> bool:
    """Return True if *r* is a finite, positive LiDAR range."""
    return math.isfinite(r) and r > 0.0


# ============================================================
# MAIN CLASS
# ============================================================
class ObstacleAvoidance:
    """
    Bug2 reactive obstacle avoidance — pure logic module (no ROS2 Node).

    Instantiated once by the mission node and fed LiDAR + pose data each
    control tick.  Returns velocity commands and an *in_avoidance* flag
    that tells the caller whether to suppress its own navigation command.

    Wall-follow side: RIGHT  (robot turns LEFT when front is blocked).
    """

    def __init__(self, logger=None) -> None:
        """
        Initialise the avoidance module.

        Args:
            logger: ROS2 logger object (``node.get_logger()``) or None
                    (falls back to ``print``).
        """
        self._log = logger

        # ---- State machine ----
        self._state = AvoidState.FREE

        # ---- Latest LiDAR sector minimums ----
        self._front_min       = float('inf')
        self._front_right_min = float('inf')
        self._front_left_min  = float('inf')
        self._right_min       = float('inf')
        self._left_min        = float('inf')
        self._scan_ready      = False

        # ---- Bug2 m-line data (set when avoidance starts) ----
        self._hit_x: float            = 0.0
        self._hit_y: float            = 0.0
        self._hit_dist_to_goal: float = float('inf')
        self._goal_x: Optional[float] = None
        self._goal_y: Optional[float] = None

        # ---- Anti-stuck ----
        self._rotation_accum:   float = 0.0
        self._rotation_ticks:   int   = 0
        self._stuck_count:      int   = 0
        self._force_turn_ticks: int   = 0   # countdown; > 0 → forced turn active

        # ---- Exit cooldown (avoids instant re-entry after leaving WALL_FOLLOW) ----
        self._exit_cooldown: int = 0
        self._wall_side: str = 'RIGHT'

    # ==========================================================
    # PUBLIC API
    # ==========================================================

    def update_scan(self, msg: LaserScan) -> None:
        """
        Ingest a new LaserScan and update per-sector minimum distances.

        NaN and Inf values are filtered before any comparison.
        Must be called from the /scan subscriber callback.

        Args:
            msg: ``sensor_msgs/LaserScan`` from /scan (BEST_EFFORT QoS).
        """
        ranges = msg.ranges
        n      = len(ranges)

        self._front_min       = self._sector_min(ranges, n, *Config.FRONT_SECTOR)
        self._front_right_min = self._sector_min(ranges, n, *Config.FRONT_RIGHT_SECTOR)
        self._front_left_min  = self._sector_min(ranges, n, *Config.FRONT_LEFT_SECTOR)
        self._right_min       = self._sector_min(ranges, n, *Config.RIGHT_SECTOR)
        self._left_min        = self._sector_min(ranges, n, *Config.LEFT_SECTOR)
        self._scan_ready      = True

    def update_force_rotate(self, delta_yaw: float) -> None:
        """
        Update the anti-stuck rotation accumulator.

        Should be called once per control tick (20 Hz) with the absolute
        yaw change since the previous tick.  Only active in WALL_FOLLOW.

        Args:
            delta_yaw: ``|yaw_now − yaw_prev|`` [rad], already normalised ≥ 0.
        """
        if self._state != AvoidState.WALL_FOLLOW:
            # Reset accumulator when not wall-following
            self._rotation_accum = 0.0
            self._rotation_ticks = 0
            return

        self._rotation_accum += abs(delta_yaw)
        self._rotation_ticks += 1

        if self._rotation_ticks >= Config.STUCK_CHECK_TICKS:
            if self._rotation_accum < Config.STUCK_ROTATION_MIN:
                self._stuck_count      += 1
                self._force_turn_ticks  = Config.STUCK_TURN_TICKS
                self._log_warn(
                    f'[AVOID] Anti-stuck triggered '
                    f'(accumulated={self._rotation_accum:.3f} rad < '
                    f'{Config.STUCK_ROTATION_MIN} rad, count={self._stuck_count})'
                )
            else:
                self._stuck_count = max(0, self._stuck_count - 1)

            # Reset period counters
            self._rotation_accum = 0.0
            self._rotation_ticks = 0

    def is_front_danger(self) -> bool:
        """
        Return True when the FRONT sector has a reading below DANGER_DIST.

        Called by the external watchdog timer at 50 Hz.  The caller is
        responsible for publishing a zero-velocity TwistStamped when True.
        """
        return self._scan_ready and (self._front_min < Config.DANGER_DIST)

    def compute(
        self,
        x:    float,
        y:    float,
        yaw:  float,
        wp_x: float,
        wp_y: float,
    ) -> Tuple[VelocityCommand, bool]:
        """
        Compute the Bug2 avoidance velocity command for this control tick.

        Call once per tick (20 Hz) from the mission node control loop.
        When *in_avoidance* is True the caller must publish the returned
        command and suppress its own navigation command.

        Args:
            x:    Robot X position [m] (SLAM or odometry).
            y:    Robot Y position [m].
            yaw:  Robot heading    [rad].
            wp_x: Active goal waypoint X [m].
            wp_y: Active goal waypoint Y [m].

        Returns:
            Tuple ``(cmd, in_avoidance)``:
              · ``cmd``          — VelocityCommand(linear_x, angular_z),
                                   clamped to hard speed limits.
              · ``in_avoidance`` — True  → WALL_FOLLOW active; caller must
                                           use this command.
                                   False → FREE; caller may use its own cmd.
        """
        if not self._scan_ready:
            return VelocityCommand(0.0, 0.0), False

        # Update cached goal
        self._goal_x = wp_x
        self._goal_y = wp_y

        # Tick down cooldown counter
        if self._exit_cooldown > 0:
            self._exit_cooldown -= 1

        if self._state == AvoidState.FREE:
            return self._step_free(x, y, yaw, wp_x, wp_y)

        if self._state == AvoidState.WALL_FOLLOW:
            return self._step_wall_follow(x, y, yaw, wp_x, wp_y)

        # DANGER is handled externally by the watchdog
        return VelocityCommand(0.0, 0.0), False

    def get_state(self) -> AvoidState:
        """Return the current ``AvoidState`` enum value."""
        return self._state

    def reset(self) -> None:
        """
        Reset to FREE state.

        Call this whenever a new waypoint is set (e.g. from mission_node)
        so stale hit-point data from the previous segment is discarded.
        """
        self._state             = AvoidState.FREE
        self._exit_cooldown     = 0
        self._stuck_count       = 0
        self._force_turn_ticks  = 0
        self._rotation_accum    = 0.0
        self._rotation_ticks    = 0
        self._hit_dist_to_goal  = float('inf')
        self._wall_side         = 'RIGHT'
        self._log_info('[AVOID] State reset → FREE')

    # ==========================================================
    # STATE: FREE
    # ==========================================================

    def _step_free(
        self,
        x: float, y: float, yaw: float,
        wp_x: float, wp_y: float,
    ) -> Tuple[VelocityCommand, bool]:
        """
        FREE state handler.

        Monitors the three forward sectors.  When an obstacle is detected
        within WARNING_DIST and the exit cooldown has expired, records the
        hit point, defines the m-line, and transitions to WALL_FOLLOW.

        Returns:
            ``(zero_cmd, False)`` while free.
            ``(first_wall_cmd, True)`` on the tick avoidance starts.
        """
        # Respect post-exit cooldown to avoid oscillation
        if self._exit_cooldown > 0:
            return VelocityCommand(0.0, 0.0), False

        # Obstacle detected in any of the three forward sectors
        front_blocked = (
            self._front_min       < Config.WARNING_DIST or
            self._front_right_min < Config.WARNING_DIST or
            self._front_left_min  < Config.WARNING_DIST
        )

        if not front_blocked:
            return VelocityCommand(0.0, 0.0), False

        # ---- Transition to WALL_FOLLOW ----
        dist_to_goal = math.sqrt((x - wp_x) ** 2 + (y - wp_y) ** 2)

        self._hit_x            = x
        self._hit_y            = y
        self._hit_dist_to_goal = dist_to_goal
        self._state            = AvoidState.WALL_FOLLOW
        self._stuck_count      = 0
        self._rotation_accum   = 0.0
        self._rotation_ticks   = 0
        self._force_turn_ticks = 0

        # Compare free space on both sides (including diagonals)
        space_right = min(self._front_right_min, self._right_min)
        space_left  = min(self._front_left_min, self._left_min)

        if space_right > space_left:
            self._wall_side = 'LEFT'   # Right is more open -> turn Right -> Wall on Left
        else:
            self._wall_side = 'RIGHT'  # Left is more open -> turn Left -> Wall on Right

        self._log_info(
            f'[AVOID] FREE → WALL_FOLLOW ({self._wall_side} SIDE)  '
            f'hit=({x:.2f}, {y:.2f})  '
            f'dist_to_goal={dist_to_goal:.2f} m  '
            f'front_min={self._front_min:.2f} m'
        )

        # Issue the first wall-follow command on this same tick
        return self._step_wall_follow(x, y, yaw, wp_x, wp_y)

    # ==========================================================
    # STATE: WALL_FOLLOW
    # ==========================================================

    def _step_wall_follow(
        self,
        x: float, y: float, yaw: float,
        wp_x: float, wp_y: float,
    ) -> Tuple[VelocityCommand, bool]:
        """
        WALL_FOLLOW state handler — Bug2 wall-following controller.

        Follows the wall on the RIGHT side.  Checks the Bug2 m-line exit
        condition each tick.  Handles the anti-stuck override.

        Returns:
            ``(cmd, True)`` — avoidance is active; caller must use this command.
        """
        # ---- Anti-stuck forced rotation override ----
        if self._force_turn_ticks > 0:
            self._force_turn_ticks -= 1
            turn_vel = Config.TURN_SPEED if self._wall_side == 'RIGHT' else -Config.TURN_SPEED
            cmd = VelocityCommand(
                0.0,
                _clamp(turn_vel, -Config.ANGULAR_MAX, Config.ANGULAR_MAX),
            )
            return cmd, True

        # ---- Check Bug2 exit condition ----
        if self._check_mline_exit(x, y, wp_x, wp_y):
            dist_final = math.sqrt((x - wp_x) ** 2 + (y - wp_y) ** 2)
            self._state         = AvoidState.FREE
            self._exit_cooldown = Config.EXIT_COOLDOWN_TICKS
            self._log_info(
                f'[AVOID] WALL_FOLLOW → FREE (m-line crossed)  '
                f'pos=({x:.2f}, {y:.2f})  '
                f'dist_to_goal={dist_final:.2f} m'
            )
            return VelocityCommand(0.0, 0.0), False

        # ---- Compute wall-follow velocity command ----
        cmd = self._wall_follow_cmd()
        return cmd, True

    def _wall_follow_cmd(self) -> VelocityCommand:
        """
        Proportional wall-following velocity command (dynamic side).
        """
        if self._wall_side == 'LEFT':
            # Case 1: Front blocked — turn right in-place
            if self._front_min < Config.WARNING_DIST:
                return VelocityCommand(
                    0.0,
                    _clamp(-Config.TURN_SPEED, -Config.ANGULAR_MAX, Config.ANGULAR_MAX),
                )

            # Case 2: Front-left corner approaching — reduce speed, bear right
            if self._front_left_min < Config.WARNING_DIST:
                return VelocityCommand(
                    _clamp(Config.WALL_FOLLOW_SPEED * 0.5, 0.0, Config.LINEAR_MAX),
                    _clamp(-Config.TURN_SPEED * Config.CORNER_TURN_FACTOR, -Config.ANGULAR_MAX, Config.ANGULAR_MAX),
                )

            # Case 3: Normal proportional wall-follow
            lateral_error = Config.WALL_FOLLOW_DIST - self._left_min
            angular_z = _clamp(-Config.KP_WALL * lateral_error, -Config.ANGULAR_MAX, Config.ANGULAR_MAX)

            # Case 4: Left wall completely absent — lean left to search for wall
            if self._left_min > Config.SAFE_DIST * 1.5:
                angular_z = _clamp(Config.LOST_WALL_TURN_SPEED, -Config.ANGULAR_MAX, Config.ANGULAR_MAX)
                return VelocityCommand(Config.WALL_FOLLOW_SPEED * 0.6, angular_z)

            return VelocityCommand(
                _clamp(Config.WALL_FOLLOW_SPEED, 0.0, Config.LINEAR_MAX),
                angular_z,
            )

        # RIGHT SIDE
        # Case 1: Front blocked — turn left in-place
        if self._front_min < Config.WARNING_DIST:
            return VelocityCommand(
                0.0,
                _clamp(Config.TURN_SPEED, -Config.ANGULAR_MAX, Config.ANGULAR_MAX),
            )

        # Case 2: Front-right corner approaching — reduce speed, bear left
        if self._front_right_min < Config.WARNING_DIST:
            return VelocityCommand(
                _clamp(
                    Config.WALL_FOLLOW_SPEED * 0.5,
                    0.0, Config.LINEAR_MAX,
                ),
                _clamp(
                    Config.TURN_SPEED * Config.CORNER_TURN_FACTOR,
                    -Config.ANGULAR_MAX, Config.ANGULAR_MAX,
                ),
            )

        # Case 3: Normal proportional wall-follow
        lateral_error = Config.WALL_FOLLOW_DIST - self._right_min
        angular_z     = _clamp(
            Config.KP_WALL * lateral_error,
            -Config.ANGULAR_MAX,
            Config.ANGULAR_MAX,
        )

        # Case 4: Right wall completely absent — lean right to search for wall
        if self._right_min > Config.SAFE_DIST * 1.5:
            angular_z = _clamp(-Config.LOST_WALL_TURN_SPEED, -Config.ANGULAR_MAX, Config.ANGULAR_MAX)
            return VelocityCommand(Config.WALL_FOLLOW_SPEED * 0.6, angular_z)

        return VelocityCommand(
            _clamp(Config.WALL_FOLLOW_SPEED, 0.0, Config.LINEAR_MAX),
            angular_z,
        )

    # ==========================================================
    # BUG2 EXIT CONDITION
    # ==========================================================

    def _check_mline_exit(
        self, x: float, y: float, wp_x: float, wp_y: float
    ) -> bool:
        """
        Evaluate the Bug2 m-line exit condition.

        The robot exits WALL_FOLLOW when **all** of the following hold:

        1. It has travelled at least MIN_TRAVEL_FROM_HIT from the hit point
           (prevents exiting immediately after entry).
        2. Its perpendicular distance to the m-line (segment hit → goal) is
           below M_LINE_THRESHOLD (robot is on the line).
        3. Its distance to the goal is at least DISTANCE_PROGRESS_MIN less
           than the distance from the hit point to the goal (robot is making
           progress — not just circling back past the start of avoidance).

        Args:
            x, y:       Current robot position [m].
            wp_x, wp_y: Active goal waypoint [m].

        Returns:
            True if all exit conditions are satisfied.
        """
        # Condition 0: must have moved away from hit point
        dist_from_hit = math.sqrt(
            (x - self._hit_x) ** 2 + (y - self._hit_y) ** 2
        )
        if dist_from_hit < Config.MIN_TRAVEL_FROM_HIT:
            return False

        # Condition 1: perpendicular distance to m-line
        d_mline = self._mline_distance(x, y, wp_x, wp_y)
        if d_mline > Config.M_LINE_THRESHOLD:
            return False

        # Condition 2: closer to goal than the hit point was
        dist_to_goal = math.sqrt((x - wp_x) ** 2 + (y - wp_y) ** 2)
        progress = self._hit_dist_to_goal - dist_to_goal
        if progress < Config.DISTANCE_PROGRESS_MIN:
            return False

        self._log_info(
            f'[AVOID] Exit conditions met  '
            f'd_mline={d_mline:.3f} m  progress={progress:.3f} m'
        )
        return True

    def _mline_distance(
        self, rx: float, ry: float, gx: float, gy: float
    ) -> float:
        """
        Perpendicular distance from robot (rx, ry) to the m-line.

        The m-line is the infinite line passing through the hit point H
        and the goal G.  Using the cross-product formula:

            d = |(robot − H) × (G − H)| / |G − H|

        Args:
            rx, ry: Robot position [m].
            gx, gy: Goal waypoint  [m].

        Returns:
            Perpendicular distance [m].
        """
        hx, hy = self._hit_x, self._hit_y
        dx     = gx - hx
        dy     = gy - hy
        length = math.sqrt(dx * dx + dy * dy)

        if length < 0.01:
            # Degenerate: hit point ≈ goal (robot is already there)
            return math.sqrt((rx - gx) ** 2 + (ry - gy) ** 2)

        cross = (rx - hx) * dy - (ry - hy) * dx
        return abs(cross) / length

    # ==========================================================
    # LIDAR SECTOR HELPER
    # ==========================================================

    @staticmethod
    def _sector_min(
        ranges: 'list[float]',
        n:      int,
        start_deg: int,
        end_deg:   int,
    ) -> float:
        """
        Minimum valid range reading within a named LiDAR sector.

        Handles wrap-around sectors (e.g. FRONT spans 340°–20°).
        TB3 LiDAR convention: index 0 = front, CCW positive.
        Signed degree input is normalised to [0, 360) via modulo.

        Args:
            ranges:    LaserScan.ranges (full 360 array).
            n:         len(ranges) — typically 360.
            start_deg: Sector start angle [deg, signed].
            end_deg:   Sector end   angle [deg, signed].

        Returns:
            Minimum valid range [m], or ``inf`` when the sector is empty
            or all readings are NaN / Inf.
        """
        start_idx = int(start_deg % 360)   # e.g. -20 → 340
        end_idx   = int(end_deg   % 360)   # e.g.  20 →  20

        minimum = float('inf')

        if start_idx <= end_idx:
            # Contiguous sector — no wrap (e.g. 20°–60°, 270°–300°)
            for i in range(start_idx, end_idx + 1):
                r = ranges[i % n]
                if _valid_range(r) and r < minimum:
                    minimum = r
        else:
            # Sector wraps through 0° / 360° (e.g. 340°–20° for FRONT)
            for i in range(start_idx, n):
                r = ranges[i % n]
                if _valid_range(r) and r < minimum:
                    minimum = r
            for i in range(0, end_idx + 1):
                r = ranges[i % n]
                if _valid_range(r) and r < minimum:
                    minimum = r

        return minimum

    # ==========================================================
    # LOGGING HELPERS
    # ==========================================================

    def _log_info(self, msg: str) -> None:
        """Log at INFO level (ROS2 logger or print fallback)."""
        if self._log:
            self._log.info(msg)
        else:
            print(f'[INFO] {msg}')

    def _log_warn(self, msg: str) -> None:
        """Log at WARN level (ROS2 logger or print fallback)."""
        if self._log:
            self._log.warning(msg)
        else:
            print(f'[WARN] {msg}')


# ============================================================
# STANDALONE SMOKE TEST  (no ROS2 required)
# ============================================================
if __name__ == '__main__':
    import math

    print('=== ObstacleAvoidance — standalone smoke test ===\n')

    # ---- Dummy LaserScan builder ----
    class FakeScan:
        """Minimal LaserScan substitute for offline testing.
        Non-overlapping index ranges so one sector never bleeds into another."""
        def __init__(self, front=2.0, front_right=2.0,
                     front_left=2.0, right=2.0, left=2.0):
            self.ranges = [2.0] * 360
            # FRONT: strictly inside 340–359, 0–20 (avoiding boundary indices)
            for i in list(range(341, 360)) + list(range(0, 20)):
                self.ranges[i] = front
            # FRONT_RIGHT: 301–339
            for i in range(301, 340):
                self.ranges[i] = front_right
            # FRONT_LEFT: 21–59
            for i in range(21, 60):
                self.ranges[i] = front_left
            # RIGHT: 271–299 (avoids 300 boundary with FRONT_RIGHT)
            for i in range(271, 300):
                self.ranges[i] = right
            # LEFT: 61–89
            for i in range(61, 90):
                self.ranges[i] = left

    avoider = ObstacleAvoidance(logger=None)

    # ---- Test 1: FREE state, clear path ----
    scan = FakeScan(front=2.0, front_right=2.0, front_left=2.0,
                    right=0.35, left=2.0)
    avoider.update_scan(scan)
    cmd, active = avoider.compute(0.0, 0.0, 0.0, 3.0, 0.0)
    assert not active, 'Should be FREE when path is clear'
    assert avoider.get_state() == AvoidState.FREE
    print('Test 1 PASS — clear path → FREE')

    # ---- Test 2: Obstacle ahead → enter WALL_FOLLOW ----
    scan_blocked = FakeScan(front=0.20, front_right=0.20, front_left=2.0,
                            right=0.35, left=2.0)
    avoider.update_scan(scan_blocked)
    cmd, active = avoider.compute(0.0, 0.0, 0.0, 3.0, 0.0)
    assert active, 'Should be WALL_FOLLOW when obstacle ahead'
    assert avoider.get_state() == AvoidState.WALL_FOLLOW
    print(f'Test 2 PASS — obstacle → WALL_FOLLOW  cmd={cmd}')

    # ---- Test 3: is_front_danger ----
    scan_danger = FakeScan(front=0.10)
    avoider2 = ObstacleAvoidance()
    avoider2.update_scan(scan_danger)
    assert avoider2.is_front_danger(), 'DANGER not triggered at 0.10 m'
    print('Test 3 PASS — is_front_danger at 0.10 m')

    scan_safe = FakeScan(front=0.50)
    avoider2.update_scan(scan_safe)
    assert not avoider2.is_front_danger(), 'DANGER false positive at 0.50 m'
    print('Test 4 PASS — no danger at 0.50 m')

    # ---- Test 5: m-line distance calculation ----
    avoider3 = ObstacleAvoidance()
    avoider3._hit_x = 0.0
    avoider3._hit_y = 0.0
    # m-line goes along X axis (hit 0,0 → goal 5,0)
    d = avoider3._mline_distance(1.0, 1.0, 5.0, 0.0)
    assert abs(d - 1.0) < 1e-9, f'Expected 1.0, got {d}'
    d2 = avoider3._mline_distance(2.0, 0.0, 5.0, 0.0)
    assert abs(d2 - 0.0) < 1e-9, f'Expected 0.0 on line, got {d2}'
    print('Test 5 PASS — m-line distance correct')

    print('\nAll tests passed.')