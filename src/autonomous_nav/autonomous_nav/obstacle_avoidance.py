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
    · Follow the wall on the chosen side (proportional lateral control).
    · Turn in-place when FRONT is blocked.
    · Exit conditions (any of them triggers FREE):
        A. Line-of-sight exit (priority):
           Minimum LiDAR range in a cone around the goal bearing exceeds
           dist(robot, goal) + LOS_CLEARANCE_MARGIN, i.e. the robot is
           no longer behind the followed wall from the goal's perspective
           ("robot between wall and goal" → drop avoidance, head to goal).
        B. Bug2 m-line exit:
           1. Perpendicular distance to m-line  < M_LINE_THRESHOLD, AND
           2. dist(robot, goal) < dist(H, goal) − DISTANCE_PROGRESS_MIN.
    · Anti-stuck: if accumulated rotation is too low over STUCK_CHECK_TICKS,
      apply a forced in-place rotation for STUCK_TURN_TICKS.

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

    # --- Bug2 m-line exit conditions ---
    M_LINE_THRESHOLD      = 0.15   # m   — perpendicular dist to m-line for "on line"
    DISTANCE_PROGRESS_MIN = 0.15   # m   — min extra progress toward goal to exit
    MIN_TRAVEL_FROM_HIT   = 0.20   # m   — min distance from hit point before checking exit

    # --- Line-of-sight early exit ("robot between wall and goal") ---
    LOS_CONE_HALF_DEG     = 15     # deg — half-aperture of cone around goal bearing
    LOS_CLEARANCE_MARGIN  = 0.20   # m   — LiDAR must see ≥ (dist_to_goal + this)
    LOS_MIN_DIST_TO_GOAL  = 0.15   # m   — skip LoS check when already at the goal

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


def _normalize_angle(a: float) -> float:
    """Normalise angle to (-π, π]."""
    while a >  math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a


# ============================================================
# MAIN CLASS
# ============================================================
class ObstacleAvoidance:
    """
    Bug2 reactive obstacle avoidance — pure logic module (no ROS2 Node).

    Instantiated once by the mission node and fed LiDAR + pose data each
    control tick.  Returns velocity commands and an *in_avoidance* flag
    that tells the caller whether to suppress its own navigation command.

    Wall-follow side: dynamic (RIGHT / LEFT) — chosen at entry based on
    which side of the robot has more free space.
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

        # ---- Latest raw LiDAR ranges (for arbitrary-bearing queries) ----
        self._latest_ranges: Optional[list] = None
        self._n_ranges: int                 = 0

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

        Also caches the raw ranges array so the line-of-sight exit check
        can query arbitrary bearings on demand.

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

        # Cache raw ranges for arbitrary-bearing queries (LoS exit)
        self._latest_ranges = ranges
        self._n_ranges      = n

        self._scan_ready    = True

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

        Exit checks, in priority order:
          1. Line-of-sight: the robot has a clear direct path to the goal
             (i.e. it is between the followed wall and the goal, so the
             wall no longer obstructs).  Exit immediately.
          2. Bug2 m-line: standard Bug2 exit on regaining the start–goal
             line with net progress.
          3. Anti-stuck forced rotation override (does NOT exit).

        Returns:
            ``(cmd, True)``  — avoidance is active; caller must use this cmd.
            ``(zero, False)`` on the tick it exits to FREE.
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

        # ---- Priority exit: line of sight to goal is clear ----
        # (Robot is between the followed wall and the waypoint.)
        if self._check_los_exit(x, y, yaw, wp_x, wp_y):
            dist_final = math.sqrt((x - wp_x) ** 2 + (y - wp_y) ** 2)
            self._state         = AvoidState.FREE
            self._exit_cooldown = Config.EXIT_COOLDOWN_TICKS
            self._log_info(
                f'[AVOID] WALL_FOLLOW → FREE (line of sight to goal clear)  '
                f'pos=({x:.2f}, {y:.2f})  '
                f'dist_to_goal={dist_final:.2f} m'
            )
            return VelocityCommand(0.0, 0.0), False

        # ---- Bug2 m-line exit condition ----
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
    # LINE-OF-SIGHT EXIT ("robot between wall and goal")
    # ==========================================================

    def _check_los_exit(
        self, x: float, y: float, yaw: float,
        wp_x: float, wp_y: float,
    ) -> bool:
        """
        Line-of-sight early exit condition.

        Returns True when the robot currently has a clear direct path to
        the goal waypoint — i.e. the nearest LiDAR return inside a cone
        of ±LOS_CONE_HALF_DEG around the goal bearing lies *beyond* the
        goal (with LOS_CLEARANCE_MARGIN of slack).

        This captures the case where the robot is "between the followed
        wall and the goal": the wall is on one side and the goal is on
        the other, with no obstacle between them.

        Gates:
          · Must have moved at least MIN_TRAVEL_FROM_HIT from the hit
            point, to avoid exiting on the very first tick of avoidance.
          · Must be at least LOS_MIN_DIST_TO_GOAL from the goal (if we
            are already there, the navigator will close the loop).

        Args:
            x, y:       Robot position [m].
            yaw:        Robot heading [rad].
            wp_x, wp_y: Goal waypoint [m].

        Returns:
            True if the line of sight to the goal is clear and the gates
            pass; False otherwise.
        """
        # ---- Gate: must have travelled away from the hit point ----
        dist_from_hit = math.sqrt(
            (x - self._hit_x) ** 2 + (y - self._hit_y) ** 2
        )
        if dist_from_hit < Config.MIN_TRAVEL_FROM_HIT:
            return False

        # ---- Gate: must not already be at the goal ----
        dist_to_goal = math.sqrt((x - wp_x) ** 2 + (y - wp_y) ** 2)
        if dist_to_goal < Config.LOS_MIN_DIST_TO_GOAL:
            return False

        # ---- Bearing to goal in robot frame (0=front, CCW+) ----
        goal_bearing = _normalize_angle(
            math.atan2(wp_y - y, wp_x - x) - yaw
        )

        # ---- Minimum LiDAR range inside cone around goal bearing ----
        half_cone_rad = math.radians(Config.LOS_CONE_HALF_DEG)
        min_range = self._range_min_in_cone(goal_bearing, half_cone_rad)

        # ---- Clear when the cone "sees past" the goal ----
        required = dist_to_goal + Config.LOS_CLEARANCE_MARGIN
        return min_range > required

    def _range_min_in_cone(
        self, center_rad: float, half_cone_rad: float,
    ) -> float:
        """
        Minimum valid LiDAR range within a cone around *center_rad*.

        Robot-frame convention: 0 rad = front, CCW positive.  Matches the
        TB3 LDS convention (``ranges[0]`` is directly ahead, index grows
        counter-clockwise).  Wrap-around (cones that straddle 0° / 360°)
        is handled the same way as ``_sector_min``.

        Args:
            center_rad:    Cone centre angle (robot frame) [rad].
            half_cone_rad: Half-aperture of the cone        [rad].

        Returns:
            Minimum valid range [m] in the cone, or ``inf`` if the cone
            contains no valid readings or no scan has been received yet.
        """
        if (not self._scan_ready or
                self._latest_ranges is None or self._n_ranges == 0):
            return float('inf')

        ranges = self._latest_ranges
        n      = self._n_ranges

        start_deg = math.degrees(center_rad - half_cone_rad)
        end_deg   = math.degrees(center_rad + half_cone_rad)

        # Use floor/ceil so the cone interval is fully covered
        start_idx = int(math.floor(start_deg)) % 360
        end_idx   = int(math.ceil(end_deg))    % 360

        minimum = float('inf')

        if start_idx <= end_idx:
            # Contiguous sector — no wrap
            for i in range(start_idx, end_idx + 1):
                r = ranges[i % n]
                if _valid_range(r) and r < minimum:
                    minimum = r
        else:
            # Sector wraps through 0° / 360°
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
    # BUG2 M-LINE EXIT CONDITION
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
        start_idx = int(start_deg) % 360   # e.g. -20 → 340
        end_idx   = int(end_deg)   % 360   # e.g.  20 →  20

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
    print('=== ObstacleAvoidance — standalone smoke test ===\n')

    # ---- Dummy LaserScan builder ----
    class FakeScan:
        """Minimal LaserScan substitute for offline testing.
        Non-overlapping index ranges so one sector never bleeds into another."""
        def __init__(self, front=2.0, front_right=2.0,
                     front_left=2.0, right=2.0, left=2.0, back=2.0):
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
            # BACK: 91–270 (everything else to the rear)
            for i in range(91, 271):
                self.ranges[i] = back

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

    # ---- Test 6: Line-of-sight early exit ----
    # Scenario: robot is wall-following on RIGHT. Goal is behind (x=-1.5).
    # Right side has the followed wall (0.16m). Back is clear at 3.0m.
    # Goal bearing ≈ 180°. LoS cone around 180° → reads 3.0m >
    # dist_to_goal (1.5) + margin (0.2) = 1.7m → CLEAR → exit.
    avoider4 = ObstacleAvoidance(logger=None)
    scan_followed = FakeScan(front=2.0, front_right=0.35, front_left=2.0,
                             right=0.16, left=2.0, back=3.0)
    avoider4.update_scan(scan_followed)
    # Force state to WALL_FOLLOW with a hit far enough to pass the gate
    avoider4._state = AvoidState.WALL_FOLLOW
    avoider4._hit_x = -3.0   # hit point far behind robot
    avoider4._hit_y =  0.0
    avoider4._hit_dist_to_goal = 1.0
    avoider4._wall_side = 'RIGHT'
    # Robot at (0,0,0), goal at (-1.5,0) → goal behind (bearing ≈ 180°)
    cmd, active = avoider4.compute(0.0, 0.0, 0.0, -1.5, 0.0)
    assert avoider4.get_state() == AvoidState.FREE, (
        'LoS exit should fire: back sector is clear (3.0m) > '
        'dist_to_goal (1.5) + margin (0.2)'
    )
    assert not active
    print('Test 6 PASS — LoS exit fires when goal is behind with clear back')

    # ---- Test 7: LoS does NOT fire when obstacle between robot and goal ----
    # Hit point offset from the robot→goal line so the m-line exit cannot
    # fire (robot is not on the m-line). This isolates the LoS check.
    avoider5 = ObstacleAvoidance(logger=None)
    # Obstacle directly in front within cone — path to goal in front is blocked
    scan_blocked_front = FakeScan(front=0.40, front_right=0.20, front_left=2.0,
                                  right=0.16, left=2.0, back=2.0)
    avoider5.update_scan(scan_blocked_front)
    avoider5._state = AvoidState.WALL_FOLLOW
    avoider5._hit_x = 0.0      # offset from m-line axis
    avoider5._hit_y = 0.5
    avoider5._hit_dist_to_goal = 3.04  # hypot(3, 0.5)
    avoider5._wall_side = 'RIGHT'
    # Goal ahead at (3,0): bearing 0°. Front reads 0.40m < dist_to_goal=3m
    # → LoS blocked. m-line distance ≈ 0.49m > 0.15 → m-line blocked too.
    cmd, active = avoider5.compute(0.0, 0.0, 0.0, 3.0, 0.0)
    assert avoider5.get_state() == AvoidState.WALL_FOLLOW, (
        'LoS should NOT fire: front=0.40m < dist_to_goal=3.0m'
    )
    assert active
    print('Test 7 PASS — LoS does not fire when goal is obstructed')

    # ---- Test 8: LoS gated by MIN_TRAVEL_FROM_HIT ----
    avoider6 = ObstacleAvoidance(logger=None)
    scan_clear = FakeScan(front=2.0, front_right=0.35, front_left=2.0,
                          right=0.16, left=2.0, back=2.0)
    avoider6.update_scan(scan_clear)
    avoider6._state = AvoidState.WALL_FOLLOW
    avoider6._hit_x = 0.0     # robot still at hit point
    avoider6._hit_y = 0.0
    avoider6._hit_dist_to_goal = 1.0
    avoider6._wall_side = 'RIGHT'
    cmd, active = avoider6.compute(0.0, 0.0, 0.0, 1.0, 0.0)
    assert avoider6.get_state() == AvoidState.WALL_FOLLOW, (
        'LoS should be gated by MIN_TRAVEL_FROM_HIT'
    )
    print('Test 8 PASS — LoS gated by MIN_TRAVEL_FROM_HIT')

    print('\nAll tests passed.')