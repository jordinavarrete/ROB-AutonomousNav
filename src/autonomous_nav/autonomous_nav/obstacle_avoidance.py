#!/usr/bin/env python3
"""
obstacle_avoidance.py — LiDAR processing, sector classification and wall-follow controller.

Responsibilities:
  - Parse raw /scan data into named angular sectors
  - Classify each sector as SAFE / WARNING / DANGER
  - Provide a wall-follow state machine for complex obstacle scenarios
  - Expose a simple query API consumed by mission_node.py

This module is a pure logic class (no ROS2 Node inheritance).
It is instantiated by mission_node.py, which passes in LiDAR messages
and retrieves velocity corrections.

State machine:
    NORMAL       → nominal navigation, no obstacles in WARNING range
    AVOID_ROTATE → obstacle in FRONT DANGER/WARNING; rotate toward free side
    WALL_FOLLOW  → lateral tracking along the obstacle wall
    RECOVERING   → front clear AND heading to waypoint improving → exit to NORMAL

Anti-stuck: if robot has not moved > STUCK_DIST_M in STUCK_TIME_S seconds,
            force a 180° rotation (state FORCE_ROTATE).
"""

# ============================================================
# CONFIGURATION — adjust these values for lab testing
# ============================================================
class Config:
    # Alert thresholds (metres)
    DANGER_DIST         = 0.25   # immediate stop / hard avoidance
    WARNING_DIST        = 0.45   # slow down, prepare avoidance
    SAFE_DIST           = 0.60   # sector fully clear

    # Sector boundaries (degrees, robot-front = 0°, CCW positive)
    #   stored as (min_deg, max_deg) inclusive
    SECTOR_FRONT_DEG        = (-25.0,  25.0)
    SECTOR_FRONT_LEFT_DEG   = ( 25.0,  70.0)
    SECTOR_FRONT_RIGHT_DEG  = (-70.0, -25.0)
    SECTOR_LEFT_DEG         = ( 70.0, 110.0)
    SECTOR_RIGHT_DEG        = (-110.0, -70.0)

    # Wall-follow parameters
    WALL_FOLLOW_DIST    = 0.35   # m — target lateral distance from wall
    WALL_FOLLOW_SPEED   = 0.10   # m/s — forward speed while wall-following
    KP_WALL             = 0.8    # proportional gain for lateral error
    MAX_WALL_ANGULAR    = 0.8    # rad/s — cap on wall-follow angular correction

    # Rotation speeds during avoidance
    AVOID_ANGULAR_SPEED = 0.50   # rad/s — speed when rotating away from obstacle
    AVOID_LINEAR_SPEED  = 0.05   # m/s — creep forward during wall-follow

    # Heading-improvement window for RECOVERING exit condition
    HEADING_WINDOW_S    = 2.0    # seconds of improvement required before exiting

    # Anti-stuck parameters
    STUCK_DIST_M        = 0.05   # m — minimum displacement to not be considered stuck
    STUCK_TIME_S        = 10.0    # s — time window for stuck detection
    FORCE_ROTATE_ANGLE  = 3.14159  # rad ≈ 180°
    FORCE_ROTATE_SPEED  = 0.50   # rad/s

    # Minimum valid LiDAR range (metres) — below this, reading is likely noise
    MIN_VALID_RANGE     = 0.12


# ============================================================
# IMPORTS
# ============================================================
import math
import time
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple


# ============================================================
# ENUMERATIONS
# ============================================================
class AlertLevel(Enum):
    """Classification of a LiDAR sector based on minimum detected distance."""
    SAFE    = auto()
    WARNING = auto()
    DANGER  = auto()


class AvoidState(Enum):
    """Wall-follow / avoidance state machine states."""
    NORMAL       = auto()   # regular navigation, no avoidance needed
    AVOID_ROTATE = auto()   # rotating in place toward free side
    WALL_FOLLOW  = auto()   # tracking lateral wall while moving forward
    RECOVERING   = auto()   # front clear, heading improving — preparing to exit
    FORCE_ROTATE = auto()   # anti-stuck: forced 180° rotation


# ============================================================
# DATA STRUCTURES
# ============================================================
class SectorData:
    """
    Holds the processed state of a single LiDAR sector.

    Attributes:
        name:        Human-readable sector name.
        min_dist:    Minimum valid range in this sector [m].
        mean_dist:   Mean of valid ranges in this sector [m].
        alert:       AlertLevel classification.
        n_points:    Number of valid scan points in this sector.
    """

    __slots__ = ('name', 'min_dist', 'mean_dist', 'alert', 'n_points')

    def __init__(self, name: str) -> None:
        self.name      = name
        self.min_dist  = float('inf')
        self.mean_dist = float('inf')
        self.alert     = AlertLevel.SAFE
        self.n_points  = 0

    def classify(self) -> None:
        """Set alert level based on min_dist and Config thresholds."""
        if self.min_dist < Config.DANGER_DIST:
            self.alert = AlertLevel.DANGER
        elif self.min_dist < Config.WARNING_DIST:
            self.alert = AlertLevel.WARNING
        else:
            self.alert = AlertLevel.SAFE

    def __repr__(self) -> str:
        return (f'SectorData({self.name}: min={self.min_dist:.3f}m '
                f'alert={self.alert.name} pts={self.n_points})')


class VelocityCommand:
    """Simple container for a linear/angular velocity pair."""

    __slots__ = ('linear_x', 'angular_z')

    def __init__(self, linear_x: float = 0.0, angular_z: float = 0.0) -> None:
        self.linear_x  = linear_x
        self.angular_z = angular_z

    @property
    def is_stop(self) -> bool:
        """True if both velocities are zero."""
        return self.linear_x == 0.0 and self.angular_z == 0.0

    def __repr__(self) -> str:
        return f'VelocityCommand(lin={self.linear_x:.3f}, ang={self.angular_z:.3f})'


# ============================================================
# MAIN CLASS
# ============================================================
class ObstacleAvoidance:
    """
    LiDAR-based obstacle detection and wall-follow avoidance controller.

    Usage (from mission_node.py):
        avoider = ObstacleAvoidance(logger=self.get_logger())
        # on each /scan callback:
        avoider.update_scan(scan_msg)
        # on each control tick:
        cmd, in_avoidance = avoider.compute(
            current_x, current_y, current_yaw,
            waypoint_x, waypoint_y
        )
        # emergency check (call from 50 Hz watchdog):
        if avoider.is_front_danger():
            publish_stop()
    """

    def __init__(self, logger=None) -> None:
        """
        Initialise the avoidance controller.

        Args:
            logger: A ROS2 logger (rclpy.impl.rcutils_logger.RcutilsLogger).
                    If None, falls back to print() for standalone testing.
        """
        self._log = logger

        # Sector containers — keyed by sector name
        self._sectors: Dict[str, SectorData] = {
            'FRONT':       SectorData('FRONT'),
            'FRONT_LEFT':  SectorData('FRONT_LEFT'),
            'FRONT_RIGHT': SectorData('FRONT_RIGHT'),
            'LEFT':        SectorData('LEFT'),
            'RIGHT':       SectorData('RIGHT'),
        }

        # Raw LiDAR snapshot (list of (angle_rad, range_m) for valid readings)
        self._valid_points: List[Tuple[float, float]] = []

        # Avoidance state machine
        self._state      = AvoidState.NORMAL
        self._wall_side  = 'LEFT'   # side we are wall-following ('LEFT' or 'RIGHT')

        # Heading improvement tracking for RECOVERING exit
        self._heading_history: List[Tuple[float, float]] = []  # (timestamp, abs_angle_error)

        # Anti-stuck tracking
        self._last_move_time  = time.time()
        self._last_check_pos  = (0.0, 0.0)
        self._stuck_rotating  = False
        self._force_rot_accumulated = 0.0   # radians rotated so far

        # Scan metadata
        self._scan_stamp  = 0.0
        self._range_min   = Config.MIN_VALID_RANGE
        self._range_max   = 3.5

    # ----------------------------------------------------------
    # Public API — called from mission_node.py
    # ----------------------------------------------------------

    def update_scan(self, scan_msg) -> None:
        """
        Ingest a new LaserScan message and recompute all sector states.

        This is the only method that reads the ROS2 message type.
        All other methods work on the pre-processed internal state.

        Args:
            scan_msg: sensor_msgs/LaserScan message.
        """
        self._range_min  = scan_msg.range_min
        self._range_max  = scan_msg.range_max
        self._scan_stamp = time.time()

        valid: List[Tuple[float, float]] = []

        # Reset sectors
        for s in self._sectors.values():
            s.min_dist  = float('inf')
            s.mean_dist = float('inf')
            s.n_points  = 0

        sector_sums: Dict[str, float] = {k: 0.0 for k in self._sectors}

        for i, r in enumerate(scan_msg.ranges):
            # --- NaN / Inf guard ---
            if math.isnan(r) or math.isinf(r):
                continue
            if r < self._range_min or r > self._range_max:
                continue

            angle_rad = scan_msg.angle_min + i * scan_msg.angle_increment
            angle_deg = math.degrees(self._normalize_angle(angle_rad))

            valid.append((angle_rad, r))

            # Assign to sector
            sector_name = self._angle_to_sector(angle_deg)
            if sector_name is None:
                continue

            s = self._sectors[sector_name]
            if r < s.min_dist:
                s.min_dist = r
            sector_sums[sector_name] += r
            s.n_points += 1

        # Compute means and classify
        for name, s in self._sectors.items():
            if s.n_points > 0:
                s.mean_dist = sector_sums[name] / s.n_points
            else:
                s.min_dist  = float('inf')
                s.mean_dist = float('inf')
            s.classify()

        self._valid_points = valid

    def compute(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        waypoint_x: float,
        waypoint_y: float,
    ) -> Tuple[VelocityCommand, bool]:
        """
        Run the avoidance state machine and return a velocity command.

        Returns:
            (VelocityCommand, in_avoidance: bool)
            in_avoidance is True whenever the state is not NORMAL,
            signalling mission_node.py to suspend its own navigation.
        """
        self._update_stuck(robot_x, robot_y)
        self._update_state(robot_x, robot_y, robot_yaw, waypoint_x, waypoint_y)
        cmd = self._compute_command(robot_yaw, waypoint_x, waypoint_y)
        in_avoidance = self._state != AvoidState.NORMAL
        return cmd, in_avoidance

    def is_front_danger(self) -> bool:
        """
        Return True if the FRONT sector is at DANGER level.

        Used by the 50 Hz safety watchdog in mission_node.py.
        """
        return self._sectors['FRONT'].alert == AlertLevel.DANGER

    def get_sectors(self) -> Dict[str, SectorData]:
        """Return a snapshot of all sector states (read-only reference)."""
        return self._sectors

    def get_state(self) -> AvoidState:
        """Return the current avoidance state."""
        return self._state

    def reset(self) -> None:
        """Force-reset to NORMAL state (call after waypoint arrival)."""
        self._state = AvoidState.NORMAL
        self._heading_history.clear()
        self._log_info('ObstacleAvoidance reset to NORMAL')

    # ----------------------------------------------------------
    # State machine transitions
    # ----------------------------------------------------------

    def _update_state(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        wp_x: float,
        wp_y: float,
    ) -> None:
        """Evaluate transitions between avoidance states."""

        # Anti-stuck takes highest priority (below the watchdog stop)
        if self._state == AvoidState.FORCE_ROTATE:
        	# Exit when 180° has been accumulated
        	if self._force_rot_accumulated >= Config.FORCE_ROTATE_ANGLE:
        		self._force_rot_accumulated = 0.0
        		# Reset anti-stuck tracker so no re-trigger immediately
        		self._last_move_time = time.time()
        		self._last_check_pos = (robot_x, robot_y)
        		self._transition(AvoidState.NORMAL)
        	return

        front   = self._sectors['FRONT']
        f_left  = self._sectors['FRONT_LEFT']
        f_right = self._sectors['FRONT_RIGHT']

        if self._state == AvoidState.NORMAL:
            if front.alert in (AlertLevel.DANGER, AlertLevel.WARNING):
                self._wall_side = self._choose_wall_side()
                self._transition(AvoidState.AVOID_ROTATE)

        elif self._state == AvoidState.AVOID_ROTATE:
            # Exit rotate when front is clear
            if front.alert == AlertLevel.SAFE:
                self._transition(AvoidState.WALL_FOLLOW)

        elif self._state == AvoidState.WALL_FOLLOW:
            if front.alert in (AlertLevel.DANGER, AlertLevel.WARNING):
                # Hit another obstacle — re-rotate
                self._wall_side = self._choose_wall_side()
                self._transition(AvoidState.AVOID_ROTATE)
            elif front.alert == AlertLevel.SAFE:
                self._transition(AvoidState.RECOVERING)

        elif self._state == AvoidState.RECOVERING:
            if front.alert in (AlertLevel.DANGER, AlertLevel.WARNING):
                self._wall_side = self._choose_wall_side()
                self._transition(AvoidState.AVOID_ROTATE)
                return
            # Track heading improvement
            abs_err = abs(self._angle_to_waypoint(robot_x, robot_y, robot_yaw, wp_x, wp_y))
            now = time.time()
            self._heading_history.append((now, abs_err))
            # Prune old entries
            cutoff = now - Config.HEADING_WINDOW_S
            self._heading_history = [(t, e) for t, e in self._heading_history if t >= cutoff]

            if self._heading_is_improving():
                self._heading_history.clear()
                self._transition(AvoidState.NORMAL)

    def _compute_command(
        self,
        robot_yaw: float,
        wp_x: float,
        wp_y: float,
    ) -> VelocityCommand:
        """Map the current state to a concrete VelocityCommand."""

        if self._state == AvoidState.NORMAL:
            return VelocityCommand(0.0, 0.0)   # navigation.py handles motion

        elif self._state == AvoidState.AVOID_ROTATE:
            direction = 1.0 if self._wall_side == 'LEFT' else -1.0
            return VelocityCommand(0.0, direction * Config.AVOID_ANGULAR_SPEED)

        elif self._state == AvoidState.WALL_FOLLOW:
            return self._wall_follow_command()

        elif self._state == AvoidState.RECOVERING:
            return self._wall_follow_command()   # keep wall-following until safe

        elif self._state == AvoidState.FORCE_ROTATE:
            return VelocityCommand(0.0, Config.FORCE_ROTATE_SPEED)

        return VelocityCommand(0.0, 0.0)

    # ----------------------------------------------------------
    # Wall-follow lateral control
    # ----------------------------------------------------------

    def _wall_follow_command(self) -> VelocityCommand:
        """
        Compute forward + lateral-correction command for wall-following.

        Lateral error = lateral_sector_min_dist − WALL_FOLLOW_DIST
        angular_correction = Kp_wall * lateral_error
        (positive error → too far from wall → turn toward wall)
        """
        if self._wall_side == 'LEFT':
            lateral_dist = self._sectors['LEFT'].min_dist
            sign = 1.0   # positive angular = turn left = toward left wall
        else:
            lateral_dist = self._sectors['RIGHT'].min_dist
            sign = -1.0  # negative angular = turn right = toward right wall

        if math.isinf(lateral_dist):
            # No wall on tracking side — just creep forward
            return VelocityCommand(Config.WALL_FOLLOW_SPEED, 0.0)

        lateral_error = lateral_dist - Config.WALL_FOLLOW_DIST
        angular_corr  = sign * Config.KP_WALL * lateral_error
        angular_corr  = max(-Config.MAX_WALL_ANGULAR,
                            min(Config.MAX_WALL_ANGULAR, angular_corr))

        return VelocityCommand(Config.WALL_FOLLOW_SPEED, angular_corr)

    # ----------------------------------------------------------
    # Helper: side selection
    # ----------------------------------------------------------

    def _choose_wall_side(self) -> str:
        """
        Choose the wall-follow side as the side with MORE free space
        (higher minimum distance in the lateral sector).

        Returns: 'LEFT' or 'RIGHT'
        """
        left_dist  = self._sectors['LEFT'].min_dist
        right_dist = self._sectors['RIGHT'].min_dist

        # If one side is completely clear (inf) prefer it
        if math.isinf(left_dist) and not math.isinf(right_dist):
            return 'LEFT'
        if math.isinf(right_dist) and not math.isinf(left_dist):
            return 'RIGHT'

        return 'LEFT' if left_dist >= right_dist else 'RIGHT'

    # ----------------------------------------------------------
    # Helper: anti-stuck
    # ----------------------------------------------------------

    def _update_stuck(self, robot_x: float, robot_y: float) -> None:
        """
        Detect if the robot has not moved for STUCK_TIME_S seconds and,
        if so, transition to FORCE_ROTATE.
        """
        now = time.time()
        dx = robot_x - self._last_check_pos[0]
        dy = robot_y - self._last_check_pos[1]
        dist = math.sqrt(dx * dx + dy * dy)

        if dist > Config.STUCK_DIST_M:
            self._last_move_time  = now
            self._last_check_pos  = (robot_x, robot_y)

        if (now - self._last_move_time > Config.STUCK_TIME_S
                and self._state != AvoidState.FORCE_ROTATE):
            self._log_warn('Anti-stuck triggered: forcing 180° rotation')
            self._force_rot_accumulated = 0.0
            self._transition(AvoidState.FORCE_ROTATE)

    def update_force_rotate(self, delta_yaw: float) -> None:
        """
        Accumulate rotated angle during FORCE_ROTATE state.

        Call this from mission_node.py every control tick, passing the
        absolute yaw change since last tick (always positive).

        Args:
            delta_yaw: Absolute yaw change in radians this tick.
        """
        if self._state == AvoidState.FORCE_ROTATE:
            self._force_rot_accumulated += abs(delta_yaw)

    # ----------------------------------------------------------
    # Helper: heading improvement check
    # ----------------------------------------------------------

    def _heading_is_improving(self) -> bool:
        """
        Return True if the heading error has been monotonically decreasing
        over the last HEADING_WINDOW_S seconds.

        Requires at least 3 data points.
        """
        if len(self._heading_history) < 3:
            return False
        errors = [e for _, e in self._heading_history]
        # Improving = last sample is smaller than the first in the window
        return errors[-1] < errors[0] - 0.05   # 0.05 rad hysteresis

    # ----------------------------------------------------------
    # Helper: angle to waypoint (in robot frame)
    # ----------------------------------------------------------

    @staticmethod
    def _angle_to_waypoint(
        rx: float, ry: float, ryaw: float,
        wx: float, wy: float,
    ) -> float:
        """
        Compute the signed angle from the robot's current heading to the
        direction toward waypoint (wx, wy), normalised to [-π, π].
        """
        desired_yaw = math.atan2(wy - ry, wx - rx)
        error = desired_yaw - ryaw
        # Normalise
        while error >  math.pi: error -= 2 * math.pi
        while error < -math.pi: error += 2 * math.pi
        return error

    # ----------------------------------------------------------
    # Helper: sector mapping
    # ----------------------------------------------------------

    @staticmethod
    def _angle_to_sector(angle_deg: float) -> Optional[str]:
        """
        Map a bearing in degrees (robot-front = 0°, CCW positive, range [-180, 180])
        to a sector name, or None if it falls outside all defined sectors.
        """
        a = angle_deg
        fl_min, fl_max = Config.SECTOR_FRONT_LEFT_DEG
        fr_min, fr_max = Config.SECTOR_FRONT_RIGHT_DEG
        f_min,  f_max  = Config.SECTOR_FRONT_DEG
        l_min,  l_max  = Config.SECTOR_LEFT_DEG
        r_min,  r_max  = Config.SECTOR_RIGHT_DEG

        if f_min  <= a <= f_max:  return 'FRONT'
        if fl_min <= a <= fl_max: return 'FRONT_LEFT'
        if fr_min <= a <= fr_max: return 'FRONT_RIGHT'
        if l_min  <= a <= l_max:  return 'LEFT'
        if r_min  <= a <= r_max:  return 'RIGHT'
        return None

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalise an angle in radians to [-π, π]."""
        while angle >  math.pi: angle -= 2 * math.pi
        while angle < -math.pi: angle += 2 * math.pi
        return angle

    # ----------------------------------------------------------
    # Logging helpers
    # ----------------------------------------------------------

    def _transition(self, new_state: AvoidState) -> None:
        """Log and perform a state transition."""
        if new_state != self._state:
            self._log_info(
                f'ObstacleAvoidance: {self._state.name} → {new_state.name}'
            )
            self._state = new_state

    def _log_info(self, msg: str) -> None:
        if self._log:
            self._log.info(msg)
        else:
            print(f'[INFO] {msg}')

    def _log_warn(self, msg: str) -> None:
        if self._log:
            self._log.warning(msg)
        else:
            print(f'[WARN] {msg}')


# ============================================================
# STANDALONE TEST (no ROS2 required)
# ============================================================
if __name__ == '__main__':
    import math

    class FakeScan:
        """Minimal LaserScan mock for unit testing."""
        def __init__(self, n=360, obstacle_angle_deg=0, obstacle_dist=0.30):
            self.range_min = 0.12
            self.range_max = 3.50
            self.angle_min = -math.pi
            self.angle_increment = 2 * math.pi / n
            self.ranges = [3.0] * n
            # Plant one obstacle
            idx = int((math.radians(obstacle_angle_deg) - self.angle_min)
                      / self.angle_increment) % n
            for di in range(-3, 4):
                self.ranges[(idx + di) % n] = obstacle_dist

    avoider = ObstacleAvoidance()

    print('=== Test 1: obstacle dead ahead at 0.20 m (DANGER) ===')
    avoider.update_scan(FakeScan(obstacle_angle_deg=0, obstacle_dist=0.20))
    for name, s in avoider.get_sectors().items():
        if s.n_points > 0:
            print(f'  {name:15s} min={s.min_dist:.3f}m  {s.alert.name}')
    print(f'  is_front_danger() → {avoider.is_front_danger()}')
    cmd, avoid = avoider.compute(0, 0, 0, 5, 5)
    print(f'  cmd={cmd}  in_avoidance={avoid}')
    print()

    avoider.reset()

    print('=== Test 2: obstacle 30° left at 0.40 m (WARNING) ===')
    avoider.update_scan(FakeScan(obstacle_angle_deg=30, obstacle_dist=0.40))
    for name, s in avoider.get_sectors().items():
        if s.n_points > 0:
            print(f'  {name:15s} min={s.min_dist:.3f}m  {s.alert.name}')
    print(f'  is_front_danger() → {avoider.is_front_danger()}')
    cmd, avoid = avoider.compute(0, 0, 0, 5, 5)
    print(f'  cmd={cmd}  in_avoidance={avoid}')
    print()

    print('=== Test 3: clear scan (all SAFE) ===')
    avoider.reset()
    avoider.update_scan(FakeScan(obstacle_angle_deg=180, obstacle_dist=3.0))
    print(f'  is_front_danger() → {avoider.is_front_danger()}')
    cmd, avoid = avoider.compute(0, 0, 0, 5, 5)
    print(f'  cmd={cmd}  in_avoidance={avoid}')
