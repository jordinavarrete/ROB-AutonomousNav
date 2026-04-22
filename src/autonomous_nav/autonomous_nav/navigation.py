#!/usr/bin/env python3
"""
navigation.py — Waypoint navigator for the autonomous mission.

State machine per waypoint:
    IDLE → ORIENT → NAVIGATE → ARRIVED

  IDLE:     waiting for a waypoint to be set
  ORIENT:   rotate in place toward target; proportional angular control
  NAVIGATE: move forward with continuous heading correction
  ARRIVED:  distance < ARRIVAL_THRESHOLD; signals mission_node to advance

Pose source priority:
  1. SLAM-corrected pose (set via set_slam_pose())
  2. Odometry fallback (set via set_odom_pose())

This class has NO ROS2 Node inheritance — it is a pure logic module
instantiated by mission_node.py, which feeds it poses and retrieves
velocity commands at each control tick.
"""

# ============================================================
# CONFIGURATION — adjust these values for lab testing
# ============================================================
class Config:
    # Speed limits
    LINEAR_SPEED        = 0.18   # m/s — max forward speed
    ANGULAR_SPEED       = 0.50   # rad/s — max rotation speed in ORIENT
    MIN_ANGULAR_SPEED   = 0.05   # rad/s — minimum rotation (avoids stalling)

    # Proportional gains
    KP_ORIENT           = 1.2    # gain for in-place rotation (ORIENT phase)
    KP_LINEAR           = 0.6    # gain for forward speed (NAVIGATE phase)
    KP_HEADING          = 0.4    # gain for heading correction while moving

    # Transition thresholds
    ORIENT_THRESHOLD    = 0.05   # rad — aligned enough to start moving
    ARRIVAL_THRESHOLD   = 0.10   # m   — waypoint considered reached

    # Slow-down zone: reduce speed when close to waypoint
    SLOWDOWN_RADIUS     = 0.50   # m   — start decelerating inside this radius
    MIN_LINEAR_SPEED    = 0.05   # m/s — minimum forward speed (avoids stalling)

    # Phase waypoint lists
    WAYPOINTS_PHASE1 = [
        (4.280,  1.735),   # Punt A
        (4.880,  2.535),   # Punt C
        (5.080,  5.740),   # Punt D
        (5.480, 10.545),   # Punt F
        (6.280, 11.685),   # Porta (Door)
    ]

    WAYPOINTS_PHASE2 = [
        (3.475, 15.390),   # P Base
        (9.115, 14.190),   # Punt Q
        (7.310, 16.190),   # Punt R
        (3.675, 14.190),   # Punt S
        (1.275, 14.990),   # Punt T
        (1.075, 16.190),   # Punt U
        (3.475, 15.390),   # Return to P Base (docking station)
    ]


# ============================================================
# IMPORTS
# ============================================================
import math
import time
from enum import Enum, auto
from typing import List, Optional, Tuple


# ============================================================
# ENUMERATIONS
# ============================================================
class NavState(Enum):
    """States of the per-waypoint navigation state machine."""
    IDLE     = auto()   # no active waypoint
    ORIENT   = auto()   # rotating in place toward target
    NAVIGATE = auto()   # moving forward with heading correction
    ARRIVED  = auto()   # within ARRIVAL_THRESHOLD of target


# ============================================================
# DATA STRUCTURES
# ============================================================
class Pose2D:
    """
    Minimal 2-D pose container.

    Attributes:
        x:         X position [m]
        y:         Y position [m]
        yaw:       Heading [rad], normalised to [-π, π]
        timestamp: Time of last update [s, monotonic]
    """
    __slots__ = ('x', 'y', 'yaw', 'timestamp')

    def __init__(self, x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> None:
        self.x         = x
        self.y         = y
        self.yaw       = yaw
        self.timestamp = time.monotonic()

    def update(self, x: float, y: float, yaw: float) -> None:
        """Update position and heading, refresh timestamp."""
        self.x         = x
        self.y         = y
        self.yaw       = normalize_angle(yaw)
        self.timestamp = time.monotonic()

    def distance_to(self, wx: float, wy: float) -> float:
        """Euclidean distance to a point."""
        return math.sqrt((wx - self.x) ** 2 + (wy - self.y) ** 2)

    def angle_to(self, wx: float, wy: float) -> float:
        """Signed angle from current heading to the direction of (wx, wy) [rad]."""
        desired = math.atan2(wy - self.y, wx - self.x)
        return normalize_angle(desired - self.yaw)

    def __repr__(self) -> str:
        return f'Pose2D(x={self.x:.3f}, y={self.y:.3f}, yaw={math.degrees(self.yaw):.1f}°)'


class VelocityCommand:
    """Simple container for a linear/angular velocity pair."""
    __slots__ = ('linear_x', 'angular_z')

    def __init__(self, linear_x: float = 0.0, angular_z: float = 0.0) -> None:
        self.linear_x  = linear_x
        self.angular_z = angular_z

    def __repr__(self) -> str:
        return f'VelocityCommand(lin={self.linear_x:.3f}, ang={self.angular_z:.3f})'


# ============================================================
# UTILITY
# ============================================================
def normalize_angle(a: float) -> float:
    """Normalise angle to [-π, π]."""
    while a >  math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value to [lo, hi]."""
    return max(lo, min(hi, value))


# ============================================================
# MAIN CLASS
# ============================================================
class WaypointNavigator:
    """
    Proportional waypoint navigator with ORIENT → NAVIGATE → ARRIVED states.

    Usage (from mission_node.py):

        nav = WaypointNavigator(logger=self.get_logger())

        # feed poses on each callback:
        nav.set_odom_pose(x, y, yaw)
        nav.set_slam_pose(x, y, yaw)    # when SLAM is available

        # set next waypoint:
        nav.set_waypoint(wp_x, wp_y)

        # each control tick (e.g. 20 Hz):
        cmd = nav.step()
        publish(cmd)

        # check arrival:
        if nav.has_arrived():
            nav.set_waypoint(next_wp_x, next_wp_y)
    """

    def __init__(self, logger=None) -> None:
        """
        Initialise navigator.

        Args:
            logger: ROS2 logger or None (falls back to print).
        """
        self._log = logger

        # Pose estimates — SLAM preferred, odometry fallback
        self._slam_pose  = Pose2D()
        self._odom_pose  = Pose2D()
        self._slam_valid = False        # True once first SLAM pose received

        # Current navigation target
        self._target_x: Optional[float] = None
        self._target_y: Optional[float] = None

        # State machine
        self._state = NavState.IDLE

        # Statistics for logging
        self._waypoints_completed = 0

    # ----------------------------------------------------------
    # Pose setters — called from ROS2 callbacks
    # ----------------------------------------------------------

    def set_odom_pose(self, x: float, y: float, yaw: float) -> None:
        """
        Update odometry-based pose estimate.

        Args:
            x:   Position X [m]
            y:   Position Y [m]
            yaw: Heading [rad]
        """
        self._odom_pose.update(x, y, yaw)

    def set_slam_pose(self, x: float, y: float, yaw: float) -> None:
        """
        Update SLAM-corrected pose estimate.

        Once called at least once, SLAM pose takes priority over odometry.

        Args:
            x:   Position X [m]
            y:   Position Y [m]
            yaw: Heading [rad]
        """
        self._slam_pose.update(x, y, yaw)
        self._slam_valid = True

    # ----------------------------------------------------------
    # Waypoint control
    # ----------------------------------------------------------

    def set_waypoint(self, x: float, y: float) -> None:
        """
        Set a new navigation target and restart the state machine.

        Args:
            x: Target X [m]
            y: Target Y [m]
        """
        self._target_x = x
        self._target_y = y
        self._transition(NavState.ORIENT)
        pose = self._active_pose()
        dist = pose.distance_to(x, y)
        self._log_info(f'[NAV] Waypoint → ({x:.2f}, {y:.2f})  dist={dist:.2f}m')

    def clear_waypoint(self) -> None:
        """Cancel current waypoint and return to IDLE."""
        self._target_x = None
        self._target_y = None
        self._transition(NavState.IDLE)

    def has_arrived(self) -> bool:
        """Return True if the robot has reached the current waypoint."""
        return self._state == NavState.ARRIVED

    def is_idle(self) -> bool:
        """Return True if no waypoint is active."""
        return self._state == NavState.IDLE

    def get_state(self) -> NavState:
        """Return the current navigation state."""
        return self._state

    def get_distance_to_target(self) -> float:
        """Return Euclidean distance to current target, or inf if none."""
        if self._target_x is None:
            return float('inf')
        pose = self._active_pose()
        return pose.distance_to(self._target_x, self._target_y)

    # ----------------------------------------------------------
    # Control step — called at each control tick
    # ----------------------------------------------------------

    def step(self) -> VelocityCommand:
        """
        Compute and return the velocity command for this control tick.

        Call this at a fixed rate (e.g. 20 Hz) from mission_node.py.
        The returned command should be published to /cmd_vel ONLY when
        obstacle_avoidance.py is NOT in avoidance mode.

        Returns:
            VelocityCommand with linear_x and angular_z.
        """
        if self._state == NavState.IDLE or self._target_x is None:
            return VelocityCommand(0.0, 0.0)

        if self._state == NavState.ARRIVED:
            return VelocityCommand(0.0, 0.0)

        pose = self._active_pose()

        if self._state == NavState.ORIENT:
            return self._step_orient(pose)

        if self._state == NavState.NAVIGATE:
            return self._step_navigate(pose)

        return VelocityCommand(0.0, 0.0)

    # ----------------------------------------------------------
    # ORIENT phase
    # ----------------------------------------------------------

    def _step_orient(self, pose: Pose2D) -> VelocityCommand:
        """
        Rotate in place toward the target.

        Proportional control: angular_speed = Kp_orient * angle_error
        Clamp to [MIN_ANGULAR_SPEED, ANGULAR_SPEED] to avoid stalling.
        Transition to NAVIGATE when |angle_error| < ORIENT_THRESHOLD.

        Args:
            pose: Current best pose estimate.

        Returns:
            VelocityCommand (linear_x = 0).
        """
        angle_error = pose.angle_to(self._target_x, self._target_y)

        if abs(angle_error) < Config.ORIENT_THRESHOLD:
            self._transition(NavState.NAVIGATE)
            return VelocityCommand(0.0, 0.0)

        # Proportional angular speed, with minimum to avoid stalling
        raw_angular = Config.KP_ORIENT * angle_error
        sign        = 1.0 if raw_angular >= 0 else -1.0
        angular_z   = sign * clamp(
            abs(raw_angular),
            Config.MIN_ANGULAR_SPEED,
            Config.ANGULAR_SPEED,
        )

        return VelocityCommand(0.0, angular_z)

    # ----------------------------------------------------------
    # NAVIGATE phase
    # ----------------------------------------------------------

    def _step_navigate(self, pose: Pose2D) -> VelocityCommand:
        """
        Move forward toward the target with continuous heading correction.

        Linear speed is proportional to distance (with slowdown near target).
        Angular correction keeps heading aligned while moving.
        Transition to ARRIVED when distance < ARRIVAL_THRESHOLD.
        If heading error becomes large (> π/3), fall back to ORIENT.

        Args:
            pose: Current best pose estimate.

        Returns:
            VelocityCommand.
        """
        dist        = pose.distance_to(self._target_x, self._target_y)
        angle_error = pose.angle_to(self._target_x, self._target_y)

        # --- Arrival check ---
        if dist < Config.ARRIVAL_THRESHOLD:
            self._waypoints_completed += 1
            self._transition(NavState.ARRIVED)
            dist = pose.distance_to(self._target_x, self._target_y)
            self._log_info(
                f'[NAV] ✓ Arrived ({self._target_x:.2f}, {self._target_y:.2f})  '
                f'error={dist:.3f}m  total={self._waypoints_completed}'
            )
            return VelocityCommand(0.0, 0.0)

        # --- Large heading error: re-orient before moving ---
        if abs(angle_error) > math.pi / 3:   # 60°
            self._transition(NavState.ORIENT)
            return VelocityCommand(0.0, 0.0)

        # --- Linear speed: proportional, with slowdown near target ---
        raw_linear = Config.KP_LINEAR * dist
        if dist < Config.SLOWDOWN_RADIUS:
            # Linear interpolation: full speed at SLOWDOWN_RADIUS, min at 0
            t = dist / Config.SLOWDOWN_RADIUS
            raw_linear = Config.MIN_LINEAR_SPEED + t * (Config.LINEAR_SPEED - Config.MIN_LINEAR_SPEED)

        linear_x = clamp(raw_linear, Config.MIN_LINEAR_SPEED, Config.LINEAR_SPEED)

        # --- Angular correction while moving ---
        angular_z = clamp(
            Config.KP_HEADING * angle_error,
            -Config.ANGULAR_SPEED,
            Config.ANGULAR_SPEED,
        )

        return VelocityCommand(linear_x, angular_z)

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

    def _active_pose(self) -> Pose2D:
        """
        Return the best available pose estimate.

        Prefers SLAM pose if it has been updated at least once;
        otherwise falls back to odometry.
        """
        if self._slam_valid:
            return self._slam_pose
        return self._odom_pose

    def _transition(self, new_state: NavState) -> None:
        """Perform a state transition (silent for ORIENT/NAVIGATE to reduce noise)."""
        if new_state != self._state:
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

    def simulate(nav: WaypointNavigator, wp_x: float, wp_y: float,
                 start_x=0.0, start_y=0.0, start_yaw=0.0,
                 dt=0.05, max_steps=2000, label='') -> bool:
        """
        Simulate the navigator driving a virtual robot to a waypoint.
        Integrates velocity commands with simple Euler kinematics.

        Returns True if the robot arrived within max_steps.
        """
        x, y, yaw = start_x, start_y, start_yaw
        nav.set_odom_pose(x, y, yaw)
        nav.set_waypoint(wp_x, wp_y)

        print(f'\n--- {label} ---')
        print(f'    Start : ({x:.2f}, {y:.2f}, {math.degrees(yaw):.0f}°)')
        print(f'    Target: ({wp_x:.2f}, {wp_y:.2f})')

        for step in range(max_steps):
            cmd = nav.step()

            # Euler integration
            x   += cmd.linear_x  * math.cos(yaw) * dt
            y   += cmd.linear_x  * math.sin(yaw) * dt
            yaw += cmd.angular_z * dt
            yaw  = normalize_angle(yaw)

            nav.set_odom_pose(x, y, yaw)

            if nav.has_arrived():
                dist = math.sqrt((x - wp_x)**2 + (y - wp_y)**2)
                print(f'    Arrived after {step*dt:.1f}s  '
                      f'final_pos=({x:.3f},{y:.3f})  '
                      f'error={dist:.3f}m ✓')
                nav.clear_waypoint()
                return True

        dist = math.sqrt((x - wp_x)**2 + (y - wp_y)**2)
        print(f'    TIMEOUT after {max_steps*dt:.1f}s  '
              f'pos=({x:.3f},{y:.3f})  remaining={dist:.3f}m ✗')
        nav.clear_waypoint()
        return False

    nav = WaypointNavigator()

    # Test 1 — straight ahead
    simulate(nav, wp_x=2.0, wp_y=0.0, start_yaw=0.0,
             label='Straight ahead 2 m')

    # Test 2 — 90° turn then forward
    simulate(nav, wp_x=0.0, wp_y=2.0, start_yaw=0.0,
             label='90° left then 2 m forward')

    # Test 3 — diagonal
    simulate(nav, wp_x=3.0, wp_y=3.0, start_yaw=math.pi,
             label='Diagonal (start facing backward)')

    # Test 4 — Phase I waypoint sequence
    print('\n--- Phase I full sequence ---')
    x, y, yaw = 0.0, 0.0, 0.0
    for i, (wx, wy) in enumerate(Config.WAYPOINTS_PHASE1):
        nav.set_odom_pose(x, y, yaw)
        nav.set_waypoint(wx, wy)
        for _ in range(10000):
            cmd = nav.step()
            x   += cmd.linear_x  * math.cos(yaw) * 0.05
            y   += cmd.linear_x  * math.sin(yaw) * 0.05
            yaw += cmd.angular_z * 0.05
            yaw  = normalize_angle(yaw)
            nav.set_odom_pose(x, y, yaw)
            if nav.has_arrived():
                dist = math.sqrt((x-wx)**2+(y-wy)**2)
                print(f'  WP{i+1} ({wx},{wy}) reached  '
                      f'error={dist:.3f}m  pos=({x:.2f},{y:.2f}) ✓')
                nav.clear_waypoint()
                break
