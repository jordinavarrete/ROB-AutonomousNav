#!/usr/bin/env python3
"""
docking.py — Precision docking controller for the charging station.

Two-phase docking sequence:
    IDLE         → waiting to be activated
    APPROACH     → navigate to a point 0.30 m in front of station centre
                   (delegates to WaypointNavigator)
    FINE_CENTRE  → live LiDAR-based centering between 4 pillars
                   proportional control on centroid offset
    DOCKED       → offset < DOCK_TOLERANCE for DOCK_CONFIRM_S seconds → done
    FAILED       → timeout or lost pillars for too long

Fine-centering control law:
    centroid (cx, cy) in robot frame from live pillar detection
    linear_x  = Kp_dock_linear  * cx   (drive toward centroid longitudinally)
    angular_z = Kp_dock_angular * atan2(cy, |cx| + eps)  (steer toward centroid)
    speeds capped at MAX_DOCK_SPEED

This class has NO ROS2 Node inheritance — pure logic, testable standalone.
It depends on:
    station_detector.StationDetector   (pillar detection each tick)
    navigation.WaypointNavigator       (approach phase)
"""

# ============================================================
# CONFIGURATION — adjust these values for lab testing
# ============================================================
class Config:
    # Approach
    APPROACH_OFFSET     = 0.30   # m — stop this far in front of station centre
    APPROACH_TOLERANCE  = 0.08   # m — approach waypoint arrival threshold

    # Fine centering proportional gains
    KP_DOCK_LINEAR      = 0.40   # gain on longitudinal centroid offset
    KP_DOCK_ANGULAR     = 0.80   # gain on lateral centroid angle

    # Speed caps (MUST be ≤ 0.05 m/s per safety spec)
    MAX_DOCK_SPEED      = 0.05   # m/s — linear
    MAX_DOCK_ANGULAR    = 0.40   # rad/s — angular

    # Docked criterion
    DOCK_TOLERANCE      = 0.03   # m — centroid offset to be considered docked
    DOCK_CONFIRM_S      = 2.0    # s — must hold tolerance this long

    # Lost-pillar recovery
    MAX_LOST_TICKS      = 20     # ticks without 4 pillars before FAILED
    LOST_CREEP_SPEED    = 0.03   # m/s — slow creep backward when pillars lost

    # Timeout
    FINE_CENTRE_TIMEOUT = 60.0   # s — abort fine centering after this


# ============================================================
# IMPORTS
# ============================================================
import math
import time
from enum import Enum, auto
from typing import List, Optional, Tuple


from autonomous_nav.station_detector import StationDetector, StationResult, dist2d
from autonomous_nav.navigation       import WaypointNavigator, VelocityCommand, normalize_angle


# ============================================================
# ENUMERATIONS
# ============================================================
class DockState(Enum):
    """States of the docking state machine."""
    IDLE        = auto()   # not yet activated
    APPROACH    = auto()   # coarse navigation to station neighbourhood
    FINE_CENTRE = auto()   # live LiDAR-based centering
    DOCKED      = auto()   # successfully centred
    FAILED      = auto()   # timeout or unrecoverable loss of pillars


# ============================================================
# DATA STRUCTURES
# ============================================================
class DockStatus:
    """
    Snapshot of docking progress for logging / mission_node.py.

    Attributes:
        state:          Current DockState.
        centroid_x/y:   Latest pillar centroid in robot frame [m].
        offset:         Distance from centroid to robot origin [m].
        docked:         True once DOCKED state is reached.
        elapsed_fine:   Seconds spent in FINE_CENTRE phase.
    """
    __slots__ = ('state', 'centroid_x', 'centroid_y', 'offset',
                 'docked', 'elapsed_fine')

    def __init__(self) -> None:
        self.state        = DockState.IDLE
        self.centroid_x   = 0.0
        self.centroid_y   = 0.0
        self.offset       = float('inf')
        self.docked       = False
        self.elapsed_fine = 0.0

    def __repr__(self) -> str:
        return (f'DockStatus(state={self.state.name}, '
                f'offset={self.offset:.4f}m, docked={self.docked})')


# ============================================================
# MAIN CLASS
# ============================================================
class DockingController:
    """
    Two-phase docking controller: coarse approach + live LiDAR fine centering.

    Usage (from mission_node.py):

        docker = DockingController(
            navigator=self.navigator,
            detector=self.station_detector,
            logger=self.get_logger(),
        )

        # Activate once station position is known:
        docker.activate(station_map_x, station_map_y)

        # Feed scan each callback:
        docker.update_scan(scan_msg)

        # Each control tick (~20 Hz):
        cmd    = docker.step(robot_x, robot_y, robot_yaw)
        status = docker.get_status()
        publish(cmd)

        if docker.is_docked():
            transition to MISSION_COMPLETE
    """

    def __init__(
        self,
        navigator: WaypointNavigator,
        detector:  StationDetector,
        logger=None,
    ) -> None:
        """
        Initialise docking controller.

        Args:
            navigator: Shared WaypointNavigator instance (from mission_node).
            detector:  Shared StationDetector instance (from mission_node).
            logger:    ROS2 logger or None.
        """
        self._nav     = navigator
        self._det     = detector
        self._log     = logger

        self._state   = DockState.IDLE
        self._status  = DockStatus()

        # Station position (map frame)
        self._station_map_x: Optional[float] = None
        self._station_map_y: Optional[float] = None

        # Fine-centering timing
        self._fine_start_time:    Optional[float] = None
        self._in_tolerance_since: Optional[float] = None

        # Lost-pillar counter
        self._lost_ticks = 0

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def activate(self, station_map_x: float, station_map_y: float) -> None:
        """
        Start the docking sequence toward the given station centre.

        Computes the approach waypoint (APPROACH_OFFSET m in front of
        station, along the robot→station vector) and hands it to the
        WaypointNavigator.

        Args:
            station_map_x: Station centre X in map frame [m].
            station_map_y: Station centre Y in map frame [m].
        """
        self._station_map_x = station_map_x
        self._station_map_y = station_map_y
        self._transition(DockState.APPROACH)
        self._log_info(
            f'Docking activated → station map=({station_map_x:.3f},{station_map_y:.3f})'
        )

    def update_scan(self, scan_msg) -> None:
        """
        Forward a new LaserScan to the internal StationDetector.

        Args:
            scan_msg: sensor_msgs/LaserScan message.
        """
        self._det.update_scan(scan_msg)

    def step(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
    ) -> VelocityCommand:
        """
        Compute one control step.

        Args:
            robot_x/y:   Robot position in map frame [m].
            robot_yaw:   Robot heading in map frame [rad].

        Returns:
            VelocityCommand to publish to /cmd_vel.
        """
        self._status.state = self._state

        if self._state == DockState.IDLE:
            return VelocityCommand(0.0, 0.0)

        if self._state == DockState.APPROACH:
            return self._step_approach(robot_x, robot_y, robot_yaw)

        if self._state == DockState.FINE_CENTRE:
            return self._step_fine_centre(robot_x, robot_y, robot_yaw)

        # DOCKED or FAILED — stay still
        return VelocityCommand(0.0, 0.0)

    def is_docked(self) -> bool:
        """Return True when docking is successfully complete."""
        return self._state == DockState.DOCKED

    def has_failed(self) -> bool:
        """Return True if docking has failed and mission_node should intervene."""
        return self._state == DockState.FAILED

    def get_status(self) -> DockStatus:
        """Return a snapshot of current docking progress."""
        return self._status

    def get_state(self) -> DockState:
        """Return the current docking state."""
        return self._state

    # ----------------------------------------------------------
    # APPROACH phase
    # ----------------------------------------------------------

    def _step_approach(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
    ) -> VelocityCommand:
        """
        Drive to the approach waypoint using the WaypointNavigator.

        The approach waypoint is APPROACH_OFFSET metres behind the
        station centre along the robot→station bearing at activation time.

        Transitions to FINE_CENTRE once the navigator reports arrival.
        """
        # Compute approach waypoint on first entry
        if self._nav.is_idle() and self._state == DockState.APPROACH:
            ap_x, ap_y = self._compute_approach_wp(robot_x, robot_y)
            self._nav.set_waypoint(ap_x, ap_y)
            self._log_info(
                f'Approach waypoint: ({ap_x:.3f},{ap_y:.3f})'
            )

        cmd = self._nav.step()

        if self._nav.has_arrived():
            self._nav.clear_waypoint()
            self._transition(DockState.FINE_CENTRE)
            self._fine_start_time    = time.monotonic()
            self._in_tolerance_since = None
            self._lost_ticks         = 0
            # Reset detector so it re-detects pillars in fine-centre mode
            self._det.reset()

        return cmd

    def _compute_approach_wp(
        self, robot_x: float, robot_y: float
    ) -> Tuple[float, float]:
        """
        Compute the approach waypoint: APPROACH_OFFSET metres in front
        of the station centre, along the robot → station vector.

        Args:
            robot_x/y: Current robot position in map frame.

        Returns:
            (ap_x, ap_y) approach waypoint in map frame.
        """
        dx = self._station_map_x - robot_x
        dy = self._station_map_y - robot_y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 1e-6:
            # Already at station — just go slightly in front
            return self._station_map_x + Config.APPROACH_OFFSET, self._station_map_y

        # Unit vector robot → station
        ux = dx / dist
        uy = dy / dist

        # Approach point = station centre − APPROACH_OFFSET * unit_vector
        ap_x = self._station_map_x - ux * Config.APPROACH_OFFSET
        ap_y = self._station_map_y - uy * Config.APPROACH_OFFSET
        return ap_x, ap_y

    # ----------------------------------------------------------
    # FINE_CENTRE phase
    # ----------------------------------------------------------

    def _step_fine_centre(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
    ) -> VelocityCommand:
        """
        Live LiDAR-based centering between the 4 pillars.

        1. Detect 4 pillars in current scan (robot frame).
        2. Compute centroid of 4 pillar positions.
        3. Apply proportional control to drive robot to centroid.
        4. Declare DOCKED once offset < DOCK_TOLERANCE for DOCK_CONFIRM_S.
        5. FAILED if timeout or pillars lost for too long.

        Args:
            robot_x/y/yaw: Current robot pose in map frame.

        Returns:
            VelocityCommand.
        """
        now = time.monotonic()

        # --- Timeout guard ---
        if self._fine_start_time and (now - self._fine_start_time) > Config.FINE_CENTRE_TIMEOUT:
            self._log_warn(f'Fine-centre timeout ({Config.FINE_CENTRE_TIMEOUT}s) — FAILED')
            self._transition(DockState.FAILED)
            return VelocityCommand(0.0, 0.0)

        self._status.elapsed_fine = now - (self._fine_start_time or now)

        # --- Detect pillars (robot frame) ---
        pillars = self._detect_pillars_robot_frame(robot_x, robot_y, robot_yaw)

        if pillars is None:
            self._lost_ticks += 1
            self._log_warn(f'Pillars not detected ({self._lost_ticks}/{Config.MAX_LOST_TICKS})')

            if self._lost_ticks >= Config.MAX_LOST_TICKS:
                self._transition(DockState.FAILED)
                return VelocityCommand(0.0, 0.0)

            # Creep backward slowly while trying to reacquire
            return VelocityCommand(-Config.LOST_CREEP_SPEED, 0.0)

        self._lost_ticks = 0   # reset on successful detection

        # --- Centroid in robot frame ---
        cx = sum(px for px, _ in pillars) / 4
        cy = sum(py for _, py in pillars) / 4
        offset = math.sqrt(cx * cx + cy * cy)

        self._status.centroid_x = cx
        self._status.centroid_y = cy
        self._status.offset     = offset

        self._log_info(
            f'Fine-centre: centroid=({cx:.4f},{cy:.4f}) offset={offset:.4f}m'
        )

        # --- Docked check ---
        if offset < Config.DOCK_TOLERANCE:
            if self._in_tolerance_since is None:
                self._in_tolerance_since = now
            elif now - self._in_tolerance_since >= Config.DOCK_CONFIRM_S:
                self._status.docked = True
                self._transition(DockState.DOCKED)
                return VelocityCommand(0.0, 0.0)
        else:
            self._in_tolerance_since = None   # reset if we drift out

        # --- Control law ---
        #   linear_x  = Kp_linear  * cx  (positive cx = station ahead → go forward)
        #   angular_z = Kp_angular * angle toward centroid
        angle_to_centroid = math.atan2(cy, abs(cx) + 1e-6)

        linear_x  = self._clamp(
            Config.KP_DOCK_LINEAR  * cx,
            -Config.MAX_DOCK_SPEED, Config.MAX_DOCK_SPEED,
        )
        angular_z = self._clamp(
            Config.KP_DOCK_ANGULAR * angle_to_centroid,
            -Config.MAX_DOCK_ANGULAR, Config.MAX_DOCK_ANGULAR,
        )

        return VelocityCommand(linear_x, angular_z)

    # ----------------------------------------------------------
    # Pillar detection in robot frame (fine-centre mode)
    # ----------------------------------------------------------

    def _detect_pillars_robot_frame(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Run the StationDetector and return the 4 pillar positions
        in robot frame from the latest confirmed or candidate detection.

        During fine centering we bypass the N_CONFIRM requirement and
        use the raw quad from the latest scan for responsiveness.
        We call detect() which accumulates confirmations; if it returns
        a result we use its pillars_robot directly.  Otherwise we fall
        back to a single-scan quad extraction.

        Args:
            robot_x/y/yaw: Current robot pose.

        Returns:
            List of 4 (x, y) in robot frame, or None if not detected.
        """
        # Try the full confirmed pipeline first
        result: Optional[StationResult] = self._det.detect(robot_x, robot_y, robot_yaw)
        if result is not None:
            return result.pillars_robot

        # During fine centering, also accept a single-scan quad
        # (bypasses N_CONFIRM for responsiveness)
        raw_quad = self._det._find_square_quad(
            self._det._filter_pillar_candidates(
                self._det._cluster_scan(self._det._scan_points)
            )
        )
        if raw_quad is not None:
            return [(c.centroid_x, c.centroid_y) for c in raw_quad]

        return None

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

    @staticmethod
    def _clamp(value: float, lo: float, hi: float) -> float:
        """Clamp value to [lo, hi]."""
        return max(lo, min(hi, value))

    def _transition(self, new_state: DockState) -> None:
        """Log and perform a state transition."""
        if new_state != self._state:
            self._log_info(f'Docking: {self._state.name} → {new_state.name}')
            self._state = new_state
            self._status.state = new_state

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

    # --------------------------------------------------------
    # Minimal stubs for testing without the full ROS2 stack
    # --------------------------------------------------------

    class FakeScan:
        """Synthetic 360° scan with 4 pillars at given robot-frame positions."""
        def __init__(self, pillars, n_rays=360, bg=3.0, diameter=0.05):
            self.range_min       = 0.12
            self.range_max       = 3.50
            self.angle_min       = -math.pi
            self.angle_increment = 2 * math.pi / n_rays
            self.ranges          = [bg] * n_rays
            for px, py in pillars:
                d    = math.sqrt(px**2 + py**2)
                bear = math.atan2(py, px)
                hw   = math.atan2(diameter / 2, d)
                for i in range(n_rays):
                    a = self.angle_min + i * self.angle_increment
                    delta = a - bear
                    while delta >  math.pi: delta -= 2 * math.pi
                    while delta < -math.pi: delta += 2 * math.pi
                    if abs(delta) <= hw:
                        self.ranges[i] = d - diameter / 2

    # --------------------------------------------------------
    # Simulate the docking sequence
    # --------------------------------------------------------
    def simulate_docking(
        station_robot_x: float = 0.60,
        station_robot_y: float = 0.00,
        robot_map_x: float = 0.0,
        robot_map_y: float = 0.0,
        robot_map_yaw: float = 0.0,
        dt: float = 0.05,
        max_steps: int = 3000,
    ) -> None:
        """
        Simulate the full docking sequence with a virtual robot.

        The robot starts offset from the station centre and must drive
        into the centre using only the centroid feedback.
        """
        half = 0.20   # half of 40 cm station side

        # Build navigator and detector
        nav = WaypointNavigator()
        det = StationDetector()
        docker = DockingController(navigator=nav, detector=det)

        # Station centre in map frame (just offset from robot start for simplicity)
        st_map_x = robot_map_x + station_robot_x
        st_map_y = robot_map_y + station_robot_y

        print(f'\n=== Docking simulation ===')
        print(f'    Robot start : ({robot_map_x:.3f},{robot_map_y:.3f},{math.degrees(robot_map_yaw):.0f}°)')
        print(f'    Station map : ({st_map_x:.3f},{st_map_y:.3f})')

        # Virtual robot state
        x, y, yaw = robot_map_x, robot_map_y, robot_map_yaw
        nav.set_odom_pose(x, y, yaw)

        docker.activate(st_map_x, st_map_y)

        for step in range(max_steps):
            # Pillars in robot frame (move with robot)
            cos_y, sin_y = math.cos(-yaw), math.sin(-yaw)

            def to_robot(mx, my):
                dx, dy = mx - x, my - y
                return dx * cos_y - dy * (-sin_y), dx * (-sin_y) + dy * cos_y

            def to_robot_correct(mx, my):
                dx, dy = mx - x, my - y
                rx =  dx * math.cos(yaw) + dy * math.sin(yaw)
                ry = -dx * math.sin(yaw) + dy * math.cos(yaw)
                return rx, ry

            pillars_robot = [
                to_robot_correct(st_map_x - half, st_map_y + half),
                to_robot_correct(st_map_x + half, st_map_y + half),
                to_robot_correct(st_map_x + half, st_map_y - half),
                to_robot_correct(st_map_x - half, st_map_y - half),
            ]

            scan = FakeScan(pillars_robot)
            docker.update_scan(scan)
            nav.set_odom_pose(x, y, yaw)

            cmd = docker.step(x, y, yaw)

            # Integrate kinematics
            x   += cmd.linear_x * math.cos(yaw) * dt
            y   += cmd.linear_x * math.sin(yaw) * dt
            yaw += cmd.angular_z * dt
            yaw  = normalize_angle(yaw)

            status = docker.get_status()

            # Progress report every 2 s
            if step % 40 == 0:
                dist_to_station = math.sqrt((x - st_map_x)**2 + (y - st_map_y)**2)
                print(f'    t={step*dt:5.1f}s  state={status.state.name:12s}  '
                      f'pos=({x:.3f},{y:.3f})  '
                      f'dist_to_stn={dist_to_station:.3f}m  '
                      f'offset={status.offset:.4f}m')

            if docker.is_docked():
                dist_to_station = math.sqrt((x - st_map_x)**2 + (y - st_map_y)**2)
                print(f'\n    ✓ DOCKED at t={step*dt:.1f}s')
                print(f'      Final pos  : ({x:.4f},{y:.4f})')
                print(f'      Dist to ctr: {dist_to_station:.4f}m  '
                      f'(tolerance {Config.DOCK_TOLERANCE}m)')
                print(f'      Centroid   : ({status.centroid_x:.4f},{status.centroid_y:.4f})')
                return

            if docker.has_failed():
                print(f'\n    ✗ FAILED at t={step*dt:.1f}s')
                return

        print(f'\n    ✗ TIMEOUT after {max_steps*dt:.1f}s')

    # Run simulations
    simulate_docking(station_robot_x=0.80, station_robot_y=0.00)
    simulate_docking(station_robot_x=0.80, station_robot_y=0.15,
                     robot_map_yaw=math.radians(15))
