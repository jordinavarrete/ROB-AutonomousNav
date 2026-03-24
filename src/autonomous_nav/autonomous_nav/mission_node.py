#!/usr/bin/env python3
"""
mission_node.py — Main ROS2 mission orchestrator for TurtleBot3 Burger.

Top-level mission state machine:
    INIT → PHASE_I → PHASE_II_EXPLORE → PHASE_II_RETURN → PHASE_III_DOCK → MISSION_COMPLETE

Safety guarantees:
  - 50 Hz watchdog timer: publishes zero-velocity if FRONT LiDAR sector enters DANGER
  - Graceful shutdown: KeyboardInterrupt → stop command → rclpy.shutdown()
  - All speeds enforced via module-level Config classes

ROS2 topics:
  Subscribed:
    /scan       sensor_msgs/LaserScan    BEST_EFFORT QoS
    /odom       nav_msgs/Odometry        RELIABLE QoS
  Published:
    /cmd_vel    geometry_msgs/TwistStamped  RELIABLE QoS
    /mission_state  std_msgs/String         RELIABLE QoS

SLAM pose is read from the TF tree (map → base_footprint).
Falls back to /odom if TF is not yet available.
"""

# ============================================================
# CONFIGURATION — adjust these values for lab testing
# ============================================================
class Config:
    # Control loop rates
    CONTROL_HZ          = 20      # Hz — main control loop
    WATCHDOG_HZ         = 50      # Hz — safety watchdog

    # Phase timeouts (seconds) — log WARNING but continue best-effort
    PHASE_I_TIMEOUT_S   = 300.0   # 5 minutes
    PHASE_II_TIMEOUT_S  = 480.0   # 8 minutes
    PHASE_III_TIMEOUT_S = 120.0   # 2 minutes

    # SLAM TF frame names
    MAP_FRAME           = 'map'
    BASE_FRAME          = 'base_footprint'

    # Map save path (no extension — map_saver_cli appends .yaml/.pgm)
    MAP_SAVE_PATH       = '~/mission_map'

    # Waypoints — Phase I (global navigation, Zone 1 → Zone 2)
    WAYPOINTS_PHASE1 = [
        (3.72,  2.55),   # Punt B
        (5.92,  8.12),   # Porta (door)
        (5.10, 12.61),   # Punt O
        (5.00, 11.69),   # Punt Base
    ]

    # Waypoints — Phase II exploration (Passadís sweep)
    WAYPOINTS_PHASE2_EXPLORE = [
        (0.30, 11.01),   # Punt P
        (1.90, 12.21),   # Punt Q
        (7.12, 12.61),   # Punt R
    ]
    WAYPOINTS_PHASE2_RETURN = [
        (5.00, 11.69),   # Punt Base
    ]

    # Logger interval
    LOG_CSV_INTERVAL_S  = 1.0


# ============================================================
# IMPORTS
# ============================================================
import math
import subprocess
import sys
import time
from enum import Enum, auto
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

# TF2 for SLAM pose
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

# ROS2 message types
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

# Our modules (same package)
from autonomous_nav.navigation       import WaypointNavigator, normalize_angle
from autonomous_nav.obstacle_avoidance import ObstacleAvoidance, AvoidState
from autonomous_nav.station_detector  import StationDetector
from autonomous_nav.docking           import DockingController, DockState
from autonomous_nav.mission_logger    import MissionLogger


# ============================================================
# MISSION STATES
# ============================================================
class MissionPhase(Enum):
    """Top-level mission state machine states."""
    INIT              = auto()
    PHASE_I           = auto()   # Global navigation Zone 1 → Zone 2
    PHASE_II_EXPLORE  = auto()   # Passadís sweep + station detection
    PHASE_II_RETURN   = auto()   # Return to Punt Base after station found
    PHASE_III_DOCK    = auto()   # Precision docking
    MISSION_COMPLETE  = auto()   # Done


# ============================================================
# MAIN NODE
# ============================================================
class MissionNode(Node):
    """
    Top-level ROS2 node that orchestrates the 3-phase autonomous mission.

    Instantiates and coordinates:
        WaypointNavigator  — waypoint-to-waypoint navigation
        ObstacleAvoidance  — LiDAR sector classification + wall-follow
        StationDetector    — 4-pillar cluster detection
        DockingController  — approach + live centering
        MissionLogger      — CSV telemetry
    """

    def __init__(self) -> None:
        super().__init__('mission_node')

        # ----------------------------------------------------------
        # QoS profiles
        # ----------------------------------------------------------
        qos_reliable    = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,    depth=10)
        qos_best_effort = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)

        # ----------------------------------------------------------
        # Publishers
        # ----------------------------------------------------------
        self._cmd_pub   = self.create_publisher(TwistStamped, '/cmd_vel',       qos_reliable)
        self._state_pub = self.create_publisher(String,        '/mission_state', qos_reliable)

        # ----------------------------------------------------------
        # Subscribers
        # ----------------------------------------------------------
        self.create_subscription(LaserScan, '/scan', self._scan_cb,  qos_best_effort)
        self.create_subscription(Odometry,  '/odom', self._odom_cb,  qos_reliable)

        # ----------------------------------------------------------
        # TF2 for SLAM-corrected pose
        # ----------------------------------------------------------
        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self._slam_valid  = False

        # ----------------------------------------------------------
        # Robot pose (updated by odom + TF callbacks)
        # ----------------------------------------------------------
        self._x   = 0.0
        self._y   = 0.0
        self._yaw = 0.0
        self._prev_yaw = 0.0    # for anti-stuck delta_yaw accumulation

        # ----------------------------------------------------------
        # Sub-modules
        # ----------------------------------------------------------
        self._navigator = WaypointNavigator(logger=self.get_logger())
        self._avoider   = ObstacleAvoidance(logger=self.get_logger())
        self._detector  = StationDetector(logger=self.get_logger())
        self._docker    = DockingController(
            navigator=self._navigator,
            detector=self._detector,
            logger=self.get_logger(),
        )
        self._csv_logger = MissionLogger()

        # ----------------------------------------------------------
        # Mission state
        # ----------------------------------------------------------
        self._phase         = MissionPhase.INIT
        self._phase_start_t = time.time()
        self._wp_queue      = list(Config.WAYPOINTS_PHASE1)   # mutable copy
        self._station_map_x: Optional[float] = None
        self._station_map_y: Optional[float] = None
        self._scan_ready    = False

        # ----------------------------------------------------------
        # Timers
        # ----------------------------------------------------------
        control_dt  = 1.0 / Config.CONTROL_HZ
        watchdog_dt = 1.0 / Config.WATCHDOG_HZ

        self._control_timer  = self.create_timer(control_dt,  self._control_loop)
        self._watchdog_timer = self.create_timer(watchdog_dt, self._watchdog)

        self.get_logger().info('MissionNode initialised — waiting for first scan and odom')

    # ==========================================================
    # ROS2 CALLBACKS
    # ==========================================================

    def _scan_cb(self, msg: LaserScan) -> None:
        """Forward LiDAR scan to all modules that need it."""
        self._avoider.update_scan(msg)
        self._detector.update_scan(msg)
        self._docker.update_scan(msg)
        self._scan_ready = True

    def _odom_cb(self, msg: Odometry) -> None:
        """
        Update robot pose from odometry.

        Also attempts to read the SLAM-corrected pose from TF.
        Falls back silently to odometry if TF is not yet available.
        """
        # Odometry pose (always available)
        odom_x   = msg.pose.pose.position.x
        odom_y   = msg.pose.pose.position.y
        odom_yaw = self._quat_to_yaw(msg.pose.pose.orientation)

        self._navigator.set_odom_pose(odom_x, odom_y, odom_yaw)

        # Attempt SLAM pose via TF
        slam_x, slam_y, slam_yaw = self._try_slam_pose()
        if slam_x is not None:
            self._navigator.set_slam_pose(slam_x, slam_y, slam_yaw)
            self._x, self._y, self._yaw = slam_x, slam_y, slam_yaw
            self._slam_valid = True
        else:
            self._x, self._y, self._yaw = odom_x, odom_y, odom_yaw

    # ==========================================================
    # WATCHDOG (50 Hz)
    # ==========================================================

    def _watchdog(self) -> None:
        """
        Safety watchdog running at 50 Hz.

        If the FRONT LiDAR sector is in DANGER, publish an immediate
        zero-velocity stop — regardless of the current mission phase.
        """
        if not self._scan_ready:
            return
        if self._avoider.is_front_danger():
            self._publish_stop()
            self.get_logger().warn(
                f'[WATCHDOG] FRONT DANGER — emergency stop '
                f'(phase={self._phase.name})'
            )

    # ==========================================================
    # MAIN CONTROL LOOP (20 Hz)
    # ==========================================================

    def _control_loop(self) -> None:
        """Dispatch to the correct phase handler each control tick."""
        if not self._scan_ready:
            return

        # Update anti-stuck in obstacle avoider
        delta_yaw = abs(normalize_angle(self._yaw - self._prev_yaw))
        self._avoider.update_force_rotate(delta_yaw)
        self._prev_yaw = self._yaw

        # Check phase timeouts
        self._check_timeouts()

        # Publish mission state string
        self._publish_mission_state()

        # Dispatch
        if self._phase == MissionPhase.INIT:
            self._init_phase()

        elif self._phase == MissionPhase.PHASE_I:
            self._phase_i()

        elif self._phase == MissionPhase.PHASE_II_EXPLORE:
            self._phase_ii_explore()

        elif self._phase == MissionPhase.PHASE_II_RETURN:
            self._phase_ii_return()

        elif self._phase == MissionPhase.PHASE_III_DOCK:
            self._phase_iii_dock()

        elif self._phase == MissionPhase.MISSION_COMPLETE:
            self._publish_stop()

        # Log telemetry
        self._update_logger()

    # ==========================================================
    # PHASE HANDLERS
    # ==========================================================

    def _init_phase(self) -> None:
        """
        INIT: wait for scan + odom, then begin Phase I.
        """
        if self._scan_ready:
            self._transition(MissionPhase.PHASE_I)
            self._wp_queue = list(Config.WAYPOINTS_PHASE1)
            wx, wy = self._wp_queue.pop(0)
            self._navigator.set_waypoint(wx, wy)

    # ----------------------------------------------------------

    def _phase_i(self) -> None:
        """
        PHASE I — Global navigation with obstacle avoidance.

        Drives through WAYPOINTS_PHASE1 using the navigator.
        ObstacleAvoidance pre-empts navigation when obstacles detected.
        """
        cmd, in_avoidance = self._avoider.compute(
            self._x, self._y, self._yaw,
            *self._next_waypoint()
        )

        if in_avoidance:
            # Avoidance module controls velocity
            self._publish(cmd.linear_x, cmd.angular_z)
        else:
            # Navigator controls velocity
            nav_cmd = self._navigator.step()
            self._publish(nav_cmd.linear_x, nav_cmd.angular_z)

        # Advance waypoints
        if self._navigator.has_arrived():
            self._avoider.reset()
            if self._wp_queue:
                wx, wy = self._wp_queue.pop(0)
                self._navigator.set_waypoint(wx, wy)
                self.get_logger().info(
                    f'Phase I: next waypoint ({wx:.2f},{wy:.2f}), '
                    f'{len(self._wp_queue)} remaining'
                )
            else:
                # All Phase I waypoints done → Punt Base reached
                self._transition(MissionPhase.PHASE_II_EXPLORE)
                self._wp_queue = list(Config.WAYPOINTS_PHASE2_EXPLORE)
                wx, wy = self._wp_queue.pop(0)
                self._navigator.set_waypoint(wx, wy)

    # ----------------------------------------------------------

    def _phase_ii_explore(self) -> None:
        """
        PHASE II — Exploration with concurrent station detection.

        Sweeps through WAYPOINTS_PHASE2_EXPLORE while the StationDetector
        runs on every scan.  On confirmation, records station position and
        transitions to PHASE_II_RETURN.
        """
        cmd, in_avoidance = self._avoider.compute(
            self._x, self._y, self._yaw,
            *self._next_waypoint()
        )

        if in_avoidance:
            self._publish(cmd.linear_x, cmd.angular_z)
        else:
            nav_cmd = self._navigator.step()
            self._publish(nav_cmd.linear_x, nav_cmd.angular_z)

        # --- Station detection ---
        result = self._detector.detect(self._x, self._y, self._yaw)
        if result is not None:
            self._station_map_x = result.centre_map_x
            self._station_map_y = result.centre_map_y
            self.get_logger().info(
                f'Station FOUND at map=({self._station_map_x:.3f},'
                f'{self._station_map_y:.3f}) — returning to Punt Base'
            )
            self._transition(MissionPhase.PHASE_II_RETURN)
            self._wp_queue = list(Config.WAYPOINTS_PHASE2_RETURN)
            wx, wy = self._wp_queue.pop(0)
            self._navigator.set_waypoint(wx, wy)
            self._avoider.reset()
            return

        # Advance exploration waypoints
        if self._navigator.has_arrived():
            self._avoider.reset()
            if self._wp_queue:
                wx, wy = self._wp_queue.pop(0)
                self._navigator.set_waypoint(wx, wy)
                self.get_logger().info(
                    f'Phase II explore: next ({wx:.2f},{wy:.2f}), '
                    f'{len(self._wp_queue)} remaining'
                )
            else:
                # Exploration exhausted without finding station
                # Loop back to first exploration waypoint
                self.get_logger().warn(
                    'Phase II: station not found after full sweep — repeating'
                )
                self._wp_queue = list(Config.WAYPOINTS_PHASE2_EXPLORE)
                wx, wy = self._wp_queue.pop(0)
                self._navigator.set_waypoint(wx, wy)

    # ----------------------------------------------------------

    def _phase_ii_return(self) -> None:
        """
        PHASE II RETURN — Navigate back to Punt Base.

        Also saves the SLAM map on arrival.
        """
        cmd, in_avoidance = self._avoider.compute(
            self._x, self._y, self._yaw,
            *self._next_waypoint()
        )

        if in_avoidance:
            self._publish(cmd.linear_x, cmd.angular_z)
        else:
            nav_cmd = self._navigator.step()
            self._publish(nav_cmd.linear_x, nav_cmd.angular_z)

        if self._navigator.has_arrived():
            self._avoider.reset()
            self.get_logger().info('Arrived at Punt Base — saving map')
            self._save_map()
            self._transition(MissionPhase.PHASE_III_DOCK)
            # Activate docking
            self._docker.activate(self._station_map_x, self._station_map_y)

    # ----------------------------------------------------------

    def _phase_iii_dock(self) -> None:
        """
        PHASE III — Precision docking.

        Delegates entirely to DockingController.
        Watchdog still fires independently.
        """
        if self._docker.is_docked():
            self.get_logger().info('DOCKED successfully — mission complete!')
            self._publish_stop()
            self._transition(MissionPhase.MISSION_COMPLETE)
            return

        if self._docker.has_failed():
            self.get_logger().error(
                'Docking FAILED — stopping. Manual intervention required.'
            )
            self._publish_stop()
            self._transition(MissionPhase.MISSION_COMPLETE)
            return

        cmd = self._docker.step(self._x, self._y, self._yaw)

        # During APPROACH we still run obstacle avoidance
        if self._docker.get_state().name == 'APPROACH':
            _, in_avoidance = self._avoider.compute(
                self._x, self._y, self._yaw,
                self._station_map_x, self._station_map_y,
            )
            if in_avoidance:
                avd_cmd, _ = self._avoider.compute(
                    self._x, self._y, self._yaw,
                    self._station_map_x, self._station_map_y,
                )
                self._publish(avd_cmd.linear_x, avd_cmd.angular_z)
                return

        self._publish(cmd.linear_x, cmd.angular_z)

    # ==========================================================
    # HELPERS
    # ==========================================================

    def _next_waypoint(self):
        """
        Return the current navigation target as (x, y).

        Used by ObstacleAvoidance.compute() to determine heading improvement.
        Falls back to current position if no waypoint is active.
        """
        if self._navigator._target_x is not None:
            return self._navigator._target_x, self._navigator._target_y
        return self._x, self._y

    def _publish(self, linear_x: float, angular_z: float) -> None:
        """
        Publish a TwistStamped velocity command.

        Hard-enforces global speed limits from the safety spec.
        """
        LINEAR_MAX  = 0.20   # m/s
        ANGULAR_MAX = 1.00   # rad/s

        cmd = TwistStamped()
        cmd.header.stamp    = self.get_clock().now().to_msg()
        cmd.header.frame_id = ''
        cmd.twist.linear.x  = max(-LINEAR_MAX,  min(LINEAR_MAX,  linear_x))
        cmd.twist.angular.z = max(-ANGULAR_MAX, min(ANGULAR_MAX, angular_z))
        self._cmd_pub.publish(cmd)

    def _publish_stop(self) -> None:
        """Publish a zero-velocity TwistStamped immediately."""
        cmd = TwistStamped()
        cmd.header.stamp    = self.get_clock().now().to_msg()
        cmd.header.frame_id = ''
        self._cmd_pub.publish(cmd)

    def _publish_mission_state(self) -> None:
        """Publish current phase name to /mission_state."""
        msg      = String()
        msg.data = self._phase.name
        self._state_pub.publish(msg)

    def _transition(self, new_phase: MissionPhase) -> None:
        """Log and perform a mission phase transition."""
        if new_phase != self._phase:
            self.get_logger().info(
                f'Mission: {self._phase.name} → {new_phase.name}'
            )
            self._phase         = new_phase
            self._phase_start_t = time.time()

    def _check_timeouts(self) -> None:
        """Log a warning if the current phase has exceeded its timeout."""
        elapsed = time.time() - self._phase_start_t
        limits = {
            MissionPhase.PHASE_I:          Config.PHASE_I_TIMEOUT_S,
            MissionPhase.PHASE_II_EXPLORE: Config.PHASE_II_TIMEOUT_S,
            MissionPhase.PHASE_II_RETURN:  Config.PHASE_II_TIMEOUT_S,
            MissionPhase.PHASE_III_DOCK:   Config.PHASE_III_TIMEOUT_S,
        }
        limit = limits.get(self._phase)
        if limit and elapsed > limit:
            self.get_logger().warn(
                f'[TIMEOUT] {self._phase.name} exceeded {limit:.0f}s '
                f'(elapsed {elapsed:.0f}s) — continuing best-effort'
            )

    def _update_logger(self) -> None:
        """Push latest telemetry to the CSV logger."""
        phase_str = {
            MissionPhase.INIT:             'I',
            MissionPhase.PHASE_I:          'I',
            MissionPhase.PHASE_II_EXPLORE: 'II',
            MissionPhase.PHASE_II_RETURN:  'II',
            MissionPhase.PHASE_III_DOCK:   'III',
            MissionPhase.MISSION_COMPLETE: 'III',
        }.get(self._phase, 'I')

        n_obs = sum(
            1 for s in self._avoider.get_sectors().values()
            if s.alert.name != 'SAFE'
        )

        self._csv_logger.update(
            phase      = phase_str,
            robot_x    = self._x,
            robot_y    = self._y,
            robot_yaw  = self._yaw,
            n_obstacles= n_obs,
            station_x  = self._station_map_x  if self._station_map_x is not None else -1.0,
            station_y  = self._station_map_y  if self._station_map_y is not None else -1.0,
        )

    def _save_map(self) -> None:
        """
        Save the SLAM map using map_saver_cli in a background process.

        Produces ~/mission_map.yaml and ~/mission_map.pgm.
        Errors are logged but do not abort the mission.
        """
        import os
        path = os.path.expanduser(Config.MAP_SAVE_PATH)
        cmd  = ['ros2', 'run', 'nav2_map_server', 'map_saver_cli', '-f', path]
        try:
            subprocess.Popen(cmd)
            self.get_logger().info(f'Map save requested → {path}')
        except Exception as exc:
            self.get_logger().error(f'Map save failed: {exc}')

    # ----------------------------------------------------------
    # SLAM pose via TF
    # ----------------------------------------------------------

    def _try_slam_pose(self):
        """
        Try to get the SLAM-corrected pose from the TF tree.

        Returns (x, y, yaw) if available, or (None, None, None) on failure.
        """
        try:
            tf = self._tf_buffer.lookup_transform(
                Config.MAP_FRAME,
                Config.BASE_FRAME,
                rclpy.time.Time(),
            )
            x   = tf.transform.translation.x
            y   = tf.transform.translation.y
            yaw = self._quat_to_yaw(tf.transform.rotation)
            return x, y, yaw
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None, None, None

    # ----------------------------------------------------------
    # Utility
    # ----------------------------------------------------------

    @staticmethod
    def _quat_to_yaw(q) -> float:
        """Convert a quaternion to a yaw angle [rad]."""
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    # ----------------------------------------------------------
    # Shutdown
    # ----------------------------------------------------------

    def shutdown(self) -> None:
        """
        Graceful shutdown handler.

        Publishes a stop command, flushes the CSV log, and tears down timers.
        """
        self.get_logger().info('Shutting down MissionNode — stopping robot')
        self._publish_stop()
        self._csv_logger.close()


# ============================================================
# ENTRY POINT
# ============================================================
def main(args=None) -> None:
    """ROS2 entry point registered in setup.py."""
    rclpy.init(args=args)
    node = MissionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt received')
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
