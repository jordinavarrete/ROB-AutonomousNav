#!/usr/bin/env python3
"""
debug_explore_dock_node.py — Node de debug per testejar Fase II + Fase III.

Executa NOMÉS:
  1. Exploració (sweep pels waypoints de la zona Passadís)
  2. Detecció de l'estació de càrrega (4 pilars)
  3. Retorn a Punt Base + guardat del mapa
  4. Docking de precisió (approach + fine centering)

Útil per provar al lab sense haver de fer tota la Fase I.

Ús:
    ros2 run autonomous_nav debug_explore_dock_node
    ros2 run autonomous_nav debug_explore_dock_node --ros-args \
        -p base_x:=5.0 -p base_y:=11.69

Topics:
  Subscriu:  /scan  (LaserScan, BEST_EFFORT)
             /odom  (Odometry,  RELIABLE)
  Publica:   /cmd_vel (TwistStamped, RELIABLE)
"""

# ============================================================
# CONFIGURACIÓ
# ============================================================
class Config:
    # Punt Base (on comença aquesta prova)
    BASE_X          = 5.00
    BASE_Y          = 11.69

    # Waypoints d'exploració del Passadís
    WAYPOINTS_EXPLORE = [
        (0.30, 11.01),   # Punt P
        (1.90, 12.21),   # Punt Q
        (7.12, 12.61),   # Punt R
    ]

    # Punt Base retorn
    WAYPOINTS_RETURN = [
        (5.00, 11.69),   # Punt Base
    ]

    # Rates
    CONTROL_HZ      = 20
    WATCHDOG_HZ     = 50

    # Frames TF
    MAP_FRAME       = 'map'
    BASE_FRAME      = 'base_footprint'

    # Map save path
    MAP_SAVE_PATH   = '~/mission_map'


# ============================================================
# IMPORTS
# ============================================================
import math
import subprocess
import os
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from enum import Enum, auto

from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from autonomous_nav.navigation import WaypointNavigator, normalize_angle
from autonomous_nav.obstacle_avoidance import ObstacleAvoidance, AvoidState
from autonomous_nav.station_detector import StationDetector
from autonomous_nav.docking import DockingController, DockState


# ============================================================
# ESTATS
# ============================================================
class DebugPhase(Enum):
    EXPLORE         = auto()   # Explorant passadís + buscant estació
    RETURN_BASE     = auto()   # Tornant a Punt Base (estació trobada)
    DOCKING         = auto()   # Approach + Fine Centering
    DONE            = auto()   # Acabat (docked o failed)


# ============================================================
# NODE PRINCIPAL
# ============================================================
class DebugExploreDockNode(Node):
    """
    Node de debug que executa la Fase II (exploració + detecció)
    i la Fase III (docking) sense necessitat de fer la Fase I.

    Estat:
      EXPLORE → RETURN_BASE → DOCKING → DONE
    """

    def __init__(self) -> None:
        super().__init__('debug_explore_dock_node')

        # Paràmetres
        self.declare_parameter('base_x', Config.BASE_X)
        self.declare_parameter('base_y', Config.BASE_Y)

        self._base_x = self.get_parameter('base_x').value
        self._base_y = self.get_parameter('base_y').value

        self.get_logger().info('=' * 60)
        self.get_logger().info('  DEBUG EXPLORE + DOCK (Fase II + III)')
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'  Punt Base: ({self._base_x:.2f}, {self._base_y:.2f})')
        self.get_logger().info(f'  Waypoints exploració: {Config.WAYPOINTS_EXPLORE}')
        self.get_logger().info('=' * 60)

        # QoS
        qos_r = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)
        qos_b = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)

        # Publisher
        self._cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', qos_r)

        # Subscribers
        self.create_subscription(LaserScan, '/scan', self._scan_cb, qos_b)
        self.create_subscription(Odometry,  '/odom', self._odom_cb, qos_r)

        # TF2
        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # Estat intern
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self._prev_yaw = 0.0
        self._scan_ready = False
        self._odom_ready = False

        # Sub-mòduls
        self._navigator = WaypointNavigator(logger=self.get_logger())
        self._avoider   = ObstacleAvoidance(logger=self.get_logger())
        self._detector  = StationDetector(logger=self.get_logger())
        self._docker    = DockingController(
            navigator=self._navigator,
            detector=self._detector,
            logger=self.get_logger(),
        )

        # Fase actual
        self._phase = DebugPhase.EXPLORE
        self._phase_start = time.time()
        self._wp_queue = list(Config.WAYPOINTS_EXPLORE)
        self._station_map_x = None
        self._station_map_y = None
        self._explore_sweeps = 0

        # Timers
        self._control_timer  = self.create_timer(1.0 / Config.CONTROL_HZ, self._control_loop)
        self._watchdog_timer = self.create_timer(1.0 / Config.WATCHDOG_HZ, self._watchdog)

        self.get_logger().info(
            'Node inicialitzat. Esperant /scan i /odom...'
        )

    # ==========================================================
    # CALLBACKS
    # ==========================================================

    def _scan_cb(self, msg: LaserScan) -> None:
        self._avoider.update_scan(msg)
        self._detector.update_scan(msg)
        self._docker.update_scan(msg)
        self._scan_ready = True

    def _odom_cb(self, msg: Odometry) -> None:
        odom_x   = msg.pose.pose.position.x
        odom_y   = msg.pose.pose.position.y
        odom_yaw = self._quat_to_yaw(msg.pose.pose.orientation)

        self._navigator.set_odom_pose(odom_x, odom_y, odom_yaw)
        self._odom_ready = True

        slam_x, slam_y, slam_yaw = self._try_slam_pose()
        if slam_x is not None:
            self._navigator.set_slam_pose(slam_x, slam_y, slam_yaw)
            self._x, self._y, self._yaw = slam_x, slam_y, slam_yaw
        else:
            self._x, self._y, self._yaw = odom_x, odom_y, odom_yaw

    # ==========================================================
    # WATCHDOG (50 Hz)
    # ==========================================================

    def _watchdog(self) -> None:
        if not self._scan_ready:
            return
        # No interrompre durant fine centering
        if self._phase == DebugPhase.DOCKING:
            dock_state = self._docker.get_state()
            if dock_state == DockState.FINE_CENTRE:
                return
        if self._avoider.is_front_danger():
            self._publish_stop()
            self.get_logger().warn(
                f'[WATCHDOG] PERILL AL FRONT — parada! (fase={self._phase.name})'
            )

    # ==========================================================
    # CONTROL LOOP (20 Hz)
    # ==========================================================

    def _control_loop(self) -> None:
        if not self._scan_ready or not self._odom_ready:
            return
        if self._phase == DebugPhase.DONE:
            return

        # Anti-stuck
        delta_yaw = abs(normalize_angle(self._yaw - self._prev_yaw))
        self._avoider.update_force_rotate(delta_yaw)
        self._prev_yaw = self._yaw

        # Dispatch
        if self._phase == DebugPhase.EXPLORE:
            self._handle_explore()
        elif self._phase == DebugPhase.RETURN_BASE:
            self._handle_return()
        elif self._phase == DebugPhase.DOCKING:
            self._handle_docking()

        # Log cada segon
        if not hasattr(self, '_tick'):
            self._tick = 0
        self._tick += 1
        if self._tick % Config.CONTROL_HZ == 0:
            elapsed = time.time() - self._phase_start
            dist_wp = self._navigator.get_distance_to_target()
            self.get_logger().info(
                f'  [{self._phase.name}] '
                f'pos=({self._x:.2f},{self._y:.2f}) '
                f'dist_wp={dist_wp:.2f}m '
                f'nav={self._navigator.get_state().name} '
                f'avoid={self._avoider.get_state().name} '
                f't={elapsed:.0f}s'
            )

    # ----------------------------------------------------------
    # EXPLORE
    # ----------------------------------------------------------

    def _handle_explore(self) -> None:
        """Explora i busca l'estació simultàniament."""

        # Primer tick: posar primer waypoint
        if self._navigator.is_idle() and self._wp_queue:
            wx, wy = self._wp_queue.pop(0)
            self._navigator.set_waypoint(wx, wy)
            self.get_logger().info(
                f'  📍 Explorant cap a ({wx:.2f},{wy:.2f}), '
                f'{len(self._wp_queue)} restants'
            )

        # Obstacle avoidance + navegació
        wp_x, wp_y = self._next_waypoint()
        cmd, in_avoidance = self._avoider.compute(
            self._x, self._y, self._yaw, wp_x, wp_y,
        )
        if in_avoidance:
            self._publish(cmd.linear_x, cmd.angular_z)
        else:
            nav_cmd = self._navigator.step()
            self._publish(nav_cmd.linear_x, nav_cmd.angular_z)

        # Detecció d'estació (concurrent)
        result = self._detector.detect(self._x, self._y, self._yaw)
        if result is not None:
            self._station_map_x = result.centre_map_x
            self._station_map_y = result.centre_map_y
            self.get_logger().info('=' * 60)
            self.get_logger().info(
                f'  🔋 ESTACIÓ TROBADA! map=({self._station_map_x:.3f},'
                f'{self._station_map_y:.3f})'
            )
            for i, (px, py) in enumerate(result.pillars_map):
                self.get_logger().info(f'    Pilar {i+1}: ({px:.3f}, {py:.3f})')
            self.get_logger().info('=' * 60)

            # Transició a RETURN_BASE
            self._transition(DebugPhase.RETURN_BASE)
            self._wp_queue = list(Config.WAYPOINTS_RETURN)
            wx, wy = self._wp_queue.pop(0)
            self._navigator.set_waypoint(wx, wy)
            self._avoider.reset()
            return

        # Avançar waypoints
        if self._navigator.has_arrived():
            self._avoider.reset()
            if self._wp_queue:
                wx, wy = self._wp_queue.pop(0)
                self._navigator.set_waypoint(wx, wy)
                self.get_logger().info(
                    f'  📍 Explorant cap a ({wx:.2f},{wy:.2f}), '
                    f'{len(self._wp_queue)} restants'
                )
            else:
                # Sweep complet sense trobar estació — repetir
                self._explore_sweeps += 1
                self.get_logger().warn(
                    f'  Sweep #{self._explore_sweeps} complet sense estació — repetint'
                )
                self._wp_queue = list(Config.WAYPOINTS_EXPLORE)
                wx, wy = self._wp_queue.pop(0)
                self._navigator.set_waypoint(wx, wy)

    # ----------------------------------------------------------
    # RETURN_BASE
    # ----------------------------------------------------------

    def _handle_return(self) -> None:
        """Torna a Punt Base i guarda el mapa."""

        wp_x, wp_y = self._next_waypoint()
        cmd, in_avoidance = self._avoider.compute(
            self._x, self._y, self._yaw, wp_x, wp_y,
        )
        if in_avoidance:
            self._publish(cmd.linear_x, cmd.angular_z)
        else:
            nav_cmd = self._navigator.step()
            self._publish(nav_cmd.linear_x, nav_cmd.angular_z)

        if self._navigator.has_arrived():
            self._avoider.reset()
            self.get_logger().info('  ✓ Arribat a Punt Base — guardant mapa')
            self._save_map()

            # Transició a DOCKING
            self._transition(DebugPhase.DOCKING)
            self._docker.activate(self._station_map_x, self._station_map_y)

    # ----------------------------------------------------------
    # DOCKING
    # ----------------------------------------------------------

    def _handle_docking(self) -> None:
        """Approach + Fine centering."""

        cmd = self._docker.step(self._x, self._y, self._yaw)

        # Durant APPROACH, obstacle avoidance actiu
        dock_state = self._docker.get_state()
        if dock_state == DockState.APPROACH:
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

        # Log extra per docking
        if not hasattr(self, '_dock_tick'):
            self._dock_tick = 0
        self._dock_tick += 1
        if self._dock_tick % (Config.CONTROL_HZ // 2) == 0:  # cada 0.5s
            status = self._docker.get_status()
            dist_stn = math.sqrt(
                (self._x - self._station_map_x)**2 +
                (self._y - self._station_map_y)**2
            )
            self.get_logger().info(
                f'    🔧 dock={dock_state.name} '
                f'dist_stn={dist_stn:.3f}m '
                f'offset={status.offset:.4f}m '
                f'centroid=({status.centroid_x:.3f},{status.centroid_y:.3f})'
            )

        # Comprovar fi
        if self._docker.is_docked():
            self._transition(DebugPhase.DONE)
            status = self._docker.get_status()
            dist_stn = math.sqrt(
                (self._x - self._station_map_x)**2 +
                (self._y - self._station_map_y)**2
            )
            self.get_logger().info('=' * 60)
            self.get_logger().info('  ✓ DOCKED CORRECTAMENT!')
            self.get_logger().info(f'  Posició final  : ({self._x:.4f}, {self._y:.4f})')
            self.get_logger().info(f'  Dist a estació : {dist_stn:.4f} m')
            self.get_logger().info(f'  Offset centroid: {status.offset:.4f} m')
            self.get_logger().info('=' * 60)
            self._publish_stop()

        elif self._docker.has_failed():
            self._transition(DebugPhase.DONE)
            self.get_logger().error('=' * 60)
            self.get_logger().error('  ✗ DOCKING FALLIT!')
            self.get_logger().error('=' * 60)
            self._publish_stop()

    # ==========================================================
    # HELPERS
    # ==========================================================

    def _next_waypoint(self):
        if self._navigator._target_x is not None:
            return self._navigator._target_x, self._navigator._target_y
        return self._x, self._y

    def _transition(self, new_phase: DebugPhase) -> None:
        self.get_logger().info(
            f'  ── Transició: {self._phase.name} → {new_phase.name} ──'
        )
        self._phase = new_phase
        self._phase_start = time.time()

    def _save_map(self) -> None:
        path = os.path.expanduser(Config.MAP_SAVE_PATH)
        cmd  = ['ros2', 'run', 'nav2_map_server', 'map_saver_cli', '-f', path]
        try:
            subprocess.Popen(cmd)
            self.get_logger().info(f'  Mapa guardat → {path}')
        except Exception as exc:
            self.get_logger().error(f'  Error guardant mapa: {exc}')

    def _publish(self, linear_x: float, angular_z: float) -> None:
        cmd = TwistStamped()
        cmd.header.stamp    = self.get_clock().now().to_msg()
        cmd.header.frame_id = ''
        cmd.twist.linear.x  = max(-0.20, min(0.20, linear_x))
        cmd.twist.angular.z = max(-1.00, min(1.00, angular_z))
        self._cmd_pub.publish(cmd)

    def _publish_stop(self) -> None:
        cmd = TwistStamped()
        cmd.header.stamp    = self.get_clock().now().to_msg()
        cmd.header.frame_id = ''
        self._cmd_pub.publish(cmd)

    def _try_slam_pose(self):
        try:
            tf = self._tf_buffer.lookup_transform(
                Config.MAP_FRAME, Config.BASE_FRAME, rclpy.time.Time(),
            )
            x   = tf.transform.translation.x
            y   = tf.transform.translation.y
            yaw = self._quat_to_yaw(tf.transform.rotation)
            return x, y, yaw
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None, None, None

    @staticmethod
    def _quat_to_yaw(q) -> float:
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def shutdown(self) -> None:
        self.get_logger().info('Apagant debug_explore_dock_node — parant robot...')
        self._publish_stop()


# ============================================================
# ENTRY POINT
# ============================================================
def main(args=None) -> None:
    rclpy.init(args=args)
    node = DebugExploreDockNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt rebut')
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
