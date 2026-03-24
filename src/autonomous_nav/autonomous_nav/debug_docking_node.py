#!/usr/bin/env python3
"""
debug_docking_node.py — Node de debug per testejar el docking de precisió.

Permet indicar la posició de l'estació de càrrega i executar NOMÉS
la maniobra de docking (approach + fine centering) sense nécessitat
d'executar cap altra fase de la missió.

Ús:
    ros2 run autonomous_nav debug_docking_node
    ros2 run autonomous_nav debug_docking_node --ros-args \
        -p station_x:=5.0 -p station_y:=12.0

Topics:
  Subscriu:  /scan  (LaserScan, BEST_EFFORT)
             /odom  (Odometry,  RELIABLE)
  Publica:   /cmd_vel (TwistStamped, RELIABLE)
"""

# ============================================================
# CONFIGURACIÓ
# ============================================================
class Config:
    # Posició de l'estació de càrrega al mapa
    STATION_X       = 5.00
    STATION_Y       = 12.00

    # Rates de control
    CONTROL_HZ      = 20
    WATCHDOG_HZ     = 50

    # Frames TF
    MAP_FRAME       = 'map'
    BASE_FRAME      = 'base_footprint'


# ============================================================
# IMPORTS
# ============================================================
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from autonomous_nav.navigation import WaypointNavigator, normalize_angle
from autonomous_nav.obstacle_avoidance import ObstacleAvoidance
from autonomous_nav.station_detector import StationDetector
from autonomous_nav.docking import DockingController, DockState


# ============================================================
# NODE PRINCIPAL
# ============================================================
class DebugDockingNode(Node):
    """
    Node per debuggar el docking sense executar la missió sencera.

    Flux:
      1. Rep la posició de l'estació via paràmetres
      2. Activa el DockingController directament
      3. Mostra en temps real: estat del docking, offset al centroid, pilars detectats
      4. Para quan arriba a DOCKED o FAILED
    """

    def __init__(self) -> None:
        super().__init__('debug_docking_node')

        # Paràmetres
        self.declare_parameter('station_x', Config.STATION_X)
        self.declare_parameter('station_y', Config.STATION_Y)

        self._station_x = self.get_parameter('station_x').value
        self._station_y = self.get_parameter('station_y').value

        self.get_logger().info('=' * 60)
        self.get_logger().info('  DEBUG DOCKING NODE')
        self.get_logger().info('=' * 60)
        self.get_logger().info(
            f'  Estació objectiu: ({self._station_x:.2f}, {self._station_y:.2f})'
        )
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

        # Estat
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self._scan_ready = False
        self._odom_ready = False
        self._done = False

        # Sub-mòduls
        self._navigator = WaypointNavigator(logger=self.get_logger())
        self._detector  = StationDetector(logger=self.get_logger())
        self._docker    = DockingController(
            navigator=self._navigator,
            detector=self._detector,
            logger=self.get_logger(),
        )

        # Tracking
        self._last_dock_state = DockState.IDLE
        self._activated = False

        # Timers
        self._control_timer  = self.create_timer(1.0 / Config.CONTROL_HZ, self._control_loop)
        self._watchdog_timer = self.create_timer(1.0 / Config.WATCHDOG_HZ, self._watchdog)

        self.get_logger().info(
            'Node inicialitzat. Esperant /scan i /odom... '
            'Un cop rebuts, s\'activarà el docking automàticament.'
        )

    # ==========================================================
    # CALLBACKS
    # ==========================================================

    def _scan_cb(self, msg: LaserScan) -> None:
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
        """Watchdog no interfera amb fine centering."""
        pass

    # ==========================================================
    # CONTROL LOOP (20 Hz)
    # ==========================================================

    def _control_loop(self) -> None:
        if not self._scan_ready or not self._odom_ready:
            return
        if self._done:
            return

        # Activar docking al primer tick
        if not self._activated:
            self.get_logger().info(
                f'Activant docking cap a ({self._station_x:.2f}, {self._station_y:.2f})'
            )
            self._docker.activate(self._station_x, self._station_y)
            self._activated = True

        # Step docking
        cmd = self._docker.step(self._x, self._y, self._yaw)

        # Detectar canvi d'estat
        current_state = self._docker.get_state()
        if current_state != self._last_dock_state:
            self.get_logger().info(
                f'  Docking: {self._last_dock_state.name} → {current_state.name}'
            )
            self._last_dock_state = current_state

        # Comprovar si ha acabat
        if self._docker.is_docked():
            self._done = True
            status = self._docker.get_status()
            dist_to_station = math.sqrt(
                (self._x - self._station_x)**2 +
                (self._y - self._station_y)**2
            )
            self.get_logger().info('=' * 60)
            self.get_logger().info('  ✓ DOCKED CORRECTAMENT!')
            self.get_logger().info(f'  Posició final  : ({self._x:.4f}, {self._y:.4f})')
            self.get_logger().info(f'  Dist a estació : {dist_to_station:.4f} m')
            self.get_logger().info(f'  Centroid offset: {status.offset:.4f} m')
            self.get_logger().info(f'  Temps fine cent: {status.elapsed_fine:.1f} s')
            self.get_logger().info('=' * 60)
            self._publish_stop()
            return

        if self._docker.has_failed():
            self._done = True
            status = self._docker.get_status()
            self.get_logger().error('=' * 60)
            self.get_logger().error('  ✗ DOCKING FALLIT!')
            self.get_logger().error(f'  Últim offset: {status.offset:.4f} m')
            self.get_logger().error(f'  Temps fine:   {status.elapsed_fine:.1f} s')
            self.get_logger().error('=' * 60)
            self._publish_stop()
            return

        self._publish(cmd.linear_x, cmd.angular_z)

        # Log cada segon
        if not hasattr(self, '_tick'):
            self._tick = 0
        self._tick += 1
        if self._tick % Config.CONTROL_HZ == 0:
            status = self._docker.get_status()
            dist = math.sqrt(
                (self._x - self._station_x)**2 +
                (self._y - self._station_y)**2
            )
            self.get_logger().info(
                f'  pos=({self._x:.2f},{self._y:.2f}) '
                f'dist_stn={dist:.3f}m '
                f'dock={current_state.name} '
                f'offset={status.offset:.4f}m '
                f'cx=({status.centroid_x:.3f},{status.centroid_y:.3f})'
            )

    # ==========================================================
    # HELPERS
    # ==========================================================

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
        self.get_logger().info('Apagant debug_docking_node — parant robot...')
        self._publish_stop()


# ============================================================
# ENTRY POINT
# ============================================================
def main(args=None) -> None:
    rclpy.init(args=args)
    node = DebugDockingNode()
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
