#!/usr/bin/env python3
"""
debug_nav_node.py — Node de debug per testejar la navegació bàsica.

Permet indicar:
  - Posició inicial del robot (x, y, yaw)
  - La navegació va automàticament a un punt 3m endavant del robot

Ús:
    ros2 run autonomous_nav debug_nav_node
    ros2 run autonomous_nav debug_nav_node --ros-args \
        -p start_x:=1.0 -p start_y:=2.0 -p start_yaw_deg:=90.0

Topics:
  Subscriu:  /scan  (LaserScan, BEST_EFFORT)
             /odom  (Odometry,  RELIABLE)
  Publica:   /cmd_vel (TwistStamped, RELIABLE)
"""

# ============================================================
# CONFIGURACIÓ — canvia aquí la posició inicial i distància
# ============================================================
class Config:
    # Posició inicial del robot al mapa (en metres i graus)
    # Canvia aquests valors segons on poses el robot físicament
    START_X         = 0.0    # m — posició X inicial al mapa
    START_Y         = 0.0    # m — posició Y inicial al mapa
    START_YAW_DEG   = 0.0    # graus — cap on mira (0° = eix X positiu)

    # Distància del waypoint de test (3m endavant de la direcció inicial)
    TEST_DISTANCE   = 3.0    # m

    # Rates de control
    CONTROL_HZ      = 20     # Hz
    WATCHDOG_HZ     = 50     # Hz

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

# Importa els mòduls del teu paquet
from autonomous_nav.navigation import WaypointNavigator, normalize_angle
from autonomous_nav.obstacle_avoidance import ObstacleAvoidance


# ============================================================
# NODE PRINCIPAL
# ============================================================
class DebugNavNode(Node):
    """
    Node mínim per debuggar la navegació a un punt de test.

    Flux:
      1. Llegeix posició inicial via paràmetres ROS2
      2. Calcula el waypoint de test (3m endavant)
      3. Navega fins al waypoint amb obstacle avoidance actiu
      4. Para i mostra resultat quan arriba
    """

    def __init__(self) -> None:
        super().__init__('debug_nav_node')

        # ----------------------------------------------------------
        # Declara paràmetres ROS2 (es poden sobreescriure des de CLI)
        # ----------------------------------------------------------
        self.declare_parameter('start_x',       Config.START_X)
        self.declare_parameter('start_y',       Config.START_Y)
        self.declare_parameter('start_yaw_deg', Config.START_YAW_DEG)
        self.declare_parameter('test_distance', Config.TEST_DISTANCE)

        start_x       = self.get_parameter('start_x').value
        start_y       = self.get_parameter('start_y').value
        start_yaw_deg = self.get_parameter('start_yaw_deg').value
        test_distance = self.get_parameter('test_distance').value

        start_yaw_rad = math.radians(start_yaw_deg)

        # ----------------------------------------------------------
        # Calcula el waypoint de test: 3m endavant de la direcció inicial
        #
        #   Si el robot mira cap a 0° (eix X), el waypoint és (3, 0)
        #   Si el robot mira cap a 90° (eix Y), el waypoint és (0, 3)
        #   etc.
        # ----------------------------------------------------------
        wp_x = start_x + test_distance * math.cos(start_yaw_rad)
        wp_y = start_y + test_distance * math.sin(start_yaw_rad)

        self.get_logger().info('=' * 55)
        self.get_logger().info('  DEBUG NAV NODE')
        self.get_logger().info('=' * 55)
        self.get_logger().info(
            f'  Posició inicial : ({start_x:.2f}, {start_y:.2f})'
            f'  yaw={start_yaw_deg:.1f}°'
        )
        self.get_logger().info(
            f'  Waypoint test   : ({wp_x:.2f}, {wp_y:.2f})'
            f'  (a {test_distance:.1f}m endavant)'
        )
        self.get_logger().info('=' * 55)

        # ----------------------------------------------------------
        # QoS profiles
        # ----------------------------------------------------------
        qos_reliable    = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,    depth=10)
        qos_best_effort = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)

        # ----------------------------------------------------------
        # Publisher
        # ----------------------------------------------------------
        self._cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', qos_reliable)

        # ----------------------------------------------------------
        # Subscribers
        # ----------------------------------------------------------
        self.create_subscription(LaserScan, '/scan', self._scan_cb, qos_best_effort)
        self.create_subscription(Odometry,  '/odom', self._odom_cb, qos_reliable)

        # ----------------------------------------------------------
        # TF2 per llegir pose SLAM (opcional, cau en odometria si no hi ha SLAM)
        # ----------------------------------------------------------
        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # ----------------------------------------------------------
        # Estat intern
        # ----------------------------------------------------------
        self._x   = start_x
        self._y   = start_y
        self._yaw = start_yaw_rad

        self._scan_ready  = False
        self._odom_ready  = False
        self._arrived     = False
        self._prev_yaw    = start_yaw_rad

        # ----------------------------------------------------------
        # Sub-mòduls
        # ----------------------------------------------------------
        self._navigator = WaypointNavigator(logger=self.get_logger())
        self._avoider   = ObstacleAvoidance(logger=self.get_logger())

        # Posa la posició inicial i el waypoint objectiu
        self._navigator.set_odom_pose(start_x, start_y, start_yaw_rad)
        self._navigator.set_waypoint(wp_x, wp_y)

        self._wp_x = wp_x
        self._wp_y = wp_y

        # ----------------------------------------------------------
        # Timers
        # ----------------------------------------------------------
        self._control_timer  = self.create_timer(
            1.0 / Config.CONTROL_HZ, self._control_loop
        )
        self._watchdog_timer = self.create_timer(
            1.0 / Config.WATCHDOG_HZ, self._watchdog
        )

        self.get_logger().info('Node inicialitzat. Esperant primer /scan i /odom...')

    # ==========================================================
    # CALLBACKS ROS2
    # ==========================================================

    def _scan_cb(self, msg: LaserScan) -> None:
        """Rep el scan del LiDAR i l'envia a l'esquivador d'obstacles."""
        self._avoider.update_scan(msg)
        self._scan_ready = True

    def _odom_cb(self, msg: Odometry) -> None:
        """
        Actualitza la posició del robot.
        Primer intenta llegir del TF (SLAM), si no usa odometria.
        """
        # Posició de l'odometria
        odom_x   = msg.pose.pose.position.x
        odom_y   = msg.pose.pose.position.y
        odom_yaw = self._quat_to_yaw(msg.pose.pose.orientation)

        self._navigator.set_odom_pose(odom_x, odom_y, odom_yaw)
        self._odom_ready = True

        # Intenta SLAM via TF (millor precisió)
        slam_x, slam_y, slam_yaw = self._try_slam_pose()
        if slam_x is not None:
            self._navigator.set_slam_pose(slam_x, slam_y, slam_yaw)
            self._x, self._y, self._yaw = slam_x, slam_y, slam_yaw
        else:
            self._x, self._y, self._yaw = odom_x, odom_y, odom_yaw

    # ==========================================================
    # WATCHDOG (50 Hz) — SEGURETAT
    # ==========================================================

    def _watchdog(self) -> None:
        """
        Para el robot immediatament si el sector FRONT entra en DANGER.
        S'executa a 50Hz independentment del loop de control.
        """
        if not self._scan_ready:
            return
        if self._avoider.is_front_danger():
            self._publish_stop()
            self.get_logger().warn('[WATCHDOG] PERILL AL FRONT — parada d\'emergència!')

    # ==========================================================
    # LOOP DE CONTROL (20 Hz)
    # ==========================================================

    def _control_loop(self) -> None:
        """
        Lògica principal de navegació.
        S'executa a 20Hz.
        """
        # Espera que hi hagi dades
        if not self._scan_ready or not self._odom_ready:
            return

        # Ja hem arribat, no fem res més
        if self._arrived:
            return

        # Actualitza anti-stuck (delta de yaw acumulat)
        delta_yaw = abs(normalize_angle(self._yaw - self._prev_yaw))
        self._avoider.update_force_rotate(delta_yaw)
        self._prev_yaw = self._yaw

        # --- Obstacle avoidance ---
        cmd, in_avoidance = self._avoider.compute(
            self._x, self._y, self._yaw,
            self._wp_x, self._wp_y,
        )

        if in_avoidance:
            # L'esquivador té el control
            self._publish(cmd.linear_x, cmd.angular_z)
        else:
            # El navegador té el control
            nav_cmd = self._navigator.step()
            self._publish(nav_cmd.linear_x, nav_cmd.angular_z)

        # --- Comprova si hem arribat ---
        if self._navigator.has_arrived():
            self._arrived = True
            dist_final = math.sqrt(
                (self._x - self._wp_x) ** 2 +
                (self._y - self._wp_y) ** 2
            )
            self.get_logger().info('=' * 55)
            self.get_logger().info('  ✓ WAYPOINT ASSOLIT!')
            self.get_logger().info(
                f'  Posició final : ({self._x:.3f}, {self._y:.3f})'
            )
            self.get_logger().info(
                f'  Error final   : {dist_final:.3f} m'
            )
            self.get_logger().info('=' * 55)
            self._publish_stop()

        # --- Log de progrés cada segon (cada 20 ticks) ---
        if not hasattr(self, '_tick'):
            self._tick = 0
        self._tick += 1
        if self._tick % Config.CONTROL_HZ == 0:
            dist = math.sqrt(
                (self._x - self._wp_x) ** 2 +
                (self._y - self._wp_y) ** 2
            )
            nav_state = self._navigator.get_state().name
            avoid_state = self._avoider.get_state().name
            self.get_logger().info(
                f'  pos=({self._x:.2f},{self._y:.2f})'
                f'  dist_wp={dist:.2f}m'
                f'  nav={nav_state}'
                f'  avoid={avoid_state}'
            )

    # ==========================================================
    # HELPERS
    # ==========================================================

    def _publish(self, linear_x: float, angular_z: float) -> None:
        """Publica una comanda de velocitat amb límits de seguretat."""
        LINEAR_MAX  = 0.20   # m/s
        ANGULAR_MAX = 1.00   # rad/s

        cmd = TwistStamped()
        cmd.header.stamp    = self.get_clock().now().to_msg()
        cmd.header.frame_id = ''
        cmd.twist.linear.x  = max(-LINEAR_MAX,  min(LINEAR_MAX,  linear_x))
        cmd.twist.angular.z = max(-ANGULAR_MAX, min(ANGULAR_MAX, angular_z))
        self._cmd_pub.publish(cmd)

    def _publish_stop(self) -> None:
        """Publica velocitat zero per parar el robot."""
        cmd = TwistStamped()
        cmd.header.stamp    = self.get_clock().now().to_msg()
        cmd.header.frame_id = ''
        self._cmd_pub.publish(cmd)

    def _try_slam_pose(self):
        """
        Intenta llegir la pose corregida pel SLAM des del TF tree.
        Retorna (x, y, yaw) si disponible, o (None, None, None) si no.
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

    @staticmethod
    def _quat_to_yaw(q) -> float:
        """Converteix quaternion a angle yaw [rad]."""
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def shutdown(self) -> None:
        """Atura el robot en apagar el node."""
        self.get_logger().info('Apagant debug_nav_node — parant robot...')
        self._publish_stop()


# ============================================================
# ENTRY POINT
# ============================================================
def main(args=None) -> None:
    rclpy.init(args=args)
    node = DebugNavNode()
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
