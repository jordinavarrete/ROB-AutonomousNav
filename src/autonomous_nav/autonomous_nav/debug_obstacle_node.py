#!/usr/bin/env python3
"""
debug_obstacle_node.py — Node de debug per testejar l'obstacle avoidance (Bug2).

Permet navegar cap a un waypoint concret i veure com reacciona l'avoidance
davant d'obstacles reals. Mostra l'estat del Bug2 (hit point, m-line, etc.)

Ús:
    ros2 run autonomous_nav debug_obstacle_node
    ros2 run autonomous_nav debug_obstacle_node --ros-args \
        -p target_x:=5.92 -p target_y:=8.12

Topics:
  Subscriu:  /scan  (LaserScan, BEST_EFFORT)
             /odom  (Odometry,  RELIABLE)
  Publica:   /cmd_vel (TwistStamped, RELIABLE)
"""

# ============================================================
# CONFIGURACIÓ
# ============================================================
class Config:
    # Waypoint objectiu per defecte (la porta)
    TARGET_X        = 5.92
    TARGET_Y        = 8.12

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
from autonomous_nav.obstacle_avoidance import ObstacleAvoidance, AvoidState


# ============================================================
# NODE PRINCIPAL
# ============================================================
class DebugObstacleNode(Node):
    """
    Node per debuggar l'obstacle avoidance amb Bug2.

    Flux:
      1. Navega cap al waypoint especificat
      2. Quan detecta un obstacle, mostra hit_point i m-line
      3. Mostra l'estat de l'avoidance en temps real
      4. Mostra quan creua la m-line i reprèn la navegació
    """

    def __init__(self) -> None:
        super().__init__('debug_obstacle_node')

        # Paràmetres ROS2
        self.declare_parameter('target_x', Config.TARGET_X)
        self.declare_parameter('target_y', Config.TARGET_Y)

        self._wp_x = self.get_parameter('target_x').value
        self._wp_y = self.get_parameter('target_y').value

        self.get_logger().info('=' * 60)
        self.get_logger().info('  DEBUG OBSTACLE AVOIDANCE (Bug2)')
        self.get_logger().info('=' * 60)
        self.get_logger().info(f'  Waypoint objectiu: ({self._wp_x:.2f}, {self._wp_y:.2f})')
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
        self._prev_yaw = 0.0
        self._scan_ready = False
        self._odom_ready = False
        self._arrived = False

        # Sub-mòduls
        self._navigator = WaypointNavigator(logger=self.get_logger())
        self._avoider   = ObstacleAvoidance(logger=self.get_logger())

        # Tracking d'estats per log
        self._last_avoid_state = AvoidState.NORMAL
        self._avoidance_count = 0

        # Timers
        self._control_timer  = self.create_timer(1.0 / Config.CONTROL_HZ, self._control_loop)
        self._watchdog_timer = self.create_timer(1.0 / Config.WATCHDOG_HZ, self._watchdog)

        self.get_logger().info('Node inicialitzat. Esperant /scan i /odom...')

    # ==========================================================
    # CALLBACKS
    # ==========================================================

    def _scan_cb(self, msg: LaserScan) -> None:
        self._avoider.update_scan(msg)
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
        if self._avoider.is_front_danger():
            self._publish_stop()
            self.get_logger().warn('[WATCHDOG] PERILL AL FRONT — parada!')

    # ==========================================================
    # CONTROL LOOP (20 Hz)
    # ==========================================================

    def _control_loop(self) -> None:
        if not self._scan_ready or not self._odom_ready:
            return
        if self._arrived:
            return

        # Primer tick: assignar waypoint
        if self._navigator.is_idle():
            self._navigator.set_waypoint(self._wp_x, self._wp_y)

        # Anti-stuck
        delta_yaw = abs(normalize_angle(self._yaw - self._prev_yaw))
        self._avoider.update_force_rotate(delta_yaw)
        self._prev_yaw = self._yaw

        # Obstacle avoidance
        cmd, in_avoidance = self._avoider.compute(
            self._x, self._y, self._yaw,
            self._wp_x, self._wp_y,
        )

        # Detectar canvi d'estat per loggejar
        current_state = self._avoider.get_state()
        if current_state != self._last_avoid_state:
            if current_state == AvoidState.AVOID_ROTATE:
                self._avoidance_count += 1
                self.get_logger().info(
                    f'  ⚠ OBSTACLE #{self._avoidance_count}! '
                    f'hit_point=({self._avoider._hit_x:.2f},{self._avoider._hit_y:.2f}) '
                    f'wp=({self._avoider._wp_hit_x:.2f},{self._avoider._wp_hit_y:.2f})'
                )
            elif current_state == AvoidState.NORMAL and self._last_avoid_state != AvoidState.NORMAL:
                self.get_logger().info(
                    f'  ✓ AVOIDANCE COMPLETAT — tornant a navegació normal'
                )
            self._last_avoid_state = current_state

        if in_avoidance:
            self._publish(cmd.linear_x, cmd.angular_z)
        else:
            nav_cmd = self._navigator.step()
            self._publish(nav_cmd.linear_x, nav_cmd.angular_z)

        # Arribada?
        if self._navigator.has_arrived():
            self._arrived = True
            dist_final = math.sqrt(
                (self._x - self._wp_x)**2 + (self._y - self._wp_y)**2
            )
            self.get_logger().info('=' * 60)
            self.get_logger().info('  ✓ WAYPOINT ASSOLIT!')
            self.get_logger().info(f'  Posició final  : ({self._x:.3f}, {self._y:.3f})')
            self.get_logger().info(f'  Error final    : {dist_final:.3f} m')
            self.get_logger().info(f'  Avoidances fets: {self._avoidance_count}')
            self.get_logger().info('=' * 60)
            self._publish_stop()

        # Log cada segon
        if not hasattr(self, '_tick'):
            self._tick = 0
        self._tick += 1
        if self._tick % Config.CONTROL_HZ == 0:
            dist = math.sqrt(
                (self._x - self._wp_x)**2 + (self._y - self._wp_y)**2
            )
            sectors = self._avoider.get_sectors()
            front_d = sectors['FRONT'].min_dist
            left_d  = sectors['LEFT'].min_dist
            right_d = sectors['RIGHT'].min_dist
            self.get_logger().info(
                f'  pos=({self._x:.2f},{self._y:.2f}) '
                f'dist={dist:.2f}m '
                f'avoid={current_state.name} '
                f'F={front_d:.2f} L={left_d:.2f} R={right_d:.2f}'
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
        self.get_logger().info('Apagant debug_obstacle_node — parant robot...')
        self._publish_stop()


# ============================================================
# ENTRY POINT
# ============================================================
def main(args=None) -> None:
    rclpy.init(args=args)
    node = DebugObstacleNode()
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
