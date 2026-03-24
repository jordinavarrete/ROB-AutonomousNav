#!/usr/bin/env python3
"""
debug_route_planner_node.py — Debug node per testejar el route planner.

Navega 3m endavant amb obstacle avoidance + fallback A* del route planner.
Si l'avoidance porta >15s sense progrés, activa el route planner.

Ús:
    ros2 run autonomous_nav debug_route_planner_node
    ros2 run autonomous_nav debug_route_planner_node --ros-args \
        -p distance:=5.0 -p timeout:=20.0
"""

# ============================================================
# CONFIGURACIÓ
# ============================================================
class Config:
    DISTANCE            = 3.0     # m endavant
    AVOIDANCE_TIMEOUT_S = 15.0    # s — trigger A* si avoidance porta tant
    CONTROL_HZ          = 20
    WATCHDOG_HZ         = 50
    MAP_FRAME           = 'map'
    BASE_FRAME          = 'base_footprint'


# ============================================================
# IMPORTS
# ============================================================
import math
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan

from autonomous_nav.navigation import WaypointNavigator, normalize_angle
from autonomous_nav.obstacle_avoidance import ObstacleAvoidance, AvoidState
from autonomous_nav.route_planner import RoutePlanner


# ============================================================
# NODE
# ============================================================
class DebugRoutePlannerNode(Node):

    def __init__(self) -> None:
        super().__init__('debug_route_planner_node')

        self.declare_parameter('distance', Config.DISTANCE)
        self.declare_parameter('timeout', Config.AVOIDANCE_TIMEOUT_S)

        self._dist    = self.get_parameter('distance').value
        self._timeout = self.get_parameter('timeout').value

        qos_r = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)
        qos_b = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)

        self._cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', qos_r)
        self.create_subscription(LaserScan,    '/scan', self._scan_cb, qos_b)
        self.create_subscription(Odometry,     '/odom', self._odom_cb, qos_r)
        self.create_subscription(OccupancyGrid, '/map', self._map_cb,  qos_r)

        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._x = self._y = self._yaw = self._prev_yaw = 0.0
        self._scan_ready = self._odom_ready = False
        self._done = False
        self._wp_set = False

        self._navigator = WaypointNavigator(logger=self.get_logger())
        self._avoider   = ObstacleAvoidance(logger=self.get_logger())
        self._planner   = RoutePlanner(logger=self.get_logger())

        # Route planner state
        self._avoid_start_t = None
        self._avoid_dist0   = float('inf')
        self._using_temp    = False
        self._original_wp   = None
        self._wp_queue      = []
        self._planner_uses  = 0

        self.create_timer(1.0 / Config.CONTROL_HZ, self._control)
        self.create_timer(1.0 / Config.WATCHDOG_HZ, self._watchdog)

        self.get_logger().info('=' * 60)
        self.get_logger().info('  DEBUG ROUTE PLANNER')
        self.get_logger().info(f'  Distància: {self._dist:.1f}m endavant')
        self.get_logger().info(f'  A* timeout: {self._timeout:.0f}s')
        self.get_logger().info('=' * 60)

    # Callbacks
    def _scan_cb(self, msg):
        self._avoider.update_scan(msg)
        self._scan_ready = True

    def _odom_cb(self, msg):
        ox = msg.pose.pose.position.x
        oy = msg.pose.pose.position.y
        oyaw = self._quat_to_yaw(msg.pose.pose.orientation)
        self._navigator.set_odom_pose(ox, oy, oyaw)
        self._odom_ready = True
        sx, sy, syaw = self._try_slam()
        if sx is not None:
            self._navigator.set_slam_pose(sx, sy, syaw)
            self._x, self._y, self._yaw = sx, sy, syaw
        else:
            self._x, self._y, self._yaw = ox, oy, oyaw

    def _map_cb(self, msg):
        self._planner.update_map(msg)
        self.get_logger().info(
            f'[MAP] Rebut: {msg.info.width}x{msg.info.height} '
            f'res={msg.info.resolution:.3f}m'
        )

    def _watchdog(self):
        if self._scan_ready and self._avoider.is_front_danger():
            self._stop()

    # Control loop
    def _control(self):
        if not self._scan_ready or not self._odom_ready or self._done:
            return

        # Primer tick: calcular waypoint 3m endavant
        if not self._wp_set:
            wx = self._x + self._dist * math.cos(self._yaw)
            wy = self._y + self._dist * math.sin(self._yaw)
            self._navigator.set_waypoint(wx, wy)
            self._wp_set = True
            self.get_logger().info(
                f'  Waypoint: ({wx:.2f},{wy:.2f}) — {self._dist:.1f}m endavant'
            )

        delta_yaw = abs(normalize_angle(self._yaw - self._prev_yaw))
        self._avoider.update_force_rotate(delta_yaw)
        self._prev_yaw = self._yaw

        wp_x = self._navigator._target_x
        wp_y = self._navigator._target_y

        cmd, in_avoidance = self._avoider.compute(
            self._x, self._y, self._yaw, wp_x, wp_y,
        )

        if in_avoidance:
            self._pub(cmd.linear_x, cmd.angular_z)
            self._track_avoidance(wp_x, wp_y)
        else:
            self._reset_tracker()
            nav_cmd = self._navigator.step()
            self._pub(nav_cmd.linear_x, nav_cmd.angular_z)

        # Arrived?
        if self._navigator.has_arrived():
            self._avoider.reset()
            self._reset_tracker()
            if self._using_temp and not self._wp_queue:
                self._finish_temp()
            elif self._wp_queue:
                nw = self._wp_queue.pop(0)
                self._navigator.set_waypoint(*nw)
            else:
                self._done = True
                d = math.sqrt((self._x - wp_x)**2 + (self._y - wp_y)**2)
                self.get_logger().info('=' * 60)
                self.get_logger().info('  ✓ WAYPOINT ASSOLIT!')
                self.get_logger().info(f'  Error: {d:.3f}m')
                self.get_logger().info(f'  A* activat: {self._planner_uses} vegades')
                self.get_logger().info('=' * 60)
                self._stop()

        # Log
        if not hasattr(self, '_t'):
            self._t = 0
        self._t += 1
        if self._t % Config.CONTROL_HZ == 0:
            d = math.sqrt((self._x - wp_x)**2 + (self._y - wp_y)**2)
            has_map = '✓' if self._planner.has_map() else '✗'
            self.get_logger().info(
                f'  pos=({self._x:.2f},{self._y:.2f}) dist={d:.2f}m '
                f'avoid={self._avoider.get_state().name} '
                f'map={has_map} temp={self._using_temp} '
                f'A*={self._planner_uses}'
            )

    # Route planner fallback
    def _track_avoidance(self, wp_x, wp_y):
        now = time.time()
        if self._avoid_start_t is None:
            self._avoid_start_t = now
            self._avoid_dist0 = math.sqrt(
                (self._x - wp_x)**2 + (self._y - wp_y)**2
            )
            return
        elapsed = now - self._avoid_start_t
        if elapsed < self._timeout:
            return
        cur_d = math.sqrt((self._x - wp_x)**2 + (self._y - wp_y)**2)
        progress = self._avoid_dist0 - cur_d
        if progress > 0.50:
            self._avoid_start_t = now
            self._avoid_dist0 = cur_d
            return
        self.get_logger().warn(
            f'  ⚠ Avoidance {elapsed:.0f}s sense progrés — activant A*'
        )
        self._trigger_planner(wp_x, wp_y)

    def _trigger_planner(self, wp_x, wp_y):
        if not self._planner.has_map():
            self.get_logger().warn('  No hi ha mapa — seguint Bug2')
            self._avoid_start_t = time.time()
            return
        temp = self._planner.compute_path(self._x, self._y, wp_x, wp_y)
        if not temp:
            self.get_logger().warn('  A* no ha trobat camí — seguint Bug2')
            self._avoid_start_t = time.time()
            return
        self._planner_uses += 1
        self._original_wp = (wp_x, wp_y)
        self._using_temp = True
        self._wp_queue = list(temp)
        first = self._wp_queue.pop(0)
        self._navigator.set_waypoint(*first)
        self._avoider.reset()
        self._avoid_start_t = None
        self.get_logger().info(
            f'  🗺 A* #{self._planner_uses}: {len(temp)} waypoints. '
            f'Primer: ({first[0]:.2f},{first[1]:.2f})'
        )

    def _finish_temp(self):
        self._using_temp = False
        if self._original_wp:
            ox, oy = self._original_wp
            self._navigator.set_waypoint(ox, oy)
            self._original_wp = None
            self.get_logger().info(
                f'  ✓ Temp WPs acabats — tornant a ({ox:.2f},{oy:.2f})'
            )

    def _reset_tracker(self):
        self._avoid_start_t = None

    # Helpers
    def _pub(self, lx, az):
        c = TwistStamped()
        c.header.stamp = self.get_clock().now().to_msg()
        c.twist.linear.x = max(-0.20, min(0.20, lx))
        c.twist.angular.z = max(-1.00, min(1.00, az))
        self._cmd_pub.publish(c)

    def _stop(self):
        c = TwistStamped()
        c.header.stamp = self.get_clock().now().to_msg()
        self._cmd_pub.publish(c)

    def _try_slam(self):
        try:
            tf = self._tf_buffer.lookup_transform(
                Config.MAP_FRAME, Config.BASE_FRAME, rclpy.time.Time())
            return (tf.transform.translation.x, tf.transform.translation.y,
                    self._quat_to_yaw(tf.transform.rotation))
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None, None, None

    @staticmethod
    def _quat_to_yaw(q):
        return math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))

    def shutdown(self):
        self._stop()


def main(args=None):
    rclpy.init(args=args)
    node = DebugRoutePlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
