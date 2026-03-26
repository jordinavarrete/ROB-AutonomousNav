#!/usr/bin/env python3
"""
debug_station_node.py — Node de debug per provar la detecció de l'estació.

Aquest node:
  - rep /scan i /odom
  - manté una estimació de la pose del robot (TF map->base_footprint si disponible,
    sinó odometria)
  - executa StationDetector.detect() a una taxa fixa
  - publica la posició detectada de l'estació a /debug/station_pose

Ús:
  ros2 run autonomous_nav debug_station_node
  ros2 run autonomous_nav debug_station_node --ros-args -p use_tf_pose:=false
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from autonomous_nav.station_detector import StationDetector


class Config:
    CONTROL_HZ = 20
    LOG_EVERY_S = 1.0

    MAP_FRAME = 'map'
    BASE_FRAME = 'base_footprint'


class DebugStationNode(Node):
    """Node per validar la detecció de l'estació amb el robot real/simulat."""

    def __init__(self) -> None:
        super().__init__('debug_station_node')

        self.declare_parameter('use_tf_pose', True)
        self._use_tf_pose = bool(self.get_parameter('use_tf_pose').value)

        qos_reliable = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)
        qos_best_effort = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)

        self._detector = StationDetector(logger=self.get_logger())

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0

        self._scan_ready = False
        self._odom_ready = False
        self._last_result = None

        self._station_pub = self.create_publisher(PoseStamped, '/debug/station_pose', qos_reliable)

        self.create_subscription(LaserScan, '/scan', self._scan_cb, qos_best_effort)
        self.create_subscription(Odometry, '/odom', self._odom_cb, qos_reliable)

        self._tick = 0
        self._timer = self.create_timer(1.0 / Config.CONTROL_HZ, self._control_loop)

        self.get_logger().info('================ DEBUG STATION NODE ================')
        self.get_logger().info('Esperant /scan i /odom per iniciar deteccio d\'estacio')
        self.get_logger().info(f'use_tf_pose={self._use_tf_pose}')
        self.get_logger().info('===================================================')

    def _scan_cb(self, msg: LaserScan) -> None:
        self._detector.update_scan(msg)
        self._scan_ready = True

    def _odom_cb(self, msg: Odometry) -> None:
        odom_x = msg.pose.pose.position.x
        odom_y = msg.pose.pose.position.y
        odom_yaw = self._quat_to_yaw(msg.pose.pose.orientation)

        self._x, self._y, self._yaw = odom_x, odom_y, odom_yaw
        self._odom_ready = True

        if self._use_tf_pose:
            tf_x, tf_y, tf_yaw = self._try_tf_pose()
            if tf_x is not None:
                self._x, self._y, self._yaw = tf_x, tf_y, tf_yaw

    def _control_loop(self) -> None:
        if not self._scan_ready or not self._odom_ready:
            return

        result = self._detector.detect(self._x, self._y, self._yaw)
        self._tick += 1

        if result is not None:
            if self._last_result is None:
                self.get_logger().info('')
                self.get_logger().info('********** ESTACIO DETECTADA I CONFIRMADA **********')
                self.get_logger().info(
                    f'Centre robot=({result.centre_robot_x:.3f}, {result.centre_robot_y:.3f})'
                )
                self.get_logger().info(
                    f'Centre map=({result.centre_map_x:.3f}, {result.centre_map_y:.3f})'
                )
                self.get_logger().info('****************************************************')

            self._last_result = result
            self._publish_station_pose(result.centre_map_x, result.centre_map_y)
            return

        if self._tick % int(Config.CONTROL_HZ * Config.LOG_EVERY_S) == 0:
            self.get_logger().info(
                f'Buscant estacio... pose=({self._x:.2f}, {self._y:.2f}, yaw={math.degrees(self._yaw):.1f}deg)'
            )

    def _publish_station_pose(self, x_map: float, y_map: float) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = Config.MAP_FRAME
        msg.pose.position.x = x_map
        msg.pose.position.y = y_map
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0
        self._station_pub.publish(msg)

    def _try_tf_pose(self):
        try:
            tf = self._tf_buffer.lookup_transform(
                Config.MAP_FRAME,
                Config.BASE_FRAME,
                rclpy.time.Time(),
            )
            x = tf.transform.translation.x
            y = tf.transform.translation.y
            yaw = self._quat_to_yaw(tf.transform.rotation)
            return x, y, yaw
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None, None, None

    @staticmethod
    def _quat_to_yaw(q) -> float:
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DebugStationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
