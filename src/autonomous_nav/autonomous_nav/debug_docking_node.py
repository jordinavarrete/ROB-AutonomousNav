#!/usr/bin/env python3
"""
debug_docking_node.py — Testing standalone node for Phase III precision docking.

Subscribes to /scan to run StationDetector.
Once detected, invokes DockingController to park accurately.
Does NOT use Dynamic A* or PGO.
"""

import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from autonomous_nav.navigation import normalize_angle
from autonomous_nav.station_detector import StationDetector
from autonomous_nav.docking_controller import DockingController
from autonomous_nav.obstacle_avoidance import ObstacleAvoidance


class Config:
    CONTROL_HZ = 20
    WATCHDOG_HZ = 50

class DebugDockingNode(Node):
    def __init__(self):
        super().__init__('debug_docking_node')
        qos_reliable = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)
        qos_best_effort = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)

        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', qos_reliable)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, qos_reliable)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_best_effort)

        self.detector = StationDetector(logger=self.get_logger())
        self.docker = DockingController(logger=self.get_logger())
        self.avoider = ObstacleAvoidance(logger=self.get_logger())

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.has_odom = False
        self.has_scan = False
        self._scan_count = 0
        self._tick = 0
        self._docking_started = False
        self._last_cmd_log_tick = 0
        self._prev_yaw = 0.0

        self.timer = self.create_timer(1.0 / Config.CONTROL_HZ, self.control_loop)
        self.watchdog_timer = self.create_timer(1.0 / Config.WATCHDOG_HZ, self.watchdog_loop)

        self.get_logger().info('================ DEBUG DOCKING NODE ================')
        self.get_logger().info('Esperant /scan i /odom per iniciar docking')
        self.get_logger().info('QoS: /scan=BEST_EFFORT, /odom=RELIABLE, /cmd_vel=RELIABLE')
        self.get_logger().info('====================================================')

    def odom_callback(self, msg: Odometry):
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.current_yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.current_x = pos.x
        self.current_y = pos.y
        self.has_odom = True

    def scan_callback(self, msg: LaserScan):
        self.has_scan = True
        self._scan_count += 1
        self.avoider.update_scan(msg)

        if self._scan_count == 1:
            self.get_logger().info(
                f'Primer /scan rebut: n_ranges={len(msg.ranges)}, '
                f'angle_min={msg.angle_min:.3f}, angle_max={msg.angle_max:.3f}'
            )

        if not self.has_odom:
            return

        self.detector.update_scan(msg)
        
        # In debug we assume odometry frame = map frame
        if not self.detector.is_confirmed():
            result = self.detector.detect(self.current_x, self.current_y, self.current_yaw)
            if result is not None:
                self.docker.start_docking(result.centre_map_x, result.centre_map_y)
                self.avoider.reset()
                if not self._docking_started:
                    self._docking_started = True
                    self.get_logger().info(
                        f'Estacio confirmada. Iniciant docking cap a '
                        f'({result.centre_map_x:.3f}, {result.centre_map_y:.3f})'
                    )

    def watchdog_loop(self):
        if not self.has_scan:
            return
        if self.avoider.is_front_danger() and self.detector.is_confirmed() and not self.docker.is_docked():
            self.publish_stop()
            self.get_logger().warn('[WATCHDOG] PERILL AL FRONT durant docking — parada d\'emergencia')

    def control_loop(self):
        self._tick += 1

        if not self.has_odom or not self.has_scan:
            if self._tick % 20 == 0:
                self.get_logger().warn(
                    f'Esperant dades... odom={self.has_odom}, scan={self.has_scan}, scan_count={self._scan_count}'
                )
            return

        if not self.detector.is_confirmed() and self._tick % 20 == 0:
            self.get_logger().info(
                f'Buscant estacio... pose=({self.current_x:.2f}, {self.current_y:.2f}, '
                f'yaw={math.degrees(self.current_yaw):.1f}deg), scans={self._scan_count}'
            )

        delta_yaw = abs(normalize_angle(self.current_yaw - self._prev_yaw))
        self.avoider.update_force_rotate(delta_yaw)
        self._prev_yaw = self.current_yaw
        
        # If detected, dock.
        if self.detector.is_confirmed() and not self.docker.is_docked():
            dock_cmd = self.docker.step(self.current_x, self.current_y, self.current_yaw)

            target_x = self.docker.target_x if self.docker.target_x is not None else self.current_x
            target_y = self.docker.target_y if self.docker.target_y is not None else self.current_y
            avoid_cmd, in_avoidance = self.avoider.compute(
                self.current_x,
                self.current_y,
                self.current_yaw,
                target_x,
                target_y,
            )

            cmd = avoid_cmd if in_avoidance else dock_cmd
            self.publish_cmd(cmd.linear_x, cmd.angular_z)

            if self._tick - self._last_cmd_log_tick >= 20:
                self._last_cmd_log_tick = self._tick
                dist = math.hypot(target_x - self.current_x, target_y - self.current_y)
                avoid_state = self.avoider.get_state().name
                mode = 'AVOID' if in_avoidance else 'DOCK'
                self.get_logger().info(
                    f'[DOCKING] mode={mode} avoid={avoid_state} '
                    f'cmd: v={cmd.linear_x:.3f} m/s, w={cmd.angular_z:.3f} rad/s, '
                    f'dist={dist:.3f} m, cmd_subs={self.cmd_pub.get_subscription_count()}'
                )
        elif self.docker.is_docked():
            # Stop cleanly
            self.publish_stop()
            self.get_logger().info("Debug Docking completed!")
            raise SystemExit

    def publish_cmd(self, linear_x: float, angular_z: float):
        linear_max = 0.20
        angular_max = 1.00

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = max(-linear_max, min(linear_max, linear_x))
        msg.twist.angular.z = max(-angular_max, min(angular_max, angular_z))
        self.cmd_pub.publish(msg)

    def publish_stop(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        self.cmd_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = DebugDockingNode()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
