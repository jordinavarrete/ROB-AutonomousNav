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
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from autonomous_nav.station_detector import StationDetector
from autonomous_nav.docking_controller import DockingController

class DebugDockingNode(Node):
    def __init__(self):
        super().__init__('debug_docking_node')
        self.get_logger().info('Debug Docking Node Started.')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        self.detector = StationDetector(logger=self.get_logger())
        self.docker = DockingController(logger=self.get_logger())

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.has_odom = False

        self.timer = self.create_timer(0.05, self.control_loop)

    def odom_callback(self, msg: Odometry):
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.current_yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.current_x = pos.x
        self.current_y = pos.y
        self.has_odom = True

    def scan_callback(self, msg: LaserScan):
        if not self.has_odom: return
        self.detector.update_scan(msg)
        
        # In debug we assume odometry frame = map frame
        if not self.detector.is_confirmed():
            result = self.detector.detect(self.current_x, self.current_y, self.current_yaw)
            if result is not None:
                self.docker.start_docking(result.centre_map_x, result.centre_map_y)

    def control_loop(self):
        if not self.has_odom: return
        
        # If detected, dock.
        if self.detector.is_confirmed() and not self.docker.is_docked():
            cmd = self.docker.step(self.current_x, self.current_y, self.current_yaw)
            msg = Twist()
            msg.linear.x = cmd.linear_x
            msg.angular.z = cmd.angular_z
            self.cmd_pub.publish(msg)
        elif self.docker.is_docked():
            # Stop cleanly
            msg = Twist()
            self.cmd_pub.publish(msg)
            self.get_logger().info("Debug Docking completed!")
            raise SystemExit

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
