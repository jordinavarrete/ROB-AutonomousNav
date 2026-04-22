#!/usr/bin/env python3
"""
mission_node.py — Simplified main orchestrator for Phase I, II and III.

Phase I:  Navigate through waypoints A → C → D → F → Door
Phase II: Explore corridor (P Base → Q → R → S → T → U → P Base) and detect station
Phase III: Precision docking at the charging station

No PGO or D* — uses simple odometry and direct waypoint navigation.
"""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from autonomous_nav.navigation import WaypointNavigator, Config
from autonomous_nav.mission_logger import MissionLogger
from autonomous_nav.station_detector import StationDetector
from autonomous_nav.docking_controller import DockingController

INITIAL_X = 4.280
INITIAL_Y = 1.735
INITIAL_YAW = 0.0

class MissionNode(Node):
    def __init__(self):
        super().__init__('mission_node')
        self.get_logger().info(
            f"Mission Node Initialising | Initial Pose: ({INITIAL_X:.3f}, {INITIAL_Y:.3f})"
        )
        
        # ROS2 Publishers & Subscribers
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Core Logic Modules
        self.navigator = WaypointNavigator(logger=self.get_logger())
        self.logger = MissionLogger()
        self.station_detector = StationDetector(logger=self.get_logger())
        self.docker = DockingController(logger=self.get_logger())
        
        # Pose State (odometry-based)
        self.current_x = INITIAL_X
        self.current_y = INITIAL_Y
        self.current_yaw = INITIAL_YAW
        self.odom_initialized = False
        self.odom_offset_x = 0.0
        self.odom_offset_y = 0.0
        self.odom_offset_yaw = 0.0
        
        # Initialize navigator with initial pose
        self.navigator.set_odom_pose(INITIAL_X, INITIAL_Y, INITIAL_YAW)
        
        # Mission State
        self.current_phase = 1
        self.global_waypoint_idx = 0
        self.global_waypoints = Config.WAYPOINTS_PHASE1
        
        # Station detection (Phase II/III)
        self.station_x = -1.0
        self.station_y = -1.0
        
        # Set first waypoint for Phase I
        self._set_next_waypoint()

        # Control loop: 20 Hz
        self.timer = self.create_timer(0.05, self.control_loop)
        self.get_logger().info("Mission Node Ready.")

    def odom_callback(self, msg: Odometry):
        """Update robot pose from odometry with map alignment."""
        odom_x = msg.pose.pose.position.x
        odom_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        
        # Extract yaw from quaternion
        odom_yaw = math.atan2(
            2 * (q.w * q.z + q.x * q.y),
            1 - 2 * (q.y * q.y + q.z * q.z)
        )
        
        # First odometry: calculate offset to align with initial position
        if not self.odom_initialized:
            self.odom_offset_x = INITIAL_X - odom_x
            self.odom_offset_y = INITIAL_Y - odom_y
            self.odom_offset_yaw = INITIAL_YAW - odom_yaw
            self.odom_initialized = True
            self.get_logger().info(
                f"Odometry offset calibrated: ({self.odom_offset_x:.3f}, {self.odom_offset_y:.3f})"
            )
            return
        
        # Apply offset to align odometry with map frame
        self.current_x = odom_x + self.odom_offset_x
        self.current_y = odom_y + self.odom_offset_y
        self.current_yaw = odom_yaw + self.odom_offset_yaw
        
        # Feed pose to navigator
        self.navigator.set_odom_pose(self.current_x, self.current_y, self.current_yaw)

    def scan_callback(self, msg: LaserScan):
        """Process LiDAR scan for station detection."""
        pose = (self.current_x, self.current_y, self.current_yaw)
        
        # Phase II/III: Detect charging station
        if self.current_phase in [1, 2]:
            self.station_detector.update_scan(msg)
            if not self.station_detector.is_confirmed():
                res = self.station_detector.detect(pose[0], pose[1], pose[2])
                if res is not None:
                    self.station_x = res.centre_map_x
                    self.station_y = res.centre_map_y
                    self.get_logger().info(
                        f"Station detected at ({self.station_x:.3f}, {self.station_y:.3f})"
                    )

    def _set_next_waypoint(self):
        """Set the next waypoint from the current phase list."""
        if self.global_waypoint_idx < len(self.global_waypoints):
            wp_x, wp_y = self.global_waypoints[self.global_waypoint_idx]
            self.navigator.set_waypoint(wp_x, wp_y)
            self.get_logger().info(
                f"Phase {self.current_phase} | WP#{self.global_waypoint_idx}: "
                f"({wp_x:.3f}, {wp_y:.3f})"
            )

    def _advance_phase(self):
        """Transition to the next phase."""
        if self.current_phase == 1:
            self.get_logger().info("✓ Phase I Complete! Starting Phase II (Exploration).")
            self.current_phase = 2
            self.global_waypoint_idx = 0
            self.global_waypoints = Config.WAYPOINTS_PHASE2
            self._set_next_waypoint()
            
        elif self.current_phase == 2:
            self.get_logger().info("✓ Phase II Complete! Checking for station...")
            if self.station_detector.is_confirmed():
                self.get_logger().info(
                    f"✓ Station confirmed at ({self.station_x:.3f}, {self.station_y:.3f})"
                )
                self.get_logger().info("Starting Phase III (Precision Docking).")
                self.current_phase = 3
                self.docker.start_docking(self.station_x, self.station_y)
            else:
                self.get_logger().error("✗ Station NOT detected after exploration! Mission aborted.")
                self.terminate_mission()

    def control_loop(self):
        """Main control loop @ 20 Hz."""
        # Log mission state
        phase_str = 'I' if self.current_phase == 1 else 'II' if self.current_phase == 2 else 'III'
        self.logger.update(
            phase=phase_str,
            robot_x=self.current_x,
            robot_y=self.current_y,
            robot_yaw=self.current_yaw,
            n_obstacles=0,
            station_x=self.station_x,
            station_y=self.station_y
        )
        
        # Display current pose
        self.get_logger().info(
            f"Pose: x={self.current_x:.3f}, y={self.current_y:.3f}, yaw={self.current_yaw:.3f}"
        )
        
        # Phase I & II: Waypoint Navigation
        if self.current_phase in [1, 2]:
            if self.navigator.has_arrived():
                self.global_waypoint_idx += 1
                if self.global_waypoint_idx < len(self.global_waypoints):
                    self._set_next_waypoint()
                else:
                    self._advance_phase()
            
            cmd = self.navigator.step()
            self.publish_cmd(cmd.linear_x, cmd.angular_z)
            
        # Phase III: Docking
        elif self.current_phase == 3:
            cmd = self.docker.step(self.current_x, self.current_y, self.current_yaw)
            self.publish_cmd(cmd.linear_x, cmd.angular_z)
            if self.docker.is_docked():
                self.get_logger().info("✓ Docking Complete! Mission Successful.")
                self.terminate_mission()

    def publish_cmd(self, lin_x, ang_z):
        """Publish velocity command."""
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.twist.linear.x = lin_x
        msg.twist.angular.z = ang_z
        self.cmd_pub.publish(msg)

    def terminate_mission(self):
        """Cleanly shut down the mission."""
        self.publish_cmd(0.0, 0.0)
        self.logger.close()
        self.get_logger().info("Mission Terminated.")
        raise SystemExit

def main(args=None):
    rclpy.init(args=args)
    node = MissionNode()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        node.logger.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
