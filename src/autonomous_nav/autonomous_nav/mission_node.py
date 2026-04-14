#!/usr/bin/env python3
"""
mission_node.py — Main orchestrator for Phase I, II and III.
Integrates PGO (Pose Graph SLAM), Dynamic A*, Map Building, 
WaypointNavigator, Mission Logger, Station Detector, and DockingController.
"""

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from autonomous_nav.navigation import WaypointNavigator, Config
from autonomous_nav.slam_pgo import PoseGraph
from autonomous_nav.map_builder import MapBuilder
from autonomous_nav.dynamic_astar import DynamicAStar
from autonomous_nav.mission_logger import MissionLogger
from autonomous_nav.station_detector import StationDetector
from autonomous_nav.docking_controller import DockingController

INITIAL_X = 2.52
INITIAL_Y = 1.35
INITIAL_YAW = 0.0

class MissionNode(Node):
    def __init__(self):
        super().__init__('mission_node')
        self.get_logger().info("Mission Node Initialising with Initial Pose: {:.2f}, {:.2f}".format(INITIAL_X, INITIAL_Y))
        
        # ROS2 Publiser & Subscribers
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

        # Logic Modules
        self.navigator = WaypointNavigator(logger=self.get_logger())
        self.pgo = PoseGraph()
        self.map_b = MapBuilder()
        self.dastar = DynamicAStar()
        self.logger = MissionLogger()
        self.station_detector = StationDetector(logger=self.get_logger())
        self.docker = DockingController(logger=self.get_logger())
        
        # Set up Initial State
        self.last_odom = None
        self.current_x = INITIAL_X
        self.current_y = INITIAL_Y
        self.current_yaw = INITIAL_YAW
        
        self.pgo_idx = self.pgo.add_pose(INITIAL_X, INITIAL_Y, INITIAL_YAW)
        self.navigator.set_slam_pose(INITIAL_X, INITIAL_Y, INITIAL_YAW)
        
        self.current_phase = 1
        self.global_waypoint_idx = 0
        self.global_waypoints = Config.WAYPOINTS_PHASE1
        self.local_path = []
        
        self.n_obstacles = 0
        self.station_x = -1.0
        self.station_y = -1.0
        
        self.replan()

        self.timer = self.create_timer(0.05, self.control_loop)
        self.plan_timer = self.create_timer(2.0, self.slow_loop)

    def replan(self):
        if self.global_waypoint_idx < len(self.global_waypoints):
            target_x, target_y = self.global_waypoints[self.global_waypoint_idx]
            latest_pose = self.current_x, self.current_y, self.current_yaw
            
            path = self.dastar.plan(latest_pose[0], latest_pose[1], target_x, target_y)
            if path is not None and len(path) > 0:
                self.local_path = path[1:] # Skip first node
                if len(self.local_path) > 0:
                    next_wx, next_wy = self.local_path.pop(0)
                    self.navigator.set_waypoint(next_wx, next_wy)
            else:
                self.get_logger().warning("D* Planner failed to find a route! Coasting.")

    def odom_callback(self, msg: Odometry):
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        
        if self.last_odom is None:
            self.last_odom = (pos.x, pos.y, yaw)
            return
            
        dx = pos.x - self.last_odom[0]
        dy = pos.y - self.last_odom[1]
        dyaw = yaw - self.last_odom[2]
        
        if math.hypot(dx, dy) > 0.05 or abs(dyaw) > 0.05:
            latest = self.pgo.get_latest_pose()
            nx = latest[0] + dx * math.cos(latest[2]) - dy * math.sin(latest[2])
            ny = latest[1] + dx * math.sin(latest[2]) + dy * math.cos(latest[2])
            nyaw = latest[2] + dyaw
            
            new_idx = self.pgo.add_pose(nx, ny, nyaw)
            self.pgo.add_odometry_edge(self.pgo_idx, new_idx, dx, dy, dyaw)
            self.pgo_idx = new_idx
            self.last_odom = (pos.x, pos.y, yaw)
            
            new_pgo_pose = self.pgo.get_latest_pose()
            self.current_x, self.current_y, self.current_yaw = new_pgo_pose
            self.navigator.set_slam_pose(*new_pgo_pose)

    def scan_callback(self, msg: LaserScan):
        pose = self.current_x, self.current_y, self.current_yaw
        angles, ranges, obstacles_world = [], [], []
        
        for i, r in enumerate(msg.ranges):
            if math.isnan(r) or math.isinf(r): continue
            ang = msg.angle_min + i * msg.angle_increment
            if r < 3.0:
                obs_x = pose[0] + r * math.cos(pose[2] + ang)
                obs_y = pose[1] + r * math.sin(pose[2] + ang)
                obstacles_world.append((obs_x, obs_y))
            angles.append(ang)
            ranges.append(r)
                
        self.map_b.update_scan(pose[0], pose[1], pose[2], angles, ranges)
        
        if self.dastar.update_obstacles(obstacles_world):
            if not self.dastar.is_path_valid() and not self.navigator.is_idle() and self.current_phase != 3:
                self.replan()
                
        # Phase 2 Exploration -> Use Detector
        if self.current_phase == 2 or self.current_phase == 1:
            self.station_detector.update_scan(msg)
            if not self.station_detector.is_confirmed():
                res = self.station_detector.detect(pose[0], pose[1], pose[2])
                if res is not None:
                    self.station_x = res.centre_map_x
                    self.station_y = res.centre_map_y

    def slow_loop(self):
        # Could run pgo.optimize() periodically here
        pass

    def transition_waypoint(self):
        if len(self.local_path) > 0:
            next_wx, next_wy = self.local_path.pop(0)
            self.navigator.set_waypoint(next_wx, next_wy)
        else:
            self.global_waypoint_idx += 1
            if self.global_waypoint_idx < len(self.global_waypoints):
                self.replan()
            else:
                if self.current_phase == 1:
                    self.get_logger().info('Phase I complete! Starting Phase II.')
                    self.map_b.export('/home/jordi/ros2_ws/src/autonomous_nav/mission_map_phase1')
                    self.current_phase = 2
                    self.global_waypoint_idx = 0
                    self.global_waypoints = Config.WAYPOINTS_PHASE2
                    self.replan()
                elif self.current_phase == 2:
                    self.get_logger().info('Phase II Complete. Final Map Exported.')
                    self.map_b.export('/home/jordi/ros2_ws/src/autonomous_nav/mission_map_final')
                    
                    if self.station_detector.is_confirmed():
                        self.get_logger().info("Starting Phase III: Precision Docking.")
                        self.current_phase = 3
                        self.docker.start_docking(self.station_x, self.station_y)
                    else:
                        self.get_logger().error("Phase II Complete but Station not detected! Aborting.")
                        self.terminate_mission()

    def control_loop(self):
        # Update Logger seamlessly
        phase_str = 'I' if self.current_phase == 1 else 'II' if self.current_phase == 2 else 'III'
        self.logger.update(
            phase=phase_str,
            robot_x=self.current_x, robot_y=self.current_y, robot_yaw=self.current_yaw,
            n_obstacles=0, # Simplified tracking
            station_x=self.station_x, station_y=self.station_y
        )
        
        # Execute Maneuvers
        if self.current_phase in [1, 2]:
            if self.navigator.has_arrived():
                self.transition_waypoint()
            cmd = self.navigator.step()
            self.publish_cmd(cmd.linear_x, cmd.angular_z)
            
        elif self.current_phase == 3:
            cmd = self.docker.step(self.current_x, self.current_y, self.current_yaw)
            self.publish_cmd(cmd.linear_x, cmd.angular_z)
            if self.docker.is_docked():
                self.terminate_mission()

    def terminate_mission(self):
        self.publish_cmd(0.0, 0.0)
        self.logger.close()
        self.get_logger().info("Mission Cleanly Terminated.")
        raise SystemExit

    def publish_cmd(self, lin_x, ang_z):
        msg = Twist()
        msg.linear.x = lin_x
        msg.angular.z = ang_z
        self.cmd_pub.publish(msg)

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
