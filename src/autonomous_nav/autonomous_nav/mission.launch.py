#!/usr/bin/env python3
"""
mission.launch.py — Launches the full autonomous navigation mission.

Starts:
  1. SLAM Toolbox  (online_async mode, Jazzy compatible)
  2. mission_node  (this package)

Usage:
    ros2 launch autonomous_nav mission.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Build and return the LaunchDescription."""

    # ----------------------------------------------------------
    # Launch arguments
    # ----------------------------------------------------------
    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation clock (set true for Gazebo)',
    )
    use_sim_time = LaunchConfiguration('use_sim_time')

    # ----------------------------------------------------------
    # SLAM Toolbox — online_async mode
    # ----------------------------------------------------------
    slam_toolbox_node = Node(
        package    = 'slam_toolbox',
        executable = 'async_slam_toolbox_node',
        name       = 'slam_toolbox',
        output     = 'screen',
        parameters = [
            {
                'use_sim_time':          use_sim_time,
                'solver_plugin':         'solver_plugins::CeresSolver',
                'ceres_linear_solver':   'SPARSE_NORMAL_CHOLESKY',
                'ceres_preconditioner':  'SCHUR_JACOBI',
                'ceres_trust_strategy':  'LEVENBERG_MARQUARDT',
                'ceres_dogleg_type':     'TRADITIONAL_DOGLEG',
                'ceres_loss_function':   'None',

                # Map parameters
                'resolution':            0.05,     # m/cell
                'max_laser_range':       3.5,      # m — TB3 Burger LiDAR limit
                'minimum_travel_distance': 0.5,    # m before adding new node
                'minimum_travel_heading':  0.5,    # rad

                # Update intervals
                'map_update_interval':   3.0,      # seconds

                # Scan matching
                'use_scan_matching':     True,
                'use_scan_barycenter':   True,
                'scan_buffer_size':      10,
                'scan_buffer_maximum_scan_distance': 10.0,

                # Odometry / TF
                'odom_frame':            'odom',
                'map_frame':             'map',
                'base_frame':            'base_footprint',
                'transform_publish_period': 0.02,

                # Serialization (map saving)
                'do_loop_closing':       True,
                'loop_search_distance':  3.0,
            }
        ],
    )

    # ----------------------------------------------------------
    # Mission node
    # ----------------------------------------------------------
    mission_node = Node(
        package    = 'autonomous_nav',
        executable = 'mission_node',
        name       = 'mission_node',
        output     = 'screen',
        parameters = [{'use_sim_time': use_sim_time}],
    )

    # ----------------------------------------------------------
    # Assemble
    # ----------------------------------------------------------
    return LaunchDescription([
        use_sim_time_arg,
        LogInfo(msg='Launching SLAM Toolbox (online_async)...'),
        slam_toolbox_node,
        LogInfo(msg='Launching Mission Node...'),
        mission_node,
    ])
