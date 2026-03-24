from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'autonomous_nav'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jordi',
    maintainer_email='jordinavarreteamer@gmail.com',
    description='Autonomous navigation for TurtleBot3 Burger',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'mission_node = autonomous_nav.mission_node:main',
            'debug_nav_node = autonomous_nav.debug_nav_node:main',
            'debug_obstacle_node = autonomous_nav.debug_obstacle_node:main',
            'debug_docking_node = autonomous_nav.debug_docking_node:main',
            'debug_lidar_node = autonomous_nav.debug_lidar_node:main',
            'debug_station_node = autonomous_nav.debug_station_node:main',
            'debug_explore_dock_node = autonomous_nav.debug_explore_dock_node:main',
            'debug_route_planner_node = autonomous_nav.debug_route_planner_node:main',
        ],
    },
)
