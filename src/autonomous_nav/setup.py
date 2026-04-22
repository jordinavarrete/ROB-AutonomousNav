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
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
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
            'debug_station_node = autonomous_nav.debug_station_node:main',
            'debug_docking_node = autonomous_nav.debug_docking_node:main',
            'debug_core_modules = autonomous_nav.debug_core_modules:main',
            'debug_phase1_node = autonomous_nav.debug_phase1_node:main',
            'debug_phase2_node = autonomous_nav.debug_phase2_node:main',
        ],
    },
)
