# SIMULATION - Guia Simple i Ordenada

## 1) Inicialitzacio del sistema (nomes la primera vegada)
```bash
chmod +x install_ros2_turtlebot.sh && ./install_ros2_turtlebot.sh && source ~/.bashrc
```

Configura el domini ROS2 al teu `~/.bashrc`:
```bash
export ROS_DOMAIN_ID=N
```

Recarrega configuracio:
```bash
source ~/.bashrc
```

## 2) Robot real (TurtleBot3)

Connecta per SSH a la Raspberry del robot:
```bash
ssh ubuntu@10.10.73.2xx
# password: turtlebot
```

Lanca el robot (a la terminal SSH):
```bash
ros2 launch turtlebot3_bringup robot.launch.py
```

Important abans d'apagar el robot (a la terminal SSH):
```bash
sudo shutdown now
```

## 3) Simulacio Gazebo (PC)

Escull un mon:
```bash
ros2 launch turtlebot3_gazebo empty_world.launch.py
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
ros2 launch turtlebot3_gazebo turtlebot3_house.launch.py
```

Opcional (visualitzacio i control manual):
```bash
ros2 launch turtlebot3_bringup rviz2.launch.py
ros2 run turtlebot3_teleop teleop_keyboard
```

## 4) Crear un package Python

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
source /opt/ros/jazzy/setup.bash
ros2 pkg create --build-type ament_python <nom_package> --node-name <nom_node> --dependencies rclpy geometry_msgs sensor_msgs nav_msgs tf2_ros
```

Si necessites crear el fitxer manualment:
```bash
cd ~/ros2_ws/src/<nom_package>/<nom_package>
touch <nom_node>.py
chmod +x <nom_node>.py
```

## 5) Entry point a setup.py (si no surt automatic)

```python
entry_points={
	'console_scripts': [
		'<nom_node> = <nom_package>.<nom_node>:main',
	],
},
```

## 6) Build i execucio del package

```bash
cd ~/ros2_ws && colcon build --packages-select autonomous_nav && source install/setup.bash
ros2 run autonomous_nav mission_node
ros2 run autonomous_nav debug_nav_node
ros2 run autonomous_nav debug_station_node
```

## 7) Nota rapida de terminals

- Terminal SSH del robot: nomes comandes del robot (`bringup`, `shutdown`, etc.).
- Terminal del PC: simulacio Gazebo, RViz, teleop, build i execucio del teu codi.
