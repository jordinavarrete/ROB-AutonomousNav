# Guia de Simulació a Gazebo

Has de seguir aquests passos en terminals separades per provar el teu codi.

## Terminal 1: Simulador Gazebo
Aquesta terminal obre el món virtual on el robot es mourà.

```bash
export TURTLEBOT3_MODEL=burger
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
```

## Terminal 2: Missió Autònoma (El teu codi)
Aquesta terminal executa el SLAM i el teu node de missió. Hem posat `use_sim_time:=true` perquè estem en simulació.

```bash
cd ~/ros2_ws
colcon build --symlink-install --packages-select autonomous_nav
source install/setup.bash
ros2 launch autonomous_nav mission.launch.py use_sim_time:=true
```

## Terminal 3 (Opcional): Visualització RViz
Aquesta terminal et permetrà veure el mapa que està construint el SLAM en temps real.

```bash
ros2 run rviz2 rviz2
```
> [!TIP]
> A RViz, afegeix el component **Map** (tòpic `/map`) i el component **RobotModel** (amb base_footprint) per veure-ho tot bé.

## Comandes Extres Útils

### Crear un paquet nou (exmple)
```bash
ros2 pkg create --build-type ament_python nom_del_paquet
```

### Construir tot el workspace
```bash
cd ~/ros2_ws
colcon build --symlink-install
```

### Inspeccionar el sistema (Depuració)
```bash
ros2 node list
ros2 topic list
ros2 topic echo /mission_state
```

### Guardar el mapa al finalitzar la missió
```bash
ros2 run nav2_map_server map_saver_cli -f ~/mission_map
```

### Netejar el workspace (si hi ha errors estranys)
```bash
cd ~/ros2_ws
rm -rf build/ install/ log/
```

## Canvis realitzats
1. He creat la carpeta `launch/` a `src/autonomous_nav/`.
2. He mogut `mission.launch.py` a la carpeta `launch/`.
3. He reconstruït el paquet amb `colcon build`.
