# PROMPT: Autonomous Navigation & Multi-Stage Mission Execution — TurtleBot3 Burger (ROS2)

---

## CONTEXT

You are an expert ROS2 robotics engineer. You must implement a complete autonomous navigation system for a **TurtleBot3 Burger** robot using **ROS2 (Humble or Iron)**. The code must be clean, modular, well-commented, and ready to run in a real laboratory environment.

This is a **university final project** evaluated live on a real TB3 robot. **Any physical collision with the environment will result in a severe grade penalty.** Collision avoidance is the highest-priority constraint in the entire system.

---

## HARDWARE & PLATFORM

- **Robot:** TurtleBot3 Burger
- **Sensor:** 360° LiDAR (LDS-01 or LDS-02), ~1° angular resolution, range 0.12–3.5m
- **Actuators:** Two differential-drive wheels
- **ROS2 Distribution:** **Jazzy Jalisco** (ROS2 Jazzy) — use Jazzy-compatible APIs and packages exclusively
- **ROS2 topics available:**
  - `/scan` → `sensor_msgs/LaserScan` (LiDAR data, use `BEST_EFFORT` QoS)
  - `/odom` → `nav_msgs/Odometry` (wheel odometry, use `RELIABLE` QoS)
  - `/cmd_vel` → `geometry_msgs/TwistStamped` (velocity commands, use `RELIABLE` QoS)
  - `/map` → `nav_msgs/OccupancyGrid` (SLAM map output)
  - `/slam_toolbox/...` → SLAM Toolbox services

> **CRITICAL:** The robot uses `TwistStamped` (NOT plain `Twist`) for `/cmd_vel`. Always fill `header.stamp` and set `header.frame_id = ''`.

---

## MISSION DESCRIPTION

The mission has **three sequential phases**. The robot must complete them autonomously without any human intervention.

### Phase I — Global Navigation & Obstacle Avoidance

- Start at an assigned initial pose inside **Zone 1** (the classroom, "Aula S203")
- Navigate to a destination pose inside **Zone 2** ("Passadís"), passing through:
  - **Waypoint B** `(3.72, 2.55)`
  - **Waypoint O** `(5.10, 12.61)`
- The path must pass through the door at `(5.92, 8.12)`
- The environment **may contain static or dynamic obstacles** not present in the initial map
- Real-time obstacle detection and avoidance is **mandatory**
- While executing Phase I, the robot must **build its own map** using SLAM

### Phase II — Area Exploration & Charging Station Detection

- Once in Zone 2, explore the **"Passadís" area** (max 12×12 meters)
- Locate a **charging station**: a 40 cm × 40 cm square structure supported by **four cylindrical pillars (~5 cm diameter)**
- Once the station is found, **return to "Punt Base"** `(5.00, 11.69)`
- Continue building the SLAM map during this phase
- Save the map (`.yaml` + `.pgm`) when the phase completes

### Phase III — Precision Docking

- From "Punt Base", execute a **high-precision docking maneuver**
- Park the robot **exactly in the center** of the charging station (equidistant from all 4 pillars)
- This completes the mission

---

## ENVIRONMENT MAP & KEY COORDINATES

All coordinates are in **meters relative to the map origin**.

```
Key Reference Points:
  Punt A     (2.52,  1.35)    P Base    (5.00, 11.69)
  Punt B     (3.72,  2.55)    Punt O    (5.10, 12.61)
  Punt C     (1.32,  0.95)    Punt P    (0.30, 11.01)
  Punt D     (3.32,  0.95)    Punt Q    (1.90, 12.21)
  Porta      (5.92,  8.12)    Punt R    (7.12, 12.61)
```

**Zone 1** = Aula S203 (large classroom, origin area)
**Zone 2** = Passadís (corridor area, ~12×12m, to the right of the classroom)
**Punt Base** = reference point in Zone 2 where Phase II ends and Phase III starts

The Passadís area can be explored using points P, Q, R as coverage waypoints.

---

## TECHNICAL REQUIREMENTS

### General
- All configurable parameters (speeds, distances, thresholds, waypoints) must be grouped in a **dedicated `CONFIG` section** at the top of each file
- Code must be **modular**: separate concerns into clearly named classes/methods
- Use **explicit state machines** with named states (strings or enums)
- Use ROS2 `declare_parameter()` for key parameters where appropriate
- All logs must use `self.get_logger()`, never `print()`
- Use log levels correctly: `INFO` for normal flow, `WARN` for anomalies, `ERROR` for failures

### QoS Profiles
```python
from rclpy.qos import QoSProfile, ReliabilityPolicy
qos_reliable = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)
qos_best_effort = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
# /scan → qos_best_effort
# /odom, /cmd_vel → qos_reliable
```

### TwistStamped (mandatory format)
```python
from geometry_msgs.msg import TwistStamped
cmd = TwistStamped()
cmd.header.stamp = self.get_clock().now().to_msg()
cmd.header.frame_id = ''
cmd.twist.linear.x = 0.2
cmd.twist.angular.z = 0.0
self.cmd_pub.publish(cmd)
```

### Odometry / Yaw extraction
```python
def quaternion_to_yaw(self, q):
    siny_cosp = 2 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)
```

---

## EXISTING REFERENCE CODE

These are simpler exercises from previous lab sessions. Use them as style and pattern references.

### `cuadrat.py` — Time-based square trajectory
```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.clock import Clock
from geometry_msgs.msg import TwistStamped
import math

class SquarePath(Node):
    def __init__(self):
        super().__init__('square_path_node')
        qos_profile_r = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)
        self.publisher = self.create_publisher(TwistStamped, '/cmd_vel', qos_profile_r)
        self.timer = self.create_timer(0.02, self.timer_callback)
        self.state = 'FORWARD'
        self.state_start_time = self.get_clock().now()
        self.forward_duration = 5.0
        self.linear_speed = 0.2
        self.turn_duration = 3.0
        target_angle_radians = math.radians(90.0)
        self.angular_speed = target_angle_radians / self.turn_duration

    def timer_callback(self):
        move_msg = TwistStamped()
        now = self.get_clock().now()
        move_msg.header.stamp = now.to_msg()
        move_msg.header.frame_id = ''
        elapsed_time = (now - self.state_start_time).nanoseconds / 1e9
        if self.state == 'FORWARD':
            if elapsed_time < self.forward_duration:
                move_msg.twist.linear.x = self.linear_speed
            else:
                self.state = 'TURN'
                self.state_start_time = now
        elif self.state == 'TURN':
            if elapsed_time < self.turn_duration:
                move_msg.twist.angular.z = self.angular_speed
            else:
                self.state = 'FORWARD'
                self.state_start_time = now
        self.publisher.publish(move_msg)
```

### `square_move_odom.py` — Odometry-based square with proportional turn control
```python
import rclpy, math
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.clock import Clock
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry

class SquareOdom(Node):
    def __init__(self):
        super().__init__('square_odom_node')
        qos_profile_r = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)
        self.publisher = self.create_publisher(TwistStamped, '/cmd_vel', qos_profile_r)
        self.odom_subscription = self.create_subscription(Odometry, '/odom', self.odom_callback, qos_profile_r)
        self.timer = self.create_timer(0.02, self.control_callback)
        self.linear_speed = 0.20
        self.angular_speed = 0.4
        self.target_distance = 1.0
        self.target_angle = math.pi / 2
        self.state = 'FORWARD'
        self.odom_x = self.odom_y = self.odom_yaw = None
        self.start_x = self.start_y = self.start_yaw = None

    def quaternion_to_yaw(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def odom_callback(self, msg):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        self.odom_yaw = self.quaternion_to_yaw(msg.pose.pose.orientation)

    def control_callback(self):
        if self.odom_x is None:
            return
        if self.start_x is None:
            self.start_x, self.start_y, self.start_yaw = self.odom_x, self.odom_y, self.odom_yaw
        move_msg = TwistStamped()
        move_msg.header.stamp = self.get_clock().now().to_msg()
        move_msg.header.frame_id = ''
        if self.state == 'FORWARD':
            traveled = math.sqrt((self.odom_x-self.start_x)**2 + (self.odom_y-self.start_y)**2)
            if traveled < self.target_distance:
                move_msg.twist.linear.x = self.linear_speed
            else:
                self.state = 'TURN'
                self.start_yaw = self.odom_yaw
        elif self.state == 'TURN':
            angle_turned = abs(self.odom_yaw - self.start_yaw)
            angle_turned = min(angle_turned, 2 * math.pi - angle_turned)
            if angle_turned < self.target_angle:
                error = self.target_angle - angle_turned
                move_msg.twist.angular.z = max(min(2.0 * error, 0.4), 0.05)
            else:
                self.state = 'FORWARD'
                self.start_x, self.start_y = self.odom_x, self.odom_y
        self.publisher.publish(move_msg)
```

### `lidar_caracterization.py` — LiDAR sampling and statistics
```python
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan

N_SAMPLES = 10

class LidarCharacterization(Node):
    def __init__(self):
        super().__init__('lidar_characterization_node')
        qos_profile_b = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        self.subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_b)
        self.samples = []

    def scan_callback(self, msg):
        if len(self.samples) >= N_SAMPLES:
            return
        value = msg.ranges[0]
        if msg.range_min < value < msg.range_max:
            self.samples.append(value)
        if len(self.samples) == N_SAMPLES:
            self.get_logger().info(f'Min: {min(self.samples):.4f} Max: {max(self.samples):.4f} Mean: {sum(self.samples)/N_SAMPLES:.4f}')
```

### `seguir_objecte.py` — Reflector follower using LiDAR intensity + proportional control
```python
import rclpy, math
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.clock import Clock
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry

MIN_DISTANCE = 0.30; MAX_DISTANCE = 0.60; MIN_INTENSITY = 7000.0
TARGET_DISTANCE = 0.15; KP_ANGULAR = 1.2; KP_LINEAR = 0.5
MAX_LINEAR_SPEED = 0.20; MAX_ANGULAR_SPEED = 1.0

class ReflectorFollower(Node):
    def __init__(self):
        super().__init__('reflector_follower_node')
        qos_b = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        qos_r = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_b)
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', qos_r)

    def scan_callback(self, msg):
        valid_points = 0; sum_dist = sum_sin = sum_cos = 0.0
        for i, (dist, intensity) in enumerate(zip(msg.ranges, msg.intensities)):
            if math.isinf(dist) or math.isnan(dist): continue
            if MIN_DISTANCE <= dist <= MAX_DISTANCE and intensity >= MIN_INTENSITY:
                angle = msg.angle_min + i * msg.angle_increment
                while angle > math.pi: angle -= 2*math.pi
                while angle < -math.pi: angle += 2*math.pi
                sum_dist += dist; sum_sin += math.sin(angle); sum_cos += math.cos(angle)
                valid_points += 1
        detected = valid_points > 0
        dist = sum_dist / valid_points if detected else 0
        angle = math.atan2(sum_sin, sum_cos) if detected else 0
        self.control_robot(detected, dist, angle)

    def control_robot(self, detected, dist, angle):
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = ''
        if detected:
            cmd.twist.angular.z = max(min(KP_ANGULAR * angle, MAX_ANGULAR_SPEED), -MAX_ANGULAR_SPEED)
            if abs(angle) < 0.8:
                dist_error = dist - TARGET_DISTANCE
                cmd.twist.linear.x = max(min(KP_LINEAR * dist_error, MAX_LINEAR_SPEED), -MAX_LINEAR_SPEED)
                if abs(dist_error) < 0.05: cmd.twist.linear.x = 0.0
            if abs(angle) < 0.05: cmd.twist.angular.z = 0.0
        self.cmd_pub.publish(cmd)
```

---

## WHAT YOU MUST IMPLEMENT

Implement the **complete mission system** as a single ROS2 Python package. Structure it as follows:

### File Structure
```
autonomous_nav/
├── autonomous_nav/
│   ├── __init__.py
│   ├── mission_node.py          ← Main mission controller (entry point)
│   ├── navigation.py            ← Waypoint navigation + orientation control
│   ├── obstacle_avoidance.py    ← LiDAR processing, sector classification, wall-follow
│   ├── station_detector.py      ← Pillar clustering, station geometry validation
│   ├── docking.py               ← Precision centering inside station
│   └── mission_logger.py        ← CSV log writer
├── config/
│   └── mission_params.yaml      ← All configurable parameters
├── launch/
│   └── mission.launch.py        ← Launches all nodes including SLAM Toolbox
├── package.xml
└── setup.py
```

### `mission_node.py` — Main orchestrator

Implement a ROS2 node that:
- Manages the top-level **mission state machine**:
  ```
  PHASE_I → PHASE_II_EXPLORE → PHASE_II_RETURN → PHASE_III_DOCK → MISSION_COMPLETE
  ```
- Delegates to sub-modules: navigation, obstacle avoidance, station detector, docking
- Has a **safety watchdog** that runs at 50Hz independently: if any LiDAR sector enters `DANGER` zone, publishes a zero-velocity stop command regardless of current state
- Publishes mission state to `/mission_state` (std_msgs/String)
- Handles `KeyboardInterrupt` by publishing a stop command before shutdown

### `navigation.py` — Waypoint navigator

State machine per waypoint:
```
ORIENT → NAVIGATE → ARRIVED
```
- **ORIENT phase:** rotate in place toward target using proportional angular control
  - `angular_speed = Kp_orient * angle_error`, clamped to `[MIN_ANG, MAX_ANG]`
  - Transition to NAVIGATE when `|angle_error| < ORIENT_THRESHOLD` (e.g., 0.05 rad)
- **NAVIGATE phase:** move forward, continuously correcting heading
  - `linear_speed = Kp_linear * distance_error`, clamped
  - `angular_correction = Kp_heading * heading_error` applied while moving
  - Transition to ARRIVED when `distance < ARRIVAL_THRESHOLD` (e.g., 0.10 m)
- Uses **SLAM-corrected pose** if available, falls back to odometry
- Waypoint list for Phase I: `[(3.72, 2.55), (5.92, 8.12), (5.10, 12.61)]` then `(5.00, 11.69)`
- Waypoints for Phase II exploration (Passadís sweep): `[(0.30, 11.01), (1.90, 12.21), (7.12, 12.61)]`

### `obstacle_avoidance.py` — LiDAR processor + avoidance controller

**Sector definitions** (angles relative to robot front = 0°):
```
FRONT:        -25° to  +25°   (most critical)
FRONT_LEFT:   +25° to  +70°
FRONT_RIGHT:  -70° to  -25°
LEFT:         +70° to +110°
RIGHT:       -110° to  -70°
```

**Alert levels per sector:**
```python
DANGER_DIST  = 0.25   # m  → immediate stop
WARNING_DIST = 0.45   # m  → slow down and prepare avoidance
SAFE_DIST    = 0.60   # m  → clear
```

**Wall-follow controller:**
```
WALL_FOLLOW states:
  AVOID_ROTATE  → rotate in place toward the side with most free space
  WALL_FOLLOW   → maintain lateral distance to wall while moving forward
  RECOVERING    → front clear, heading toward waypoint improving → exit
```

Wall-follow side selection: choose left or right based on which side has greater minimum distance in the lateral sector.

Wall-follow lateral control:
- Target lateral distance: `WALL_FOLLOW_DIST = 0.35` m
- `lateral_error = lateral_dist - WALL_FOLLOW_DIST`
- `angular_correction = Kp_wall * lateral_error`
- Forward speed reduced to `WALL_FOLLOW_SPEED = 0.10` m/s

Exit wall-follow condition: `FRONT` sector is SAFE **and** angle toward next waypoint is improving (decreasing over last 2 seconds).

Anti-stuck logic: if robot has not moved more than 3 cm in 5 seconds, force a 180° rotation.

### `station_detector.py` — Charging station detection

The station is: **4 cylindrical pillars, ~5 cm diameter, arranged in a 40×40 cm square**.

**Detection algorithm:**
1. From raw `/scan` data, extract all valid range readings
2. **Cluster adjacent scan points** that are within `CLUSTER_DIST = 0.08` m of each other
3. Filter clusters by size: keep clusters whose arc width is consistent with a ~5 cm cylinder at the measured distance
   - Expected angular width at distance `d`: `~arctan(0.05 / d)` radians → typically 2–6 consecutive scan points at 1–2 m
4. Estimate cluster centroid (x, y) in robot frame from range + angle
5. **Validate geometry**: from all detected clusters, find a group of 4 whose pairwise distances match a 40×40 cm square:
   - 4 sides ≈ 0.40 m (tolerance ±0.05 m)
   - 2 diagonals ≈ 0.566 m (tolerance ±0.07 m)
6. Require **N_CONFIRM = 5** consecutive scans confirming the same station position before accepting
7. Compute station **center** as centroid of the 4 pillar positions
8. Transform station center from robot frame to map frame using current SLAM pose

**Output:** `(station_x, station_y)` in map frame, plus individual pillar positions for logging.

### `docking.py` — Precision centering

Two-phase docking:
1. **Approach phase**: navigate to a point 0.30 m in front of the station center (using `navigation.py`)
2. **Fine centering phase**:
   - At each control cycle, detect the 4 pillars in the current scan
   - Compute centroid of 4 pillars in robot frame
   - Apply proportional control to minimize distance to centroid:
     - `linear_x = Kp_dock_linear * centroid_dist * cos(centroid_angle)`
     - `angular_z = Kp_dock_angular * centroid_angle`
   - Max docking speed: `MAX_DOCK_SPEED = 0.05` m/s
   - **Docked criterion:** robot centroid offset < `DOCK_TOLERANCE = 0.03` m for 2 consecutive seconds
3. On docking complete: publish zero velocity, log final pose, transition to `MISSION_COMPLETE`

### `mission_logger.py` — CSV logger

Write a CSV log file to `~/mission_log.csv` with one row per second containing:
```
timestamp, phase, robot_x, robot_y, robot_yaw, n_obstacles_detected, station_x, station_y
```
- `phase` is one of: `I`, `II`, `III`
- `station_x / station_y` = `-1.0` until station is found
- File must be finalized and flushed on shutdown

### `mission.launch.py` — Launch file

Must launch:
1. **SLAM Toolbox** in `online_async` mode with appropriate parameters:
   - `use_sim_time: false`
   - `max_laser_range: 3.5`
   - `resolution: 0.05`
   - `map_update_interval: 3.0`
2. **mission_node** from this package
3. Map saver: save map to `~/mission_map` when mission completes (or on signal)

---

## CONFIG BLOCK TEMPLATE

Each file must start with a config block similar to:

```python
# ============================================================
# CONFIGURATION — adjust these values for lab testing
# ============================================================
class Config:
    # Navigation
    LINEAR_SPEED       = 0.18   # m/s — max forward speed
    ANGULAR_SPEED      = 0.50   # rad/s — max rotation speed
    ARRIVAL_THRESHOLD  = 0.10   # m — waypoint reached if dist < this
    ORIENT_THRESHOLD   = 0.05   # rad — aligned if angle error < this
    KP_LINEAR          = 0.6    # proportional gain for distance
    KP_ANGULAR         = 1.2    # proportional gain for orientation
    KP_HEADING         = 0.4    # heading correction while moving

    # Obstacle avoidance
    DANGER_DIST        = 0.25   # m
    WARNING_DIST       = 0.45   # m
    SAFE_DIST          = 0.60   # m
    WALL_FOLLOW_DIST   = 0.35   # m — target lateral distance from wall
    WALL_FOLLOW_SPEED  = 0.10   # m/s
    KP_WALL            = 0.8

    # Station detection
    CLUSTER_DIST       = 0.08   # m — max distance between adjacent scan points in same cluster
    PILLAR_DIAMETER    = 0.05   # m
    STATION_SIDE       = 0.40   # m
    STATION_TOL        = 0.06   # m — geometric validation tolerance
    N_CONFIRM          = 5      # consecutive detections required

    # Docking
    MAX_DOCK_SPEED     = 0.05   # m/s
    KP_DOCK_LINEAR     = 0.4
    KP_DOCK_ANGULAR    = 0.8
    DOCK_TOLERANCE     = 0.03   # m

    # Waypoints — Phase I
    WAYPOINTS_PHASE1 = [
        (3.72,  2.55),   # Punt B
        (5.92,  8.12),   # Porta (door)
        (5.10, 12.61),   # Punt O
        (5.00, 11.69),   # P Base
    ]

    # Waypoints — Phase II exploration (Passadís sweep)
    WAYPOINTS_PHASE2 = [
        (0.30, 11.01),   # Punt P
        (1.90, 12.21),   # Punt Q
        (7.12, 12.61),   # Punt R
        (5.00, 11.69),   # Return to P Base
    ]
```

---

## SAFETY REQUIREMENTS (NON-NEGOTIABLE)

1. **Emergency stop watchdog**: a dedicated timer at 50Hz that reads the latest LiDAR sectors. If `FRONT` sector minimum distance < `DANGER_DIST`, **immediately publish zero velocity** regardless of state machine state.

2. **Graceful shutdown**: all nodes must catch `KeyboardInterrupt` and `finally` publish a stop `TwistStamped` before calling `rclpy.shutdown()`.

3. **NaN/Inf filtering**: always filter `math.isinf()` and `math.isnan()` from `/scan` ranges before any computation.

4. **Speed limits**: no command may exceed `LINEAR_SPEED = 0.20 m/s` or `ANGULAR_SPEED = 1.0 rad/s` anywhere in the codebase.

5. **Timeout guards**: each phase has a maximum duration. If Phase I exceeds 5 minutes or Phase II exceeds 8 minutes, log a warning and continue best-effort.

---

## DELIVERABLES EXPECTED FROM YOU

Provide the following files, fully implemented and ready to run:

1. `mission_node.py` — main orchestrator with mission state machine
2. `navigation.py` — waypoint navigator
3. `obstacle_avoidance.py` — LiDAR processing, avoidance, wall-follow
4. `station_detector.py` — pillar clustering and station geometry validation
5. `docking.py` — precision docking controller
6. `mission_logger.py` — CSV log writer
7. `mission.launch.py` — ROS2 launch file
8. `package.xml` — ROS2 package descriptor (Jazzy compatible: `<exec_depend>slam_toolbox</exec_depend>`, `<exec_depend>nav2_map_server</exec_depend>`)
9. `setup.py` — Python package setup

**Code quality requirements (graded):**
- PEP8 compliant
- Every class and non-trivial method has a docstring
- State transitions are logged with `self.get_logger().info()`
- No hardcoded magic numbers outside the Config class
- No bare `except:` clauses

---

## IMPLEMENTATION NOTES & HINTS

- **ROS2 Jazzy note:** In Jazzy, `rclpy` and all standard message packages are unchanged. Use `slam_toolbox` (available in Jazzy via `apt install ros-jazzy-slam-toolbox`). For nav2 map saver use `ros-jazzy-nav2-map-server`. Avoid deprecated APIs from older distros.

- **SLAM localization vs odometry:** Subscribe to the SLAM-corrected pose via `/slam_toolbox/pose` or by listening to the `map → odom` TF transform. If SLAM is not yet initialized, fall back to `/odom`.

- **Angle normalization:** always normalize angle differences to `[-π, π]`:
  ```python
  def normalize_angle(a):
      while a > math.pi:  a -= 2*math.pi
      while a < -math.pi: a += 2*math.pi
      return a
  ```

- **LiDAR index to angle:** `angle = msg.angle_min + i * msg.angle_increment`. For the TB3 Burger, index 0 = front, indices increase counterclockwise.

- **Pillar detection at close range:** at 1 m distance, a 5 cm pillar subtends ~2.9°, which is ~3 scan points. At 2 m it's ~1.4° (~1–2 points). Cluster validation must account for this.

- **Wall-follow exit:** do NOT exit wall-follow just because the front is clear — the robot may be in a gap in the wall. Also require that the heading toward the next waypoint is improving.

- **Station centering:** the interior of the station is 40×40 cm. The TB3 Burger is ~20 cm wide. Centering must be precise (< 3 cm error). Use live LiDAR feedback, not a stored position, during the final centering phase.

- **Map saving:** use the `nav2_map_server` or `map_saver_cli` tool:
  ```bash
  ros2 run nav2_map_server map_saver_cli -f ~/mission_map
  ```
  Trigger this programmatically using a subprocess call at the end of Phase II.

---

## SUMMARY OF MISSION FLOW

```
[START] Robot at initial pose in Zone 1
    │
    ├─ SLAM Toolbox starts building map
    │
    ▼
[PHASE I] Navigate: initial → B(3.72,2.55) → Door(5.92,8.12) → O(5.10,12.61) → PBase(5.00,11.69)
    │  • Real-time obstacle avoidance active
    │  • Wall-follow for complex scenarios
    │  • SLAM map being built
    │
    ▼
[PHASE II] Explore Passadís: sweep P→Q→R→PBase
    │  • LiDAR scanning for 4-pillar station pattern
    │  • Cluster detection + geometry validation
    │  • Once station found: record position, return to PBase
    │  • Save map (.yaml + .pgm) on arrival at PBase
    │
    ▼
[PHASE III] Precision docking: PBase → approach station → fine centering
    │  • Navigate to station neighborhood
    │  • Live LiDAR-based centering between 4 pillars
    │  • Stop when equidistant (< 3 cm error)
    │
    ▼
[MISSION COMPLETE] Flush CSV log, robot stopped
```

---

*Generated context based on: TurtleBot3 Burger platform, ROS2 Jazzy Jalisco, lab exercise reference code, and project specification document "Autonomous Navigation & Multi-Stage Mission Execution" (UPC FIB Robotics, VERSION 1, March 18th 2026).*
