#!/usr/bin/env python3
"""
debug_door_node.py — Node de debug per a la detecció de porta.

Comportament:
  1. El robot avança en línia recta cap endavant (sense waypoints).
  2. Obstacle avoidance (Bug2) SEMPRE actiu — inclou wall-following.
  3. DoorDetector busca la porta contínuament al LiDAR.
  4. Quan detecta el centre de la porta:
     a. Surt de wall-follow si cal (l'avoider rep el nou waypoint).
     b. Navega directament al centre de la porta detectada.
     c. S'atura en arribar-hi.
  5. Durant APPROACH, si entra en wall-follow i després en surt,
     re-detecta la porta i actualitza el target.

Màquina d'estats:
    WAITING       → esperant primer /scan i /odom
    GO_STRAIGHT   → avança recte amb avoidance actiu + DoorDetector buscant
    APPROACH_DOOR → porta detectada; navega al centre real
    DONE          → al centre de la porta, robot aturat

Característiques:
  · Localització: SLAM Toolbox (TF map→base_footprint) amb fallback a odometria.
  · Obstacle avoidance reactiu: Bug2 via ObstacleAvoidance.
  · Watchdog de seguretat a 50 Hz (DANGER_DIST = 0.14 m → stop d'emergència).
  · MissionLogger: CSV a ~/mission_log_door_debug.csv.
  · Telemetria contínua cada segon.

Topics:
  Subscriu:  /scan      (LaserScan,     BEST_EFFORT)
             /odom      (Odometry,      RELIABLE)
  Publica:   /cmd_vel   (TwistStamped,  RELIABLE)

Paràmetres ROS2 (tots opcionals):
  start_x              float  Posició X inicial al mapa       [defecte: 4.280]
  start_y              float  Posició Y inicial al mapa       [defecte: 1.735]
  start_yaw_deg        float  Orientació inicial (graus)      [defecte: 0.0]
  straight_speed       float  Velocitat recta (m/s)           [defecte: 0.15]
  virtual_wp_distance  float  Dist waypoint virtual (m)       [defecte: 5.0]

Ús:
    ros2 run autonomous_nav debug_door_node
    ros2 run autonomous_nav debug_door_node --ros-args \\
        -p start_yaw_deg:=90.0 -p straight_speed:=0.10
"""

# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    # Posició inicial (coordenades del mapa)
    START_X           = 4.280   # m
    START_Y           = 1.735   # m
    START_YAW_DEG     = 0.0     # graus

    # Velocitat en línia recta
    STRAIGHT_SPEED    = 0.15    # m/s
    # Distància del waypoint virtual (projectat endavant)
    VIRTUAL_WP_DIST   = 5.0     # m

    # Rates de control
    CONTROL_HZ  = 20    # Hz — loop principal
    WATCHDOG_HZ = 50    # Hz — vigilant de seguretat

    # Frames TF
    MAP_FRAME  = 'map'
    BASE_FRAME = 'base_footprint'

    # Límits de velocitat publicats
    LINEAR_MAX  = 0.20   # m/s
    ANGULAR_MAX = 1.00   # rad/s

    # Log prefix
    LOG_PATH = '~/mission_log_door_debug.csv'


# ============================================================
# IMPORTS
# ============================================================
import math
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

# Mòduls del paquet
from autonomous_nav.navigation import WaypointNavigator, normalize_angle
from autonomous_nav.obstacle_avoidance import ObstacleAvoidance
from autonomous_nav.mission_logger import MissionLogger
from autonomous_nav.door_detector import DoorDetector


# ============================================================
# NODE PRINCIPAL
# ============================================================
class DebugDoorNode(Node):
    """
    Node de debug per a la detecció de porta.

    Màquina d'estats:
        WAITING       → esperant primer /scan i /odom
        GO_STRAIGHT   → avança recte amb avoidance actiu + DoorDetector
        APPROACH_DOOR → porta detectada; navega al centre real
        DONE          → al centre, robot aturat
    """

    # ---- Estats de la missió ----
    _WAITING       = 'WAITING'
    _GO_STRAIGHT   = 'GO_STRAIGHT'
    _APPROACH_DOOR = 'APPROACH_DOOR'
    _DONE          = 'DONE'

    def __init__(self) -> None:
        super().__init__('debug_door_node')

        # ----------------------------------------------------------
        # Paràmetres ROS2
        # ----------------------------------------------------------
        self.declare_parameter('start_x',             Config.START_X)
        self.declare_parameter('start_y',             Config.START_Y)
        self.declare_parameter('start_yaw_deg',       Config.START_YAW_DEG)
        self.declare_parameter('straight_speed',      Config.STRAIGHT_SPEED)
        self.declare_parameter('virtual_wp_distance', Config.VIRTUAL_WP_DIST)

        start_x         = float(self.get_parameter('start_x').value)
        start_y         = float(self.get_parameter('start_y').value)
        start_yaw_deg   = float(self.get_parameter('start_yaw_deg').value)
        start_yaw_rad   = math.radians(start_yaw_deg)
        self._straight_speed = float(self.get_parameter('straight_speed').value)
        self._virtual_wp_dist = float(self.get_parameter('virtual_wp_distance').value)

        # ----------------------------------------------------------
        # QoS profiles
        # ----------------------------------------------------------
        qos_reliable    = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,    depth=10)
        qos_best_effort = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)

        # ----------------------------------------------------------
        # Publisher
        # ----------------------------------------------------------
        self._cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', qos_reliable)

        # ----------------------------------------------------------
        # Subscribers
        # ----------------------------------------------------------
        self.create_subscription(LaserScan, '/scan', self._scan_cb, qos_best_effort)
        self.create_subscription(Odometry,  '/odom', self._odom_cb, qos_reliable)

        # ----------------------------------------------------------
        # TF2 (SLAM Toolbox → map → base_footprint)
        # ----------------------------------------------------------
        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        # ----------------------------------------------------------
        # Estat intern — pose
        # ----------------------------------------------------------
        self._x   = start_x
        self._y   = start_y
        self._yaw = start_yaw_rad
        self._prev_yaw    = start_yaw_rad
        self._slam_active = False

        # Offset odometria
        self._map_start_x     = start_x
        self._map_start_y     = start_y
        self._map_start_yaw   = start_yaw_rad
        self._odom_origin_x   = None
        self._odom_origin_y   = None
        self._odom_origin_yaw = None

        # Offset SLAM
        self._slam_offset_x:   float = 0.0
        self._slam_offset_y:   float = 0.0
        self._slam_offset_yaw: float = 0.0
        self._slam_offset_set: bool  = False

        # ----------------------------------------------------------
        # Estat intern — dades
        # ----------------------------------------------------------
        self._scan_ready = False
        self._odom_ready = False

        # ----------------------------------------------------------
        # Sub-mòduls
        # ----------------------------------------------------------
        self._navigator     = WaypointNavigator(logger=self.get_logger())
        self._avoider       = ObstacleAvoidance(logger=self.get_logger())
        self._csv_logger    = MissionLogger()
        self._door_detector = DoorDetector(logger=self.get_logger())

        # Posa la pose inicial al navegador
        self._navigator.set_odom_pose(start_x, start_y, start_yaw_rad)

        # ----------------------------------------------------------
        # Estat de la missió
        # ----------------------------------------------------------
        self._mission_state    = self._WAITING
        self._tick             = 0
        self._n_obstacles      = 0

        # ---- Estat porta ----
        self._door_target_x    = 0.0
        self._door_target_y    = 0.0
        self._door_found       = False

        # Rastreja transicions wall-following
        self._was_wall_following = False

        # ----------------------------------------------------------
        # Banner inicial
        # ----------------------------------------------------------
        self._print_banner(start_x, start_y, start_yaw_deg)

        # ----------------------------------------------------------
        # Timers
        # ----------------------------------------------------------
        self._ctrl_timer = self.create_timer(
            1.0 / Config.CONTROL_HZ, self._control_loop
        )
        self._wdog_timer = self.create_timer(
            1.0 / Config.WATCHDOG_HZ, self._watchdog
        )

        self.get_logger().info(
            'Node inicialitzat. Esperant primer /scan i /odom...'
        )

    # ==========================================================
    # BANNER
    # ==========================================================

    def _print_banner(self, sx: float, sy: float, syaw: float) -> None:
        log = self.get_logger().info
        log('=' * 60)
        log('  DEBUG DOOR NODE — Detecció de Porta')
        log('=' * 60)
        log(f'  Posició inicial       : ({sx:.3f}, {sy:.3f})'
            f'  yaw={syaw:.1f}°')
        log(f'  Velocitat recta       : {self._straight_speed:.2f} m/s')
        log(f'  WP virtual dist       : {self._virtual_wp_dist:.1f} m')
        log('=' * 60)
        log('  Localització : SLAM Toolbox (fallback: odometria)')
        log('  Avoidance    : Bug2 reactiu (SEMPRE actiu)')
        log(f'  CSV log      : {Config.LOG_PATH}')
        log('=' * 60)
        log('  Flux: GO_STRAIGHT → [detecta porta] → APPROACH_DOOR → DONE')
        log('  DoorDetector : ACTIU des de l\'inici (LiDAR gap 0.80m)')
        log('=' * 60)

    # ==========================================================
    # CALLBACKS ROS2
    # ==========================================================

    def _scan_cb(self, msg: LaserScan) -> None:
        """Ingestió del scan LiDAR → ObstacleAvoidance + DoorDetector."""
        self._avoider.update_scan(msg)
        self._door_detector.update_scan(msg)
        self._scan_ready = True
        self._n_obstacles = self._count_near_obstacles(msg)

    def _odom_cb(self, msg: Odometry) -> None:
        """
        Actualitza la pose del robot.
        Prioritza SLAM (TF map → base_footprint); si no disponible, usa
        odometria corregida amb l'offset del Punt A.
        """
        raw_x   = msg.pose.pose.position.x
        raw_y   = msg.pose.pose.position.y
        raw_yaw = self._quat_to_yaw(msg.pose.pose.orientation)

        # Inicialitza l'origen odom en el primer missatge rebut
        if self._odom_origin_x is None:
            self._odom_origin_x   = raw_x
            self._odom_origin_y   = raw_y
            self._odom_origin_yaw = raw_yaw
            self.get_logger().info(
                f'[ODOM] Origen odom fixat: '
                f'({raw_x:.4f}, {raw_y:.4f}, {math.degrees(raw_yaw):.1f}°)'
            )

        # Delta en frame odom
        dx    = raw_x   - self._odom_origin_x
        dy    = raw_y   - self._odom_origin_y
        d_yaw = normalize_angle(raw_yaw - self._odom_origin_yaw)

        # Transforma delta al frame mapa (rotació per start_yaw)
        cos_s = math.cos(self._map_start_yaw)
        sin_s = math.sin(self._map_start_yaw)
        map_x   = self._map_start_x + cos_s * dx - sin_s * dy
        map_y   = self._map_start_y + sin_s * dx + cos_s * dy
        map_yaw = normalize_angle(self._map_start_yaw + d_yaw)

        self._navigator.set_odom_pose(map_x, map_y, map_yaw)
        self._odom_ready = True

        # Intenta SLAM via TF
        slam_x, slam_y, slam_yaw = self._read_slam_tf()
        if slam_x is not None:
            if not self._slam_offset_set:
                self._slam_offset_x   = map_x   - slam_x
                self._slam_offset_y   = map_y   - slam_y
                self._slam_offset_yaw = normalize_angle(map_yaw - slam_yaw)
                self._slam_offset_set = True
                self.get_logger().info(
                    f'[SLAM] Offset calculat: '
                    f'Δx={self._slam_offset_x:.3f}m  '
                    f'Δy={self._slam_offset_y:.3f}m  '
                    f'Δyaw={math.degrees(self._slam_offset_yaw):.2f}°'
                )

            if not self._slam_active:
                self.get_logger().info(
                    '[SLAM] TF map→base_footprint disponible — '
                    'usant localització SLAM corregida.'
                )
                self._slam_active = True

            corr_x   = slam_x   + self._slam_offset_x
            corr_y   = slam_y   + self._slam_offset_y
            corr_yaw = normalize_angle(slam_yaw + self._slam_offset_yaw)

            self._navigator.set_slam_pose(corr_x, corr_y, corr_yaw)
            self._x, self._y, self._yaw = corr_x, corr_y, corr_yaw
        else:
            if self._slam_active:
                self.get_logger().warn(
                    '[SLAM] TF perdut — fallback a odometria corregida.'
                )
                self._slam_active = False
            self._x, self._y, self._yaw = map_x, map_y, map_yaw

    # ==========================================================
    # WATCHDOG (50 Hz)
    # ==========================================================

    def _watchdog(self) -> None:
        """Para el robot immediatament si el sector FRONT entra en DANGER."""
        if not self._scan_ready:
            return
        if self._avoider.is_front_danger():
            self._publish_stop()
            self.get_logger().warn(
                '[WATCHDOG] ⚠ PERILL AL FRONT — parada d\'emergència!'
            )

    # ==========================================================
    # WAYPOINT VIRTUAL (projectat endavant)
    # ==========================================================

    def _compute_virtual_waypoint(self):
        """
        Calcula un waypoint virtual projectat endavant en la direcció
        actual del robot. Això fa que l'avoider tingui un objectiu
        coherent per al Bug2 (m-line), i el robot avanci recte
        quan no hi ha obstacles.
        """
        vwp_x = self._x + self._virtual_wp_dist * math.cos(self._yaw)
        vwp_y = self._y + self._virtual_wp_dist * math.sin(self._yaw)
        return vwp_x, vwp_y

    # ==========================================================
    # LOOP DE CONTROL (20 Hz)
    # ==========================================================

    def _control_loop(self) -> None:
        """
        Lògica principal.

        Flux per tick:
          1. Comprova dades scan i odom.
          2. DONE → no fa res.
          3. WAITING → passa a GO_STRAIGHT.
          4. GO_STRAIGHT:
             · Avança recte (waypoint virtual projectat endavant).
             · Avoidance actiu: si detecta obstacle → wall-follow.
             · DoorDetector busca porta contínuament.
             · Si detecta porta → APPROACH_DOOR.
          5. APPROACH_DOOR:
             · Navega al centre real de la porta.
             · Si wall-follow, en sortir re-detecta i actualitza target.
             · Si arriba al centre → DONE.
        """
        self._tick += 1

        # ---- Espera dades inicials ----
        if not self._scan_ready or not self._odom_ready:
            return

        # ---- Missió completada ----
        if self._mission_state == self._DONE:
            return

        # ---- Primer tick amb dades: engega GO_STRAIGHT ----
        if self._mission_state == self._WAITING:
            self._mission_state = self._GO_STRAIGHT
            vwp_x, vwp_y = self._compute_virtual_waypoint()
            self._navigator.set_waypoint(vwp_x, vwp_y)
            self.get_logger().info('─' * 50)
            self.get_logger().info('  ▶ GO_STRAIGHT — avançant recte + buscant porta')
            self.get_logger().info('─' * 50)

        # ---- Anti-stuck: delta yaw per a l'avoider ----
        delta_yaw = abs(normalize_angle(self._yaw - self._prev_yaw))
        self._avoider.update_force_rotate(delta_yaw)
        self._prev_yaw = self._yaw

        # ---- Determina waypoint objectiu ----
        if self._mission_state == self._GO_STRAIGHT:
            # Waypoint virtual: es recalcula contínuament per mantenir
            # la direcció recta. Només s'actualitza al navigator quan
            # NO estem en wall-following (per no trencar la m-line del Bug2).
            wp_x, wp_y = self._compute_virtual_waypoint()
        else:
            # APPROACH_DOOR: objectiu és el centre de la porta
            wp_x, wp_y = self._door_target_x, self._door_target_y

        # ---- Obstacle avoidance ----
        cmd, in_avoidance = self._avoider.compute(
            self._x, self._y, self._yaw, wp_x, wp_y
        )

        if in_avoidance:
            self._publish(cmd.linear_x, cmd.angular_z)
        else:
            if self._mission_state == self._GO_STRAIGHT:
                # En GO_STRAIGHT sense avoidance: avancem recte
                # Actualitzem el waypoint virtual perquè el navigator
                # sempre apunti endavant
                self._navigator.set_waypoint(wp_x, wp_y)
                self._publish(self._straight_speed, 0.0)
            else:
                # APPROACH_DOOR sense avoidance: el navigator controla
                nav_cmd = self._navigator.step()
                self._publish(nav_cmd.linear_x, nav_cmd.angular_z)

        # ---- Detecta transicions wall-following ----
        now_wall_following = in_avoidance
        if self._mission_state == self._APPROACH_DOOR:
            if now_wall_following and not self._was_wall_following:
                self.get_logger().info(
                    '[DOOR] Wall-following activat durant APPROACH — '
                    'resetejant detector per confirmar des de nova posició.'
                )
                self._door_detector.reset()
        self._was_wall_following = now_wall_following

        # ---- Lògica específica per estat ----
        if self._mission_state == self._GO_STRAIGHT:
            self._step_go_straight()

        elif self._mission_state == self._APPROACH_DOOR:
            self._step_approach_door()

        # ---- CSV logger ----
        self._csv_logger.update(
            phase='DOOR_DBG',
            robot_x=self._x,
            robot_y=self._y,
            robot_yaw=self._yaw,
            n_obstacles=self._n_obstacles,
        )

        # ---- Telemetria (1 cop per segon) ----
        if self._tick % Config.CONTROL_HZ == 0:
            self._print_telemetry(wp_x, wp_y)

    # ==========================================================
    # LÒGICA D'ESTATS
    # ==========================================================

    def _step_go_straight(self) -> None:
        """
        En GO_STRAIGHT: el DoorDetector busca la porta contínuament.
        Si la detecta → passa a APPROACH_DOOR.
        """
        door = self._door_detector.detect(self._x, self._y, self._yaw)
        if door is not None:
            self._door_found = True
            self._launch_door_approach(door)

    def _step_approach_door(self) -> None:
        """
        En APPROACH_DOOR: navega al centre real detectat.
        Si wall-follow ha acabat i re-detecta la porta amb un centre
        diferent, actualitza el target.
        Si arriba al centre → DONE.
        """
        # Re-detecta la porta (pot actualitzar el centre si ha canviat)
        door = self._door_detector.detect(self._x, self._y, self._yaw)
        if door is not None:
            dist_update = math.hypot(
                door.centre_map_x - self._door_target_x,
                door.centre_map_y - self._door_target_y,
            )
            # Actualitza si el nou centre difereix > 5 cm
            if dist_update > 0.05:
                self.get_logger().info(
                    f'[DOOR] Target actualitzat durant APPROACH: '
                    f'({self._door_target_x:.3f},{self._door_target_y:.3f}) → '
                    f'({door.centre_map_x:.3f},{door.centre_map_y:.3f})  '
                    f'Δ={dist_update:.3f}m'
                )
                self._door_target_x = door.centre_map_x
                self._door_target_y = door.centre_map_y
                self._navigator.set_waypoint(door.centre_map_x, door.centre_map_y)

        # Comprova arribada
        if self._navigator.has_arrived():
            self._on_door_centre_reached()

    def _launch_door_approach(self, door) -> None:
        """
        Porta real detectada: redirigeix el robot al centre real.
        Surt de wall-following si cal (l'avoider es reset automàticament
        quan rep el nou waypoint al navigator).
        """
        self._mission_state  = self._APPROACH_DOOR
        self._door_target_x  = door.centre_map_x
        self._door_target_y  = door.centre_map_y

        dist = math.hypot(
            door.centre_map_x - self._x,
            door.centre_map_y - self._y,
        )

        self.get_logger().info('=' * 50)
        self.get_logger().info('  ✓ PORTA DETECTADA!')
        self.get_logger().info(
            f'    Centre real (mapa)  : '
            f'({door.centre_map_x:.3f}, {door.centre_map_y:.3f})'
        )
        self.get_logger().info(
            f'    Amplada mesurada    : {door.gap_width:.3f} m'
        )
        self.get_logger().info(
            f'    Heading porta (mapa): {math.degrees(door.heading_map):.1f}°'
        )
        self.get_logger().info(
            f'    Distància al centre : {dist:.3f} m'
        )
        self.get_logger().info('  ▶ APPROACH_DOOR — navegant al centre real')
        self.get_logger().info('=' * 50)

        self._avoider.reset()
        self._navigator.set_waypoint(door.centre_map_x, door.centre_map_y)

    def _on_door_centre_reached(self) -> None:
        """Robot al centre de la porta → missió completada."""
        dist = math.hypot(
            self._x - self._door_target_x,
            self._y - self._door_target_y,
        )
        self._mission_state = self._DONE
        self._publish_stop()

        self.get_logger().info('★' * 50)
        self.get_logger().info('  ✓✓ CENTRE PORTA ASSOLIT — Robot aturat!')
        self.get_logger().info(
            f'  Posició final : ({self._x:.3f}, {self._y:.3f}, '
            f'{math.degrees(self._yaw):.1f}°)'
        )
        self.get_logger().info(
            f'  Target porta  : ({self._door_target_x:.3f}, {self._door_target_y:.3f})'
        )
        self.get_logger().info(
            f'  Error posició : {dist:.3f} m'
        )
        self.get_logger().info(
            f'  Localització  : {"SLAM" if self._slam_active else "Odometria"}'
        )
        self.get_logger().info('★' * 50)

    # ==========================================================
    # TELEMETRIA
    # ==========================================================

    def _print_telemetry(self, wp_x: float, wp_y: float) -> None:
        """Imprimeix una línia de telemetria cada segon."""
        dist      = math.hypot(self._x - wp_x, self._y - wp_y)
        nav_state = self._navigator.get_state().name
        av_state  = self._avoider.get_state().name
        loc_src   = 'SLAM' if self._slam_active else 'ODOM'
        ms        = self._mission_state

        # Etiqueta del destí actiu
        if ms == self._GO_STRAIGHT:
            lbl = 'RECTE(virtual)'
        elif ms == self._APPROACH_DOOR:
            lbl = 'Porta(REAL)'
        else:
            lbl = ms

        # Info del detector
        door_info = ''
        conf = self._door_detector._confirm_count
        door_info = f'  door_conf={conf}/{self._door_detector._confirmed is not None}'

        self.get_logger().info(
            f'[TELEM]'
            f'  pos=({self._x:.3f},{self._y:.3f})'
            f'  yaw={math.degrees(self._yaw):6.1f}°'
            f'  dest={lbl}'
            f'  dist={dist:.3f}m'
            f'  nav={nav_state}'
            f'  avoid={av_state}'
            f'  loc={loc_src}'
            f'  obs≈{self._n_obstacles}'
            f'{door_info}'
        )

    # ==========================================================
    # HELPERS — VELOCITAT
    # ==========================================================

    def _publish(self, linear_x: float, angular_z: float) -> None:
        """Publica TwistStamped amb límits de seguretat."""
        cmd = TwistStamped()
        cmd.header.stamp    = self.get_clock().now().to_msg()
        cmd.header.frame_id = ''
        cmd.twist.linear.x  = max(-Config.LINEAR_MAX,
                                   min(Config.LINEAR_MAX,  linear_x))
        cmd.twist.angular.z = max(-Config.ANGULAR_MAX,
                                   min(Config.ANGULAR_MAX, angular_z))
        self._cmd_pub.publish(cmd)

    def _publish_stop(self) -> None:
        """Publica velocitat zero."""
        cmd = TwistStamped()
        cmd.header.stamp    = self.get_clock().now().to_msg()
        cmd.header.frame_id = ''
        self._cmd_pub.publish(cmd)

    # ==========================================================
    # HELPERS — TF / POSES
    # ==========================================================

    def _read_slam_tf(self):
        """
        Llegeix la pose corregida pel SLAM Toolbox via TF.
        Retorna (x, y, yaw) si disponible, o (None, None, None).
        """
        try:
            tf = self._tf_buffer.lookup_transform(
                Config.MAP_FRAME,
                Config.BASE_FRAME,
                rclpy.time.Time(),
            )
            x   = tf.transform.translation.x
            y   = tf.transform.translation.y
            yaw = self._quat_to_yaw(tf.transform.rotation)
            return x, y, yaw
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None, None, None

    @staticmethod
    def _quat_to_yaw(q) -> float:
        """Converteix quaternion a yaw [rad]."""
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    # ==========================================================
    # HELPERS — OBSTACLES
    # ==========================================================

    @staticmethod
    def _count_near_obstacles(scan: LaserScan, threshold: float = 1.0) -> int:
        """
        Compta el nombre de grups de lectures LiDAR per sota del llindar.
        Retorna una estimació ràpida del nombre d'obstacles propers.
        """
        ranges  = scan.ranges
        in_obs  = False
        count   = 0
        for r in ranges:
            valid = math.isfinite(r) and r > 0.0
            if valid and r < threshold:
                if not in_obs:
                    count  += 1
                    in_obs  = True
            else:
                in_obs = False
        return count

    # ==========================================================
    # SHUTDOWN
    # ==========================================================

    def shutdown(self) -> None:
        """Atura el robot i tanca el logger en sortir."""
        self.get_logger().info('Apagant debug_door_node — parant robot...')
        self._publish_stop()
        self._csv_logger.close()


# ============================================================
# ENTRY POINT
# ============================================================
def main(args=None) -> None:
    rclpy.init(args=args)
    node = DebugDoorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt rebut.')
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()