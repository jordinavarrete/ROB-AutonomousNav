#!/usr/bin/env python3
"""
debug_phase1_node.py — Node de debug per a la Fase I de la missió autònoma.

Executa la seqüència completa de la Fase I:
    Punt A (inici) → C → D → F → [cerca porta real] → centre porta

Característiques:
  · Localització: SLAM Toolbox (TF map→base_footprint) amb fallback a odometria.
  · Obstacle avoidance reactiu: Bug2 via ObstacleAvoidance.
  · Watchdog de seguretat a 50 Hz (DANGER_DIST = 0.14 m → stop d'emergència).
  · MissionLogger: CSV a ~/mission_log_phase1.csv, un registre per segon.
  · Exportació del mapa: PGM + YAML al finalitzar (amb delay configurable).
  · Telemetria contínua cada segon: posició, waypoint actiu, distància, estat.

Topics:
  Subscriu:  /scan      (LaserScan,     BEST_EFFORT)
             /odom      (Odometry,      RELIABLE)
             /map       (OccupancyGrid, RELIABLE + TRANSIENT_LOCAL)
  Publica:   /cmd_vel   (TwistStamped,  RELIABLE)

Paràmetres ROS2 (tots opcionals):
  start_x              float  Posició X inicial al mapa       [defecte: 4.280]
  start_y              float  Posició Y inicial al mapa       [defecte: 1.735]
  start_yaw_deg        float  Orientació inicial (graus)      [defecte: 0.0]
  map_output_prefix    str    Prefix fitxers mapa             [defecte: 'phase1_map']
  map_save_delay_sec   float  Espera post-arribada per mapa   [defecte: 3.0]
  map_mode             str    'scale' o 'trinary'             [defecte: 'scale']
  map_occupied_thresh  float  Llindar ocupació mapa           [defecte: 0.65]
  map_free_thresh      float  Llindar lliure mapa             [defecte: 0.25]
  map_overwrite        bool   Sobreescriu si ja existeix      [defecte: False]

Ús:
    ros2 run autonomous_nav debug_phase1_node
    ros2 run autonomous_nav debug_phase1_node --ros-args \\
        -p start_yaw_deg:=90.0 -p map_output_prefix:=run1_map
"""

# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    # Posició inicial: Punt A (coordenades del mapa)
    START_X           = 4.280   # m
    START_Y           = 1.735   # m
    START_YAW_DEG     = 0.0     # graus

    # Waypoints Fase I (Punt A és la posició inicial, no un objectiu)
    # Etiquetes corresponents a cada waypoint:
    WAYPOINT_LABELS = ['C', 'D', 'F', 'Q']
    WAYPOINTS = [
        (4.880,  2.535),   # Punt C
        (5.080,  5.740),   # Punt D
        (5.480, 10.545),   # Punt F
        (9.115, 14.190),   # Punt Q  ← darrer WP; després: finalitzar
    ]

    # Coordenades nominals de la Porta (usades com a destí de seek si no
    # es detecta la porta real abans d'arribar-hi)
    # ELIMINAT: Ja no busquem porta, anem directament a Q

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
    LOG_PATH = '~/mission_log_phase1.csv'


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
from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan

# Mòduls del paquet
from autonomous_nav.navigation import WaypointNavigator, normalize_angle
from autonomous_nav.obstacle_avoidance import ObstacleAvoidance
from autonomous_nav.mission_logger import MissionLogger


# ============================================================
# NODE PRINCIPAL
# ============================================================
class DebugPhase1Node(Node):
    """
    Node de debug per a la Fase I completa de la missió autònoma.

    Màquina d'estats de missió:
        WAITING       → esperant primer /scan i /odom
        NAVIGATE      → navegant cap als waypoints C, D, F, Q
        DONE          → Fase I completada

    El WaypointNavigator gestiona la màquina d'estats per waypoint
    (IDLE → ORIENT → NAVIGATE → ARRIVED).  ObstacleAvoidance sobreescriu
    les comandes quan detecta obstacles.
    """

    # ---- Estats de la missió del node ----
    _WAITING       = 'WAITING'
    _NAVIGATE      = 'NAVIGATE'
    _DONE          = 'DONE'

    def __init__(self) -> None:
        super().__init__('debug_phase1_node')

        # ----------------------------------------------------------
        # Paràmetres ROS2
        # ----------------------------------------------------------
        self.declare_parameter('start_x',             Config.START_X)
        self.declare_parameter('start_y',             Config.START_Y)
        self.declare_parameter('start_yaw_deg',       Config.START_YAW_DEG)
        self.declare_parameter('map_output_prefix',   'phase1_map')
        self.declare_parameter('map_save_delay_sec',  3.0)
        self.declare_parameter('map_mode',            'scale')
        self.declare_parameter('map_occupied_thresh', 0.65)
        self.declare_parameter('map_free_thresh',     0.25)
        self.declare_parameter('map_overwrite',       False)

        start_x         = float(self.get_parameter('start_x').value)
        start_y         = float(self.get_parameter('start_y').value)
        start_yaw_deg   = float(self.get_parameter('start_yaw_deg').value)
        start_yaw_rad   = math.radians(start_yaw_deg)

        self._map_prefix        = str(self.get_parameter('map_output_prefix').value)
        self._map_delay         = float(self.get_parameter('map_save_delay_sec').value)
        self._map_mode          = str(self.get_parameter('map_mode').value).lower()
        self._map_occ_thresh    = float(self.get_parameter('map_occupied_thresh').value)
        self._map_free_thresh   = float(self.get_parameter('map_free_thresh').value)
        self._map_overwrite     = bool(self.get_parameter('map_overwrite').value)

        # Valida mode
        if self._map_mode not in ('trinary', 'scale'):
            self.get_logger().warn(
                f"map_mode='{self._map_mode}' no vàlid → s'usarà 'scale'."
            )
            self._map_mode = 'scale'

        # Valida thresholds
        self._map_occ_thresh  = min(max(self._map_occ_thresh,  0.0), 1.0)
        self._map_free_thresh = min(max(self._map_free_thresh, 0.0), 1.0)
        if self._map_free_thresh >= self._map_occ_thresh:
            self.get_logger().warn(
                'map_free_thresh >= map_occupied_thresh → s\'ajusten a valors per defecte.'
            )
            self._map_free_thresh = 0.25
            self._map_occ_thresh  = 0.65

        # ----------------------------------------------------------
        # QoS profiles
        # ----------------------------------------------------------
        qos_reliable    = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,    depth=10)
        qos_best_effort = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        qos_map = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )

        # ----------------------------------------------------------
        # Publisher
        # ----------------------------------------------------------
        self._cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', qos_reliable)

        # ----------------------------------------------------------
        # Subscribers
        # ----------------------------------------------------------
        self.create_subscription(LaserScan,     '/scan', self._scan_cb, qos_best_effort)
        self.create_subscription(Odometry,      '/odom', self._odom_cb, qos_reliable)
        self.create_subscription(OccupancyGrid, '/map',  self._map_cb,  qos_map)

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
        self._slam_active = False   # True en quan es rep el primer TF vàlid

        # Offset odometria:
        # L'odometria del TB3 comença sempre a (0, 0, 0) en arrencar.
        # Guardem la primera lectura odom com a origen i calculem la pose
        # real al mapa com:
        #   delta  = odom_now - odom_origin          (en frame odom)
        #   map_xy = Punt_A + R(start_yaw) * delta   (rotat al frame mapa)
        # Així la telemetria i el navegador sempre treballen en coordenades
        # absolutes del mapa fins que SLAM estigui disponible.
        self._map_start_x     = start_x
        self._map_start_y     = start_y
        self._map_start_yaw   = start_yaw_rad
        self._odom_origin_x   = None   # primera lectura odom X
        self._odom_origin_y   = None   # primera lectura odom Y
        self._odom_origin_yaw = None   # primera lectura odom yaw

        # Offset SLAM:
        # SLAM Toolbox construeix el seu mapa des del seu propi origen,
        # que normalment NO coincideix amb el Punt A (4.280, 1.735).
        # En el primer TF vàlid calculem:
        #   slam_offset = pose_odom_corregida - pose_SLAM_rebuda
        # i l'apliquem a totes les lectures SLAM posteriors:
        #   map_pose = slam_pose + slam_offset
        # Si SLAM ja estan alineats, l'offset serà ~(0,0,0).
        self._slam_offset_x:   float = 0.0
        self._slam_offset_y:   float = 0.0
        self._slam_offset_yaw: float = 0.0
        self._slam_offset_set: bool  = False   # True un cop calculat

        # ----------------------------------------------------------
        # Estat intern — dades
        # ----------------------------------------------------------
        self._scan_ready = False
        self._odom_ready = False
        self._map_ready  = False

        self._latest_map      = None
        self._map_saved       = False
        self._map_update_cnt  = 0

        # ----------------------------------------------------------
        # Waypoints Fase I
        # ----------------------------------------------------------
        self._waypoints = list(Config.WAYPOINTS)         # còpia mutable
        self._wp_labels = list(Config.WAYPOINT_LABELS)
        self._wp_idx    = 0                              # índex actual
        self._total_wps = len(self._waypoints)

        # ----------------------------------------------------------
        # Sub-mòduls
        # ----------------------------------------------------------
        self._navigator   = WaypointNavigator(logger=self.get_logger())
        self._avoider     = ObstacleAvoidance(logger=self.get_logger())
        self._csv_logger  = MissionLogger()   # CSV logger (no _logger: conflicte amb rclpy Node)

        # Posa la pose inicial al navegador
        self._navigator.set_odom_pose(start_x, start_y, start_yaw_rad)

        # ----------------------------------------------------------
        # Estat de la missió
        # ----------------------------------------------------------
        self._mission_state  = self._WAITING
        self._arrived_time   = None   # nanoseconds quan arriba a Q
        self._tick           = 0      # comptador de ticks (20 Hz)
        self._n_obstacles    = 0      # actualitzat per ObstacleAvoidance

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
        self.get_logger().info(
            f'Mapa de sortida: {self._map_prefix}.pgm/.yaml  '
            f'(delay={self._map_delay:.1f}s)'
        )

    # ==========================================================
    # BANNER
    # ==========================================================

    def _print_banner(self, sx: float, sy: float, syaw: float) -> None:
        log = self.get_logger().info
        log('=' * 60)
        log('  DEBUG PHASE I NODE — Autonomous Navigation')
        log('=' * 60)
        log(f'  Posició inicial (Punt A) : ({sx:.3f}, {sy:.3f})'
            f'  yaw={syaw:.1f}°')
        log(f'  Waypoints a recórrer     : {self._total_wps}')
        for i, ((wx, wy), lbl) in enumerate(
                zip(self._waypoints, self._wp_labels)):
            log(f'    [{i+1}/{self._total_wps}]  Punt {lbl:5s} → ({wx:.3f}, {wy:.3f})')
        log('=' * 60)
        log('  Localització : SLAM Toolbox (fallback: odometria)')
        log('  Avoidance    : Bug2 reactiu')
        log(f'  CSV log      : {Config.LOG_PATH}')
        log('=' * 60)
        log('  Flux final   : C → D → F → Q → DONE')
        log('=' * 60)

    # ==========================================================
    # CALLBACKS ROS2
    # ==========================================================

    def _scan_cb(self, msg: LaserScan) -> None:
        """Ingestió del scan LiDAR → ObstacleAvoidance."""
        self._avoider.update_scan(msg)
        self._scan_ready = True

        # Estimació ràpida del nombre d'obstacles a prop (sectors actius)
        self._n_obstacles = self._count_near_obstacles(msg)

    def _odom_cb(self, msg: Odometry) -> None:
        """
        Actualitza la pose del robot.
        Prioritza SLAM (TF map → base_footprint); si no disponible, usa
        odometria corregida amb l'offset del Punt A.

        Correcció odometria (quan SLAM no és actiu):
          El TB3 publica /odom sempre des de (0, 0, 0).
          Guardem la primera lectura com a origen i calculem:
            delta_x   = odom_x   - odom_origin_x
            delta_y   = odom_y   - odom_origin_y
            delta_yaw = odom_yaw - odom_origin_yaw   (normalitzat)
          Llavors transformem el delta al frame mapa (rotat per start_yaw):
            map_x = start_x + cos(start_yaw)*delta_x - sin(start_yaw)*delta_y
            map_y = start_y + sin(start_yaw)*delta_x + cos(start_yaw)*delta_y
            map_yaw = start_yaw + delta_yaw
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
        dx      = raw_x   - self._odom_origin_x
        dy      = raw_y   - self._odom_origin_y
        d_yaw   = normalize_angle(raw_yaw - self._odom_origin_yaw)

        # Transforma delta al frame mapa (rotació per start_yaw)
        cos_s = math.cos(self._map_start_yaw)
        sin_s = math.sin(self._map_start_yaw)
        map_x   = self._map_start_x + cos_s * dx - sin_s * dy
        map_y   = self._map_start_y + sin_s * dx + cos_s * dy
        map_yaw = normalize_angle(self._map_start_yaw + d_yaw)

        self._navigator.set_odom_pose(map_x, map_y, map_yaw)
        self._odom_ready = True

        # Intenta SLAM via TF (millor precisió)
        slam_x, slam_y, slam_yaw = self._read_slam_tf()
        if slam_x is not None:
            # Primer TF SLAM vàlid: calcula l'offset entre el frame SLAM
            # i les coordenades reals del mapa (on sabem que estem perquè
            # l'odometria corregida ja és correcta en aquest instant).
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

            # Aplica l'offset per alinear SLAM al frame del mapa real
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

    def _map_cb(self, msg: OccupancyGrid) -> None:
        """Desa l'últim OccupancyGrid per exportació posterior."""
        self._latest_map     = msg
        self._map_ready      = True
        self._map_update_cnt += 1

    # ==========================================================
    # WATCHDOG (50 Hz)
    # ==========================================================

    def _watchdog(self) -> None:
        """
        Para el robot immediatament si el sector FRONT entra en DANGER.
        S'executa a 50 Hz, independentment del loop de control.
        """
        if not self._scan_ready:
            return
        if self._avoider.is_front_danger():
            self._publish_stop()
            self.get_logger().warn(
                '[WATCHDOG] ⚠ PERILL AL FRONT — parada d\'emergència!'
            )

    # ==========================================================
    # LOOP DE CONTROL (20 Hz)
    # ==========================================================

    def _control_loop(self) -> None:
        """
        Lògica principal de navegació de la Fase I.

        Flux per tick:
          1. Comprova que hi ha dades de scan i odom.
          2. Si la missió ha acabat, espera delay i exporta mapa.
          3. WAITING → engega primer waypoint (C).
          4. NAVIGATE → navega C→D→F; en arribar a F entra SEEK_DOOR.
          5. SEEK_DOOR → navega cap a Porta nominal amb DoorDetector actiu:
               · Si DoorDetector confirma porta real → APPROACH_DOOR.
               · Si arriba a Porta nominal sense detecció → usa nominal.
          6. APPROACH_DOOR → navega al centre real de la porta detectada;
               en arribar → missió completada.
          7. Anti-stuck, obstacle avoidance i CSV logger en tots els estats actius.
        """
        self._tick += 1

        # ---- Espera dades inicials ----
        if not self._scan_ready or not self._odom_ready:
            return

        # ---- Missió completada: espera delay i desa mapa ----
        if self._mission_state == self._DONE:
            if not self._map_saved and self._arrived_time is not None:
                elapsed = (
                    self.get_clock().now().nanoseconds - self._arrived_time
                ) / 1e9
                if elapsed >= self._map_delay:
                    self._save_map()
            return

        # ---- Primer tick amb dades: engega la missió ----
        if self._mission_state == self._WAITING:
            self._mission_state = self._NAVIGATE
            self._launch_waypoint(self._wp_idx)

        # ---- Anti-stuck: calcula delta yaw per a l'avoider ----
        delta_yaw = abs(normalize_angle(self._yaw - self._prev_yaw))
        self._avoider.update_force_rotate(delta_yaw)
        self._prev_yaw = self._yaw

        # ---- Tria el waypoint objectiu actiu ----
        wp_x, wp_y = self._waypoints[self._wp_idx]

        # ---- Obstacle avoidance ----
        cmd, in_avoidance = self._avoider.compute(
            self._x, self._y, self._yaw, wp_x, wp_y
        )

        if in_avoidance:
            self._publish(cmd.linear_x, cmd.angular_z)
        else:
            nav_cmd = self._navigator.step()
            self._publish(nav_cmd.linear_x, nav_cmd.angular_z)

        # ---- Lògica específica per estat ----
        if self._mission_state == self._NAVIGATE:
            if self._navigator.has_arrived():
                self._on_waypoint_reached(wp_x, wp_y)

        # ---- Actualitza CSV logger ----
        self._csv_logger.update(
            phase='I',
            robot_x=self._x,
            robot_y=self._y,
            robot_yaw=self._yaw,
            n_obstacles=self._n_obstacles,
        )

        # ---- Telemetria per pantalla (1 cop per segon) ----
        if self._tick % Config.CONTROL_HZ == 0:
            dist      = math.hypot(self._x - wp_x, self._y - wp_y)
            nav_state = self._navigator.get_state().name
            av_state  = self._avoider.get_state().name
            loc_src   = 'SLAM' if self._slam_active else 'ODOM'
            ms        = self._mission_state

            # Etiqueta del destí actiu
            if ms == self._NAVIGATE:
                lbl = f'Punt{self._wp_labels[self._wp_idx]}' if self._wp_idx < self._total_wps else 'Q'
            else:
                lbl = ms

            self.get_logger().info(
                f'[TELEM]'
                f'  pos=({self._x:.3f},{self._y:.3f})'
                f'  yaw={math.degrees(self._yaw):6.1f}°'
                f'  dest={lbl}'
                f'  dist={dist:.3f}m'
                f'  nav={nav_state}'
                f'  avoid={av_state}'
                f'  loc={loc_src}'
                f'  obs={self._n_obstacles}'
            )

    # ==========================================================
    # GESTIÓ WAYPOINTS
    # ==========================================================

    def _launch_waypoint(self, idx: int) -> None:
        """Configura el navegador per al waypoint d'índex idx."""
        wx, wy = self._waypoints[idx]
        lbl    = self._wp_labels[idx]
        dist   = math.hypot(wx - self._x, wy - self._y)

        self.get_logger().info('─' * 50)
        self.get_logger().info(
            f'  ▶ Waypoint [{idx+1}/{self._total_wps}]: '
            f'Punt {lbl} → ({wx:.3f}, {wy:.3f})'
        )
        self.get_logger().info(
            f'    Distància actual: {dist:.2f} m  '
            f'  Pose: ({self._x:.3f}, {self._y:.3f}, '
            f'{math.degrees(self._yaw):.1f}°)'
        )
        self.get_logger().info('─' * 50)

        self._avoider.reset()
        self._navigator.set_waypoint(wx, wy)

    def _on_waypoint_reached(self, wp_x: float, wp_y: float) -> None:
        """Gestiona l'arribada al waypoint actual i avança al següent."""
        lbl  = self._wp_labels[self._wp_idx]
        dist = math.hypot(self._x - wp_x, self._y - wp_y)

        self.get_logger().info('=' * 50)
        self.get_logger().info(
            f'  ✓ WAYPOINT ASSOLIT: Punt {lbl} '
            f'[{self._wp_idx+1}/{self._total_wps}]'
        )
        self.get_logger().info(
            f'    Posició final : ({self._x:.3f}, {self._y:.3f})'
        )
        self.get_logger().info(
            f'    Error posició : {dist:.3f} m'
        )
        self.get_logger().info(
            f'    Localització  : {"SLAM" if self._slam_active else "Odometria"}'
        )
        self.get_logger().info('=' * 50)

        self._wp_idx += 1

        if self._wp_idx >= self._total_wps:
            # Hem assolit Q (darrer WP) → missió completada
            self._mission_done()
        else:
            # Llança el següent waypoint normal
            self._launch_waypoint(self._wp_idx)

    def _mission_done(self) -> None:
        """Marca la fi de la Fase I i atura el robot."""
        self._mission_state = self._DONE
        self._publish_stop()
        self._arrived_time  = self.get_clock().now().nanoseconds

        self.get_logger().info('★' * 50)
        self.get_logger().info('  ✓✓ FASE I COMPLETADA — Robot al punt Q!')
        self.get_logger().info(
            f'  Posició final : ({self._x:.3f}, {self._y:.3f}, '
            f'{math.degrees(self._yaw):.1f}°)'
        )
        self.get_logger().info(
            f'  Localització  : {"SLAM" if self._slam_active else "Odometria"}'
        )
        self.get_logger().info(
            f'  Esperant {self._map_delay:.1f}s per exportar el mapa final...'
        )
        self.get_logger().info('★' * 50)

    # ==========================================================
    # TELEMETRIA (pantalla)
    # ==========================================================

    def _print_telemetry(self, wp_x: float, wp_y: float) -> None:
        """Imprimeix una línia de telemetria cada segon."""
        dist      = math.hypot(self._x - wp_x, self._y - wp_y)
        nav_state = self._navigator.get_state().name
        av_state  = self._avoider.get_state().name
        loc_src   = 'SLAM' if self._slam_active else 'ODOM'
        ms        = self._mission_state

        # Etiqueta del destí actiu
        if ms == self._NAVIGATE:
            lbl = f'Punt{self._wp_labels[self._wp_idx]}' if self._wp_idx < self._total_wps else '?'
        elif ms == self._SEEK_DOOR:
            lbl = 'Porta(nominal)'
        elif ms == self._APPROACH_DOOR:
            lbl = 'Porta(REAL)'
        else:
            lbl = ms

        # Informació addicional del detector en mode cerca
        door_info = ''
        if ms in (self._SEEK_DOOR, self._APPROACH_DOOR):
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
        n       = len(ranges)
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
    # EXPORTACIÓ DEL MAPA
    # ==========================================================

    def _save_map(self) -> None:
        """Escriu el darrer OccupancyGrid rebut a fitxers PGM + YAML."""
        if self._map_saved:
            return

        if not self._map_ready or self._latest_map is None:
            self.get_logger().warn(
                'No s\'ha rebut cap missatge a /map; '
                'no es pot exportar el mapa.'
            )
            return

        grid   = self._latest_map
        width  = grid.info.width
        height = grid.info.height
        data   = grid.data

        if len(data) != width * height:
            self.get_logger().error(
                f'Mapa inconsistent ({len(data)} ≠ {width}×{height}); '
                'exportació cancel·lada.'
            )
            return

        base_prefix = Path.cwd() / self._map_prefix
        pgm_path, yaml_path = self._output_paths(base_prefix)

        occ_int  = int(round(self._map_occ_thresh  * 100.0))
        free_int = int(round(self._map_free_thresh * 100.0))

        try:
            with pgm_path.open('wb') as pgm:
                pgm.write(f'P5\n{width} {height}\n255\n'.encode('ascii'))
                for row_y in range(height - 1, -1, -1):   # flip Y (ROS → imatge)
                    row_base = row_y * width
                    row      = bytearray(width)
                    for col_x in range(width):
                        v = data[row_base + col_x]
                        if v < 0:
                            row[col_x] = 205                    # desconegut
                        elif self._map_mode == 'trinary':
                            if v >= occ_int:
                                row[col_x] = 0                  # ocupat
                            elif v <= free_int:
                                row[col_x] = 254                # lliure
                            else:
                                row[col_x] = 205                # desconegut
                        else:   # scale
                            row[col_x] = int(round((100 - v) * 255 / 100.0))
                    pgm.write(row)

            origin = grid.info.origin.position
            orig_yaw = self._quat_to_yaw(grid.info.origin.orientation)
            yaml_content = (
                f'image: {pgm_path.name}\n'
                f'mode: {self._map_mode}\n'
                f'resolution: {grid.info.resolution}\n'
                f'origin: [{origin.x}, {origin.y}, {orig_yaw}]\n'
                f'negate: 0\n'
                f'occupied_thresh: {self._map_occ_thresh}\n'
                f'free_thresh: {self._map_free_thresh}\n'
            )
            yaml_path.write_text(yaml_content, encoding='ascii')

            self._map_saved = True
            self.get_logger().info(
                f'Mapa exportat: {pgm_path}  +  {yaml_path}  '
                f'(actualizacions /map: {self._map_update_cnt})'
            )
        except OSError as exc:
            self.get_logger().error(f'Error exportant mapa: {exc}')

    def _output_paths(self, base: Path):
        """
        Retorna (pgm_path, yaml_path) lliures.
        Si map_overwrite=True, sobreescriu directament.
        """
        pgm  = base.with_suffix('.pgm')
        yaml = base.with_suffix('.yaml')

        if self._map_overwrite or (not pgm.exists() and not yaml.exists()):
            return pgm, yaml

        for idx in range(1, 10000):
            candidate = base.parent / f'{base.name}_{idx:03d}'
            cp = candidate.with_suffix('.pgm')
            cy = candidate.with_suffix('.yaml')
            if not cp.exists() and not cy.exists():
                return cp, cy

        # Fallback extremadament improbable
        import time as _time
        stamp = str(int(_time.time()))
        fb = base.parent / f'{base.name}_{stamp}'
        return fb.with_suffix('.pgm'), fb.with_suffix('.yaml')

    # ==========================================================
    # SHUTDOWN
    # ==========================================================

    def shutdown(self) -> None:
        """Atura el robot, desa el mapa i tanca el logger en sortir."""
        self.get_logger().info('Apagant debug_phase1_node — parant robot...')
        self._publish_stop()
        self._save_map()
        self._csv_logger.close()


# ============================================================
# ENTRY POINT
# ============================================================
def main(args=None) -> None:
    rclpy.init(args=args)
    node = DebugPhase1Node()
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