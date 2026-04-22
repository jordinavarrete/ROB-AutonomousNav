#!/usr/bin/env python3
"""
mission_node.py — Node principal de la missió autònoma (Fase I + Fase II).

Executa la seqüència completa de la missió:

    Fase I  (inici a Punt A, yaw=0°):
        A → C → D → F → Q
        (a Q: alineació yaw → 0° per enllaçar amb la Fase II)

    Fase II (inici a Q amb yaw=0°):
        Q → R → U → T  (bucle d'exploració amb StationDetector)
        → estació detectada: guarda posició
        → GO_BASE (P) → RETURN_DETECT (torna al punt de detecció)
        → re-detecció de l'estació (comparació de precisió)
        → DOCKING de precisió (amb obstacle avoidance)
        → DONE

Màquina d'estats:
    WAITING        → esperant primer /scan i /odom
    NAV_P1         → navegant C, D, F, Q (Fase I)
    ALIGN_Q        → in-place rotation a Q fins yaw = 0° (transició I → II)
    EXPLORE        → bucle R, U, T amb StationDetector actiu (Fase II)
    GO_BASE        → estació detectada; navega cap a P Base
    RETURN_DETECT  → torna al punt on es va detectar l'estació;
                     StationDetector resetejat per re-detectar
    DOCKING        → DockingController + ObstacleAvoidance actiu
    DONE           → Missió completada, robot estacionat

Característiques:
  · Localització: SLAM Toolbox (TF map→base_footprint) amb fallback a odometria.
  · Obstacle avoidance reactiu: Bug2 via ObstacleAvoidance (actiu també en DOCKING).
  · Detecció d'estació: StationDetector (4 pilars, quadrat 40×40 cm).
  · Re-detecció: compara la posició de l'estació amb la primera detecció.
  · Docking de precisió: DockingController amb obstacle avoidance.
  · Watchdog de seguretat a 50 Hz.
  · MissionLogger: CSV a ~/mission_log.csv (phase = 'I' o 'II' segons estat).
  · Exportació del mapa: PGM + YAML al finalitzar.

Topics:
  Subscriu:  /scan      (LaserScan,     BEST_EFFORT)
             /odom      (Odometry,      RELIABLE)
             /map       (OccupancyGrid, RELIABLE + TRANSIENT_LOCAL)
  Publica:   /cmd_vel   (TwistStamped,  RELIABLE)

Paràmetres ROS2 (tots opcionals):
  start_x              float  Posició X inicial al mapa       [defecte: 4.280]
  start_y              float  Posició Y inicial al mapa       [defecte: 1.735]
  start_yaw_deg        float  Orientació inicial (graus)      [defecte: 0.0]
  map_output_prefix    str    Prefix fitxers mapa             [defecte: 'mission_map']
  map_save_delay_sec   float  Espera post-arribada per mapa   [defecte: 3.0]
  map_mode             str    'scale' o 'trinary'             [defecte: 'scale']
  map_occupied_thresh  float  Llindar ocupació mapa           [defecte: 0.65]
  map_free_thresh      float  Llindar lliure mapa             [defecte: 0.25]
  map_overwrite        bool   Sobreescriu si ja existeix      [defecte: False]

Ús:
    ros2 run autonomous_nav mission_node
    ros2 run autonomous_nav mission_node --ros-args \\
        -p start_yaw_deg:=0.0 -p map_output_prefix:=run_map
"""

# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    # ---- Posició inicial: Punt A ----
    START_X           = 4.280   # m
    START_Y           = 1.735   # m
    START_YAW_DEG     = 0.0     # graus

    # ---- Fase I: Waypoints (A és la posició inicial, no un objectiu) ----
    P1_LABELS = ['C', 'D', 'E', 'F', 'Q']
    P1_WAYPOINTS = [
        (4.880,  2.535),   # Punt C
        (5.080,  5.740),   # Punt D
        (5.880,  8.145),   # Punt E
        (5.480, 10.545),   # Punt F
        (9.115, 14.190),   # Punt Q  ← darrer WP de la Fase I
    ]

    # ---- Transició I → II: alineació yaw a Q ----
    ALIGN_Q_YAW_RAD       = 0.0    # objectiu yaw a Q [rad]
    ALIGN_Q_THRESHOLD     = 0.03   # rad (≈ 1.7°) — tolerància per acabar ALIGN_Q
    ALIGN_Q_KP            = 1.2    # guany proporcional rotació
    ALIGN_Q_MIN_ANG_SPEED = 0.05   # rad/s — rotació mínima (evita stall)
    ALIGN_Q_MAX_ANG_SPEED = 0.60   # rad/s — rotació màxima
    ALIGN_Q_HOLD_TICKS    = 10     # ticks @20 Hz → 0.5 s dins tolerància

    # ---- Fase II: Bucle d'exploració (Q → R → U → T) ----
    P2_LOOP_LABELS = ['R', 'U', 'T']
    P2_LOOP_WAYPOINTS = [
        (7.310, 16.190),   # Punt R
        (1.075, 16.190),   # Punt U
        (1.275, 14.990),   # Punt T
    ]

    # ---- Punt Base (destí intermedi de la Fase II) ----
    BASE_LABEL = 'P_Base'
    BASE_X     = 3.475    # m
    BASE_Y     = 15.390   # m

    # ---- Rates de control ----
    CONTROL_HZ  = 20    # Hz — loop principal
    WATCHDOG_HZ = 50    # Hz — vigilant de seguretat

    # ---- Frames TF ----
    MAP_FRAME  = 'map'
    BASE_FRAME = 'base_footprint'

    # ---- Límits de velocitat publicats ----
    LINEAR_MAX  = 0.20   # m/s
    ANGULAR_MAX = 1.00   # rad/s

    # ---- Anti-orbita durant DOCKING (sense desactivar avoidance) ----
    DOCK_PROGRESS_EPSILON_M      = 0.01   # m per considerar progrés real
    DOCK_NO_PROGRESS_TICKS       = 80     # 4.0 s @ 20 Hz
    DOCK_WALL_FOLLOW_RESET_TICKS = 50     # 2.5 s continus en WALL_FOLLOW
    DOCK_RECOVERY_COOLDOWN_TICKS = 40     # 2.0 s entre recuperacions


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
from autonomous_nav.station_detector import StationDetector
from autonomous_nav.docking_controller import DockingController


# ============================================================
# NODE PRINCIPAL
# ============================================================
class MissionNode(Node):
    """
    Node principal que encadena la Fase I i la Fase II de la missió.

    Flux de missió:
        WAITING → NAV_P1 (C,D,F,Q) → ALIGN_Q (yaw→0°) →
        EXPLORE (R,U,T) → GO_BASE → RETURN_DETECT → DOCKING → DONE
    """

    # ---- Estats de la missió ----
    _WAITING        = 'WAITING'
    _NAV_P1         = 'NAV_P1'
    _ALIGN_Q        = 'ALIGN_Q'
    _EXPLORE        = 'EXPLORE'
    _GO_BASE        = 'GO_BASE'
    _RETURN_DETECT  = 'RETURN_DETECT'
    _DOCKING        = 'DOCKING'
    _DONE           = 'DONE'

    # ---- Estats que pertanyen a la Fase I (etiqueta CSV) ----
    _PHASE1_STATES = frozenset({_WAITING, _NAV_P1, _ALIGN_Q})

    def __init__(self) -> None:
        super().__init__('mission_node')

        # ----------------------------------------------------------
        # Paràmetres ROS2
        # ----------------------------------------------------------
        self.declare_parameter('start_x',             Config.START_X)
        self.declare_parameter('start_y',             Config.START_Y)
        self.declare_parameter('start_yaw_deg',       Config.START_YAW_DEG)
        self.declare_parameter('map_output_prefix',   'mission_map')
        self.declare_parameter('map_save_delay_sec',  3.0)
        self.declare_parameter('map_mode',            'scale')
        self.declare_parameter('map_occupied_thresh', 0.65)
        self.declare_parameter('map_free_thresh',     0.25)
        self.declare_parameter('map_overwrite',       False)

        start_x       = float(self.get_parameter('start_x').value)
        start_y       = float(self.get_parameter('start_y').value)
        start_yaw_deg = float(self.get_parameter('start_yaw_deg').value)
        start_yaw_rad = math.radians(start_yaw_deg)

        self._map_prefix      = str(self.get_parameter('map_output_prefix').value)
        self._map_delay       = float(self.get_parameter('map_save_delay_sec').value)
        self._map_mode        = str(self.get_parameter('map_mode').value).lower()
        self._map_occ_thresh  = float(self.get_parameter('map_occupied_thresh').value)
        self._map_free_thresh = float(self.get_parameter('map_free_thresh').value)
        self._map_overwrite   = bool(self.get_parameter('map_overwrite').value)

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
        self._slam_active = False   # True quan es rep el primer TF vàlid

        # Offset odometria (el TB3 publica /odom sempre des de (0,0,0))
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
        self._map_ready  = False

        self._latest_map      = None
        self._map_saved       = False
        self._map_update_cnt  = 0

        # ----------------------------------------------------------
        # Fase I: Waypoints
        # ----------------------------------------------------------
        self._p1_waypoints = list(Config.P1_WAYPOINTS)
        self._p1_labels    = list(Config.P1_LABELS)
        self._p1_idx       = 0
        self._p1_total     = len(self._p1_waypoints)

        # ----------------------------------------------------------
        # Fase II: Bucle d'exploració
        # ----------------------------------------------------------
        self._loop_wps    = list(Config.P2_LOOP_WAYPOINTS)
        self._loop_labels = list(Config.P2_LOOP_LABELS)
        self._loop_idx    = 0
        self._loop_count  = 0
        self._legs_done   = 0

        # ----------------------------------------------------------
        # Sub-mòduls
        # ----------------------------------------------------------
        self._navigator   = WaypointNavigator(logger=self.get_logger())
        self._avoider     = ObstacleAvoidance(logger=self.get_logger())
        self._csv_logger  = MissionLogger()
        self._station_det = StationDetector(logger=self.get_logger())
        self._docker      = DockingController(logger=self.get_logger())

        # Posa la pose inicial al navegador
        self._navigator.set_odom_pose(start_x, start_y, start_yaw_rad)

        # ----------------------------------------------------------
        # Estat de la missió
        # ----------------------------------------------------------
        self._mission_state  = self._WAITING
        self._arrived_time   = None
        self._tick           = 0
        self._n_obstacles    = 0

        # ---- Comptadors per a l'alineació a Q ----
        self._align_hold_ticks = 0

        # ---- Estat estació — primera detecció ----
        self._first_station_result = None
        self._first_station_map_x  = None
        self._first_station_map_y  = None

        # ---- Posició on es va detectar l'estació per primera vegada ----
        self._detect_pose_x   = None
        self._detect_pose_y   = None
        self._detect_pose_yaw = None

        # ---- Estat estació — re-detecció ----
        self._redetect_result  = None
        self._dock_station_x   = None
        self._dock_station_y   = None
        self._waiting_redetect_at_pose = False

        # ---- Anti-loop específic de docking ----
        self._dock_best_dist         = float('inf')
        self._dock_no_progress_ticks = 0
        self._dock_wall_follow_ticks = 0
        self._dock_recovery_cooldown = 0

        # ---- Pilars en temps real durant docking ----
        self._live_pillars_map = []
        self._inside_pillars   = False

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
        log('  MISSION NODE — Autonomous Navigation (Phase I + II)')
        log('=' * 60)
        log(f'  Posició inicial (Punt A) : ({sx:.3f}, {sy:.3f})'
            f'  yaw={syaw:.1f}°')
        log('')
        log(f'  FASE I — Waypoints       : {self._p1_total}')
        for i, ((wx, wy), lbl) in enumerate(
                zip(self._p1_waypoints, self._p1_labels)):
            log(f'    [{i+1}/{self._p1_total}]  Punt {lbl:5s} → ({wx:.3f}, {wy:.3f})')
        log(f'  Alineació a Q            : yaw → '
            f'{math.degrees(Config.ALIGN_Q_YAW_RAD):.1f}°')
        log('')
        log(f'  FASE II — Exploració     : Q → R → U → T '
            f'(fins detectar estació)')
        all_labels = ['Q'] + self._loop_labels
        all_wps = [(Config.P1_WAYPOINTS[-1][0], Config.P1_WAYPOINTS[-1][1])] + self._loop_wps
        for lbl, (wx, wy) in zip(all_labels, all_wps):
            log(f'    Punt {lbl:5s} → ({wx:.3f}, {wy:.3f})')
        log(f'  Punt Base                : ({Config.BASE_X:.3f}, {Config.BASE_Y:.3f})')
        log('=' * 60)
        log('  Localització : SLAM Toolbox (fallback: odometria)')
        log('  Avoidance    : Bug2 reactiu (actiu també en DOCKING)')
        log('  Detector     : StationDetector (4 pilars, 40×40 cm)')
        log('  CSV log      : ~/mission_log.csv')
        log('=' * 60)
        log('  Flux:')
        log('    NAV_P1 (C→D→F→Q) → ALIGN_Q (yaw=0°) →')
        log('    EXPLORE (R→U→T) → detecta estació → guarda posició →')
        log('    GO_BASE (P) → RETURN_DETECT → re-detecta estació →')
        log('    DOCKING (detecció contínua; avoidance OFF dins pilars) →')
        log('    DONE')
        log('=' * 60)

    # ==========================================================
    # CALLBACKS ROS2
    # ==========================================================

    def _scan_cb(self, msg: LaserScan) -> None:
        """Ingestió del scan LiDAR."""
        self._avoider.update_scan(msg)
        self._station_det.update_scan(msg)
        self._scan_ready = True
        self._n_obstacles = self._count_near_obstacles(msg)

    def _odom_cb(self, msg: Odometry) -> None:
        """
        Actualitza la pose del robot.
        Prioritza SLAM (TF map → base_footprint); fallback a odometria
        corregida amb l'offset del Punt A.
        """
        raw_x   = msg.pose.pose.position.x
        raw_y   = msg.pose.pose.position.y
        raw_yaw = self._quat_to_yaw(msg.pose.pose.orientation)

        if self._odom_origin_x is None:
            self._odom_origin_x   = raw_x
            self._odom_origin_y   = raw_y
            self._odom_origin_yaw = raw_yaw
            self.get_logger().info(
                f'[ODOM] Origen odom fixat: '
                f'({raw_x:.4f}, {raw_y:.4f}, {math.degrees(raw_yaw):.1f}°)'
            )

        dx    = raw_x   - self._odom_origin_x
        dy    = raw_y   - self._odom_origin_y
        d_yaw = normalize_angle(raw_yaw - self._odom_origin_yaw)

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
        En DOCKING dins els pilars: watchdog inactiu (pilars no són obstacles).
        En ALIGN_Q: rotació pura, no cal control extra (però sí watchdog).
        """
        if not self._scan_ready:
            return
        if self._mission_state == self._DONE:
            return

        if self._mission_state == self._DOCKING:
            if self._inside_pillars:
                return
            if self._avoider.is_front_danger() and not self._docker.is_docked():
                self._publish_stop()
                self.get_logger().warn(
                    '[WATCHDOG] ⚠ PERILL AL FRONT durant docking — '
                    'parada d\'emergència!'
                )
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
        Lògica principal de la missió.

        Flux per tick:
          1. Espera dades inicials (/scan + /odom).
          2. Missió completada: espera delay, exporta mapa.
          3. WAITING → arrenca NAV_P1 (waypoint C).
          4. Actualitza anti-stuck de l'avoider.
          5. Dispatch segons estat:
               · NAV_P1   → navegador + avoidance, C→D→F→Q.
               · ALIGN_Q  → rotació pura fins yaw=0°, després EXPLORE.
               · EXPLORE  → bucle R→U→T, StationDetector actiu.
               · GO_BASE  → navega a P Base.
               · RETURN_DETECT → torna a detect_pose, re-detecció activa.
               · DOCKING  → DockingController + avoidance (via _step_docking).
          6. CSV logger i telemetria (1 Hz).
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
            self._mission_state = self._NAV_P1
            self._launch_p1_waypoint(self._p1_idx)

        # ---- Anti-stuck: calcula delta yaw per a l'avoider ----
        delta_yaw = abs(normalize_angle(self._yaw - self._prev_yaw))
        self._avoider.update_force_rotate(delta_yaw)
        self._prev_yaw = self._yaw

        # ---- ALIGN_Q: rotació pura a yaw = 0° (sense avoidance ni nav) ----
        if self._mission_state == self._ALIGN_Q:
            self._step_align_q()
            self._update_csv()
            if self._tick % Config.CONTROL_HZ == 0:
                self._print_telemetry_align()
            return

        # ---- DOCKING: DockingController + ObstacleAvoidance ----
        if self._mission_state == self._DOCKING:
            self._step_docking()
            self._update_csv()
            if self._tick % Config.CONTROL_HZ == 0:
                self._print_telemetry_docking()
            return

        # ---- Waypoint objectiu actiu segons estat ----
        if self._mission_state == self._NAV_P1:
            wp_x, wp_y = self._p1_waypoints[self._p1_idx]
        elif self._mission_state == self._EXPLORE:
            wp_x, wp_y = self._loop_wps[self._loop_idx]
        elif self._mission_state == self._GO_BASE:
            wp_x, wp_y = Config.BASE_X, Config.BASE_Y
        elif self._mission_state == self._RETURN_DETECT:
            wp_x, wp_y = self._detect_pose_x, self._detect_pose_y
        else:
            wp_x, wp_y = self._x, self._y

        # ---- Obstacle avoidance ----
        cmd, in_avoidance = self._avoider.compute(
            self._x, self._y, self._yaw, wp_x, wp_y
        )

        if in_avoidance:
            self._publish(cmd.linear_x, cmd.angular_z)
        else:
            nav_cmd = self._navigator.step()
            self._publish(nav_cmd.linear_x, nav_cmd.angular_z)

        # ---- StationDetector: actiu en EXPLORE i RETURN_DETECT ----
        if self._mission_state == self._EXPLORE:
            if self._first_station_result is None:
                result = self._station_det.detect(self._x, self._y, self._yaw)
                if result is not None:
                    self._on_station_found(result)

        elif self._mission_state == self._RETURN_DETECT:
            if self._redetect_result is None:
                result = self._station_det.detect(self._x, self._y, self._yaw)
                if result is not None:
                    self._on_station_redetected(result)

        # ---- Transicions per arribada ----
        if self._mission_state == self._NAV_P1:
            if self._navigator.has_arrived():
                self._on_p1_waypoint_reached(wp_x, wp_y)

        elif self._mission_state == self._EXPLORE:
            if self._navigator.has_arrived():
                self._on_loop_waypoint_reached()

        elif self._mission_state == self._GO_BASE:
            if self._navigator.has_arrived():
                self._on_base_reached()

        elif self._mission_state == self._RETURN_DETECT:
            if self._navigator.has_arrived() and self._redetect_result is None:
                self._on_return_detect_arrived()

        # ---- CSV logger ----
        self._update_csv()

        # ---- Telemetria (1 Hz) ----
        if self._tick % Config.CONTROL_HZ == 0:
            self._print_telemetry(wp_x, wp_y)

    # ==========================================================
    # FASE I — GESTIÓ DE WAYPOINTS
    # ==========================================================

    def _launch_p1_waypoint(self, idx: int) -> None:
        """Configura el navegador per al waypoint idx de la Fase I."""
        wx, wy = self._p1_waypoints[idx]
        lbl    = self._p1_labels[idx]
        dist   = math.hypot(wx - self._x, wy - self._y)

        self.get_logger().info('─' * 50)
        self.get_logger().info(
            f'  ▶ FASE I · Waypoint [{idx+1}/{self._p1_total}]: '
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

    def _on_p1_waypoint_reached(self, wp_x: float, wp_y: float) -> None:
        """Gestiona l'arribada a un waypoint de la Fase I."""
        lbl  = self._p1_labels[self._p1_idx]
        dist = math.hypot(self._x - wp_x, self._y - wp_y)

        self.get_logger().info('=' * 50)
        self.get_logger().info(
            f'  ✓ WAYPOINT ASSOLIT: Punt {lbl} '
            f'[{self._p1_idx+1}/{self._p1_total}]'
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

        self._p1_idx += 1

        if self._p1_idx >= self._p1_total:
            # Hem assolit Q (darrer WP de la Fase I) → alineació yaw
            self._start_align_q()
        else:
            # Següent waypoint de la Fase I
            self._launch_p1_waypoint(self._p1_idx)

    # ==========================================================
    # ALINEACIÓ A Q (transició Fase I → Fase II)
    # ==========================================================

    def _start_align_q(self) -> None:
        """Inicia l'estat ALIGN_Q: rotació pura a Q fins yaw = 0°."""
        self._mission_state = self._ALIGN_Q
        self._align_hold_ticks = 0
        self._publish_stop()
        self._navigator.clear_waypoint()
        self._avoider.reset()

        err = normalize_angle(Config.ALIGN_Q_YAW_RAD - self._yaw)
        self.get_logger().info('★' * 50)
        self.get_logger().info(
            f'  ✓ FASE I COMPLETADA — Robot al Punt Q '
            f'({self._x:.3f}, {self._y:.3f})'
        )
        self.get_logger().info(
            f'  ▶ ALIGN_Q — alineant yaw a '
            f'{math.degrees(Config.ALIGN_Q_YAW_RAD):.1f}°  '
            f'(actual: {math.degrees(self._yaw):.1f}°, '
            f'error: {math.degrees(err):+.1f}°)'
        )
        self.get_logger().info('★' * 50)

    def _step_align_q(self) -> None:
        """
        Rotació en lloc fins que |yaw - 0°| < ALIGN_Q_THRESHOLD durant
        ALIGN_Q_HOLD_TICKS ticks consecutius (evita falsos positius per
        soroll d'odometria/SLAM).
        """
        err = normalize_angle(Config.ALIGN_Q_YAW_RAD - self._yaw)

        if abs(err) < Config.ALIGN_Q_THRESHOLD:
            # Dins tolerància: mantenir quiet i comptar ticks
            self._align_hold_ticks += 1
            self._publish_stop()

            if self._align_hold_ticks >= Config.ALIGN_Q_HOLD_TICKS:
                self._finish_align_q()
            return

        # Fora de tolerància: trenca el hold i roda
        self._align_hold_ticks = 0

        raw_angular = Config.ALIGN_Q_KP * err
        sign        = 1.0 if raw_angular >= 0 else -1.0
        angular_z   = sign * max(
            Config.ALIGN_Q_MIN_ANG_SPEED,
            min(Config.ALIGN_Q_MAX_ANG_SPEED, abs(raw_angular)),
        )

        self._publish(0.0, angular_z)

    def _finish_align_q(self) -> None:
        """Alineació completada — comença la Fase II (EXPLORE)."""
        self._publish_stop()

        self.get_logger().info('★' * 50)
        self.get_logger().info(
            f'  ✓ ALIGN_Q COMPLETAT — yaw={math.degrees(self._yaw):.2f}°'
        )
        self.get_logger().info(
            f'    Posició final : ({self._x:.3f}, {self._y:.3f})'
        )
        self.get_logger().info(
            '  ▶ INICIANT FASE II — Exploració Q → R → U → T'
        )
        self.get_logger().info('★' * 50)

        # Inicia el bucle d'exploració de la Fase II
        self._mission_state = self._EXPLORE
        self._loop_idx   = 0
        self._loop_count = 0
        self._legs_done  = 0
        self._launch_loop_waypoint()

    # ==========================================================
    # FASE II — BUCLE D'EXPLORACIÓ
    # ==========================================================

    def _launch_loop_waypoint(self) -> None:
        """Configura el navegador per al waypoint actual del bucle."""
        wx, wy = self._loop_wps[self._loop_idx]
        lbl    = self._loop_labels[self._loop_idx]
        dist   = math.hypot(wx - self._x, wy - self._y)

        self.get_logger().info('─' * 50)
        self.get_logger().info(
            f'  ▶ EXPLORE → Punt {lbl} ({wx:.3f}, {wy:.3f})'
            f'  [tram {self._legs_done + 1}, volta {self._loop_count + 1}]'
        )
        self.get_logger().info(
            f'    Distància: {dist:.2f} m  '
            f'Pose: ({self._x:.3f}, {self._y:.3f}, '
            f'{math.degrees(self._yaw):.1f}°)'
        )
        self.get_logger().info('─' * 50)

        self._avoider.reset()
        self._navigator.set_waypoint(wx, wy)

    def _on_loop_waypoint_reached(self) -> None:
        """Gestiona l'arribada a un waypoint del bucle d'exploració."""
        lbl    = self._loop_labels[self._loop_idx]
        wx, wy = self._loop_wps[self._loop_idx]
        dist   = math.hypot(self._x - wx, self._y - wy)

        self._legs_done += 1

        self.get_logger().info('=' * 50)
        self.get_logger().info(
            f'  ✓ WAYPOINT ASSOLIT: Punt {lbl}'
            f'  [tram {self._legs_done}]'
        )
        self.get_logger().info(
            f'    Posició final : ({self._x:.3f}, {self._y:.3f})'
            f'    Error: {dist:.3f} m'
        )
        self.get_logger().info('=' * 50)

        # Següent waypoint del bucle
        self._loop_idx += 1

        # Si hem completat la seqüència R → U → T sense detectar estació
        if self._loop_idx >= len(self._loop_wps):
            self.get_logger().info(
                '  ✓ SEQÜÈNCIA COMPLETADA: Q → R → U → T'
            )
            self.get_logger().info(
                '  Estació no detectada durant l\'exploració — '
                'finalitzant missió'
            )
            self._mission_done()
            return

        if self._first_station_result is not None:
            return

        self._launch_loop_waypoint()

    # ==========================================================
    # PRIMERA DETECCIÓ → GO_BASE
    # ==========================================================

    def _on_station_found(self, result) -> None:
        """Primera detecció de l'estació; guarda pose i va a P Base."""
        self._first_station_result = result
        self._first_station_map_x  = result.centre_map_x
        self._first_station_map_y  = result.centre_map_y

        self._detect_pose_x   = self._x
        self._detect_pose_y   = self._y
        self._detect_pose_yaw = self._yaw

        dist_station = math.hypot(
            result.centre_map_x - self._x,
            result.centre_map_y - self._y,
        )
        dist_base = math.hypot(
            Config.BASE_X - self._x,
            Config.BASE_Y - self._y,
        )

        self.get_logger().info('★' * 50)
        self.get_logger().info('  ✓ ESTACIÓ DETECTADA (1a vegada)!')
        self.get_logger().info(
            f'    Centre estació (mapa) : '
            f'({result.centre_map_x:.3f}, {result.centre_map_y:.3f})')
        self.get_logger().info(
            f'    Confiança             : {result.confidence} confirmacions')
        self.get_logger().info(
            f'    Distància estació     : {dist_station:.3f} m')
        self.get_logger().info(
            f'    Pilars (mapa):')
        for i, (px, py) in enumerate(result.pillars_map):
            self.get_logger().info(
                f'      Pilar {i+1}: ({px:.3f}, {py:.3f})')
        self.get_logger().info(
            f'    Posició detecció guardada : '
            f'({self._detect_pose_x:.3f}, {self._detect_pose_y:.3f}, '
            f'{math.degrees(self._detect_pose_yaw):.1f}°)')
        self.get_logger().info(
            f'  ▶ GO_BASE → P Base ({Config.BASE_X:.3f}, {Config.BASE_Y:.3f})'
            f'  dist={dist_base:.2f} m')
        self.get_logger().info('★' * 50)

        self._mission_state = self._GO_BASE
        self._avoider.reset()
        self._navigator.set_waypoint(Config.BASE_X, Config.BASE_Y)

    # ==========================================================
    # GO_BASE → RETURN_DETECT
    # ==========================================================

    def _on_base_reached(self) -> None:
        """Arribada a P Base → torna al punt de detecció per re-detectar."""
        dist = math.hypot(self._x - Config.BASE_X, self._y - Config.BASE_Y)
        dist_return = math.hypot(
            self._detect_pose_x - self._x,
            self._detect_pose_y - self._y,
        )

        self.get_logger().info('=' * 50)
        self.get_logger().info('  ✓ P BASE ASSOLIT')
        self.get_logger().info(
            f'    Posició final : ({self._x:.3f}, {self._y:.3f})')
        self.get_logger().info(
            f'    Error posició : {dist:.3f} m')
        self.get_logger().info(
            f'  ▶ RETURN_DETECT → tornant a '
            f'({self._detect_pose_x:.3f}, {self._detect_pose_y:.3f})'
            f'  dist={dist_return:.2f} m')
        self.get_logger().info(
            '    StationDetector RESETEJAT per re-detecció')
        self.get_logger().info('=' * 50)

        self._station_det.reset()
        self._waiting_redetect_at_pose = False

        self._mission_state = self._RETURN_DETECT
        self._avoider.reset()
        self._navigator.set_waypoint(self._detect_pose_x, self._detect_pose_y)

    # ==========================================================
    # RETURN_DETECT → re-detecció
    # ==========================================================

    def _on_station_redetected(self, result) -> None:
        """Re-detecció confirmada → inicia DOCKING."""
        self._redetect_result  = result
        self._dock_station_x   = result.centre_map_x
        self._dock_station_y   = result.centre_map_y

        dx = result.centre_map_x - self._first_station_map_x
        dy = result.centre_map_y - self._first_station_map_y
        drift = math.hypot(dx, dy)

        self.get_logger().info('★' * 50)
        self.get_logger().info('  ✓ ESTACIÓ RE-DETECTADA!')
        self.get_logger().info(
            f'    1a detecció (mapa) : '
            f'({self._first_station_map_x:.3f}, {self._first_station_map_y:.3f})')
        self.get_logger().info(
            f'    Re-detecció (mapa) : '
            f'({result.centre_map_x:.3f}, {result.centre_map_y:.3f})')
        self.get_logger().info(
            f'    Δx={dx:.4f}m  Δy={dy:.4f}m  '
            f'Drift total={drift:.4f}m')

        if drift < 0.05:
            self.get_logger().info(
                '    Precisió: EXCEL·LENT (< 5 cm)')
        elif drift < 0.10:
            self.get_logger().info(
                '    Precisió: BONA (< 10 cm)')
        elif drift < 0.20:
            self.get_logger().info(
                '    Precisió: ACCEPTABLE (< 20 cm)')
        else:
            self.get_logger().warn(
                f'    Precisió: BAIXA ({drift:.3f}m > 20 cm)')

        self.get_logger().info(
            f'    Pilars re-detecció (mapa):')
        for i, (px, py) in enumerate(result.pillars_map):
            self.get_logger().info(
                f'      Pilar {i+1}: ({px:.3f}, {py:.3f})')

        self.get_logger().info(
            f'  ▶ DOCKING — usant re-detecció '
            f'({result.centre_map_x:.3f}, {result.centre_map_y:.3f})')
        self.get_logger().info(
            '    Obstacle avoidance ACTIU durant docking')
        self.get_logger().info('★' * 50)

        self._mission_state = self._DOCKING
        self._avoider.reset()
        self._docker.start_docking(result.centre_map_x, result.centre_map_y)
        self._live_pillars_map       = list(result.pillars_map)
        self._inside_pillars         = False
        self._waiting_redetect_at_pose = False
        self._dock_best_dist         = float('inf')
        self._dock_no_progress_ticks = 0
        self._dock_wall_follow_ticks = 0
        self._dock_recovery_cooldown = 0

    def _on_return_detect_arrived(self) -> None:
        """Arribat al punt de detecció original sense re-detecció confirmada."""
        if self._waiting_redetect_at_pose:
            return

        self._waiting_redetect_at_pose = True
        self._publish_stop()
        self.get_logger().warn(
            '[RETURN_DETECT] Arribat al punt de detecció original; '
            'esperant re-detecció de l\'estació per iniciar docking.'
        )
        self.get_logger().info(
            '  StationDetector continua ACTIU; docking començarà amb '
            'la NOVA posició quan es confirmi.'
        )

    # ==========================================================
    # DOCKING (amb avoidance)
    # ==========================================================

    def _step_docking(self) -> None:
        """DockingController + ObstacleAvoidance + detecció contínua."""
        if self._docker.is_docked():
            self._publish_stop()
            self._mission_done()
            return

        # ---- Detecció contínua de l'estació durant docking ----
        live_result = self._station_det.detect_single(
            self._x, self._y, self._yaw
        )
        if live_result is not None:
            old_x = self._dock_station_x
            old_y = self._dock_station_y
            self._dock_station_x   = live_result.centre_map_x
            self._dock_station_y   = live_result.centre_map_y
            self._live_pillars_map = list(live_result.pillars_map)
            self._docker.update_target(
                live_result.centre_map_x, live_result.centre_map_y
            )
            drift = math.hypot(
                live_result.centre_map_x - old_x,
                live_result.centre_map_y - old_y,
            )
            if drift > 0.02 and self._tick % Config.CONTROL_HZ == 0:
                self.get_logger().info(
                    f'[DOCKING] Estació actualitzada: '
                    f'({live_result.centre_map_x:.3f}, '
                    f'{live_result.centre_map_y:.3f})  '
                    f'drift={drift:.4f}m'
                )

        # ---- Comprovar si som dins els 4 pilars ----
        was_inside = self._inside_pillars
        if len(self._live_pillars_map) == 4:
            self._inside_pillars = self._point_in_quad(
                self._x, self._y, self._live_pillars_map
            )
        else:
            self._inside_pillars = False

        if self._inside_pillars and not was_inside:
            self.get_logger().info(
                '★ [DOCKING] Robot DINS els pilars — '
                'avoidance DESACTIVAT, anant directe al centre!'
            )
        if not self._inside_pillars and was_inside:
            self.get_logger().info(
                '[DOCKING] Robot FORA dels pilars — '
                'avoidance RE-ACTIVAT.'
            )

        # ---- DockingController sempre calcula el seu command ----
        dock_cmd = self._docker.step(self._x, self._y, self._yaw)

        # ---- Si dins els pilars: anar directe al centre ----
        if self._inside_pillars:
            self._publish(dock_cmd.linear_x, dock_cmd.angular_z)
            return

        # ---- Fora dels pilars: avoidance actiu ----
        target_x = self._docker.target_x if self._docker.target_x is not None else self._x
        target_y = self._docker.target_y if self._docker.target_y is not None else self._y
        avoid_cmd, in_avoidance = self._avoider.compute(
            self._x, self._y, self._yaw, target_x, target_y
        )

        dist_to_target = math.hypot(target_x - self._x, target_y - self._y)
        if dist_to_target + Config.DOCK_PROGRESS_EPSILON_M < self._dock_best_dist:
            self._dock_best_dist = dist_to_target
            self._dock_no_progress_ticks = 0
        else:
            self._dock_no_progress_ticks += 1

        if in_avoidance:
            self._dock_wall_follow_ticks += 1
        else:
            self._dock_wall_follow_ticks = 0

        if self._dock_recovery_cooldown > 0:
            self._dock_recovery_cooldown -= 1

        # Recuperació anti-orbita
        should_recover = (
            self._dock_recovery_cooldown == 0 and
            in_avoidance and
            self._dock_wall_follow_ticks >= Config.DOCK_WALL_FOLLOW_RESET_TICKS and
            self._dock_no_progress_ticks >= Config.DOCK_NO_PROGRESS_TICKS
        )
        if should_recover:
            self.get_logger().warn(
                '[DOCKING] Recuperació anti-orbita: WALL_FOLLOW persistent '
                'sense progrés. Reiniciant avoidance i re-alineant docking.'
            )
            self._avoider.reset()
            self._docker.start_docking(target_x, target_y)
            self._dock_best_dist         = dist_to_target
            self._dock_no_progress_ticks = 0
            self._dock_wall_follow_ticks = 0
            self._dock_recovery_cooldown = Config.DOCK_RECOVERY_COOLDOWN_TICKS
            self._publish_stop()
            return

        if in_avoidance:
            self._publish(avoid_cmd.linear_x, avoid_cmd.angular_z)
        else:
            self._publish(dock_cmd.linear_x, dock_cmd.angular_z)

    # ==========================================================
    # HELPERS — POINT IN QUADRILATERAL
    # ==========================================================

    @staticmethod
    def _point_in_quad(
        px: float, py: float,
        vertices: list,
    ) -> bool:
        """Comprova si el punt (px, py) és dins el quadrilàter convex."""
        if len(vertices) != 4:
            return False

        cx = sum(v[0] for v in vertices) / 4
        cy = sum(v[1] for v in vertices) / 4

        sorted_verts = sorted(
            vertices,
            key=lambda v: math.atan2(v[1] - cy, v[0] - cx)
        )

        n = len(sorted_verts)
        for i in range(n):
            x1, y1 = sorted_verts[i]
            x2, y2 = sorted_verts[(i + 1) % n]
            cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
            if cross < 0:
                return False

        return True

    # ==========================================================
    # MISSIÓ COMPLETADA
    # ==========================================================

    def _mission_done(self) -> None:
        """Marca la fi de la missió i atura el robot."""
        self._mission_state = self._DONE
        self._publish_stop()
        self._arrived_time  = self.get_clock().now().nanoseconds

        self.get_logger().info('★' * 50)
        self.get_logger().info('  ✓✓ MISSIÓ COMPLETADA!')
        self.get_logger().info(
            f'  Posició final    : ({self._x:.3f}, {self._y:.3f}, '
            f'{math.degrees(self._yaw):.1f}°)')

        if self._dock_station_x is not None:
            dist = math.hypot(
                self._x - self._dock_station_x,
                self._y - self._dock_station_y,
            )
            self.get_logger().info(
                f'  Centre estació   : ({self._dock_station_x:.3f}, '
                f'{self._dock_station_y:.3f})')
            self.get_logger().info(
                f'  Error vs estació : {dist:.3f} m')

        self.get_logger().info(
            f'  Voltes exploració: {self._loop_count}  '
            f'Trams: {self._legs_done}')

        if self._redetect_result is not None:
            drift = math.hypot(
                self._redetect_result.centre_map_x - self._first_station_map_x,
                self._redetect_result.centre_map_y - self._first_station_map_y,
            )
            self.get_logger().info(
                f'  Drift 1a↔2a detecció: {drift:.4f} m')
            self.get_logger().info(
                f'  Docking basat en    : re-detecció')
        elif self._first_station_result is not None:
            self.get_logger().info(
                f'  Docking basat en    : 1a detecció (fallback)')

        self.get_logger().info(
            f'  Localització     : {"SLAM" if self._slam_active else "Odometria"}')
        self.get_logger().info(
            f'  Esperant {self._map_delay:.1f}s per exportar el mapa final...')
        self.get_logger().info('★' * 50)

    # ==========================================================
    # TELEMETRIA
    # ==========================================================

    def _phase_tag(self) -> str:
        """Retorna 'I' si l'estat actual és de Fase I, sinó 'II'."""
        return 'I' if self._mission_state in self._PHASE1_STATES else 'II'

    def _print_telemetry(self, wp_x: float, wp_y: float) -> None:
        """Telemetria per estats amb waypoint (NAV_P1, EXPLORE, GO_BASE, RETURN_DETECT)."""
        dist      = math.hypot(self._x - wp_x, self._y - wp_y)
        nav_state = self._navigator.get_state().name
        av_state  = self._avoider.get_state().name
        loc_src   = 'SLAM' if self._slam_active else 'ODOM'
        ms        = self._mission_state

        if ms == self._NAV_P1:
            lbl = (f'Punt{self._p1_labels[self._p1_idx]}'
                   if self._p1_idx < self._p1_total else 'Q')
        elif ms == self._EXPLORE:
            lbl = f'Punt{self._loop_labels[self._loop_idx]}'
        elif ms == self._GO_BASE:
            lbl = 'P_Base'
        elif ms == self._RETURN_DETECT:
            lbl = 'DetectPose'
        else:
            lbl = ms

        det_info = ''
        if ms in (self._EXPLORE, self._RETURN_DETECT):
            conf = self._station_det._confirm_count
            det_info = (f'  det={conf}/'
                        f'{self._station_det._confirmed is not None}')

        self.get_logger().info(
            f'[TELEM|P{self._phase_tag()}]'
            f'  pos=({self._x:.3f},{self._y:.3f})'
            f'  yaw={math.degrees(self._yaw):6.1f}°'
            f'  dest={lbl}'
            f'  dist={dist:.3f}m'
            f'  nav={nav_state}'
            f'  avoid={av_state}'
            f'  loc={loc_src}'
            f'  obs≈{self._n_obstacles}'
            f'{det_info}'
        )

    def _print_telemetry_align(self) -> None:
        """Telemetria específica de l'estat ALIGN_Q."""
        err = normalize_angle(Config.ALIGN_Q_YAW_RAD - self._yaw)
        loc_src = 'SLAM' if self._slam_active else 'ODOM'

        self.get_logger().info(
            f'[TELEM|PI]'
            f'  pos=({self._x:.3f},{self._y:.3f})'
            f'  yaw={math.degrees(self._yaw):6.1f}°'
            f'  dest=ALIGN_Q(0°)'
            f'  err={math.degrees(err):+6.2f}°'
            f'  hold={self._align_hold_ticks}/{Config.ALIGN_Q_HOLD_TICKS}'
            f'  loc={loc_src}'
        )

    def _print_telemetry_docking(self) -> None:
        """Telemetria específica de l'estat DOCKING."""
        dist = math.hypot(
            self._x - self._dock_station_x,
            self._y - self._dock_station_y,
        )
        av_state = self._avoider.get_state().name
        loc_src  = 'SLAM' if self._slam_active else 'ODOM'
        inside   = 'YES' if self._inside_pillars else 'no'

        self.get_logger().info(
            f'[TELEM|PII]'
            f'  pos=({self._x:.3f},{self._y:.3f})'
            f'  yaw={math.degrees(self._yaw):6.1f}°'
            f'  dest=DOCKING'
            f'  dist_station={dist:.3f}m'
            f'  dock_state={self._docker.state}'
            f'  avoid={av_state}'
            f'  inside={inside}'
            f'  loc={loc_src}'
        )

    # ==========================================================
    # CSV LOGGER
    # ==========================================================

    def _update_csv(self) -> None:
        """Actualitza l'estat del CSV logger amb phase=I/II segons estat."""
        sx = (self._dock_station_x if self._dock_station_x is not None else
              (self._first_station_map_x if self._first_station_map_x is not None else -1.0))
        sy = (self._dock_station_y if self._dock_station_y is not None else
              (self._first_station_map_y if self._first_station_map_y is not None else -1.0))
        self._csv_logger.update(
            phase=self._phase_tag(),
            robot_x=self._x,
            robot_y=self._y,
            robot_yaw=self._yaw,
            n_obstacles=self._n_obstacles,
            station_x=sx,
            station_y=sy,
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
        """Llegeix la pose corregida pel SLAM Toolbox via TF."""
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
        """Estimació ràpida del nombre d'obstacles propers."""
        in_obs = False
        count  = 0
        for r in scan.ranges:
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
                'No s\'ha rebut cap missatge a /map; no es pot exportar el mapa.'
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

            origin   = grid.info.origin.position
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
        """Retorna (pgm_path, yaml_path) lliures (amb sufix si cal)."""
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

        import time as _time
        stamp = str(int(_time.time()))
        fb = base.parent / f'{base.name}_{stamp}'
        return fb.with_suffix('.pgm'), fb.with_suffix('.yaml')

    # ==========================================================
    # SHUTDOWN
    # ==========================================================

    def shutdown(self) -> None:
        """Atura el robot, desa el mapa i tanca el logger en sortir."""
        self.get_logger().info('Apagant mission_node — parant robot...')
        self._publish_stop()
        self._save_map()
        self._csv_logger.close()


# ============================================================
# ENTRY POINT
# ============================================================
def main(args=None) -> None:
    rclpy.init(args=args)
    node = MissionNode()
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