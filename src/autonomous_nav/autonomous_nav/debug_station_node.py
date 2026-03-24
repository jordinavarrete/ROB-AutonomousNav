#!/usr/bin/env python3
"""
debug_station_node.py — Node de debug per testejar la detecció de l'estació de càrrega.

NO mou el robot. Llegeix el LiDAR i intenta detectar els 4 pilars
de l'estació de càrrega. Mostra informació detallada sobre:
  - Clusters detectats
  - Candidats a pilar
  - Validació geomètrica del quadrat
  - Confirmació progressiva

Ús:
    ros2 run autonomous_nav debug_station_node

Topics:
  Subscriu:  /scan  (LaserScan, BEST_EFFORT)
             /odom  (Odometry,  RELIABLE)
"""

# ============================================================
# CONFIGURACIÓ
# ============================================================
class Config:
    LOG_HZ          = 2    # Hz
    MAP_FRAME       = 'map'
    BASE_FRAME      = 'base_footprint'


# ============================================================
# IMPORTS
# ============================================================
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from autonomous_nav.station_detector import StationDetector, dist2d


# ============================================================
# NODE PRINCIPAL
# ============================================================
class DebugStationNode(Node):
    """
    Node passiu per debuggar la detecció de l'estació.

    Mostra:
      - Nombre de clusters detectats
      - Nombre de candidats a pilar (post-filtratge)
      - Si s'ha trobat un quadruplet vàlid
      - Progrés de confirmació (N/5)
      - Posició de l'estació un cop confirmada
    """

    def __init__(self) -> None:
        super().__init__('debug_station_node')

        qos_r = QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10)
        qos_b = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)

        self.create_subscription(LaserScan, '/scan', self._scan_cb, qos_b)
        self.create_subscription(Odometry,  '/odom', self._odom_cb, qos_r)

        self._tf_buffer   = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._detector = StationDetector(logger=self.get_logger())

        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self._scan_ready = False
        self._odom_ready = False
        self._station_found = False

        self._timer = self.create_timer(1.0 / Config.LOG_HZ, self._detect_and_log)

        self.get_logger().info('=' * 60)
        self.get_logger().info('  DEBUG STATION DETECTOR (no mou el robot)')
        self.get_logger().info('=' * 60)
        self.get_logger().info('Esperant /scan i /odom...')

    # ==========================================================
    # CALLBACKS
    # ==========================================================

    def _scan_cb(self, msg: LaserScan) -> None:
        self._detector.update_scan(msg)
        self._scan_ready = True

    def _odom_cb(self, msg: Odometry) -> None:
        odom_x   = msg.pose.pose.position.x
        odom_y   = msg.pose.pose.position.y
        odom_yaw = self._quat_to_yaw(msg.pose.pose.orientation)

        self._odom_ready = True

        slam_x, slam_y, slam_yaw = self._try_slam_pose()
        if slam_x is not None:
            self._x, self._y, self._yaw = slam_x, slam_y, slam_yaw
        else:
            self._x, self._y, self._yaw = odom_x, odom_y, odom_yaw

    # ==========================================================
    # DETECT & LOG
    # ==========================================================

    def _detect_and_log(self) -> None:
        if not self._scan_ready or not self._odom_ready:
            return

        if self._station_found:
            return

        # Executar pipeline manualment per obtenir info intermèdia
        clusters   = self._detector._cluster_scan(self._detector._scan_points)
        candidates = self._detector._filter_pillar_candidates(clusters)
        quad       = self._detector._find_square_quad(candidates)

        # Executar detect() habitual per acumular confirmacions
        result = self._detector.detect(self._x, self._y, self._yaw)

        # Log
        n_clusters   = len(clusters)
        n_candidates = len(candidates)
        has_quad     = quad is not None
        confirm      = self._detector._confirm_count

        lines = ['┌─ STATION DETECTOR ────────────────────────────┐']
        lines.append(f'│  Robot pos: ({self._x:.2f}, {self._y:.2f})                    │')
        lines.append(f'│  Clusters:       {n_clusters:3d}                            │')
        lines.append(f'│  Candidats:      {n_candidates:3d}                            │')
        lines.append(f'│  Quadrat vàlid:  {"SÍ ✓" if has_quad else "NO ✗":6s}                         │')
        lines.append(f'│  Confirmació:    {confirm}/5                            │')

        if has_quad:
            for i, c in enumerate(quad):
                lines.append(
                    f'│    Pilar {i+1}: ({c.centroid_x:.3f}, {c.centroid_y:.3f}) '
                    f'range={c.mean_range:.2f}m pts={c.n_points}  │'
                )
            # Centre
            cx = sum(c.centroid_x for c in quad) / 4
            cy = sum(c.centroid_y for c in quad) / 4
            lines.append(f'│    Centre robot: ({cx:.3f}, {cy:.3f})              │')

        lines.append('└───────────────────────────────────────────────┘')

        for line in lines:
            self.get_logger().info(line)

        if result is not None:
            self._station_found = True
            self.get_logger().info('=' * 60)
            self.get_logger().info('  ✓ ESTACIÓ CONFIRMADA!')
            self.get_logger().info(
                f'  Posició mapa: ({result.centre_map_x:.3f}, {result.centre_map_y:.3f})'
            )
            self.get_logger().info(
                f'  Posició robot: ({result.centre_robot_x:.3f}, {result.centre_robot_y:.3f})'
            )
            for i, (px, py) in enumerate(result.pillars_map):
                self.get_logger().info(f'  Pilar {i+1} mapa: ({px:.3f}, {py:.3f})')
            self.get_logger().info('=' * 60)

    # ==========================================================
    # HELPERS
    # ==========================================================

    def _try_slam_pose(self):
        try:
            tf = self._tf_buffer.lookup_transform(
                Config.MAP_FRAME, Config.BASE_FRAME, rclpy.time.Time(),
            )
            x   = tf.transform.translation.x
            y   = tf.transform.translation.y
            yaw = self._quat_to_yaw(tf.transform.rotation)
            return x, y, yaw
        except (LookupException, ConnectivityException, ExtrapolationException):
            return None, None, None

    @staticmethod
    def _quat_to_yaw(q) -> float:
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def shutdown(self) -> None:
        self.get_logger().info('Apagant debug_station_node')


# ============================================================
# ENTRY POINT
# ============================================================
def main(args=None) -> None:
    rclpy.init(args=args)
    node = DebugStationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt rebut')
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
