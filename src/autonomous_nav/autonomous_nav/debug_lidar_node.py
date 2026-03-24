#!/usr/bin/env python3
"""
debug_lidar_node.py — Node de debug per visualitzar sectors LiDAR en temps real.

NO mou el robot. Només llegeix el LiDAR i mostra les distàncies mínimes
de cada sector, els nivells d'alerta i els punts crus del scan.
Útil per calibrar els llindars DANGER/WARNING/SAFE abans de provar
la navegació o obstacle avoidance.

Ús:
    ros2 run autonomous_nav debug_lidar_node

Topics:
  Subscriu:  /scan  (LaserScan, BEST_EFFORT)
"""

# ============================================================
# CONFIGURACIÓ
# ============================================================
class Config:
    LOG_HZ = 2   # Hz — freqüència de log (no cal anar molt ràpid)


# ============================================================
# IMPORTS
# ============================================================
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import LaserScan
from autonomous_nav.obstacle_avoidance import ObstacleAvoidance, AlertLevel


# ============================================================
# NODE PRINCIPAL
# ============================================================
class DebugLidarNode(Node):
    """
    Node passiu que mostra l'estat dels sectors LiDAR sense moure el robot.

    Mostra:
      - Distància mínima de cada sector (FRONT, FRONT_LEFT, FRONT_RIGHT, LEFT, RIGHT)
      - Nivell d'alerta (SAFE / WARNING / DANGER)
      - Nombre de punts vàlids per sector
      - Distàncies crues a 0°, 90°, 180°, 270° (front, esquerra, darrera, dreta)
    """

    def __init__(self) -> None:
        super().__init__('debug_lidar_node')

        qos_b = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=10)
        self.create_subscription(LaserScan, '/scan', self._scan_cb, qos_b)

        self._avoider = ObstacleAvoidance(logger=self.get_logger())
        self._scan_data = None

        self._timer = self.create_timer(1.0 / Config.LOG_HZ, self._log_callback)

        self.get_logger().info('=' * 60)
        self.get_logger().info('  DEBUG LIDAR NODE (no mou el robot)')
        self.get_logger().info('=' * 60)
        self.get_logger().info('Esperant primer /scan...')

    def _scan_cb(self, msg: LaserScan) -> None:
        self._avoider.update_scan(msg)
        self._scan_data = msg

    def _log_callback(self) -> None:
        if self._scan_data is None:
            return

        sectors = self._avoider.get_sectors()
        msg = self._scan_data

        # Sector summary
        lines = ['┌─ SECTORS ─────────────────────────────────────┐']
        for name in ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'LEFT', 'RIGHT']:
            s = sectors[name]
            alert_icon = {
                AlertLevel.SAFE: '🟢',
                AlertLevel.WARNING: '🟡',
                AlertLevel.DANGER: '🔴',
            }.get(s.alert, '?')
            min_d = f'{s.min_dist:.3f}' if not math.isinf(s.min_dist) else ' inf '
            lines.append(
                f'│  {alert_icon} {name:15s}  min={min_d}m  '
                f'mean={s.mean_dist:.3f}m  pts={s.n_points:3d}  │'
            )
        lines.append('└───────────────────────────────────────────────┘')

        # Raw distances at cardinal directions
        n = len(msg.ranges)
        if n > 0:
            # Index 0 = front (0°), n//4 = 90° (left), n//2 = 180° (back), 3*n//4 = 270° (right)
            def safe_range(idx):
                r = msg.ranges[idx % n]
                if math.isnan(r) or math.isinf(r):
                    return 'inf'
                return f'{r:.3f}'

            lines.append(
                f'  Raw: front={safe_range(0)}m  '
                f'left={safe_range(n // 4)}m  '
                f'back={safe_range(n // 2)}m  '
                f'right={safe_range(3 * n // 4)}m'
            )

        for line in lines:
            self.get_logger().info(line)

    def shutdown(self) -> None:
        self.get_logger().info('Apagant debug_lidar_node')


# ============================================================
# ENTRY POINT
# ============================================================
def main(args=None) -> None:
    rclpy.init(args=args)
    node = DebugLidarNode()
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
