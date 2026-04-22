#!/usr/bin/env python3
"""
station_detector.py — Charging station detection via LiDAR pillar clustering.

The charging station consists of 4 cylindrical pillars (~5 cm diameter)
arranged at the corners of a 40 cm × 40 cm square.

Detection pipeline:
    1. Ingest raw /scan ranges → convert to (x, y) Cartesian points
    2. Cluster adjacent scan points within CLUSTER_DIST of each other
    3. Filter clusters by angular width (consistent with ~5 cm cylinder)
    4. Estimate centroid (x, y) of each candidate pillar in robot frame
    5. From all candidate pillars, find a group of 4 whose pairwise
       distances match a 40×40 cm square (4 sides + 2 diagonals)
    6. Require N_CONFIRM consecutive scans confirming the same station
    7. Transform station centre from robot frame to map frame

Output:
    StationResult with station centre (map frame) + 4 pillar positions
    Returns None until N_CONFIRM threshold is met.

This class has NO ROS2 Node inheritance — pure logic, testable standalone.
"""

# ============================================================
# CONFIGURATION — adjust these values for lab testing
# ============================================================
class Config:
    # Clustering
    CLUSTER_DIST        = 0.08   # m — max gap between adjacent points in cluster
    MIN_CLUSTER_POINTS  = 1      # minimum scan points to form a candidate cluster
    MAX_CLUSTER_POINTS  = 12     # discard clusters larger than this (walls, not pillars)

    # Pillar geometry
    PILLAR_DIAMETER     = 0.05   # m — nominal pillar diameter
    PILLAR_WIDTH_FACTOR = 3.0    # expected_width = factor * arctan(D/2 / dist)
                                 # generous multiplier for noisy real LiDAR

    # Station geometry (40 × 40 cm square)
    STATION_SIDE        = 0.40   # m — expected side length
    STATION_DIAGONAL    = 0.566  # m — expected diagonal (√2 × 0.40)
    SIDE_TOL            = 0.07   # m — tolerance on side length
    DIAG_TOL            = 0.09   # m — tolerance on diagonal length

    # Confirmation
    N_CONFIRM           = 5      # consecutive detections required before accepting
    MAX_CENTRE_DRIFT    = 0.10   # m — max shift between confirmations (same station)

    # Range limits for detection (ignore very close / very far returns)
    MIN_DETECT_RANGE    = 0.15   # m
    MAX_DETECT_RANGE    = 3.50   # m


# ============================================================
# IMPORTS
# ============================================================
import math
import itertools
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ============================================================
# DATA STRUCTURES
# ============================================================
@dataclass
class Cluster:
    """
    A group of adjacent LiDAR scan points that may represent a pillar.

    Attributes:
        points:       List of (angle_rad, range_m) raw scan points.
        centroid_x:   X of cluster centroid in robot frame [m].
        centroid_y:   Y of cluster centroid in robot frame [m].
        mean_range:   Mean range of all points in cluster [m].
        angular_width: Total angular span of the cluster [rad].
        n_points:     Number of scan points.
    """
    points:        List[Tuple[float, float]] = field(default_factory=list)
    centroid_x:    float = 0.0
    centroid_y:    float = 0.0
    mean_range:    float = 0.0
    angular_width: float = 0.0
    n_points:      int   = 0

    def compute_centroid(self) -> None:
        """Compute Cartesian centroid and statistics from raw (angle, range) points."""
        if not self.points:
            return
        xs = [r * math.cos(a) for a, r in self.points]
        ys = [r * math.sin(a) for a, r in self.points]
        self.centroid_x    = sum(xs) / len(xs)
        self.centroid_y    = sum(ys) / len(ys)
        self.mean_range    = sum(r for _, r in self.points) / len(self.points)
        self.angular_width = abs(self.points[-1][0] - self.points[0][0])
        self.n_points      = len(self.points)

    def __repr__(self) -> str:
        return (f'Cluster(n={self.n_points}, '
                f'cx={self.centroid_x:.3f}, cy={self.centroid_y:.3f}, '
                f'range={self.mean_range:.3f}m, '
                f'width={math.degrees(self.angular_width):.1f}°)')


@dataclass
class StationResult:
    """
    Detected charging station with confirmed position.

    Attributes:
        centre_robot_x/y: Station centre in robot frame [m].
        centre_map_x/y:   Station centre in map frame [m].
        pillars_robot:    List of 4 (x, y) pillar positions in robot frame.
        pillars_map:      List of 4 (x, y) pillar positions in map frame.
        confidence:       Number of consecutive confirmations.
    """
    centre_robot_x: float = 0.0
    centre_robot_y: float = 0.0
    centre_map_x:   float = 0.0
    centre_map_y:   float = 0.0
    pillars_robot:  List[Tuple[float, float]] = field(default_factory=list)
    pillars_map:    List[Tuple[float, float]] = field(default_factory=list)
    confidence:     int = 0

    def __repr__(self) -> str:
        return (f'StationResult(map=({self.centre_map_x:.3f},{self.centre_map_y:.3f}), '
                f'robot=({self.centre_robot_x:.3f},{self.centre_robot_y:.3f}), '
                f'conf={self.confidence})')


# ============================================================
# UTILITY
# ============================================================
def dist2d(ax: float, ay: float, bx: float, by: float) -> float:
    """Euclidean distance between two 2-D points."""
    return math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)


def robot_to_map(
    rx: float, ry: float,
    robot_pose_x: float, robot_pose_y: float, robot_pose_yaw: float,
) -> Tuple[float, float]:
    """
    Transform a point from robot frame to map frame.

    Args:
        rx, ry:           Point in robot frame [m].
        robot_pose_x/y:   Robot position in map frame [m].
        robot_pose_yaw:   Robot heading in map frame [rad].

    Returns:
        (map_x, map_y) in map frame.
    """
    cos_y = math.cos(robot_pose_yaw)
    sin_y = math.sin(robot_pose_yaw)
    map_x = robot_pose_x + rx * cos_y - ry * sin_y
    map_y = robot_pose_y + rx * sin_y + ry * cos_y
    return map_x, map_y


# ============================================================
# MAIN CLASS
# ============================================================
class StationDetector:
    """
    Detects the 4-pillar charging station from LiDAR scans.

    Usage (from mission_node.py):

        detector = StationDetector(logger=self.get_logger())

        # in /scan callback:
        detector.update_scan(scan_msg)

        # in exploration control tick:
        result = detector.detect(robot_x, robot_y, robot_yaw)
        if result is not None:
            # station confirmed — result.centre_map_x/y is the target
    """

    def __init__(self, logger=None) -> None:
        """
        Initialise the station detector.

        Args:
            logger: ROS2 logger or None (falls back to print).
        """
        self._log = logger

        # Raw scan snapshot: list of (angle_rad, range_m) valid points
        self._scan_points: List[Tuple[float, float]] = []
        self._scan_ready   = False

        # Confirmation buffer
        self._confirm_count  = 0
        self._last_centre_x  = None
        self._last_centre_y  = None

        # Final confirmed result (set once N_CONFIRM reached)
        self._confirmed: Optional[StationResult] = None

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def update_scan(self, scan_msg) -> None:
        """
        Ingest a new LaserScan message.

        Filters NaN/Inf and out-of-range values.
        Stores valid (angle_rad, range_m) pairs for the next detect() call.

        Args:
            scan_msg: sensor_msgs/LaserScan message.
        """
        valid = []
        for i, r in enumerate(scan_msg.ranges):
            if math.isnan(r) or math.isinf(r):
                continue
            if r < Config.MIN_DETECT_RANGE or r > Config.MAX_DETECT_RANGE:
                continue
            angle = scan_msg.angle_min + i * scan_msg.angle_increment
            valid.append((angle, r))

        self._scan_points = valid
        self._scan_ready  = True

    def detect(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
    ) -> Optional[StationResult]:
        """
        Run one detection cycle on the latest scan.

        Returns a StationResult once N_CONFIRM consecutive cycles agree
        on the same station position, or None otherwise.

        Args:
            robot_x/y:   Robot position in map frame [m].
            robot_yaw:   Robot heading in map frame [rad].

        Returns:
            StationResult or None.
        """
        if not self._scan_ready:
            return None

        # Already confirmed in a previous cycle — return cached result
        if self._confirmed is not None:
            return self._confirmed

        # --- Pipeline ---
        clusters   = self._cluster_scan(self._scan_points)
        candidates = self._filter_pillar_candidates(clusters)
        quad       = self._find_square_quad(candidates)

        if quad is None:
            self._confirm_count = 0
            return None

        # Compute centre in robot frame
        cx_r = sum(p.centroid_x for p in quad) / 4
        cy_r = sum(p.centroid_y for p in quad) / 4

        # Check that this detection is consistent with the previous one
        if self._last_centre_x is not None:
            drift = dist2d(cx_r, cy_r, self._last_centre_x, self._last_centre_y)
            if drift > Config.MAX_CENTRE_DRIFT:
                # Different station candidate — reset counter
                self._log_warn(
                    f'[DETECT] Candidat ha canviat {drift:.3f}m — resettejant confirmació'
                )
                self._confirm_count = 0

        self._last_centre_x = cx_r
        self._last_centre_y = cy_r
        self._confirm_count += 1

        self._log_info(
            f'[DETECT] Candidat [{self._confirm_count}/{Config.N_CONFIRM}] '
            f'centre=({cx_r:.3f},{cy_r:.3f})'
        )

        if self._confirm_count >= Config.N_CONFIRM:
            result = self._build_result(quad, cx_r, cy_r, robot_x, robot_y, robot_yaw)
            self._confirmed = result
            self._log_info(
                f'[DETECT] ✓ ESTACIÓ CONFIRMADA! map=({result.centre_map_x:.3f},'
                f'{result.centre_map_y:.3f})'
            )
            return result

        return None

    def is_confirmed(self) -> bool:
        """Return True if the station has been confirmed."""
        return self._confirmed is not None

    def get_confirmed(self) -> Optional[StationResult]:
        """Return the confirmed StationResult, or None if not yet found."""
        return self._confirmed

    def detect_single(
        self,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
    ) -> Optional[StationResult]:
        """
        Run one detection cycle WITHOUT confirmation.

        Returns a StationResult if 4 pillars forming a valid square are
        found in the current scan, regardless of confirmation history.
        Used during DOCKING for live station tracking.

        Args:
            robot_x/y:   Robot position in map frame [m].
            robot_yaw:   Robot heading in map frame [rad].

        Returns:
            StationResult or None.
        """
        if not self._scan_ready:
            return None

        clusters   = self._cluster_scan(self._scan_points)
        candidates = self._filter_pillar_candidates(clusters)
        quad       = self._find_square_quad(candidates)

        if quad is None:
            return None

        cx_r = sum(p.centroid_x for p in quad) / 4
        cy_r = sum(p.centroid_y for p in quad) / 4

        return self._build_result(quad, cx_r, cy_r, robot_x, robot_y, robot_yaw)

    def reset(self) -> None:
        """Reset detector state (use if re-scanning after false positive)."""
        self._confirm_count = 0
        self._last_centre_x = None
        self._last_centre_y = None
        self._confirmed     = None
        self._log_info('[DETECT] Reset')

    # ----------------------------------------------------------
    # Step 1 — Clustering
    # ----------------------------------------------------------

    def _cluster_scan(
        self, points: List[Tuple[float, float]]
    ) -> List[Cluster]:
        """
        Group adjacent scan points into clusters.

        Two consecutive scan points belong to the same cluster if the
        Euclidean distance between their Cartesian positions is ≤ CLUSTER_DIST.

        Args:
            points: List of (angle_rad, range_m) sorted by angle (as from /scan).

        Returns:
            List of Cluster objects with computed centroids.
        """
        if not points:
            return []

        clusters: List[Cluster] = []
        current   = Cluster(points=[points[0]])

        for i in range(1, len(points)):
            a_prev, r_prev = points[i - 1]
            a_curr, r_curr = points[i]

            # Cartesian positions
            x_prev = r_prev * math.cos(a_prev)
            y_prev = r_prev * math.sin(a_prev)
            x_curr = r_curr * math.cos(a_curr)
            y_curr = r_curr * math.sin(a_curr)

            gap = dist2d(x_prev, y_prev, x_curr, y_curr)

            if gap <= Config.CLUSTER_DIST:
                current.points.append(points[i])
            else:
                current.compute_centroid()
                clusters.append(current)
                current = Cluster(points=[points[i]])

        # Don't forget the last cluster
        current.compute_centroid()
        clusters.append(current)

        return clusters

    # ----------------------------------------------------------
    # Step 2 — Pillar candidate filtering
    # ----------------------------------------------------------

    def _filter_pillar_candidates(
        self, clusters: List[Cluster]
    ) -> List[Cluster]:
        """
        Keep only clusters whose size is consistent with a ~5 cm cylinder.

        Expected angular width at distance d:
            expected = PILLAR_WIDTH_FACTOR × arctan(PILLAR_DIAMETER / (2 × d))

        A cluster passes if:
          - MIN_CLUSTER_POINTS ≤ n_points ≤ MAX_CLUSTER_POINTS
          - angular_width ≤ expected_max_width  (not too wide → not a wall)

        Args:
            clusters: All clusters from _cluster_scan().

        Returns:
            Filtered list of pillar candidate clusters.
        """
        candidates = []
        for c in clusters:
            if c.n_points < Config.MIN_CLUSTER_POINTS:
                continue
            if c.n_points > Config.MAX_CLUSTER_POINTS:
                continue

            # Angular width consistent with a 5 cm cylinder?
            if c.mean_range > 0:
                expected_half = math.atan2(
                    Config.PILLAR_DIAMETER / 2, c.mean_range
                )
                max_width = Config.PILLAR_WIDTH_FACTOR * 2 * expected_half
                if c.angular_width > max_width:
                    continue   # too wide — probably a wall segment

            candidates.append(c)

        return candidates

    # ----------------------------------------------------------
    # Step 3 — Square quadruplet search
    # ----------------------------------------------------------

    def _find_square_quad(
        self, candidates: List[Cluster]
    ) -> Optional[List[Cluster]]:
        """
        Find a group of 4 candidate pillars whose pairwise distances
        match a STATION_SIDE × STATION_SIDE square.

        Validation criteria for 4 points A, B, C, D:
          - Exactly 4 pairwise distances ≈ STATION_SIDE  (the 4 sides)
          - Exactly 2 pairwise distances ≈ STATION_DIAGONAL  (the 2 diagonals)

        Args:
            candidates: Filtered pillar candidates.

        Returns:
            List of 4 Cluster objects forming a valid square, or None.
        """
        if len(candidates) < 4:
            return None

        s  = Config.STATION_SIDE
        d  = Config.STATION_DIAGONAL
        st = Config.SIDE_TOL
        dt = Config.DIAG_TOL

        # Try all combinations of 4 clusters
        for quad in itertools.combinations(candidates, 4):
            pts = [(c.centroid_x, c.centroid_y) for c in quad]

            # Compute all 6 pairwise distances
            dists = [
                dist2d(pts[i][0], pts[i][1], pts[j][0], pts[j][1])
                for i, j in itertools.combinations(range(4), 2)
            ]
            dists.sort()

            # For a square: 4 sides (shorter) + 2 diagonals (longer)
            # After sorting: dists[0..3] should be sides, dists[4..5] diagonals
            sides     = dists[:4]
            diagonals = dists[4:]

            sides_ok = all(abs(d_s - s) <= st for d_s in sides)
            diags_ok = all(abs(d_d - d) <= dt for d_d in diagonals)

            if sides_ok and diags_ok:
                return list(quad)

        return None

    # ----------------------------------------------------------
    # Step 4 — Build final result
    # ----------------------------------------------------------

    def _build_result(
        self,
        quad: List[Cluster],
        cx_robot: float,
        cy_robot: float,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
    ) -> StationResult:
        """
        Construct a StationResult from a confirmed quad.

        Transforms pillar positions and station centre from robot frame
        to map frame using the current robot pose.

        Args:
            quad:           4 confirmed pillar clusters (robot frame).
            cx_robot:       Station centre X in robot frame [m].
            cy_robot:       Station centre Y in robot frame [m].
            robot_x/y/yaw:  Robot pose in map frame.

        Returns:
            Fully populated StationResult.
        """
        pillars_robot = [(c.centroid_x, c.centroid_y) for c in quad]
        pillars_map   = [
            robot_to_map(px, py, robot_x, robot_y, robot_yaw)
            for px, py in pillars_robot
        ]

        cx_map, cy_map = robot_to_map(cx_robot, cy_robot, robot_x, robot_y, robot_yaw)

        return StationResult(
            centre_robot_x = cx_robot,
            centre_robot_y = cy_robot,
            centre_map_x   = cx_map,
            centre_map_y   = cy_map,
            pillars_robot  = pillars_robot,
            pillars_map    = pillars_map,
            confidence     = self._confirm_count,
        )

    # ----------------------------------------------------------
    # Logging helpers
    # ----------------------------------------------------------

    def _log_info(self, msg: str) -> None:
        if self._log:
            self._log.info(msg)
        else:
            print(f'[INFO] {msg}')

    def _log_warn(self, msg: str) -> None:
        if self._log:
            self._log.warning(msg)
        else:
            print(f'[WARN] {msg}')


# ============================================================
# STANDALONE TEST (no ROS2 required)
# ============================================================
if __name__ == '__main__':
    import math

    # --------------------------------------------------------
    # Mock LaserScan builder
    # --------------------------------------------------------
    class FakeScan:
        """
        Generates a synthetic 360° LaserScan with planted cylindrical pillars.

        Args:
            pillars:     List of (x, y) pillar positions in robot frame.
            diameter:    Pillar diameter [m].
            n_rays:      Total number of rays (default 360).
            bg_range:    Background range for empty rays [m].
        """
        def __init__(
            self,
            pillars: List[Tuple[float, float]],
            diameter: float = 0.05,
            n_rays: int = 360,
            bg_range: float = 3.0,
        ) -> None:
            self.range_min       = 0.12
            self.range_max       = 3.50
            self.angle_min       = -math.pi
            self.angle_increment = 2 * math.pi / n_rays
            self.ranges          = [bg_range] * n_rays

            for px, py in pillars:
                dist_to_pillar = math.sqrt(px**2 + py**2)
                bearing        = math.atan2(py, px)

                # Angular half-width of pillar at this distance
                half_width = math.atan2(diameter / 2, dist_to_pillar)

                for i in range(n_rays):
                    ray_angle = self.angle_min + i * self.angle_increment
                    delta     = ray_angle - bearing
                    # Normalise delta to [-π, π]
                    while delta >  math.pi: delta -= 2 * math.pi
                    while delta < -math.pi: delta += 2 * math.pi

                    if abs(delta) <= half_width:
                        # Range = distance to pillar surface
                        self.ranges[i] = dist_to_pillar - diameter / 2

    # --------------------------------------------------------
    # Helper: run N_CONFIRM cycles
    # --------------------------------------------------------
    def run_detection(
        pillars_robot: List[Tuple[float, float]],
        robot_pose: Tuple[float, float, float],
        label: str,
    ) -> None:
        detector = StationDetector()
        scan     = FakeScan(pillars_robot)

        print(f'\n=== {label} ===')
        print(f'    Pillars (robot frame): {[(round(x,3),round(y,3)) for x,y in pillars_robot]}')

        result = None
        for i in range(Config.N_CONFIRM + 2):
            detector.update_scan(scan)
            result = detector.detect(*robot_pose)
            if result:
                break

        if result:
            print(f'    ✓ Confirmed after {result.confidence} scans')
            print(f'    Centre robot : ({result.centre_robot_x:.3f}, {result.centre_robot_y:.3f})')
            print(f'    Centre map   : ({result.centre_map_x:.3f}, {result.centre_map_y:.3f})')
            sides = []
            for i in range(4):
                for j in range(i+1, 4):
                    px, py = result.pillars_robot[i]
                    qx, qy = result.pillars_robot[j]
                    sides.append(dist2d(px, py, qx, qy))
            sides.sort()
            print(f'    Pairwise distances: sides={[round(s,3) for s in sides[:4]]}'
                  f'  diags={[round(s,3) for s in sides[4:]]}')
        else:
            print(f'    ✗ Not confirmed after {Config.N_CONFIRM + 2} scans')

    # --------------------------------------------------------
    # Test 1 — Perfect square directly in front at 1.0 m
    # --------------------------------------------------------
    half = Config.STATION_SIDE / 2   # 0.20 m
    pillars_front = [
        (1.0 - half,  half),
        (1.0 + half,  half),
        (1.0 + half, -half),
        (1.0 - half, -half),
    ]
    run_detection(pillars_front, (0.0, 0.0, 0.0), 'Square at 1.0 m ahead, robot at origin')

    # --------------------------------------------------------
    # Test 2 — Station at 2.0 m, robot rotated 45°
    # --------------------------------------------------------
    cx, cy = 2.0, 1.0
    pillars_offset = [
        (cx - half, cy + half),
        (cx + half, cy + half),
        (cx + half, cy - half),
        (cx - half, cy - half),
    ]
    run_detection(
        pillars_offset,
        (1.0, 0.5, math.pi / 6),
        'Station offset (2.0,1.0) robot at (1.0,0.5,30°)'
    )

    # --------------------------------------------------------
    # Test 3 — Only 3 pillars visible (should fail)
    # --------------------------------------------------------
    print('\n=== Only 3 pillars visible (expect failure) ===')
    detector = StationDetector()
    scan = FakeScan(pillars_front[:3])   # only 3 pillars
    for _ in range(Config.N_CONFIRM + 2):
        detector.update_scan(scan)
        r = detector.detect(0.0, 0.0, 0.0)
        if r:
            print('    ✗ Unexpected detection!')
            break
    else:
        print('    ✓ Correctly rejected (only 3 pillars)')

    # --------------------------------------------------------
    # Test 4 — Geometry validation: not a square (rectangle)
    # --------------------------------------------------------
    print('\n=== Rectangle (not a square) — expect failure ===')
    pillars_rect = [
        (0.80, 0.20), (0.80, -0.20),
        (1.20, 0.20), (1.20, -0.20),   # 40 cm × 40 cm → actually a square!
    ]
    # Make it a non-square rectangle: different side lengths
    pillars_rect_bad = [
        (0.80,  0.30),
        (0.80, -0.30),   # 60 cm side
        (1.20,  0.20),
        (1.20, -0.20),   # 40 cm side → not a square
    ]
    detector2 = StationDetector()
    scan2 = FakeScan(pillars_rect_bad)
    detected = False
    for _ in range(Config.N_CONFIRM + 2):
        detector2.update_scan(scan2)
        r = detector2.detect(0.0, 0.0, 0.0)
        if r:
            detected = True
            break
    print(f'    {"✗ Unexpected detection!" if detected else "✓ Correctly rejected (non-square rectangle)"}')