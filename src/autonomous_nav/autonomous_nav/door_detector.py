#!/usr/bin/env python3
"""
door_detector.py — Door detection via LiDAR gap analysis.

Detects a doorway of ~0.8 m width in the environment by identifying
two wall-edge clusters (door jambs) separated by the expected gap,
with open space (no return) between them.

Detection pipeline:
    1. Ingest raw /scan ranges → filter valid points in detection range
    2. Cluster adjacent scan points (wall segments / jamb edges)
    3. For each consecutive pair of clusters, analyse the gap between them:
         a. Gap angular width corresponds to expected door opening
         b. Both edge clusters are at similar range (coplanar wall)
         c. The open sector between them has no returns (it is truly open)
         d. The gap midpoint bearing is within ±FRONT_FOV of the robot front
    4. Compute the midpoint position (centre of doorway) in robot frame
    5. Require N_CONFIRM consecutive scans agreeing on the same gap centre
    6. Transform door centre from robot frame to map frame

Key geometric idea:
    At distance d, a gap of width W subtends an angle:
        α = 2 * arctan(W / (2 * d))   [rad]
    We measure α from the scan and back-calculate d, then compute
    the Cartesian position of the door centre.

Output:
    DoorResult with door centre + both jamb endpoints (robot & map frame).
    Returns None until N_CONFIRM threshold is met.

This class has NO ROS2 Node inheritance — pure logic, testable standalone.
"""

# ============================================================
# IMPORTS (needed by Config constants)
# ============================================================
import math

# ============================================================
# CONFIGURATION — adjust these values for lab testing
# ============================================================
class Config:
    # ---- Door geometry ----
    DOOR_WIDTH          = 0.80   # m — nominal door opening width
    DOOR_WIDTH_TOL      = 0.15   # m — ±tolerance (accepts 0.65–0.95 m gaps)

    # ---- Jamb (edge) cluster geometry ----
    # Door jambs appear as short wall-edge clusters just at the gap boundary.
    # They must be narrow (not a long wall) and at similar range.
    CLUSTER_DIST        = 0.06   # m — max Cartesian gap inside a cluster
    MIN_JAMB_POINTS     = 2      # minimum scan points per jamb cluster
    MAX_JAMB_POINTS     = 40     # maximum (beyond this → long wall, not a jamb)
    MAX_JAMB_DEPTH      = 0.25   # m — max depth (range spread) inside a jamb cluster
    MAX_RANGE_DIFF      = 0.35   # m — max range difference between the two jambs
                                  #     (they must belong to the same wall plane)

    # ---- Gap validation ----
    # The open sector between jambs must have NO valid LiDAR returns.
    GAP_EMPTY_FRACTION  = 0.75   # fraction of gap rays that must be empty / far
    GAP_EMPTY_MIN_RANGE = 1.20   # m — a ray counts as "open" if range > this value
                                  #     or is NaN/Inf

    # ---- Field of view ----
    # Only detect doors within ±FRONT_FOV of the robot's forward direction.
    # Set to math.pi to detect doors in all directions (360°).
    FRONT_FOV           = math.pi / 2   # rad — ±90° (full front hemisphere)

    # ---- Detection range ----
    MIN_DETECT_RANGE    = 0.20   # m — ignore very close returns
    MAX_DETECT_RANGE    = 4.00   # m — ignore very far returns

    # ---- Confirmation ----
    N_CONFIRM           = 4      # consecutive detections required
    MAX_CENTRE_DRIFT    = 0.20   # m — max shift between confirmations

    # ---- How far ahead to project the door centre ----
    # The door centre in robot frame is placed at the range of the jambs,
    # projected along the gap midpoint bearing.
    # No extra parameter needed — it is derived from the scan geometry.


# ============================================================
# IMPORTS
# ============================================================
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ============================================================
# DATA STRUCTURES
# ============================================================
@dataclass
class EdgeCluster:
    """
    A compact cluster of LiDAR scan points representing one door jamb edge.

    Attributes:
        points:        List of (angle_rad, range_m) raw scan points.
        centroid_x:    Cluster centroid X in robot frame [m].
        centroid_y:    Cluster centroid Y in robot frame [m].
        mean_range:    Mean range of all cluster points [m].
        min_angle:     Minimum angle of cluster span [rad].
        max_angle:     Maximum angle of cluster span [rad].
        angular_width: Total angular span [rad].
        range_spread:  max_range - min_range within cluster [m].
        n_points:      Number of scan points.
    """
    points:        List[Tuple[float, float]] = field(default_factory=list)
    centroid_x:    float = 0.0
    centroid_y:    float = 0.0
    mean_range:    float = 0.0
    min_angle:     float = 0.0
    max_angle:     float = 0.0
    angular_width: float = 0.0
    range_spread:  float = 0.0
    n_points:      int   = 0

    def compute(self) -> None:
        """Compute all statistics from raw (angle, range) points."""
        if not self.points:
            return
        angles = [a for a, _ in self.points]
        ranges = [r for _, r in self.points]
        xs = [r * math.cos(a) for a, r in self.points]
        ys = [r * math.sin(a) for a, r in self.points]

        self.centroid_x    = sum(xs) / len(xs)
        self.centroid_y    = sum(ys) / len(ys)
        self.mean_range    = sum(ranges) / len(ranges)
        self.min_angle     = min(angles)
        self.max_angle     = max(angles)
        self.angular_width = self.max_angle - self.min_angle
        self.range_spread  = max(ranges) - min(ranges)
        self.n_points      = len(self.points)

    def __repr__(self) -> str:
        return (f'EdgeCluster(n={self.n_points}, '
                f'cx={self.centroid_x:.3f}, cy={self.centroid_y:.3f}, '
                f'range={self.mean_range:.3f}m, '
                f'width={math.degrees(self.angular_width):.1f}°, '
                f'depth={self.range_spread:.3f}m)')


@dataclass
class DoorResult:
    """
    A confirmed doorway detection.

    Attributes:
        centre_robot_x/y:  Door centre in robot frame [m].
        centre_map_x/y:    Door centre in map frame [m].
        jamb_left_robot:   Left jamb edge (x, y) in robot frame [m].
        jamb_right_robot:  Right jamb edge (x, y) in robot frame [m].
        jamb_left_map:     Left jamb edge (x, y) in map frame [m].
        jamb_right_map:    Right jamb edge (x, y) in map frame [m].
        gap_width:         Measured gap width [m].
        gap_bearing:       Bearing to door centre from robot [rad], in robot frame.
        gap_range:         Range to door centre from robot [m].
        heading_map:       Door orientation in map frame [rad] (normal to door plane).
        confidence:        Number of consecutive confirmations.
    """
    centre_robot_x:   float = 0.0
    centre_robot_y:   float = 0.0
    centre_map_x:     float = 0.0
    centre_map_y:     float = 0.0
    jamb_left_robot:  Tuple[float, float] = (0.0, 0.0)
    jamb_right_robot: Tuple[float, float] = (0.0, 0.0)
    jamb_left_map:    Tuple[float, float] = (0.0, 0.0)
    jamb_right_map:   Tuple[float, float] = (0.0, 0.0)
    gap_width:        float = 0.0
    gap_bearing:      float = 0.0   # rad in robot frame (0 = straight ahead)
    gap_range:        float = 0.0   # m
    heading_map:      float = 0.0   # rad — direction robot must travel to pass through
    confidence:       int   = 0

    def __repr__(self) -> str:
        return (
            f'DoorResult('
            f'map=({self.centre_map_x:.3f},{self.centre_map_y:.3f}), '
            f'robot=({self.centre_robot_x:.3f},{self.centre_robot_y:.3f}), '
            f'width={self.gap_width:.3f}m, '
            f'bearing={math.degrees(self.gap_bearing):.1f}°, '
            f'range={self.gap_range:.3f}m, '
            f'conf={self.confidence})'
        )


# ============================================================
# UTILITIES
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
        (map_x, map_y).
    """
    cos_y = math.cos(robot_pose_yaw)
    sin_y = math.sin(robot_pose_yaw)
    return (
        robot_pose_x + rx * cos_y - ry * sin_y,
        robot_pose_y + rx * sin_y + ry * cos_y,
    )


def normalize_angle(a: float) -> float:
    """Normalise angle to [-π, π]."""
    while a >  math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a


# ============================================================
# MAIN CLASS
# ============================================================
class DoorDetector:
    """
    Detects an 0.8 m doorway from LiDAR scans using gap analysis.

    The detector looks for two wall-edge clusters (door jambs) separated
    by an angular gap consistent with a 0.8 m opening, with empty space
    (no returns) between them.

    Usage (from mission_node.py / debug_phase1_node.py):

        detector = DoorDetector(logger=self.get_logger())

        # in /scan callback:
        detector.update_scan(scan_msg)

        # in control tick:
        result = detector.detect(robot_x, robot_y, robot_yaw)
        if result is not None:
            # door confirmed — result.centre_map_x/y is the passage target
            # result.heading_map is the direction to travel through the door
    """

    def __init__(self, logger=None) -> None:
        """
        Initialise the door detector.

        Args:
            logger: ROS2 logger or None (falls back to print).
        """
        self._log = logger

        # Raw scan snapshot
        self._raw_ranges:    List[float]              = []
        self._scan_points:   List[Tuple[float, float]] = []   # (angle, range) valid
        self._scan_angle_min: float = 0.0
        self._scan_angle_inc: float = 0.0
        self._scan_ready      = False

        # Confirmation buffer
        self._confirm_count  = 0
        self._last_centre_x: Optional[float] = None
        self._last_centre_y: Optional[float] = None

        # Confirmed result
        self._confirmed: Optional[DoorResult] = None

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def update_scan(self, scan_msg) -> None:
        """
        Ingest a new LaserScan message.

        Stores the full raw ranges array (needed for gap emptiness check)
        and the filtered (angle, range) list for clustering.

        Args:
            scan_msg: sensor_msgs/LaserScan.
        """
        self._raw_ranges     = list(scan_msg.ranges)
        self._scan_angle_min = scan_msg.angle_min
        self._scan_angle_inc = scan_msg.angle_increment

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
        robot_x:   float,
        robot_y:   float,
        robot_yaw: float,
    ) -> Optional[DoorResult]:
        """
        Run one detection cycle on the latest scan.

        Returns a DoorResult once N_CONFIRM consecutive cycles agree on
        the same door position, or None otherwise.

        Args:
            robot_x/y:   Robot position in map frame [m].
            robot_yaw:   Robot heading in map frame [rad].

        Returns:
            DoorResult or None.
        """
        if not self._scan_ready:
            return None

        # Already confirmed — return cached result
        if self._confirmed is not None:
            return self._confirmed

        # --- Pipeline ---
        clusters  = self._cluster_scan(self._scan_points)
        candidate = self._find_door_gap(clusters)

        if candidate is None:
            self._confirm_count = 0
            return None

        left_cluster, right_cluster, gap_bearing, gap_range, gap_width = candidate

        # Centre of the doorway in robot frame
        cx_r = gap_range * math.cos(gap_bearing)
        cy_r = gap_range * math.sin(gap_bearing)

        # Consistency check in MAP frame (stable even when robot moves
        # during wall-following; robot-frame coords change with robot pose)
        cx_map, cy_map = robot_to_map(cx_r, cy_r, robot_x, robot_y, robot_yaw)

        if self._last_centre_x is not None:
            drift = dist2d(cx_map, cy_map, self._last_centre_x, self._last_centre_y)
            if drift > Config.MAX_CENTRE_DRIFT:
                self._log_warn(
                    f'[DOOR] Candidat ha canviat {drift:.3f}m (frame mapa) — '
                    'resettejant confirmació.'
                )
                self._confirm_count = 0

        # Store map-frame centre for next drift check
        self._last_centre_x = cx_map
        self._last_centre_y = cy_map
        self._confirm_count += 1

        self._log_info(
            f'[DOOR] Candidat [{self._confirm_count}/{Config.N_CONFIRM}]  '
            f'centre_robot=({cx_r:.3f},{cy_r:.3f})  '
            f'width={gap_width:.3f}m  '
            f'bearing={math.degrees(gap_bearing):.1f}°  '
            f'range={gap_range:.3f}m'
        )

        if self._confirm_count >= Config.N_CONFIRM:
            result = self._build_result(
                left_cluster, right_cluster,
                cx_r, cy_r, gap_bearing, gap_range, gap_width,
                robot_x, robot_y, robot_yaw,
            )
            self._confirmed = result
            self._log_info(
                f'[DOOR] ✓ PORTA CONFIRMADA!  '
                f'map=({result.centre_map_x:.3f},{result.centre_map_y:.3f})  '
                f'width={result.gap_width:.3f}m  '
                f'heading={math.degrees(result.heading_map):.1f}°'
            )
            return result

        return None

    def is_confirmed(self) -> bool:
        """Return True if a door has been confirmed."""
        return self._confirmed is not None

    def get_confirmed(self) -> Optional[DoorResult]:
        """Return the confirmed DoorResult, or None if not yet found."""
        return self._confirmed

    def reset(self) -> None:
        """Reset detector state (e.g. after passing through the door)."""
        self._confirm_count  = 0
        self._last_centre_x  = None
        self._last_centre_y  = None
        self._confirmed      = None
        self._log_info('[DOOR] Reset')

    # ----------------------------------------------------------
    # Step 1 — Clustering
    # ----------------------------------------------------------

    def _cluster_scan(
        self, points: List[Tuple[float, float]]
    ) -> List[EdgeCluster]:
        """
        Group adjacent scan points into wall-segment clusters.

        Two consecutive points belong to the same cluster when their
        Cartesian distance is ≤ CLUSTER_DIST.  Each cluster is then
        characterised by its centroid, angular span, and range spread.

        Args:
            points: List of (angle_rad, range_m), sorted by angle (from /scan).

        Returns:
            List of EdgeCluster objects with computed statistics.
        """
        if not points:
            return []

        clusters: List[EdgeCluster] = []
        current = EdgeCluster(points=[points[0]])

        for i in range(1, len(points)):
            a_prev, r_prev = points[i - 1]
            a_curr, r_curr = points[i]

            x_prev = r_prev * math.cos(a_prev)
            y_prev = r_prev * math.sin(a_prev)
            x_curr = r_curr * math.cos(a_curr)
            y_curr = r_curr * math.sin(a_curr)

            if dist2d(x_prev, y_prev, x_curr, y_curr) <= Config.CLUSTER_DIST:
                current.points.append(points[i])
            else:
                current.compute()
                clusters.append(current)
                current = EdgeCluster(points=[points[i]])

        current.compute()
        clusters.append(current)
        return clusters

    # ----------------------------------------------------------
    # Step 2 — Gap search
    # ----------------------------------------------------------

    def _find_door_gap(
        self, clusters: List[EdgeCluster]
    ) -> Optional[Tuple]:
        """
        Scan every consecutive cluster pair for a gap matching a 0.8 m door.

        For each pair (left_cluster, right_cluster) — where "left" has the
        smaller angle (more counter-clockwise) and "right" has the larger
        angle — we check:

          1. Gap angular width consistent with DOOR_WIDTH ± DOOR_WIDTH_TOL
             at the mean range of the two jambs.
          2. Both jambs are at similar range (MAX_RANGE_DIFF).
          3. Both jambs are within FRONT_FOV of the robot forward direction.
          4. The jamb clusters are compact (not long wall segments).
          5. The open sector between them is truly empty (GAP_EMPTY_FRACTION
             of rays have no return or a far return).

        Args:
            clusters: All clusters from _cluster_scan().

        Returns:
            Tuple (left_cluster, right_cluster, gap_bearing, gap_range,
                   gap_width_m) or None if no valid gap found.
        """
        # Work with clusters that could be jamb edges
        candidates = [
            c for c in clusters
            if Config.MIN_JAMB_POINTS <= c.n_points <= Config.MAX_JAMB_POINTS
            and c.range_spread <= Config.MAX_JAMB_DEPTH
        ]

        # Sort by angle of centroid (ascending = right-to-left in ROS convention)
        candidates.sort(key=lambda c: math.atan2(c.centroid_y, c.centroid_x))

        best: Optional[Tuple] = None
        best_score = float('inf')   # prefer the gap closest to the robot front

        for i in range(len(candidates) - 1):
            right_c = candidates[i]       # smaller angle → more to the right
            left_c  = candidates[i + 1]   # larger angle  → more to the left

            # ---- 1. Gap angular extent ----
            # The gap spans from the right edge of right_c to the left edge of left_c
            gap_angle_start = right_c.max_angle   # rightmost ray of right jamb
            gap_angle_end   = left_c.min_angle    # leftmost  ray of left  jamb
            gap_angular_width = gap_angle_end - gap_angle_start

            if gap_angular_width <= 0:
                continue   # clusters overlap angularly

            # Mean range to the door plane
            mean_range = (right_c.mean_range + left_c.mean_range) / 2.0

            # Physical gap width from angular width at this range
            # W = 2 * d * tan(α/2)
            measured_width = 2.0 * mean_range * math.tan(gap_angular_width / 2.0)

            if not (
                Config.DOOR_WIDTH - Config.DOOR_WIDTH_TOL
                <= measured_width
                <= Config.DOOR_WIDTH + Config.DOOR_WIDTH_TOL
            ):
                continue

            # ---- 2. Jambs at similar range ----
            if abs(right_c.mean_range - left_c.mean_range) > Config.MAX_RANGE_DIFF:
                continue

            # ---- 3. Gap bearing within FOV ----
            gap_bearing = (gap_angle_start + gap_angle_end) / 2.0
            if abs(normalize_angle(gap_bearing)) > Config.FRONT_FOV:
                continue

            # ---- 4. Gap is truly open (no returns inside) ----
            if not self._gap_is_empty(gap_angle_start, gap_angle_end):
                continue

            # ---- All checks passed — score by proximity to front ----
            score = abs(normalize_angle(gap_bearing))
            if score < best_score:
                best_score = score
                best = (left_c, right_c, gap_bearing, mean_range, measured_width)

        return best

    # ----------------------------------------------------------
    # Step 3 — Gap emptiness check
    # ----------------------------------------------------------

    def _gap_is_empty(
        self, angle_start: float, angle_end: float
    ) -> bool:
        """
        Verify that the angular sector between angle_start and angle_end
        contains no valid close returns (i.e. is open space).

        A ray is considered "open" if its range is:
          - NaN or Inf (no return), OR
          - greater than GAP_EMPTY_MIN_RANGE (far return, not a wall inside the gap)

        The gap passes if at least GAP_EMPTY_FRACTION of its rays are open.

        Args:
            angle_start: Start of gap sector [rad, in scan frame].
            angle_end:   End   of gap sector [rad, in scan frame].

        Returns:
            True if the gap appears genuinely open.
        """
        if not self._raw_ranges or self._scan_angle_inc == 0:
            return True   # no data — assume open (conservative)

        n = len(self._raw_ranges)
        total_rays = 0
        open_rays  = 0

        for i, r in enumerate(self._raw_ranges):
            ray_angle = self._scan_angle_min + i * self._scan_angle_inc
            if not (angle_start <= ray_angle <= angle_end):
                continue
            total_rays += 1
            if math.isnan(r) or math.isinf(r) or r > Config.GAP_EMPTY_MIN_RANGE:
                open_rays += 1

        if total_rays == 0:
            return True   # sector has no rays (sparse scan) — assume open

        return (open_rays / total_rays) >= Config.GAP_EMPTY_FRACTION

    # ----------------------------------------------------------
    # Step 4 — Build result
    # ----------------------------------------------------------

    def _build_result(
        self,
        left_cluster:  EdgeCluster,
        right_cluster: EdgeCluster,
        cx_robot:   float,
        cy_robot:   float,
        gap_bearing: float,
        gap_range:   float,
        gap_width:   float,
        robot_x:    float,
        robot_y:    float,
        robot_yaw:  float,
    ) -> DoorResult:
        """
        Construct a DoorResult from a confirmed gap candidate.

        Transforms all positions from robot frame to map frame.
        The door heading in map frame is the direction the robot must travel
        to pass through the door (= gap_bearing rotated by robot_yaw).

        Args:
            left_cluster / right_cluster: The two jamb edge clusters.
            cx_robot / cy_robot:          Door centre in robot frame [m].
            gap_bearing:                  Bearing to door centre [rad, robot frame].
            gap_range:                    Range to door centre [m].
            gap_width:                    Measured gap width [m].
            robot_x/y/yaw:                Robot pose in map frame.

        Returns:
            Fully populated DoorResult.
        """
        # Jamb inner-edge positions in robot frame (closest point of each cluster)
        jlx = left_cluster.centroid_x
        jly = left_cluster.centroid_y
        jrx = right_cluster.centroid_x
        jry = right_cluster.centroid_y

        # Map frame transforms
        cx_map, cy_map   = robot_to_map(cx_robot, cy_robot, robot_x, robot_y, robot_yaw)
        jl_map           = robot_to_map(jlx, jly, robot_x, robot_y, robot_yaw)
        jr_map           = robot_to_map(jrx, jry, robot_x, robot_y, robot_yaw)

        # Door heading in map frame: direction robot travels to pass through
        heading_map = normalize_angle(robot_yaw + gap_bearing)

        return DoorResult(
            centre_robot_x   = cx_robot,
            centre_robot_y   = cy_robot,
            centre_map_x     = cx_map,
            centre_map_y     = cy_map,
            jamb_left_robot  = (jlx, jly),
            jamb_right_robot = (jrx, jry),
            jamb_left_map    = jl_map,
            jamb_right_map   = jr_map,
            gap_width        = gap_width,
            gap_bearing      = gap_bearing,
            gap_range        = gap_range,
            heading_map      = heading_map,
            confidence       = self._confirm_count,
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
    # Fake LaserScan builder
    # --------------------------------------------------------
    class FakeScan:
        """
        Generates a synthetic 360° LaserScan with:
          - Two wall segments flanking an open gap (simulated door).
          - Optional extra wall segments in the background.

        The scan represents the robot looking straight ahead (+X).
        The door is placed at distance `door_dist` directly in front,
        with `door_bearing` offset from the robot's forward direction.

        Args:
            door_dist:     Range to the door plane [m].
            door_width:    Width of the door opening [m].
            door_bearing:  Bearing to door centre in robot frame [rad].
            wall_width:    Width of wall on each side of door [m].
            bg_range:      Background range for open sectors [m].
            n_rays:        Total number of rays.
            noise_m:       Gaussian range noise std-dev [m] (0 = no noise).
        """
        def __init__(
            self,
            door_dist:    float = 1.5,
            door_width:   float = 0.80,
            door_bearing: float = 0.0,
            wall_width:   float = 0.60,
            bg_range:     float = 5.0,
            n_rays:       int   = 360,
            noise_m:      float = 0.0,
        ) -> None:
            import random
            self.range_min       = 0.12
            self.range_max       = bg_range
            self.angle_min       = -math.pi
            self.angle_increment = 2 * math.pi / n_rays
            self.ranges          = [bg_range] * n_rays

            half_door = door_width  / 2.0
            half_wall = wall_width  / 2.0

            # For each ray, check if it hits the left wall, right wall, or open gap
            for i in range(n_rays):
                ray_angle = self.angle_min + i * self.angle_increment
                # Angle of ray relative to door bearing
                rel = normalize_angle(ray_angle - door_bearing)

                # Y-intercept of this ray on the door plane at door_dist
                # (assuming wall is perpendicular to the door_bearing direction)
                if abs(math.cos(rel)) < 1e-6:
                    continue   # ray parallel to wall — no intersection
                # Project: distance along the ray to reach the door plane
                d_to_plane = door_dist / math.cos(rel)
                if d_to_plane < 0:
                    continue   # behind robot

                # Lateral offset at the door plane
                lateral = d_to_plane * math.sin(rel)

                # Door gap: |lateral| < half_door → open
                # Left wall: half_door <= lateral < half_door + half_wall
                # Right wall: -(half_door + half_wall) < lateral <= -half_door
                if abs(lateral) < half_door:
                    # Open gap — no return (leave as bg_range)
                    pass
                elif half_door <= lateral < half_door + half_wall:
                    # Left wall
                    r = d_to_plane
                    if noise_m > 0:
                        r += random.gauss(0, noise_m)
                    self.ranges[i] = max(0.12, r)
                elif -(half_door + half_wall) < lateral <= -half_door:
                    # Right wall
                    r = d_to_plane
                    if noise_m > 0:
                        r += random.gauss(0, noise_m)
                    self.ranges[i] = max(0.12, r)

    # --------------------------------------------------------
    # Helper: run N_CONFIRM cycles
    # --------------------------------------------------------
    def run_detection(
        scan:       'FakeScan',
        robot_pose: Tuple[float, float, float],
        label:      str,
        expect_detect: bool = True,
    ) -> Optional[DoorResult]:
        detector = DoorDetector()
        print(f'\n=== {label} ===')

        result = None
        for i in range(Config.N_CONFIRM + 3):
            detector.update_scan(scan)
            result = detector.detect(*robot_pose)
            if result:
                break

        if result:
            if expect_detect:
                print(f'    ✓ Porta confirmada après {result.confidence} scans')
            else:
                print(f'    ✗ Detecció inesperada!')
            print(f'    Centre robot : ({result.centre_robot_x:.3f}, {result.centre_robot_y:.3f})')
            print(f'    Centre mapa  : ({result.centre_map_x:.3f}, {result.centre_map_y:.3f})')
            print(f'    Amplada gap  : {result.gap_width:.3f} m  '
                  f'(nominal {Config.DOOR_WIDTH} m)')
            print(f'    Bearing      : {math.degrees(result.gap_bearing):.1f}°')
            print(f'    Range        : {result.gap_range:.3f} m')
            print(f'    Heading mapa : {math.degrees(result.heading_map):.1f}°')
        else:
            if not expect_detect:
                print(f'    ✓ Correctament rebutjat')
            else:
                print(f'    ✗ No detectat en {Config.N_CONFIRM + 3} scans')

        return result

    # --------------------------------------------------------
    # Test 1 — Door straight ahead at 1.5 m, robot at origin
    # --------------------------------------------------------
    run_detection(
        FakeScan(door_dist=1.5, door_width=0.80, door_bearing=0.0),
        (0.0, 0.0, 0.0),
        'Porta de 0.80m directament al davant a 1.5m',
    )

    # --------------------------------------------------------
    # Test 2 — Door at 2.5 m, slightly off-centre (+20°)
    # --------------------------------------------------------
    run_detection(
        FakeScan(door_dist=2.5, door_width=0.80, door_bearing=math.radians(20)),
        (0.0, 0.0, 0.0),
        'Porta a 2.5m, 20° a l\'esquerra del front',
    )

    # --------------------------------------------------------
    # Test 3 — Door with realistic noise
    # --------------------------------------------------------
    run_detection(
        FakeScan(door_dist=1.8, door_width=0.80, door_bearing=0.0, noise_m=0.02),
        (0.0, 0.0, 0.0),
        'Porta a 1.8m amb soroll LiDAR ±2cm',
    )

    # --------------------------------------------------------
    # Test 4 — Robot not at origin; check map transform
    # --------------------------------------------------------
    result4 = run_detection(
        FakeScan(door_dist=2.0, door_width=0.80, door_bearing=0.0),
        (6.280, 10.545, math.pi / 2),   # robot at Punt F, facing north
        'Robot a Punt F (6.28, 10.545, 90°), porta al davant a 2m',
    )
    if result4:
        expected_map_x = 6.280
        expected_map_y = 10.545 + 2.0
        err = dist2d(
            result4.centre_map_x, result4.centre_map_y,
            expected_map_x, expected_map_y,
        )
        print(f'    Error posició mapa : {err:.3f} m  '
              f'(esperat ≈({expected_map_x:.2f},{expected_map_y:.2f}))')

    # --------------------------------------------------------
    # Test 5 — Too narrow gap (0.40 m) → must be rejected
    # --------------------------------------------------------
    run_detection(
        FakeScan(door_dist=1.5, door_width=0.40, door_bearing=0.0),
        (0.0, 0.0, 0.0),
        'Gap de 0.40m (massa estret) — ha de ser rebutjat',
        expect_detect=False,
    )

    # --------------------------------------------------------
    # Test 6 — Too wide gap (1.5 m) → must be rejected
    # --------------------------------------------------------
    run_detection(
        FakeScan(door_dist=1.5, door_width=1.50, door_bearing=0.0),
        (0.0, 0.0, 0.0),
        'Gap de 1.50m (massa ample) — ha de ser rebutjat',
        expect_detect=False,
    )

    # --------------------------------------------------------
    # Test 7 — Door behind robot (> FRONT_FOV) → must be rejected
    # --------------------------------------------------------
    run_detection(
        FakeScan(door_dist=1.5, door_width=0.80, door_bearing=math.pi),
        (0.0, 0.0, 0.0),
        'Porta darrere el robot (180°) — ha de ser rebutjada',
        expect_detect=False,
    )

    print('\nTots els tests completats.')