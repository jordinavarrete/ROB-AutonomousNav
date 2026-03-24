#!/usr/bin/env python3
"""
route_planner.py — A* path planner on OccupancyGrid (map-based fallback).

This module provides a SECONDARY navigation strategy. The robot primarily
navigates using direct waypoints + Bug2 obstacle avoidance. The RoutePlanner
activates ONLY when Bug2 fails to make progress (timeout or anti-stuck trigger).

Algorithm:
  1. Receive the latest OccupancyGrid from /map
  2. Inflate obstacles by robot radius + safety margin
  3. Run A* from robot cell to target waypoint cell (8-connected)
  4. Simplify the raw A* path into a small set of intermediate waypoints
  5. Return world-coordinate waypoints for the navigator to follow

This class has NO ROS2 Node inheritance — it is a pure logic class
called by mission_node.py.
"""

# ============================================================
# CONFIGURACIÓ
# ============================================================
class Config:
    # Obstacle inflation
    INFLATE_RADIUS_M      = 0.15    # m — robot radius + safety margin
    OCCUPANCY_THRESHOLD   = 65      # grid cell values > this → occupied

    # Path simplification
    MAX_WAYPOINT_SPACING  = 1.0     # m — max spacing between simplified waypoints
    DIRECTION_CHANGE_DEG  = 30.0    # degrees — min angle change to emit a waypoint
    MAX_TEMP_WAYPOINTS    = 10      # max intermediate waypoints returned

    # Unknown cells
    UNKNOWN_AS_FREE       = True    # treat -1 (unknown) cells as free


# ============================================================
# IMPORTS
# ============================================================
import math
import heapq
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ============================================================
# DATA TYPES
# ============================================================
@dataclass
class GridInfo:
    """Stores occupancy grid metadata."""
    width:      int
    height:     int
    resolution: float   # m/cell
    origin_x:   float   # world x of cell (0,0)
    origin_y:   float   # world y of cell (0,0)
    data:       list    # flattened occupancy row-major


# ============================================================
# MAIN CLASS
# ============================================================
class RoutePlanner:
    """
    A* path planner on a ROS2 OccupancyGrid.

    Usage:
        planner = RoutePlanner(logger=node.get_logger())

        # Called from /map subscriber:
        planner.update_map(occupancy_grid_msg)

        # Called when avoidance times out:
        waypoints = planner.compute_path(robot_x, robot_y, target_x, target_y)
        if waypoints:
            # Insert these before the original target
            ...
    """

    def __init__(self, logger=None) -> None:
        self._logger = logger
        self._grid: Optional[GridInfo] = None
        self._inflated: Optional[list] = None
        self._inflate_cells: int = 0

    # ==========================================================
    # MAP UPDATE
    # ==========================================================

    def update_map(self, msg) -> None:
        """
        Store the latest OccupancyGrid message.

        Args:
            msg: nav_msgs/OccupancyGrid message
        """
        self._grid = GridInfo(
            width=msg.info.width,
            height=msg.info.height,
            resolution=msg.info.resolution,
            origin_x=msg.info.origin.position.x,
            origin_y=msg.info.origin.position.y,
            data=list(msg.data),
        )
        self._inflate_cells = max(1, int(Config.INFLATE_RADIUS_M / msg.info.resolution))
        self._inflated = None  # invalidate cache

    def has_map(self) -> bool:
        """Return True if at least one map has been received."""
        return self._grid is not None

    # ==========================================================
    # MAIN ENTRY POINT
    # ==========================================================

    def compute_path(
        self,
        robot_x: float,
        robot_y: float,
        target_x: float,
        target_y: float,
    ) -> List[Tuple[float, float]]:
        """
        Compute A*-based intermediate waypoints from robot to target.

        Returns:
            List of (x, y) world-coordinate waypoints (may be empty
            if no path found or no map available).
        """
        if self._grid is None:
            if self._logger:
                self._logger.warn('[RoutePlanner] No map available — cannot plan')
            return []

        # Inflate obstacles if not cached
        if self._inflated is None:
            self._inflated = self._inflate_grid()

        # Convert world → grid
        start = self._world_to_grid(robot_x, robot_y)
        goal  = self._world_to_grid(target_x, target_y)

        if start is None or goal is None:
            if self._logger:
                self._logger.warn(
                    f'[RoutePlanner] Start or goal outside map bounds '
                    f'robot=({robot_x:.2f},{robot_y:.2f}) '
                    f'target=({target_x:.2f},{target_y:.2f})'
                )
            return []

        # Check start/goal cells aren't blocked
        if self._is_blocked(start[0], start[1]):
            # Robot is inside an inflated obstacle — find nearest free cell
            start = self._nearest_free(start[0], start[1])
            if start is None:
                if self._logger:
                    self._logger.warn('[RoutePlanner] Robot stuck in blocked area')
                return []

        if self._is_blocked(goal[0], goal[1]):
            goal = self._nearest_free(goal[0], goal[1])
            if goal is None:
                if self._logger:
                    self._logger.warn('[RoutePlanner] Goal in blocked area')
                return []

        # Run A*
        raw_path = self._astar(start, goal)
        if not raw_path:
            if self._logger:
                self._logger.warn('[RoutePlanner] A* found no path')
            return []

        # Convert grid path → world
        world_path = [self._grid_to_world(gx, gy) for gx, gy in raw_path]

        # Simplify
        simplified = self._simplify_path(world_path)

        if self._logger:
            self._logger.info(
                f'[RoutePlanner] A* path: {len(raw_path)} cells → '
                f'{len(simplified)} waypoints'
            )

        return simplified

    # ==========================================================
    # GRID OPERATIONS
    # ==========================================================

    def _world_to_grid(self, wx: float, wy: float) -> Optional[Tuple[int, int]]:
        """Convert world coordinates to grid cell indices."""
        g = self._grid
        gx = int((wx - g.origin_x) / g.resolution)
        gy = int((wy - g.origin_y) / g.resolution)
        if 0 <= gx < g.width and 0 <= gy < g.height:
            return (gx, gy)
        return None

    def _grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """Convert grid cell indices to world coordinates (centre of cell)."""
        g = self._grid
        wx = g.origin_x + (gx + 0.5) * g.resolution
        wy = g.origin_y + (gy + 0.5) * g.resolution
        return (wx, wy)

    def _is_blocked(self, gx: int, gy: int) -> bool:
        """Check if a cell is occupied in the inflated grid."""
        grid = self._inflated if self._inflated else self._grid.data
        g = self._grid
        if 0 <= gx < g.width and 0 <= gy < g.height:
            val = grid[gy * g.width + gx]
            if val == -1:
                return not Config.UNKNOWN_AS_FREE
            return val > Config.OCCUPANCY_THRESHOLD
        return True  # out of bounds = blocked

    def _inflate_grid(self) -> list:
        """
        Create an inflated copy of the occupancy grid.

        Every cell within INFLATE_RADIUS_CELLS of an occupied cell
        is also marked as occupied (value = 100).
        """
        g = self._grid
        inflated = list(g.data)  # copy
        r = self._inflate_cells

        # Find all occupied cells
        occupied = []
        for gy in range(g.height):
            for gx in range(g.width):
                val = g.data[gy * g.width + gx]
                if val > Config.OCCUPANCY_THRESHOLD:
                    occupied.append((gx, gy))

        # Inflate
        for ox, oy in occupied:
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    nx, ny = ox + dx, oy + dy
                    if 0 <= nx < g.width and 0 <= ny < g.height:
                        if dx * dx + dy * dy <= r * r:  # circular inflation
                            inflated[ny * g.width + nx] = 100

        if self._logger:
            self._logger.info(
                f'[RoutePlanner] Grid inflated: {len(occupied)} obstacles, '
                f'radius={r} cells ({r * g.resolution:.2f}m)'
            )

        return inflated

    def _nearest_free(self, gx: int, gy: int, max_r: int = 20) -> Optional[Tuple[int, int]]:
        """Find the nearest free cell using BFS expanding from (gx, gy)."""
        g = self._grid
        for r in range(1, max_r + 1):
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if abs(dx) != r and abs(dy) != r:
                        continue  # only check perimeter
                    nx, ny = gx + dx, gy + dy
                    if 0 <= nx < g.width and 0 <= ny < g.height:
                        if not self._is_blocked(nx, ny):
                            return (nx, ny)
        return None

    # ==========================================================
    # A* SEARCH
    # ==========================================================

    def _astar(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """
        A* on the inflated grid with 8-connected neighbors.

        Returns list of (gx, gy) from start to goal, or [] if no path.
        """
        g = self._grid

        # Heuristic: Euclidean distance
        def h(a, b):
            return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

        # 8 neighbours with costs
        NEIGHBORS = [
            (-1, -1, 1.414), (-1, 0, 1.0), (-1, 1, 1.414),
            ( 0, -1, 1.0),                  ( 0, 1, 1.0),
            ( 1, -1, 1.414), ( 1, 0, 1.0),  ( 1, 1, 1.414),
        ]

        open_set = [(h(start, goal), 0.0, start)]  # (f, g, cell)
        came_from = {}
        g_score = {start: 0.0}

        max_iterations = g.width * g.height  # safety limit

        iterations = 0
        while open_set and iterations < max_iterations:
            iterations += 1
            f, cost, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            if cost > g_score.get(current, float('inf')):
                continue

            for dx, dy, step_cost in NEIGHBORS:
                nx, ny = current[0] + dx, current[1] + dy
                if not (0 <= nx < g.width and 0 <= ny < g.height):
                    continue
                if self._is_blocked(nx, ny):
                    continue

                neighbor = (nx, ny)
                new_g = cost + step_cost

                if new_g < g_score.get(neighbor, float('inf')):
                    g_score[neighbor] = new_g
                    f_score = new_g + h(neighbor, goal)
                    heapq.heappush(open_set, (f_score, new_g, neighbor))
                    came_from[neighbor] = current

        return []  # no path found

    # ==========================================================
    # PATH SIMPLIFICATION
    # ==========================================================

    def _simplify_path(
        self,
        world_path: List[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """
        Reduce a dense A* path to a small set of navigation waypoints.

        Keeps points where:
          - Direction changes by > DIRECTION_CHANGE_DEG
          - Distance from last kept point > MAX_WAYPOINT_SPACING
        Always includes the last point (near the goal).
        """
        if len(world_path) <= 2:
            return world_path

        thresh_rad = math.radians(Config.DIRECTION_CHANGE_DEG)
        max_spacing = Config.MAX_WAYPOINT_SPACING

        simplified = [world_path[0]]
        last_kept = world_path[0]
        prev_angle = math.atan2(
            world_path[1][1] - world_path[0][1],
            world_path[1][0] - world_path[0][0],
        )

        for i in range(1, len(world_path) - 1):
            px, py = world_path[i]
            nx, ny = world_path[i + 1]

            # Current segment angle
            angle = math.atan2(ny - py, nx - px)
            angle_diff = abs(self._normalize(angle - prev_angle))

            # Distance from last kept waypoint
            dist = math.sqrt(
                (px - last_kept[0])**2 + (py - last_kept[1])**2
            )

            if angle_diff > thresh_rad or dist > max_spacing:
                simplified.append((px, py))
                last_kept = (px, py)
                prev_angle = angle

        # Always include the last point
        simplified.append(world_path[-1])

        # Limit number of waypoints
        if len(simplified) > Config.MAX_TEMP_WAYPOINTS:
            # Subsample evenly
            step = len(simplified) / Config.MAX_TEMP_WAYPOINTS
            indices = [int(i * step) for i in range(Config.MAX_TEMP_WAYPOINTS)]
            if indices[-1] != len(simplified) - 1:
                indices[-1] = len(simplified) - 1
            simplified = [simplified[i] for i in indices]

        # Remove the first point (it's the robot's current position)
        if len(simplified) > 1:
            simplified = simplified[1:]

        return simplified

    @staticmethod
    def _normalize(a: float) -> float:
        """Normalize angle to [-π, π]."""
        while a > math.pi:
            a -= 2 * math.pi
        while a < -math.pi:
            a += 2 * math.pi
        return a


# ============================================================
# STANDALONE TEST
# ============================================================
if __name__ == '__main__':
    """Quick test with a synthetic 20x20 grid."""
    import sys

    # Create a simple 20x20 grid with a wall in the middle
    width, height, res = 20, 20, 0.10
    data = [0] * (width * height)

    # Wall from (10, 3) to (10, 17)
    for y in range(3, 17):
        data[y * width + 10] = 100

    class FakeMsg:
        class Info:
            def __init__(self):
                self.width = width
                self.height = height
                self.resolution = res
                class Origin:
                    class Position:
                        x = 0.0
                        y = 0.0
                    position = Position()
                self.origin = Origin()
        info = Info()
        data = data

    planner = RoutePlanner()
    planner.update_map(FakeMsg())

    # Plan from (0.5, 1.0) to (1.5, 1.0) — must go around wall
    result = planner.compute_path(0.5, 1.0, 1.5, 1.0)
    print(f'A* waypoints: {result}')
    print(f'Count: {len(result)}')

    if result:
        print('✓ Route planner works')
    else:
        print('✗ No path found')
        sys.exit(1)
