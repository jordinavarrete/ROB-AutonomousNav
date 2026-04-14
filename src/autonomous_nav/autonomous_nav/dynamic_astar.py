#!/usr/bin/env python3
"""
dynamic_astar.py — Dynamic A* (D* / D* Lite principles) path planner.

Provides grid-based route planning with dynamic obstacle avoidance.
When the map is updated (via LiDAR), it re-evaluates the path and 
rapidly computes a new optimal route to the goal using a heuristic.

Features:
- Maintains an occupancy cost grid.
- Heuristic search (A* style) replanned upon grid changes.
- Safe buffer around obstacles to ensure physical integrity (required).
"""

import math
import heapq
import numpy as np
from typing import List, Tuple, Optional

class Config:
    CELL_SIZE = 0.1       # 10 cm per cell
    SAFE_MARGIN = 0.25    # 25 cm clearance from obstacles to center of robot
    MAX_X = 15.0          # Passadis max X
    MAX_Y = 15.0          # Passadis max Y
    MIN_X = -5.0
    MIN_Y = -5.0

class DynamicAStar:
    def __init__(self):
        self.width = int((Config.MAX_X - Config.MIN_X) / Config.CELL_SIZE)
        self.height = int((Config.MAX_Y - Config.MIN_Y) / Config.CELL_SIZE)
        
        # 0 = free, 1 = obstacle
        self.grid = np.zeros((self.height, self.width), dtype=np.int8)
        self.cost_map = np.zeros((self.height, self.width), dtype=np.float32)
        
        # Keep track of last path to detect if it's invalidated
        self.current_path = []
        
    def _world_to_grid(self, wx, wy) -> Tuple[int, int]:
        gx = int((wx - Config.MIN_X) / Config.CELL_SIZE)
        gy = int((wy - Config.MIN_Y) / Config.CELL_SIZE)
        # clamp
        gx = max(0, min(self.width - 1, gx))
        gy = max(0, min(self.height - 1, gy))
        return gx, gy

    def _grid_to_world(self, gx, gy) -> Tuple[float, float]:
        wx = gx * Config.CELL_SIZE + Config.MIN_X + Config.CELL_SIZE / 2.0
        wy = gy * Config.CELL_SIZE + Config.MIN_Y + Config.CELL_SIZE / 2.0
        return wx, wy

    def update_obstacles(self, obstacles: List[Tuple[float, float]]):
        """Update map with new obstacles discovered by LiDAR."""
        changed = False
        safe_cells = int(Config.SAFE_MARGIN / Config.CELL_SIZE)
        
        for ox, oy in obstacles:
            gx, gy = self._world_to_grid(ox, oy)
            
            # Inflate obstacle by SAFE_MARGIN
            for dx in range(-safe_cells, safe_cells + 1):
                for dy in range(-safe_cells, safe_cells + 1):
                    if dx*dx + dy*dy <= safe_cells*safe_cells: # circular inflation
                        nx, ny = gx + dx, gy + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            if self.grid[ny, nx] == 0:
                                self.grid[ny, nx] = 1
                                changed = True
        return changed

    def is_path_valid(self) -> bool:
        """Check if the currently planned path goes through newly added obstacles."""
        if not self.current_path:
            return False
            
        for wx, wy in self.current_path:
            gx, gy = self._world_to_grid(wx, wy)
            if self.grid[gy, gx] == 1:
                return False
        return True

    def plan(self, start_x: float, start_y: float, goal_x: float, goal_y: float) -> Optional[List[Tuple[float, float]]]:
        """
        Dynamically plan an A* path from start to goal.
        Returns a list of (x, y) waypoints if possible, otherwise None.
        """
        start_g = self._world_to_grid(start_x, start_y)
        goal_g = self._world_to_grid(goal_x, goal_y)
        
        if self.grid[start_g[1], start_g[0]] == 1:
            # Start is inside an obstacle (or margin). Try to move to the closest free cell.
            start_g = self._find_nearest_free(*start_g)
            if not start_g:
                return None
                
        if self.grid[goal_g[1], goal_g[0]] == 1:
            # Goal is inside obstacle (or margin).
            goal_g = self._find_nearest_free(*goal_g)
            if not goal_g:
                return None

        # A* implementation
        open_set = []
        heapq.heappush(open_set, (0, start_g))
        
        came_from = {}
        g_score = {start_g: 0}
        
        def heuristic(a, b):
            return math.dist(a, b)
            
        f_score = {start_g: heuristic(start_g, goal_g)}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal_g:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                
                # Convert path to world coordinates, filtering redundant collinear points
                world_path = self._simplify_path([self._grid_to_world(cx, cy) for cx, cy in path])
                self.current_path = world_path
                return world_path
                
            for dx, dy in [(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if 0 <= neighbor[0] < self.width and 0 <= neighbor[1] < self.height:
                    if self.grid[neighbor[1], neighbor[0]] == 1:
                        continue # Obstacle
                        
                    tentative_g = g_score[current] + math.dist(current, neighbor)
                    if tentative_g < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + heuristic(neighbor, goal_g)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        
        self.current_path = []
        return None

    def _find_nearest_free(self, gx, gy) -> Optional[Tuple[int, int]]:
        """Find the nearest free cell if start/goal falls inside clearance zone."""
        for r in range(1, 10):
            for dx in range(-r, r+1):
                for dy in range(-r, r+1):
                    if max(abs(dx), abs(dy)) == r:
                        nx, ny = gx + dx, gy + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height:
                            if self.grid[ny, nx] == 0:
                                return (nx, ny)
        return None

    def _simplify_path(self, path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Removes intermediate points on straight lines."""
        if len(path) < 3:
            return path
            
        simplified = [path[0]]
        for i in range(1, len(path) - 1):
            p_prev = simplified[-1]
            p_curr = path[i]
            p_next = path[i+1]
            
            angle1 = math.atan2(p_curr[1] - p_prev[1], p_curr[0] - p_prev[0])
            angle2 = math.atan2(p_next[1] - p_curr[1], p_next[0] - p_curr[0])
            
            # If the angle changes significantly, it's a corner, so we keep the point
            if abs(angle1 - angle2) > 1e-3:
                simplified.append(p_curr)
                
        simplified.append(path[-1])
        return simplified

# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    dastar = DynamicAStar()
    obs = [(3.72, 2.55)] # Punt B but occupied
    dastar.update_obstacles(obs)
    path = dastar.plan(2.52, 1.35, 5.0, 11.69)
    print("Planned path:")
    for wp in path:
        print(f"({wp[0]:.2f}, {wp[1]:.2f})")
