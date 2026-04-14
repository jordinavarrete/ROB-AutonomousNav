#!/usr/bin/env python3
"""
map_builder.py — Constructs a 2D Occupancy Grid from PGO poses and LiDAR scans.

Provides the ability to export the grid to standard .yaml and .pgm files 
as required by the Autonomous Navigation project Phase I & II.
"""

import math
import struct
import numpy as np

class MapBuilder:
    def __init__(self, resolution=0.05, origin_x=-10.0, origin_y=-10.0, width_px=400, height_px=400):
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.width = width_px
        self.height = height_px
        
        # 127 = unknown, 0 = free, 255 = occupied (or 100/0 depending on ROS conventions)
        # Using 0 as free, 100 as occupied, 255 as unknown for ROS occupancy grid
        self.grid = np.full((self.height, self.width), -1, dtype=np.int8)

    def world_to_map(self, wx, wy):
        """Convert world coordinates (m) to map pixels."""
        mx = int((wx - self.origin_x) / self.resolution)
        my = int((wy - self.origin_y) / self.resolution)
        return mx, my
        
    def bresenham(self, x0, y0, x1, y1):
        """Yield integer coordinates on the line from (x0, y0) to (x1, y1)."""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                yield (x, y)
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                yield (x, y)
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        yield (x, y)

    def update_scan(self, pose_x, pose_y, pose_yaw, scan_angles, scan_ranges, max_range=3.0):
        """
        Update map using current pose and LiDAR arrays.
        """
        start_x, start_y = self.world_to_map(pose_x, pose_y)
        
        for angle_local, r in zip(scan_angles, scan_ranges):
            if math.isnan(r) or math.isinf(r):
                continue
                
            angle_world = pose_yaw + angle_local
            
            if r > max_range:
                # the ray is free up to max_range
                hit = False
                eff_r = max_range
            else:
                hit = True
                eff_r = r

            end_wx = pose_x + eff_r * math.cos(angle_world)
            end_wy = pose_y + eff_r * math.sin(angle_world)
            
            end_x, end_y = self.world_to_map(end_wx, end_wy)
            
            # Trace all free cells along the ray
            for cx, cy in self.bresenham(start_x, start_y, end_x, end_y):
                if 0 <= cx < self.width and 0 <= cy < self.height:
                    # If we haven't marked it occupied before, mark as free
                    if self.grid[cy, cx] != 100:
                        self.grid[cy, cx] = 0
            
            # Mark the hit cell as occupied
            if hit and 0 <= end_x < self.width and 0 <= end_y < self.height:
                self.grid[end_y, end_x] = 100

    def export(self, file_prefix: str):
        """Export map to .pgm and .yaml"""
        pgm_file = f"{file_prefix}.pgm"
        yaml_file = f"{file_prefix}.yaml"

        # Export PGM file
        # Unkonwn -> 205, Free -> 254, Occupied -> 0
        img = np.zeros((self.height, self.width), dtype=np.uint8)
        img[self.grid == -1] = 205
        img[self.grid == 0] = 254
        img[self.grid == 100] = 0
        
        # PGM expects top-left origin. NumPy array is row, col. Flip Y.
        img_flipped = np.flipud(img)

        with open(pgm_file, 'wb') as f:
            header = f"P5\n{self.width} {self.height}\n255\n"
            f.write(header.encode('ascii'))
            f.write(img_flipped.tobytes())
            
        print(f"Exported PGM to {pgm_file}")

        # Export YAML
        yaml_content = f"""image: {file_prefix}.pgm
resolution: {self.resolution}
origin: [{self.origin_x}, {self.origin_y}, 0.0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
"""
        with open(yaml_file, 'w') as f:
            f.write(yaml_content)
        print(f"Exported YAML to {yaml_file}")

if __name__ == '__main__':
    # Test builder
    mb = MapBuilder()
    rad = np.linspace(-math.pi, math.pi, 50)
    ranges = np.ones(50) * 2.0
    mb.update_scan(0, 0, 0, rad, ranges)
    mb.export('test_map')
