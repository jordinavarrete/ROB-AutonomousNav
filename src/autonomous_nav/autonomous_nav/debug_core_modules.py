#!/usr/bin/env python3
"""
debug_core_modules.py - consolidated smoke tests for core logic modules.

Checks:
  1) slam_pgo.PoseGraph optimization step behaves consistently
  2) map_builder.MapBuilder updates occupancy and exports files
  3) dynamic_astar.DynamicAStar plans and detects invalidated paths

Usage:
  ros2 run autonomous_nav debug_core_modules
"""

import math
import tempfile
from pathlib import Path

import numpy as np

from autonomous_nav.slam_pgo import PoseGraph
from autonomous_nav.map_builder import MapBuilder
from autonomous_nav.dynamic_astar import DynamicAStar


def _check_pose_graph() -> tuple[bool, str]:
    pgo = PoseGraph()

    pgo.add_pose(0.0, 0.0, 0.0)
    pgo.add_pose(1.0, 0.25, 0.10)
    pgo.add_landmark(1, 1.0, 1.0)

    pgo.add_odometry_edge(0, 1, 1.0, 0.0, 0.0)
    pgo.add_landmark_edge(0, 1, math.sqrt(2.0), math.pi / 4.0)
    pgo.add_landmark_edge(1, 1, 1.0, math.pi / 2.0)

    before = pgo.poses[1].copy()
    pgo.optimize()
    after = pgo.poses[1].copy()

    moved = float(np.linalg.norm(after - before))
    if not np.isfinite(moved):
        return False, 'slam_pgo: optimized pose is not finite'

    if moved < 1e-6:
        return False, 'slam_pgo: optimization had no effect in a constrained toy case'

    return True, f'slam_pgo: OK (pose1 moved {moved:.4f})'


def _check_map_builder() -> tuple[bool, str]:
    mb = MapBuilder(resolution=0.05, origin_x=-2.0, origin_y=-2.0, width_px=120, height_px=120)

    angles = np.linspace(-math.pi / 2.0, math.pi / 2.0, 90)
    ranges = np.ones(90, dtype=float) * 1.2
    mb.update_scan(0.0, 0.0, 0.0, angles, ranges, max_range=2.0)

    occupied = int(np.sum(mb.grid == 100))
    free = int(np.sum(mb.grid == 0))

    if occupied == 0 or free == 0:
        return False, f'map_builder: unexpected cell counts occupied={occupied}, free={free}'

    with tempfile.TemporaryDirectory(prefix='autonav_maptest_') as tmp:
        prefix = Path(tmp) / 'smoke_map'
        mb.export(str(prefix))
        pgm = prefix.with_suffix('.pgm')
        yaml = prefix.with_suffix('.yaml')
        if not pgm.exists() or not yaml.exists():
            return False, 'map_builder: export files were not created'

    return True, f'map_builder: OK (occupied={occupied}, free={free})'


def _check_dynamic_astar() -> tuple[bool, str]:
    planner = DynamicAStar()
    start = (0.0, 0.0)
    goal = (3.0, 0.0)

    path = planner.plan(start[0], start[1], goal[0], goal[1])
    if not path or len(path) < 2:
        return False, 'dynamic_astar: failed to plan a baseline path'

    if not planner.is_path_valid():
        return False, 'dynamic_astar: baseline path unexpectedly invalid'

    blocking_points = path[1:-1]
    if not blocking_points:
        blocking_points = [path[len(path) // 2]]

    changed = planner.update_obstacles(blocking_points)
    if not changed:
        return False, 'dynamic_astar: obstacle update reported no changes'

    if planner.is_path_valid():
        return False, 'dynamic_astar: path should be invalid after obstacle insertion'

    new_path = planner.plan(start[0], start[1], goal[0], goal[1])
    if not new_path or len(new_path) < 2:
        return False, 'dynamic_astar: failed to replan after invalidation'

    return True, f'dynamic_astar: OK (path points {len(path)} -> {len(new_path)})'


def main() -> None:
    checks = [_check_pose_graph, _check_map_builder, _check_dynamic_astar]

    print('=== CORE MODULES SMOKE TEST ===')
    passed = 0
    for fn in checks:
        ok, msg = fn()
        if ok:
            passed += 1
            print(f'[PASS] {msg}')
        else:
            print(f'[FAIL] {msg}')

    total = len(checks)
    print(f'=== RESULT: {passed}/{total} passed ===')
    if passed != total:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
