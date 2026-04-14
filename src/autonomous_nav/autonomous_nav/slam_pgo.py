#!/usr/bin/env python3
"""
slam_pgo.py — Pose Graph SLAM (PGO) implementation.

Constructs a graph of robot poses and landmarks.
Edges:
    - Odometry: dx, dy, dtheta between consecutive poses.
    - Landmark: r, bearing to a specific landmark ID.

An optimization step runs nonlinear least squares (via SciPy) to minimize
the sum of squared residuals, effectively distributing odometry drift
evenly across the trajectory when a loop closure (repeated landmark) occurs.

This class is pure logic.
"""

import math
import numpy as np
from scipy.optimize import least_squares
from typing import List, Tuple, Dict

# ============================================================
# UTILITY
# ============================================================
def normalize_angle(a: float) -> float:
    while a > math.pi: a -= 2 * math.pi
    while a < -math.pi: a += 2 * math.pi
    return a

# ============================================================
# MAIN CLASS
# ============================================================
class PoseGraph:
    def __init__(self):
        # State:
        # Poses: list of [x, y, yaw]
        self.poses: List[np.ndarray] = []
        
        # Landmarks: dict mapping landmark_id -> [x, y]
        self.landmarks: Dict[int, np.ndarray] = {}
        
        # Edges
        # Odom: (pose_idx_from, pose_idx_to, dx, dy, dyaw, info_matrix)
        self.odom_edges = []
        
        # Landmark: (pose_idx, landmark_id, r, bearing, info_matrix)
        self.land_edges = []
        
    def add_pose(self, x: float, y: float, yaw: float) -> int:
        """Add a pose vertex. Returns the index of the pose."""
        idx = len(self.poses)
        self.poses.append(np.array([x, y, yaw], dtype=float))
        return idx

    def add_landmark(self, lm_id: int, x: float, y: float):
        """Add or update a landmark vertex."""
        self.landmarks[lm_id] = np.array([x, y], dtype=float)

    def add_odometry_edge(self, p_from: int, p_to: int, dx: float, dy: float, dyaw: float, info=None):
        """
        Add odometry measurement between p_from and p_to.
        dx, dy, dyaw are in the local frame of p_from.
        """
        if info is None:
            info = np.eye(3) # Default identity info matrix
        self.odom_edges.append((p_from, p_to, dx, dy, normalize_angle(dyaw), info))

    def add_landmark_edge(self, p_idx: int, lm_id: int, r: float, bearing: float, info=None):
        """
        Add a landmark observation edge from p_idx to lm_id.
        """
        if info is None:
            info = np.eye(2)
        self.land_edges.append((p_idx, lm_id, r, normalize_angle(bearing), info))

    def _state_to_vector(self) -> np.ndarray:
        """Flatten poses and landmarks into a 1D vector."""
        vec = []
        for p in self.poses:
            vec.extend(p)
        for lm_id in sorted(self.landmarks.keys()):
            vec.extend(self.landmarks[lm_id])
        return np.array(vec)

    def _vector_to_state(self, vec: np.ndarray):
        """Update internal poses and landmarks from 1D vector."""
        idx = 0
        for i in range(len(self.poses)):
            self.poses[i] = vec[idx:idx+3]
            self.poses[i][2] = normalize_angle(self.poses[i][2])
            idx += 3
        for lm_id in sorted(self.landmarks.keys()):
            self.landmarks[lm_id] = vec[idx:idx+2]
            idx += 2

    def _compute_residuals(self, vec: np.ndarray) -> np.ndarray:
        """Compute the error residuals for all edges."""
        # Unpack state temporarily
        poses = []
        landmarks = {}
        idx = 0
        for _ in range(len(self.poses)):
            poses.append(vec[idx:idx+3])
            idx += 3
        for lm_id in sorted(self.landmarks.keys()):
            landmarks[lm_id] = vec[idx:idx+2]
            idx += 2

        residuals = []
        
        # Odom residuals
        for p_from, p_to, dx, dy, dyaw, info in self.odom_edges:
            xf, yf, thf = poses[p_from]
            xt, yt, tht = poses[p_to]
            
            # Predict t in frame of f
            cos_th = math.cos(thf)
            sin_th = math.sin(thf)
            
            p_dx = (xt - xf) * cos_th + (yt - yf) * sin_th
            p_dy = -(xt - xf) * sin_th + (yt - yf) * cos_th
            p_dth = normalize_angle(tht - thf)
            
            # Error
            e_x = p_dx - dx
            e_y = p_dy - dy
            e_th = normalize_angle(p_dth - dyaw)
            
            # Incorporate info matrix (sqrt)
            W = np.linalg.cholesky(info)
            err = W @ np.array([e_x, e_y, e_th])
            residuals.extend(err)

        # Landmark residuals
        for p_idx, lm_id, obs_r, obs_b, info in self.land_edges:
            xr, yr, thr = poses[p_idx]
            xl, yl = landmarks[lm_id]
            
            p_r = math.sqrt((xl - xr)**2 + (yl - yr)**2)
            p_b = normalize_angle(math.atan2(yl - yr, xl - xr) - thr)
            
            e_r = p_r - obs_r
            e_b = normalize_angle(p_b - obs_b)
            
            W = np.linalg.cholesky(info)
            err = W @ np.array([e_r, e_b])
            residuals.extend(err)
            
        # Add constraint to keep the first pose anchored (prior)
        if len(poses) > 0:
            residuals.extend((poses[0] - self.poses[0]) * 1e6)

        return np.array(residuals)

    def optimize(self):
        """Run nonlinear least squares to update the graph state."""
        if not self.odom_edges and not self.land_edges:
            return # nothing to do
        
        initial_guess = self._state_to_vector()
        result = least_squares(self._compute_residuals, initial_guess, method='lm')
        self._vector_to_state(result.x)
        
    def get_latest_pose(self) -> Tuple[float, float, float]:
        if not self.poses:
            return (0.0, 0.0, 0.0)
        return tuple(self.poses[-1])

# ============================================================
# STANDALONE TEST
# ============================================================
if __name__ == "__main__":
    pgo = PoseGraph()
    # P0
    pgo.add_pose(0.0, 0.0, 0.0)
    # P1
    pgo.add_pose(1.0, 0.1, 0.0) # slightly off Y
    # LM1 observed from P0
    pgo.add_landmark(1, 1.0, 1.0)
    
    # Odom edge P0 -> P1: x=1, y=0, th=0
    pgo.add_odometry_edge(0, 1, 1.0, 0.0, 0.0)
    
    # Observe LM1 from P0: r=1.414, b=pi/4
    pgo.add_landmark_edge(0, 1, math.sqrt(2), math.pi/4)
    
    # Observe LM1 from P1: r=1.0, b=pi/2
    pgo.add_landmark_edge(1, 1, 1.0, math.pi/2)
    
    print("Before Optimize:")
    print("P1:", pgo.poses[1])
    print("LM1:", pgo.landmarks[1])
    
    pgo.optimize()
    
    print("\nAfter Optimize:")
    print("P1:", pgo.poses[1])
    print("LM1:", pgo.landmarks[1])
