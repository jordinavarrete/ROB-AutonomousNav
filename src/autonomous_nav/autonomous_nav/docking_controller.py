#!/usr/bin/env python3
"""
docking_controller.py — Precision docking logic module for Phase III.

Once the station coordinates are locked in Phase II, this controller
handles the high-precision maneuver to park exactly in the middle 
without triggering standard obstacle avoidance inflation (to prevent false 
positive collision blocks against the 4 pillars).

This class is pure logic.
"""

import math
from autonomous_nav.navigation import Pose2D, VelocityCommand, normalize_angle, clamp

class DockingConfig:
    LINEAR_SPEED = 0.08      # Very slow and precise
    MAX_ANGULAR = 0.2
    KP_LINEAR = 0.4
    KP_ANGULAR = 0.8
    ALIGN_THRESHOLD = 0.05   # radians before moving forward
    STOP_DISTANCE = 0.03     # Stop 3 cm from center

class DockState:
    IDLE = 0
    ALIGNING = 1
    APPROACHING = 2
    DOCKED = 3

class DockingController:
    def __init__(self, logger=None):
        self._log = logger
        self.state = DockState.IDLE
        self.target_x = None
        self.target_y = None

    def start_docking(self, station_x: float, station_y: float):
        """Invoke this to begin the docking sequence."""
        self.target_x = station_x
        self.target_y = station_y
        self.state = DockState.ALIGNING
        if self._log:
            self._log.info(f"[DOCKING] Initiated precision docking towards ({station_x:.3f}, {station_y:.3f})")

    def update_target(self, station_x: float, station_y: float):
        """Update docking target without resetting the state machine."""
        self.target_x = station_x
        self.target_y = station_y

    def step(self, current_x: float, current_y: float, current_yaw: float) -> VelocityCommand:
        """
        Call this periodically at 20Hz instead of default WaypointNavigator.
        """
        if self.state == DockState.IDLE or self.state == DockState.DOCKED:
            return VelocityCommand(0.0, 0.0)

        dist = math.hypot(self.target_x - current_x, self.target_y - current_y)
        desired_yaw = math.atan2(self.target_y - current_y, self.target_x - current_x)
        angle_error = normalize_angle(desired_yaw - current_yaw)

        if self.state == DockState.ALIGNING:
            if abs(angle_error) < DockingConfig.ALIGN_THRESHOLD:
                self.state = DockState.APPROACHING
                return VelocityCommand(0.0, 0.0)
            
            w = clamp(DockingConfig.KP_ANGULAR * angle_error, -DockingConfig.MAX_ANGULAR, DockingConfig.MAX_ANGULAR)
            return VelocityCommand(0.0, w)

        if self.state == DockState.APPROACHING:
            if dist < DockingConfig.STOP_DISTANCE:
                self.state = DockState.DOCKED
                if self._log:
                    self._log.info("[DOCKING] Successfully parked at station center!")
                return VelocityCommand(0.0, 0.0)
            
            # If we drift too much while approaching, stop and realign
            if abs(angle_error) > math.radians(20):
                self.state = DockState.ALIGNING
                return VelocityCommand(0.0, 0.0)

            v = clamp(DockingConfig.KP_LINEAR * dist, 0.02, DockingConfig.LINEAR_SPEED)
            w = clamp(DockingConfig.KP_ANGULAR * angle_error, -DockingConfig.MAX_ANGULAR, DockingConfig.MAX_ANGULAR)
            return VelocityCommand(v, w)

        return VelocityCommand(0.0, 0.0)

    def is_docked(self):
        return self.state == DockState.DOCKED