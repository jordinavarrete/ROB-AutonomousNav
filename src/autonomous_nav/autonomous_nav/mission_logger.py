#!/usr/bin/env python3
"""
mission_logger.py — CSV log writer for the autonomous navigation mission.

Writes one row per second to ~/mission_log.csv with:
    timestamp, phase, robot_x, robot_y, robot_yaw,
    n_obstacles_detected, station_x, station_y

Usage:
    logger = MissionLogger()
    logger.update(phase='I', robot_x=1.0, robot_y=2.0, robot_yaw=0.5,
                  n_obstacles=3, station_x=-1.0, station_y=-1.0)
    logger.close()  # must be called on shutdown
"""

# ============================================================
# CONFIGURATION
# ============================================================
class Config:
    LOG_PATH        = '~/mission_log.csv'   # output file path (~ expanded at runtime)
    LOG_INTERVAL    = 1.0                   # seconds between rows
    STATION_UNKNOWN = -1.0                  # sentinel value before station is found
    CSV_DELIMITER   = ','
    FLOAT_PRECISION = 4                     # decimal places for coordinates


# ============================================================
# IMPORTS
# ============================================================
import csv
import math
import os
import time
import threading
from datetime import datetime
from typing import Optional


# ============================================================
# LOGGER CLASS
# ============================================================
class MissionLogger:
    """
    Periodic CSV logger for mission telemetry.

    Thread-safe: update() can be called from any ROS2 callback thread.
    A background daemon thread flushes one row every LOG_INTERVAL seconds.
    Call close() explicitly on node shutdown to finalise the file.
    """

    # CSV column headers — order must match _build_row()
    FIELDNAMES = [
        'timestamp',
        'phase',
        'robot_x',
        'robot_y',
        'robot_yaw',
        'n_obstacles_detected',
        'station_x',
        'station_y',
    ]

    def __init__(self) -> None:
        """Open the CSV file, write the header, and start the periodic flush thread."""
        self._log_path = os.path.expanduser(Config.LOG_PATH)
        self._lock = threading.Lock()
        self._running = True

        # Current telemetry snapshot — updated by update(), read by _flush_loop()
        self._state = {
            'phase':       'I',
            'robot_x':     0.0,
            'robot_y':     0.0,
            'robot_yaw':   0.0,
            'n_obstacles': 0,
            'station_x':   Config.STATION_UNKNOWN,
            'station_y':   Config.STATION_UNKNOWN,
        }

        # Open file and write header
        self._file = open(self._log_path, 'w', newline='')
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=self.FIELDNAMES,
            delimiter=Config.CSV_DELIMITER,
        )
        self._writer.writeheader()
        self._file.flush()

        # Start background flush thread
        self._thread = threading.Thread(
            target=self._flush_loop,
            name='mission_logger_flush',
            daemon=True,
        )
        self._thread.start()

        print(f'[MissionLogger] Logging to: {self._log_path}')

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def update(
        self,
        phase: str,
        robot_x: float,
        robot_y: float,
        robot_yaw: float,
        n_obstacles: int = 0,
        station_x: float = Config.STATION_UNKNOWN,
        station_y: float = Config.STATION_UNKNOWN,
    ) -> None:
        """
        Update the current telemetry snapshot.

        This method is non-blocking and thread-safe. The background thread
        will persist the latest snapshot at the next LOG_INTERVAL tick.

        Args:
            phase:       Mission phase string — 'I', 'II', or 'III'.
            robot_x:     Robot X position in map frame [m].
            robot_y:     Robot Y position in map frame [m].
            robot_yaw:   Robot heading [rad], normalised to [-π, π].
            n_obstacles: Number of obstacle clusters currently detected.
            station_x:   Station X in map frame [m]; -1.0 if not yet found.
            station_y:   Station Y in map frame [m]; -1.0 if not yet found.
        """
        with self._lock:
            self._state.update({
                'phase':       phase,
                'robot_x':     robot_x,
                'robot_y':     robot_y,
                'robot_yaw':   robot_yaw,
                'n_obstacles': n_obstacles,
                'station_x':   station_x,
                'station_y':   station_y,
            })

    def close(self) -> None:
        """
        Finalise logging: flush the last row, close the file.

        Must be called from the node's shutdown / finally block.
        """
        self._running = False
        self._thread.join(timeout=Config.LOG_INTERVAL * 2)
        self._write_row()          # write final snapshot
        self._file.flush()
        self._file.close()
        print(f'[MissionLogger] Log finalised → {self._log_path}')

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------

    def _flush_loop(self) -> None:
        """Background thread: write one CSV row every LOG_INTERVAL seconds."""
        while self._running:
            time.sleep(Config.LOG_INTERVAL)
            if self._running:   # avoid double-write on close()
                self._write_row()

    def _write_row(self) -> None:
        """Build a CSV row from the current snapshot and write it."""
        with self._lock:
            snapshot = dict(self._state)    # shallow copy under lock

        row = self._build_row(snapshot)

        try:
            self._writer.writerow(row)
            self._file.flush()
        except ValueError:
            # File already closed — can happen during shutdown race
            pass

    def _build_row(self, snapshot: dict) -> dict:
        """
        Convert a snapshot dictionary into the CSV row dictionary.

        Args:
            snapshot: Internal state dict (keys match self._state).

        Returns:
            Dict with keys matching FIELDNAMES, values formatted for CSV.
        """
        p = Config.FLOAT_PRECISION
        return {
            'timestamp':            datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
            'phase':                snapshot['phase'],
            'robot_x':              round(snapshot['robot_x'],   p),
            'robot_y':              round(snapshot['robot_y'],   p),
            'robot_yaw':            round(snapshot['robot_yaw'], p),
            'n_obstacles_detected': snapshot['n_obstacles'],
            'station_x':            round(snapshot['station_x'], p),
            'station_y':            round(snapshot['station_y'], p),
        }


# ============================================================
# STANDALONE TEST (run directly: python3 mission_logger.py)
# ============================================================
if __name__ == '__main__':
    import math

    logger = MissionLogger()
    print('Writing 5 test rows (1 s apart)…')

    test_data = [
        ('I',   1.00, 0.50, 0.00,  0, -1.0,  -1.0),
        ('I',   2.10, 1.30, 0.31,  1, -1.0,  -1.0),
        ('II',  3.50, 5.00, 1.57,  2, -1.0,  -1.0),
        ('II',  4.80, 9.20, 1.57,  0,  5.10, 12.61),
        ('III', 5.00, 11.6, 1.57,  0,  5.10, 12.61),
    ]

    for phase, x, y, yaw, obs, sx, sy in test_data:
        logger.update(
            phase=phase, robot_x=x, robot_y=y, robot_yaw=yaw,
            n_obstacles=obs, station_x=sx, station_y=sy,
        )
        time.sleep(1.1)

    logger.close()
    print('Done. Check ~/mission_log.csv')
