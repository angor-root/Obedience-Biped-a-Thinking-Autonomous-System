"""
Mission Planner for Medicine Delivery Robot.

This module implements a simple fixed-order mission planner for
the first test: go to each bed and return to charging station.

Mission Flow:
    CHARGING_STATION → BED_1 → BED_2 → BED_3 → CHARGING_STATION
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional, Tuple


class MissionState(Enum):
    """Mission execution states."""
    IDLE = auto()
    NAVIGATING = auto()
    DELIVERING = auto()
    RETURNING = auto()
    CHARGING = auto()
    COMPLETED = auto()
    ABORTED = auto()


class WaypointType(Enum):
    """Types of waypoints in the hospital."""
    CHARGING_STATION = "charging"
    PATIENT_BED = "bed"
    CORRIDOR = "corridor"


@dataclass
class Waypoint:
    """A navigation waypoint."""
    name: str
    position: np.ndarray  # [x, y] in world frame
    waypoint_type: WaypointType
    delivery_time: float = 0.0  # Time to spend at waypoint (s)
    
    def distance_to(self, other_pos: np.ndarray) -> float:
        """Compute distance to another position."""
        return np.linalg.norm(self.position - other_pos[:2])


@dataclass 
class MissionConfig:
    """Mission configuration parameters."""
    # Delivery time at each bed (seconds)
    delivery_duration: float = 5.0
    
    # Navigation parameters
    waypoint_tolerance: float = 0.3  # meters
    
    # Battery constraints
    min_battery_to_start: float = 0.25  # 25%
    return_battery_threshold: float = 0.20  # 20%


class MissionPlanner:
    """
    Simple fixed-order mission planner for medicine delivery.
    
    First test mission:
        1. Start at charging station
        2. Go to Bed 1, deliver (2s pause)
        3. Go to Bed 2, deliver (2s pause)
        4. Go to Bed 3, deliver (2s pause)
        5. Return to charging station
    """
    
    def __init__(self, config: Optional[MissionConfig] = None):
        """Initialize mission planner."""
        self.config = config or MissionConfig()
        
        # Define hospital waypoints (based on hospital_scene.xml)
        self.waypoints = self._create_hospital_waypoints()
        
        # Mission state
        self._state = MissionState.IDLE
        self._current_waypoint_idx = 0
        self._delivery_timer = 0.0
        self._mission_sequence: List[int] = []
        
        # Status tracking
        self._deliveries_completed = 0
        self._mission_start_time = 0.0
        self._total_distance = 0.0
        
    def _create_hospital_waypoints(self) -> List[Waypoint]:
        """Create waypoints based on hospital scene layout."""
        waypoints = [
            # Charging station (southeast corner)
            Waypoint(
                name="charging_station",
                position=np.array([3.5, -4.5]),
                waypoint_type=WaypointType.CHARGING_STATION,
                delivery_time=0.0
            ),
            # Bed 1 delivery point (bed at -3,-2, waypoint at 0,+1 relative)
            Waypoint(
                name="bed_1",
                position=np.array([-3.0, -1.0]),
                waypoint_type=WaypointType.PATIENT_BED,
                delivery_time=self.config.delivery_duration
            ),
            # Bed 2 delivery point (bed at -5,-2, waypoint at 0,+1 relative)
            Waypoint(
                name="bed_2", 
                position=np.array([-5.0, -1.0]),
                waypoint_type=WaypointType.PATIENT_BED,
                delivery_time=self.config.delivery_duration
            ),
            # Bed 3 delivery point (bed at -5,+2, waypoint at 0,-1 relative)
            Waypoint(
                name="bed_3",
                position=np.array([-5.0, 1.0]),
                waypoint_type=WaypointType.PATIENT_BED,
                delivery_time=self.config.delivery_duration
            ),
        ]
        return waypoints
    
    @property
    def state(self) -> MissionState:
        """Current mission state."""
        return self._state
    
    @property
    def current_target(self) -> Optional[Waypoint]:
        """Current target waypoint."""
        if self._current_waypoint_idx < len(self._mission_sequence):
            idx = self._mission_sequence[self._current_waypoint_idx]
            return self.waypoints[idx]
        return None
    
    @property
    def progress(self) -> float:
        """Mission progress (0.0 - 1.0)."""
        if not self._mission_sequence:
            return 0.0
        return self._current_waypoint_idx / len(self._mission_sequence)
    
    def start_delivery_mission(self, current_time: float) -> bool:
        """
        Start the medicine delivery mission.
        
        Args:
            current_time: Current simulation time
            
        Returns:
            True if mission started successfully
        """
        # Define mission sequence: charging → bed1 → bed2 → bed3 → charging
        self._mission_sequence = [0, 1, 2, 3, 0]  # indices into waypoints
        self._current_waypoint_idx = 1  # Start navigating to bed 1
        self._state = MissionState.NAVIGATING
        self._mission_start_time = current_time
        self._deliveries_completed = 0
        
        print(f"[MISSION] Started delivery mission at t={current_time:.1f}s")
        print(f"[MISSION] Sequence: {[self.waypoints[i].name for i in self._mission_sequence]}")
        
        return True
    
    def update(self, robot_position: np.ndarray, current_time: float, dt: float) -> Tuple[MissionState, Optional[np.ndarray]]:
        """
        Update mission state and get navigation target.
        
        Args:
            robot_position: Current robot position [x, y, z]
            current_time: Current simulation time
            dt: Time step
            
        Returns:
            Tuple of (mission_state, target_position or None if arrived)
        """
        if self._state == MissionState.IDLE:
            return self._state, None
        
        if self._state == MissionState.COMPLETED:
            return self._state, None
        
        target = self.current_target
        if target is None:
            self._state = MissionState.COMPLETED
            return self._state, None
        
        # Check if arrived at waypoint
        distance = target.distance_to(robot_position)
        
        if distance < self.config.waypoint_tolerance:
            # Arrived at waypoint
            return self._handle_arrival(target, current_time, dt)
        
        # Still navigating
        return self._state, target.position
    
    def _handle_arrival(self, waypoint: Waypoint, current_time: float, dt: float) -> Tuple[MissionState, Optional[np.ndarray]]:
        """Handle arrival at a waypoint."""
        
        if waypoint.waypoint_type == WaypointType.PATIENT_BED:
            # At patient bed - delivering
            if self._state != MissionState.DELIVERING:
                self._state = MissionState.DELIVERING
                self._delivery_timer = waypoint.delivery_time
                print(f"[MISSION] Arrived at {waypoint.name}, delivering medicine...")
            
            # Count down delivery time
            self._delivery_timer -= dt
            
            if self._delivery_timer <= 0:
                # Delivery complete
                self._deliveries_completed += 1
                print(f"[MISSION] Delivery complete at {waypoint.name} ({self._deliveries_completed}/3)")
                self._advance_to_next_waypoint()
            
            return self._state, None
        
        elif waypoint.waypoint_type == WaypointType.CHARGING_STATION:
            # Back at charging station
            if self._current_waypoint_idx >= len(self._mission_sequence) - 1:
                # Mission complete
                self._state = MissionState.COMPLETED
                elapsed = current_time - self._mission_start_time
                print(f"[MISSION] Mission COMPLETED in {elapsed:.1f}s")
                print(f"[MISSION] Deliveries: {self._deliveries_completed}")
                return self._state, None
            else:
                # Shouldn't happen in normal flow
                self._advance_to_next_waypoint()
                return self._state, self.current_target.position if self.current_target else None
        
        return self._state, None
    
    def _advance_to_next_waypoint(self):
        """Advance to the next waypoint in sequence."""
        self._current_waypoint_idx += 1
        
        if self._current_waypoint_idx >= len(self._mission_sequence):
            self._state = MissionState.COMPLETED
        else:
            next_wp = self.current_target
            if next_wp:
                if next_wp.waypoint_type == WaypointType.CHARGING_STATION:
                    self._state = MissionState.RETURNING
                    print(f"[MISSION] Returning to charging station...")
                else:
                    self._state = MissionState.NAVIGATING
                    print(f"[MISSION] Navigating to {next_wp.name}...")
    
    def abort_mission(self, reason: str = ""):
        """Abort current mission."""
        self._state = MissionState.ABORTED
        print(f"[MISSION] ABORTED: {reason}")
    
    def get_status(self) -> dict:
        """Get comprehensive mission status."""
        return {
            "state": self._state.name,
            "current_target": self.current_target.name if self.current_target else None,
            "progress": self.progress,
            "deliveries_completed": self._deliveries_completed,
            "waypoint_index": self._current_waypoint_idx,
            "total_waypoints": len(self._mission_sequence),
        }
    
    def compute_velocity_command(self, robot_position: np.ndarray, 
                                  robot_heading: float,
                                  target_position: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute velocity commands to navigate to target.
        
        Args:
            robot_position: Current [x, y, z]
            robot_heading: Current heading angle (rad)
            target_position: Target [x, y]
            
        Returns:
            Tuple of (forward_vel, lateral_vel, turn_rate)
        """
        # Vector to target
        delta = target_position - robot_position[:2]
        distance = np.linalg.norm(delta)
        
        if distance < 0.1:
            return 0.0, 0.0, 0.0
        
        # Desired heading
        target_heading = np.arctan2(delta[1], delta[0])
        
        # Heading error
        heading_error = target_heading - robot_heading
        # Normalize to [-pi, pi]
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        # Simple proportional control
        turn_rate = np.clip(1.5 * heading_error, -1.0, 1.0)
        
        # Forward velocity (reduce when turning)
        forward_vel = 0.4 * (1.0 - abs(heading_error) / np.pi)
        forward_vel = np.clip(forward_vel, 0.1, 0.45)
        
        # Lateral velocity (not used in simple navigation)
        lateral_vel = 0.0
        
        return forward_vel, lateral_vel, turn_rate
