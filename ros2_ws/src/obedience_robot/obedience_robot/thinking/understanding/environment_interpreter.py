"""
Environment Interpreter - Sensor Data Interpretation

Interprets raw sensor data into meaningful environmental understanding.
"""

from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import numpy as np
from ..common import FaultState


@dataclass
class EnvironmentState:
    """Interpreted environment state."""
    robot_stable: bool = True
    path_clear: bool = True
    at_target: bool = False
    target_direction: Optional[float] = None  # radians
    nearest_obstacle_distance: float = float('inf')
    terrain_condition: str = "normal"
    interpretation: str = ""


class EnvironmentInterpreter:
    """
    Interprets sensor data into environment understanding.
    
    Provides high-level interpretation of:
    - Stability (from IMU)
    - Navigation (from position/target)
    - Obstacles (from LiDAR)
    - Energy state (from battery)
    """
    
    # Thresholds
    STABLE_TILT_THRESHOLD = 0.3  # radians
    AT_TARGET_THRESHOLD = 0.4  # meters
    OBSTACLE_WARNING_DISTANCE = 0.5  # meters
    
    def __init__(self):
        """Initialize interpreter."""
        self.state = EnvironmentState()
        
        # Zone definitions (from hospital scene)
        self.zones = {
            "charging": (np.array([3.5, -4.5]), 0.5),
            "supply": (np.array([2.0, 0.0]), 0.6),
            "bed_1": (np.array([-3.0, -1.0]), 0.4),
            "bed_2": (np.array([-5.0, -1.0]), 0.4),
            "bed_3": (np.array([-5.0, 1.0]), 0.4),
        }
    
    def interpret(self, robot_pos: np.ndarray, robot_heading: float,
                  target_pos: Optional[np.ndarray], imu_tilt: float,
                  battery_level: float, obstacle_distance: float) -> EnvironmentState:
        """
        Interpret current environment state.
        
        Args:
            robot_pos: Robot position [x, y] or [x, y, z]
            robot_heading: Robot heading angle (radians)
            target_pos: Current navigation target [x, y]
            imu_tilt: Combined tilt angle (radians)
            battery_level: Battery state of charge (0.0 - 1.0)
            obstacle_distance: Distance to nearest obstacle (meters)
            
        Returns:
            Interpreted environment state
        """
        robot_2d = robot_pos[:2] if len(robot_pos) > 2 else robot_pos
        
        # Stability check
        self.state.robot_stable = imu_tilt < self.STABLE_TILT_THRESHOLD
        
        # Obstacle check
        self.state.path_clear = obstacle_distance > self.OBSTACLE_WARNING_DISTANCE
        self.state.nearest_obstacle_distance = obstacle_distance
        
        # Target check
        if target_pos is not None:
            distance = np.linalg.norm(target_pos - robot_2d)
            self.state.at_target = distance < self.AT_TARGET_THRESHOLD
            
            # Direction to target
            to_target = target_pos - robot_2d
            self.state.target_direction = np.arctan2(to_target[1], to_target[0]) - robot_heading
        else:
            self.state.at_target = False
            self.state.target_direction = None
        
        # Terrain condition (inferred from tilt)
        if imu_tilt < 0.1:
            self.state.terrain_condition = "flat"
        elif imu_tilt < 0.2:
            self.state.terrain_condition = "uneven"
        else:
            self.state.terrain_condition = "rough"
        
        # Generate interpretation
        self.state.interpretation = self._generate_interpretation(battery_level)
        
        return self.state
    
    def _generate_interpretation(self, battery_level: float) -> str:
        """Generate human-readable interpretation."""
        parts = []
        
        # Stability
        if not self.state.robot_stable:
            parts.append("Robot tilted - unstable")
        
        # Path
        if not self.state.path_clear:
            parts.append(f"Obstacle at {self.state.nearest_obstacle_distance:.1f}m")
        
        # Target
        if self.state.at_target:
            parts.append("At target location")
        elif self.state.target_direction is not None:
            dir_deg = np.degrees(self.state.target_direction)
            if abs(dir_deg) < 10:
                parts.append("Target ahead")
            elif dir_deg > 0:
                parts.append(f"Target {dir_deg:.0f}° left")
            else:
                parts.append(f"Target {-dir_deg:.0f}° right")
        
        # Battery
        if battery_level < 0.10:
            parts.append("CRITICAL BATTERY")
        elif battery_level < 0.20:
            parts.append("Low battery")
        
        if not parts:
            parts.append("All clear")
        
        return "; ".join(parts)
    
    def interpret_environment(self, battery_level: float, imu_data: Dict) -> str:
        """
        Simplified interface matching Thought System diagram.
        
        Args:
            battery_level: 0.0 - 1.0
            imu_data: Dictionary with pitch, roll
            
        Returns:
            Interpretation string
        """
        tilt = np.sqrt(
            imu_data.get("pitch", 0)**2 + 
            imu_data.get("roll", 0)**2
        )
        
        # Simple interpretation without full context
        parts = []
        
        if battery_level < 0.10:
            parts.append("CRITICAL: Battery depleted")
        elif battery_level < 0.20:
            parts.append("WARNING: Battery low")
        else:
            parts.append(f"Battery: {battery_level*100:.0f}%")
        
        if tilt > 0.3:
            parts.append("UNSTABLE")
        elif tilt > 0.15:
            parts.append("Tilted")
        else:
            parts.append("Stable")
        
        return " | ".join(parts)
    
    def get_current_zone(self, robot_pos: np.ndarray) -> Optional[str]:
        """Determine which zone robot is in."""
        robot_2d = robot_pos[:2] if len(robot_pos) > 2 else robot_pos
        
        for zone_name, (zone_pos, zone_radius) in self.zones.items():
            distance = np.linalg.norm(zone_pos - robot_2d)
            if distance < zone_radius:
                return zone_name
        
        return None
