"""
LiDAR Processor - Obstacle Detection

Monitors LiDAR sensor for:
- Obstruction detection (blocked sensor)
- Obstacle proximity warnings
- Environment mapping support

Fault Modes:
- Obstruction: Sensor blocked (constant close reading)
- No return: No obstacles in range (possible malfunction if unexpected)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple
from collections import deque
from ..common import FaultState, SensorStatus, SensorType


@dataclass
class LiDARConfig:
    """LiDAR processor configuration."""
    # Range limits
    min_range: float = 0.1  # meters
    max_range: float = 10.0  # meters
    
    # Obstruction detection
    obstruction_threshold: float = 0.15  # meters - too close = blocked
    obstruction_samples: int = 10  # consecutive samples to confirm
    
    # Obstacle warning
    warning_distance: float = 0.5  # meters
    critical_distance: float = 0.2  # meters
    
    # Number of rays (simulated)
    num_rays: int = 36  # 10 degree resolution


@dataclass
class LiDARScan:
    """Single LiDAR scan data."""
    ranges: np.ndarray  # Distance per ray
    angles: np.ndarray  # Angle per ray (radians)
    timestamp: float = 0.0
    
    def get_closest(self) -> Tuple[float, float]:
        """Get closest obstacle distance and angle."""
        valid = self.ranges > 0
        if not np.any(valid):
            return float('inf'), 0.0
        
        valid_ranges = self.ranges[valid]
        valid_angles = self.angles[valid]
        min_idx = np.argmin(valid_ranges)
        return valid_ranges[min_idx], valid_angles[min_idx]


class LiDARProcessor:
    """
    LiDAR sensor processor with fault detection.
    
    Simulates a simple 2D LiDAR for obstacle detection.
    """
    
    def __init__(self, config: Optional[LiDARConfig] = None):
        """Initialize LiDAR processor."""
        self.config = config or LiDARConfig()
        
        # Status
        self.status = SensorStatus(
            sensor_type=SensorType.LIDAR,
            name="lidar",
            fault_state=FaultState.NONE
        )
        
        # Current scan
        angles = np.linspace(0, 2*np.pi, self.config.num_rays, endpoint=False)
        self.scan = LiDARScan(
            ranges=np.full(self.config.num_rays, self.config.max_range),
            angles=angles
        )
        
        # Obstruction detection history
        self._close_readings: deque = deque(maxlen=self.config.obstruction_samples)
        
        # Obstacle tracking
        self.closest_obstacle: float = float('inf')
        self.closest_angle: float = 0.0
        
        # Injected fault
        self._injected_fault: Optional[str] = None
    
    def update(self, ranges: np.ndarray, timestamp: float) -> FaultState:
        """
        Update LiDAR data and check for faults.
        
        Args:
            ranges: Distance measurements per ray
            timestamp: Current simulation time
            
        Returns:
            Current fault state
        """
        # Apply injected fault
        if self._injected_fault == "obstruction":
            ranges = np.full_like(ranges, 0.1)  # Blocked sensor
        elif self._injected_fault == "no_return":
            ranges = np.full_like(ranges, self.config.max_range)  # No obstacles
        
        # Update scan
        self.scan.ranges = ranges.copy()
        self.scan.timestamp = timestamp
        self.status.last_update = timestamp
        
        # Get closest obstacle
        self.closest_obstacle, self.closest_angle = self.scan.get_closest()
        self.status.last_value = self.closest_obstacle
        
        # Check for obstruction
        min_range = np.min(ranges[ranges > 0]) if np.any(ranges > 0) else float('inf')
        self._close_readings.append(min_range < self.config.obstruction_threshold)
        
        # Determine fault state
        self.status.fault_state = self._check_faults()
        
        return self.status.fault_state
    
    def _check_faults(self) -> FaultState:
        """Check for LiDAR faults."""
        # Check for obstruction (all recent readings very close)
        if len(self._close_readings) >= self.config.obstruction_samples:
            if all(self._close_readings):
                return FaultState.TRUE  # Sensor likely blocked
        
        return FaultState.FALSE
    
    def get_obstacle_warning(self) -> Tuple[FaultState, float]:
        """
        Get obstacle proximity warning.
        
        Returns:
            (warning_state, distance_to_obstacle)
        """
        if self.closest_obstacle < self.config.critical_distance:
            return FaultState.TRUE, self.closest_obstacle
        elif self.closest_obstacle < self.config.warning_distance:
            return FaultState.SUSPECT, self.closest_obstacle
        return FaultState.FALSE, self.closest_obstacle
    
    def get_clear_direction(self) -> Optional[float]:
        """Find clearest direction to navigate."""
        if np.all(self.scan.ranges < self.config.warning_distance):
            return None  # No clear path
        
        # Find direction with maximum clearance
        max_idx = np.argmax(self.scan.ranges)
        return self.scan.angles[max_idx]
    
    def inject_fault(self, fault_type: str):
        """Inject fault (obstruction, no_return)."""
        self._injected_fault = fault_type
    
    def clear_fault(self):
        """Clear injected fault."""
        self._injected_fault = None
    
    def simulate_from_obstacles(self, robot_pos: np.ndarray, robot_heading: float,
                                obstacles: List[Tuple[np.ndarray, float]]) -> np.ndarray:
        """
        Simulate LiDAR readings from known obstacles.
        
        Args:
            robot_pos: Robot position [x, y]
            robot_heading: Robot heading angle
            obstacles: List of (position, radius) tuples
            
        Returns:
            Simulated range measurements
        """
        ranges = np.full(self.config.num_rays, self.config.max_range)
        
        for i, angle in enumerate(self.scan.angles):
            # Ray direction in world frame
            world_angle = robot_heading + angle
            ray_dir = np.array([np.cos(world_angle), np.sin(world_angle)])
            
            # Check intersection with each obstacle
            for obs_pos, obs_radius in obstacles:
                # Vector from robot to obstacle center
                to_obs = obs_pos[:2] - robot_pos[:2]
                
                # Project onto ray
                proj = np.dot(to_obs, ray_dir)
                if proj < 0:
                    continue  # Behind robot
                
                # Perpendicular distance
                perp_dist = np.abs(np.cross(ray_dir, to_obs))
                
                if perp_dist < obs_radius:
                    # Ray intersects obstacle
                    dist = proj - np.sqrt(obs_radius**2 - perp_dist**2)
                    if dist > 0 and dist < ranges[i]:
                        ranges[i] = dist
        
        return np.clip(ranges, self.config.min_range, self.config.max_range)
