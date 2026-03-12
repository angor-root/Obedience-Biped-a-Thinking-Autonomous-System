"""
IMU Processor - Inertial Measurement Unit Processing

Monitors IMU data for:
- Drift detection (gradual offset in readings)
- Anomalous readings (sudden spikes)
- Sensor health

Fault Modes:
- IMU drift: Gradual offset accumulation
- Reading unavailable: No data received
- Anomalous reading: Out of expected range
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from ..common import FaultState, SensorStatus, SensorType


@dataclass
class IMUConfig:
    """IMU processor configuration."""
    # Drift detection
    drift_threshold: float = 0.05  # rad - max acceptable drift
    drift_window: float = 5.0  # seconds to accumulate for drift check
    
    # Anomaly detection
    max_angular_velocity: float = 10.0  # rad/s
    max_acceleration: float = 50.0  # m/s^2
    
    # Health monitoring
    timeout: float = 0.1  # seconds without data = fault


@dataclass
class IMUData:
    """Processed IMU data."""
    orientation: np.ndarray = field(default_factory=lambda: np.zeros(3))  # roll, pitch, yaw
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    linear_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    timestamp: float = 0.0


class IMUProcessor:
    """
    IMU sensor processor with fault detection.
    
    Detects:
    - Drift: Accumulating offset in orientation
    - Anomalies: Out-of-range readings
    - Timeout: No data received
    """
    
    def __init__(self, config: Optional[IMUConfig] = None):
        """Initialize IMU processor."""
        self.config = config or IMUConfig()
        
        # Current state
        self.status = SensorStatus(
            sensor_type=SensorType.IMU,
            name="imu",
            fault_state=FaultState.NONE
        )
        
        # Data history for drift detection
        self._orientation_history: List[Tuple[float, np.ndarray]] = []
        self._initial_orientation: Optional[np.ndarray] = None
        
        # Current processed data
        self.data = IMUData()
        
        # Injected fault (for testing)
        self._injected_fault: Optional[str] = None
    
    def update(self, orientation: np.ndarray, angular_vel: np.ndarray,
               linear_accel: np.ndarray, timestamp: float) -> FaultState:
        """
        Update IMU data and check for faults.
        
        Args:
            orientation: [roll, pitch, yaw] in radians
            angular_vel: [wx, wy, wz] in rad/s
            linear_accel: [ax, ay, az] in m/s^2
            timestamp: Current simulation time
            
        Returns:
            Current fault state
        """
        # Check for injected fault
        if self._injected_fault == "drift":
            orientation = orientation + np.array([0.1, 0.0, 0.0])  # Add artificial drift
        elif self._injected_fault == "spike":
            angular_vel = angular_vel * 10.0  # Anomalous spike
        
        # Store data
        self.data = IMUData(
            orientation=orientation.copy(),
            angular_velocity=angular_vel.copy(),
            linear_acceleration=linear_accel.copy(),
            timestamp=timestamp
        )
        
        # Update status timestamp
        self.status.last_update = timestamp
        self.status.last_value = np.linalg.norm(orientation)
        
        # Initialize reference
        if self._initial_orientation is None:
            self._initial_orientation = orientation.copy()
        
        # Add to history
        self._orientation_history.append((timestamp, orientation.copy()))
        
        # Trim old history
        cutoff = timestamp - self.config.drift_window
        self._orientation_history = [
            (t, o) for t, o in self._orientation_history if t > cutoff
        ]
        
        # Check for faults
        fault_state = self._check_faults(angular_vel, linear_accel, timestamp)
        self.status.fault_state = fault_state
        
        return fault_state
    
    def _check_faults(self, angular_vel: np.ndarray, linear_accel: np.ndarray,
                      timestamp: float) -> FaultState:
        """Check for IMU faults."""
        # Check angular velocity bounds
        if np.any(np.abs(angular_vel) > self.config.max_angular_velocity):
            return FaultState.TRUE  # Anomalous reading
        
        # Check acceleration bounds
        if np.any(np.abs(linear_accel) > self.config.max_acceleration):
            return FaultState.SUSPECT  # Possible anomaly
        
        # Check for drift
        if len(self._orientation_history) >= 2:
            drift = self._calculate_drift()
            if drift > self.config.drift_threshold:
                return FaultState.SUSPECT  # Drift detected
        
        return FaultState.FALSE
    
    def _calculate_drift(self) -> float:
        """Calculate orientation drift over window."""
        if len(self._orientation_history) < 2:
            return 0.0
        
        # Compare oldest and newest in window
        _, oldest = self._orientation_history[0]
        _, newest = self._orientation_history[-1]
        
        # Drift in roll/pitch (yaw can change normally)
        drift = np.sqrt(
            (newest[0] - oldest[0])**2 + 
            (newest[1] - oldest[1])**2
        )
        return drift
    
    def inject_fault(self, fault_type: str):
        """Inject a fault for testing (drift, spike, timeout)."""
        self._injected_fault = fault_type
    
    def clear_fault(self):
        """Clear injected fault."""
        self._injected_fault = None
    
    def get_pitch_roll(self) -> Tuple[float, float]:
        """Get current pitch and roll angles."""
        return self.data.orientation[1], self.data.orientation[0]
    
    def is_stable(self, pitch_threshold: float = 0.3,
                  roll_threshold: float = 0.3) -> bool:
        """Check if robot orientation is within stable bounds."""
        pitch, roll = self.get_pitch_roll()
        return abs(pitch) < pitch_threshold and abs(roll) < roll_threshold
