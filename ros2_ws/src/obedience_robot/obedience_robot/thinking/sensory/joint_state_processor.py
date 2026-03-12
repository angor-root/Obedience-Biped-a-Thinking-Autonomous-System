"""
Joint State Processor - Joint Position and Torque Monitoring

Monitors joint states for:
- Motor not responding (position stuck)
- Anomalous vibration (high frequency oscillation)
- Torque overload

Fault Modes:
- Motor unresponsive: Joint not moving despite commands
- Vibration anomaly: Oscillation indicating mechanical issue
- Torque overload: Exceeding safe limits
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from collections import deque
from ..common import FaultState, SensorStatus, SensorType


@dataclass
class JointConfig:
    """Joint processor configuration."""
    # Vibration detection
    vibration_window: int = 50  # samples to analyze
    vibration_threshold: float = 0.1  # rad - high frequency amplitude
    
    # Unresponsive detection
    stuck_threshold: float = 0.01  # rad - min movement expected
    stuck_window: float = 0.5  # seconds to wait
    
    # Torque limits
    max_torque: float = 30.0  # Nm
    torque_warning: float = 25.0  # Nm


@dataclass
class JointState:
    """State of a single joint."""
    name: str
    position: float = 0.0
    velocity: float = 0.0
    torque: float = 0.0
    fault_state: FaultState = FaultState.NONE


class JointStateProcessor:
    """
    Joint state processor with fault detection.
    
    Monitors all robot joints for:
    - Unresponsive motors
    - Anomalous vibration
    - Torque overload
    """
    
    # Joint names for the biped
    JOINT_NAMES = [
        "right_hip_z_j", "right_hip_y_j", "right_hip", "right_knee", "right_foot_j",
        "left_hip_z_j", "left_hip_y_j", "left_hip", "left_knee", "left_foot_j"
    ]
    
    def __init__(self, config: Optional[JointConfig] = None):
        """Initialize joint processor."""
        self.config = config or JointConfig()
        
        # Status for each joint
        self.joint_status: Dict[str, SensorStatus] = {}
        for name in self.JOINT_NAMES:
            self.joint_status[name] = SensorStatus(
                sensor_type=SensorType.JOINT_ENCODER,
                name=name,
                fault_state=FaultState.NONE
            )
        
        # Joint states
        self.joints: Dict[str, JointState] = {
            name: JointState(name=name) for name in self.JOINT_NAMES
        }
        
        # History for vibration detection
        self._position_history: Dict[str, deque] = {
            name: deque(maxlen=self.config.vibration_window)
            for name in self.JOINT_NAMES
        }
        
        # History for stuck detection
        self._stuck_start: Dict[str, Optional[Tuple[float, float]]] = {
            name: None for name in self.JOINT_NAMES
        }
        
        # Injected faults
        self._injected_faults: Dict[str, str] = {}
    
    def update(self, positions: np.ndarray, velocities: np.ndarray,
               torques: np.ndarray, timestamp: float) -> Dict[str, FaultState]:
        """
        Update joint states and check for faults.
        
        Args:
            positions: Joint positions [10] in radians
            velocities: Joint velocities [10] in rad/s
            torques: Joint torques [10] in Nm
            timestamp: Current simulation time
            
        Returns:
            Dictionary of joint name -> fault state
        """
        fault_states = {}
        
        for i, name in enumerate(self.JOINT_NAMES):
            pos = positions[i] if i < len(positions) else 0.0
            vel = velocities[i] if i < len(velocities) else 0.0
            torque = torques[i] if i < len(torques) else 0.0
            
            # Apply injected fault
            if name in self._injected_faults:
                fault_type = self._injected_faults[name]
                if fault_type == "vibration":
                    pos += 0.2 * np.sin(100 * timestamp)  # High freq oscillation
                elif fault_type == "stuck":
                    pos = self.joints[name].position  # Keep old position
                elif fault_type == "overload":
                    torque = self.config.max_torque * 1.5
            
            # Update joint state
            self.joints[name].position = pos
            self.joints[name].velocity = vel
            self.joints[name].torque = torque
            
            # Add to history
            self._position_history[name].append(pos)
            
            # Check faults
            fault = self._check_joint_fault(name, pos, vel, torque, timestamp)
            self.joints[name].fault_state = fault
            self.joint_status[name].fault_state = fault
            self.joint_status[name].last_update = timestamp
            self.joint_status[name].last_value = pos
            
            fault_states[name] = fault
        
        return fault_states
    
    def _check_joint_fault(self, name: str, pos: float, vel: float,
                           torque: float, timestamp: float) -> FaultState:
        """Check for faults in a single joint."""
        # Check torque overload
        if abs(torque) > self.config.max_torque:
            return FaultState.TRUE  # Confirmed overload
        if abs(torque) > self.config.torque_warning:
            return FaultState.SUSPECT  # Warning
        
        # Check vibration
        if self._check_vibration(name):
            return FaultState.SUSPECT  # Possible mechanical issue
        
        # Check if stuck (motor not responding)
        if self._check_stuck(name, pos, timestamp):
            return FaultState.SUSPECT  # Motor might not be responding
        
        return FaultState.FALSE
    
    def _check_vibration(self, name: str) -> bool:
        """Check for high-frequency vibration."""
        history = list(self._position_history[name])
        if len(history) < 10:
            return False
        
        # Calculate high-frequency component (simple difference analysis)
        diffs = np.diff(history)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        
        # Many sign changes = oscillation
        oscillation_ratio = sign_changes / len(diffs)
        
        # High amplitude oscillation
        amplitude = np.std(history)
        
        return oscillation_ratio > 0.7 and amplitude > self.config.vibration_threshold
    
    def _check_stuck(self, name: str, pos: float, timestamp: float) -> bool:
        """Check if joint is stuck (not moving)."""
        start = self._stuck_start[name]
        
        if start is None:
            self._stuck_start[name] = (timestamp, pos)
            return False
        
        start_time, start_pos = start
        
        # Check if position changed significantly
        if abs(pos - start_pos) > self.config.stuck_threshold:
            self._stuck_start[name] = (timestamp, pos)
            return False
        
        # Check if enough time passed
        if timestamp - start_time > self.config.stuck_window:
            return True  # Joint hasn't moved for too long
        
        return False
    
    def inject_fault(self, joint_name: str, fault_type: str):
        """Inject fault for testing (vibration, stuck, overload)."""
        if joint_name in self.JOINT_NAMES:
            self._injected_faults[joint_name] = fault_type
    
    def clear_fault(self, joint_name: Optional[str] = None):
        """Clear injected fault."""
        if joint_name:
            self._injected_faults.pop(joint_name, None)
        else:
            self._injected_faults.clear()
    
    def get_overall_fault(self) -> FaultState:
        """Get overall joint system fault state."""
        states = [j.fault_state for j in self.joints.values()]
        
        if FaultState.TRUE in states:
            return FaultState.TRUE
        if FaultState.SUSPECT in states:
            return FaultState.SUSPECT
        if all(s == FaultState.FALSE for s in states):
            return FaultState.FALSE
        return FaultState.NONE
    
    def get_faulty_joints(self) -> List[str]:
        """Get list of joints with faults."""
        return [
            name for name, joint in self.joints.items()
            if joint.fault_state in [FaultState.TRUE, FaultState.SUSPECT]
        ]
