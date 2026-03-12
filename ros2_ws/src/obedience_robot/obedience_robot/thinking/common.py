"""
Common types and enumerations for the Thinking System.

Fault propagation follows aerospace industry standards with four states:
- TRUE: Confirmed fault
- FALSE: No fault detected
- NONE: Unknown/not evaluated
- SUSPECT: Possible fault, pending confirmation
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from datetime import datetime


class FaultState(Enum):
    """
    Fault propagation states (aerospace standard).
    
    Used for error propagation in FMEA chains.
    """
    TRUE = "TRUE"        # Fault confirmed
    FALSE = "FALSE"      # No fault
    NONE = "NONE"        # Unknown / not evaluated
    SUSPECT = "SUSPECT"  # Possible fault, pending confirmation


class SensorType(Enum):
    """Types of sensors in the system."""
    IMU = auto()
    JOINT_ENCODER = auto()
    JOINT_TORQUE = auto()
    CONTACT = auto()
    LIDAR = auto()
    BATTERY = auto()


class LocomotionState(Enum):
    """Robot locomotion states."""
    STANDING = auto()
    WALKING = auto()
    TURNING = auto()
    FALLING = auto()
    FALLEN = auto()
    RECOVERING = auto()


class MissionPhase(Enum):
    """High-level mission phases."""
    IDLE = auto()
    LOADING = auto()
    NAVIGATING = auto()
    DELIVERING = auto()
    RETURNING = auto()
    CHARGING = auto()
    EMERGENCY = auto()


@dataclass
class SensorStatus:
    """Status of a single sensor."""
    sensor_type: SensorType
    name: str
    fault_state: FaultState = FaultState.NONE
    last_value: Optional[float] = None
    last_update: Optional[float] = None  # simulation time
    confidence: float = 1.0  # 0.0 to 1.0
    
    def is_healthy(self) -> bool:
        """Check if sensor is healthy."""
        return self.fault_state == FaultState.FALSE
    
    def is_faulty(self) -> bool:
        """Check if sensor has confirmed fault."""
        return self.fault_state == FaultState.TRUE


@dataclass
class SystemHealth:
    """Overall system health status."""
    overall: FaultState = FaultState.NONE  # Computed overall state
    locomotion: FaultState = FaultState.NONE
    battery: FaultState = FaultState.NONE
    sensors: FaultState = FaultState.NONE
    mission: FaultState = FaultState.NONE
    communication: FaultState = FaultState.NONE
    
    # Track active faults
    active_faults: list = field(default_factory=list)
    
    # Detailed sensor states
    sensor_details: Dict[str, SensorStatus] = field(default_factory=dict)
    
    # Timestamps
    last_update: Optional[float] = None
    
    def get_overall_state(self) -> FaultState:
        """
        Get overall system fault state.
        
        Priority: TRUE > SUSPECT > NONE > FALSE
        """
        states = [
            self.locomotion,
            self.battery,
            self.sensors,
            self.mission,
            self.communication
        ]
        
        if FaultState.TRUE in states:
            return FaultState.TRUE
        if FaultState.SUSPECT in states:
            return FaultState.SUSPECT
        if FaultState.NONE in states:
            return FaultState.NONE
        return FaultState.FALSE
    
    def to_dict(self) -> dict:
        """Convert to dictionary for reporting."""
        return {
            "overall": self.get_overall_state().value,
            "locomotion": self.locomotion.value,
            "battery": self.battery.value,
            "sensors": self.sensors.value,
            "mission": self.mission.value,
            "communication": self.communication.value,
            "last_update": self.last_update,
        }


@dataclass
class FaultEvent:
    """Record of a fault occurrence."""
    fault_id: str
    fault_type: str
    source: str
    state: FaultState
    timestamp: float
    description: str
    propagated_from: Optional[str] = None
    
    def __str__(self) -> str:
        return f"[{self.state.value}] {self.fault_type}: {self.description}"


# Fault chains based on FMEA diagram
FAULT_CHAINS = {
    # Battery chain
    "loose_connector": ["power_loss", "anomalous_discharge"],
    "power_loss": ["anomalous_discharge", "actuators_off"],
    "anomalous_discharge": ["low_voltage"],
    "low_voltage": ["insufficient_power", "actuators_off"],
    "insufficient_power": ["reduced_torque"],
    "reduced_torque": ["balance_loss"],
    "actuators_off": ["balance_loss"],
    
    # IMU chain
    "reading_unavailable": ["cable_fault"],
    "cable_fault": ["anomalous_sensor_reading"],
    "calibration_error": ["anomalous_sensor_reading"],
    "anomalous_sensor_reading": ["balance_loss"],
    
    # System chain
    "balance_loss": ["robot_collapse", "asymmetric_gait"],
    "asymmetric_gait": ["robot_collapse"],
    "robot_collapse": ["no_command_response"],
    "overheating": ["robot_collapse"],
    
    # External
    "external_perturbation": ["balance_loss"],
}
