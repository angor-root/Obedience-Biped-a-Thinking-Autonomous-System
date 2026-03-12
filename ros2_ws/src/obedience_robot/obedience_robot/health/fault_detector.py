"""
Fault Detector - FMEA-Based Fault Detection

Implements fault detection based on Failure Mode and Effects Analysis (FMEA).

Fault Categories:
1. Locomotion: fall, trip, contact loss
2. Battery: anomalous discharge, charge failure
3. Sensors: IMU drift, LiDAR obstruction
4. Mission: wrong medicine, wrong bed
5. Communication: ROS2 connection loss

Uses four-state logic for fault propagation:
- TRUE: Confirmed fault
- FALSE: No fault
- NONE: Unknown/not evaluated
- SUSPECT: Possible fault, pending confirmation
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Callable
from datetime import datetime


class FaultType(Enum):
    """Types of faults based on FMEA categories."""
    # Locomotion faults
    FALL = auto()
    TRIP = auto()
    CONTACT_LOSS = auto()
    BALANCE_LOSS = auto()
    
    # Battery faults
    ANOMALOUS_DISCHARGE = auto()
    CHARGE_FAILURE = auto()
    LOW_VOLTAGE = auto()
    
    # Sensor faults
    IMU_DRIFT = auto()
    LIDAR_OBSTRUCTION = auto()
    SENSOR_TIMEOUT = auto()
    
    # Mission faults
    WRONG_MEDICINE = auto()
    WRONG_BED = auto()
    MISSION_TIMEOUT = auto()
    
    # Communication faults
    ROS2_DISCONNECT = auto()
    MESSAGE_TIMEOUT = auto()


class FaultState(Enum):
    """Fault propagation states."""
    TRUE = "TRUE"
    FALSE = "FALSE"
    NONE = "NONE"
    SUSPECT = "SUSPECT"


@dataclass
class Fault:
    """Representation of a detected fault."""
    fault_type: FaultType
    state: FaultState
    timestamp: float
    source: str
    description: str
    propagated_from: Optional['Fault'] = None
    
    def __str__(self) -> str:
        src = f" (from {self.propagated_from.fault_type.name})" if self.propagated_from else ""
        return f"[{self.state.value}] {self.fault_type.name}: {self.description}{src}"


@dataclass
class FaultChain:
    """Defines a fault propagation chain."""
    source: FaultType
    effects: List[FaultType]
    probability: float = 1.0  # Probability of propagation
    delay: float = 0.0  # Delay before effect manifests


class FaultDetector:
    """
    FMEA-based fault detector with propagation.
    
    Monitors system for faults and propagates effects through
    defined causal chains.
    """
    
    # Fault propagation chains based on FMEA diagram
    FAULT_CHAINS: List[FaultChain] = [
        # Battery chains
        FaultChain(FaultType.ANOMALOUS_DISCHARGE, [FaultType.LOW_VOLTAGE]),
        FaultChain(FaultType.LOW_VOLTAGE, [FaultType.BALANCE_LOSS]),
        FaultChain(FaultType.CHARGE_FAILURE, [FaultType.LOW_VOLTAGE]),
        
        # Sensor chains
        FaultChain(FaultType.IMU_DRIFT, [FaultType.BALANCE_LOSS]),
        FaultChain(FaultType.LIDAR_OBSTRUCTION, [FaultType.TRIP, FaultType.WRONG_BED]),
        FaultChain(FaultType.SENSOR_TIMEOUT, [FaultType.BALANCE_LOSS]),
        
        # Locomotion chains
        FaultChain(FaultType.CONTACT_LOSS, [FaultType.FALL]),
        FaultChain(FaultType.BALANCE_LOSS, [FaultType.FALL]),
        FaultChain(FaultType.TRIP, [FaultType.FALL]),
        
        # Mission chains
        FaultChain(FaultType.WRONG_BED, [FaultType.MISSION_TIMEOUT]),
        
        # Communication chains
        FaultChain(FaultType.ROS2_DISCONNECT, [FaultType.MISSION_TIMEOUT]),
    ]
    
    def __init__(self):
        """Initialize fault detector."""
        # Active faults
        self._active_faults: Dict[FaultType, Fault] = {}
        
        # Fault history
        self._fault_history: List[Fault] = []
        
        # Build propagation map
        self._propagation_map: Dict[FaultType, List[FaultChain]] = {}
        for chain in self.FAULT_CHAINS:
            if chain.source not in self._propagation_map:
                self._propagation_map[chain.source] = []
            self._propagation_map[chain.source].append(chain)
        
        # Callbacks
        self._on_fault_detected: Optional[Callable[[Fault], None]] = None
        self._on_fault_cleared: Optional[Callable[[FaultType], None]] = None
        
        # Injected faults for testing
        self._injected_faults: Set[FaultType] = set()
    
    def detect_fault(self, fault_type: FaultType, state: FaultState,
                     source: str, description: str, timestamp: float,
                     propagate: bool = True) -> Fault:
        """
        Detect and register a fault.
        
        Args:
            fault_type: Type of fault detected
            state: Confidence state (TRUE, FALSE, SUSPECT, NONE)
            source: Source subsystem that detected the fault
            description: Human-readable description
            timestamp: Detection time
            propagate: Whether to propagate effects
            
        Returns:
            Created Fault object
        """
        fault = Fault(
            fault_type=fault_type,
            state=state,
            timestamp=timestamp,
            source=source,
            description=description
        )
        
        # Update active faults
        if state in [FaultState.TRUE, FaultState.SUSPECT]:
            self._active_faults[fault_type] = fault
            self._fault_history.append(fault)
            
            if self._on_fault_detected:
                self._on_fault_detected(fault)
            
            # Propagate to effects
            if propagate and state == FaultState.TRUE:
                self._propagate_fault(fault, timestamp)
        
        elif state == FaultState.FALSE:
            if fault_type in self._active_faults:
                del self._active_faults[fault_type]
                if self._on_fault_cleared:
                    self._on_fault_cleared(fault_type)
        
        return fault
    
    def _propagate_fault(self, fault: Fault, timestamp: float):
        """Propagate fault to its effects."""
        if fault.fault_type not in self._propagation_map:
            return
        
        for chain in self._propagation_map[fault.fault_type]:
            for effect_type in chain.effects:
                # Create propagated fault
                effect_fault = Fault(
                    fault_type=effect_type,
                    state=FaultState.SUSPECT,  # Effects start as suspect
                    timestamp=timestamp + chain.delay,
                    source="propagation",
                    description=f"Propagated from {fault.fault_type.name}",
                    propagated_from=fault
                )
                
                # Don't override confirmed faults
                if effect_type not in self._active_faults:
                    self._active_faults[effect_type] = effect_fault
                    self._fault_history.append(effect_fault)
                    
                    if self._on_fault_detected:
                        self._on_fault_detected(effect_fault)
    
    def clear_fault(self, fault_type: FaultType):
        """Clear a fault."""
        if fault_type in self._active_faults:
            del self._active_faults[fault_type]
            if self._on_fault_cleared:
                self._on_fault_cleared(fault_type)
    
    def inject_fault(self, fault_type: FaultType, timestamp: float):
        """Inject a fault for testing."""
        self._injected_faults.add(fault_type)
        self.detect_fault(
            fault_type=fault_type,
            state=FaultState.TRUE,
            source="injection",
            description=f"Injected fault for testing",
            timestamp=timestamp
        )
    
    def clear_injected_fault(self, fault_type: FaultType):
        """Clear an injected fault."""
        self._injected_faults.discard(fault_type)
        self.clear_fault(fault_type)
    
    def is_fault_active(self, fault_type: FaultType) -> bool:
        """Check if a fault is currently active."""
        return fault_type in self._active_faults
    
    def get_active_faults(self) -> List[Fault]:
        """Get all active faults."""
        return list(self._active_faults.values())
    
    def get_fault_state(self, fault_type: FaultType) -> FaultState:
        """Get current state of a specific fault type."""
        if fault_type in self._active_faults:
            return self._active_faults[fault_type].state
        return FaultState.FALSE
    
    def get_category_state(self, category: str) -> FaultState:
        """Get overall state for a fault category."""
        category_faults = {
            "locomotion": [FaultType.FALL, FaultType.TRIP, FaultType.CONTACT_LOSS, FaultType.BALANCE_LOSS],
            "battery": [FaultType.ANOMALOUS_DISCHARGE, FaultType.CHARGE_FAILURE, FaultType.LOW_VOLTAGE],
            "sensors": [FaultType.IMU_DRIFT, FaultType.LIDAR_OBSTRUCTION, FaultType.SENSOR_TIMEOUT],
            "mission": [FaultType.WRONG_MEDICINE, FaultType.WRONG_BED, FaultType.MISSION_TIMEOUT],
            "communication": [FaultType.ROS2_DISCONNECT, FaultType.MESSAGE_TIMEOUT],
        }
        
        if category not in category_faults:
            return FaultState.NONE
        
        states = [self.get_fault_state(ft) for ft in category_faults[category]]
        
        if FaultState.TRUE in states:
            return FaultState.TRUE
        if FaultState.SUSPECT in states:
            return FaultState.SUSPECT
        if all(s == FaultState.FALSE for s in states):
            return FaultState.FALSE
        return FaultState.NONE
    
    def set_callbacks(self, on_detected: Callable[[Fault], None],
                      on_cleared: Optional[Callable[[FaultType], None]] = None):
        """Set fault event callbacks."""
        self._on_fault_detected = on_detected
        self._on_fault_cleared = on_cleared
    
    def get_fault_history(self, limit: int = 100) -> List[Fault]:
        """Get fault history."""
        return self._fault_history[-limit:]
    
    def get_status_summary(self) -> Dict[str, str]:
        """Get status summary by category."""
        return {
            "locomotion": self.get_category_state("locomotion").value,
            "battery": self.get_category_state("battery").value,
            "sensors": self.get_category_state("sensors").value,
            "mission": self.get_category_state("mission").value,
            "communication": self.get_category_state("communication").value,
            "active_faults": len(self._active_faults),
        }
