"""
Contact Processor - Foot Contact Monitoring

Monitors foot contact sensors for:
- Loss of contact (foot not touching ground)
- Unexpected contact (tripping)
- Contact asymmetry

Fault Modes:
- Contact loss: Walking without ground contact (falling)
- Trip detection: Unexpected foot contact during swing
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict
from ..common import FaultState, SensorStatus, SensorType, LocomotionState


@dataclass
class ContactConfig:
    """Contact processor configuration."""
    # Minimum time without contact to trigger fault
    loss_timeout: float = 0.3  # seconds
    
    # Minimum contact for stability
    min_stable_contacts: int = 1  # At least one foot on ground


class ContactProcessor:
    """
    Foot contact processor with fault detection.
    
    Monitors both feet for ground contact.
    """
    
    FOOT_NAMES = ["left", "right"]
    
    def __init__(self, config: Optional[ContactConfig] = None):
        """Initialize contact processor."""
        self.config = config or ContactConfig()
        
        # Contact states
        self.left_contact: bool = False
        self.right_contact: bool = False
        
        # Status
        self.status = SensorStatus(
            sensor_type=SensorType.CONTACT,
            name="foot_contacts",
            fault_state=FaultState.NONE
        )
        
        # Tracking for loss detection
        self._last_contact_time: float = 0.0
        self._no_contact_start: Optional[float] = None
        
        # Injected fault
        self._injected_fault: Optional[str] = None
    
    def update(self, left_contact: bool, right_contact: bool,
               timestamp: float) -> FaultState:
        """
        Update contact states and check for faults.
        
        Args:
            left_contact: True if left foot has ground contact
            right_contact: True if right foot has ground contact
            timestamp: Current simulation time
            
        Returns:
            Current fault state
        """
        # Apply injected fault
        if self._injected_fault == "loss_left":
            left_contact = False
        elif self._injected_fault == "loss_right":
            right_contact = False
        elif self._injected_fault == "loss_both":
            left_contact = False
            right_contact = False
        
        self.left_contact = left_contact
        self.right_contact = right_contact
        self.status.last_update = timestamp
        
        # Check for contact loss
        has_contact = left_contact or right_contact
        
        if has_contact:
            self._last_contact_time = timestamp
            self._no_contact_start = None
            self.status.fault_state = FaultState.FALSE
        else:
            # Track no contact duration
            if self._no_contact_start is None:
                self._no_contact_start = timestamp
            
            no_contact_duration = timestamp - self._no_contact_start
            
            if no_contact_duration > self.config.loss_timeout:
                # Extended loss of contact - likely falling
                self.status.fault_state = FaultState.TRUE
            elif no_contact_duration > self.config.loss_timeout / 2:
                # Potential issue
                self.status.fault_state = FaultState.SUSPECT
            else:
                # Normal during step
                self.status.fault_state = FaultState.NONE
        
        return self.status.fault_state
    
    def get_locomotion_state(self) -> LocomotionState:
        """Infer locomotion state from contacts."""
        if self.status.fault_state == FaultState.TRUE:
            return LocomotionState.FALLING
        
        if self.left_contact and self.right_contact:
            return LocomotionState.STANDING
        elif self.left_contact or self.right_contact:
            return LocomotionState.WALKING
        else:
            return LocomotionState.FALLING
    
    def inject_fault(self, fault_type: str):
        """Inject fault (loss_left, loss_right, loss_both)."""
        self._injected_fault = fault_type
    
    def clear_fault(self):
        """Clear injected fault."""
        self._injected_fault = None
    
    def get_contact_state(self) -> Dict[str, bool]:
        """Get current contact states."""
        return {
            "left": self.left_contact,
            "right": self.right_contact,
            "any": self.left_contact or self.right_contact,
            "both": self.left_contact and self.right_contact,
        }
