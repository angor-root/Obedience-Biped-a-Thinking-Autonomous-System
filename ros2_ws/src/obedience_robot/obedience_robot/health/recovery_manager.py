"""
Recovery Manager - Fault Recovery Actions

Manages recovery actions for detected faults.

Recovery actions are suggestions for:
- Automatic recovery (system can handle)
- Manual intervention required (needs human)
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable
from .fault_detector import FaultType, FaultState, Fault


class RecoveryType(Enum):
    """Types of recovery actions."""
    AUTOMATIC = auto()      # System can handle automatically
    MANUAL = auto()         # Requires human intervention
    NONE_REQUIRED = auto()  # No action needed


class RecoveryAction(Enum):
    """Specific recovery actions."""
    # Locomotion recovery
    STOP_WALKING = auto()
    SLOW_DOWN = auto()
    RETURN_HOME = auto()
    EMERGENCY_STOP = auto()
    
    # Battery recovery
    GO_TO_CHARGER = auto()
    REDUCE_POWER = auto()
    
    # Sensor recovery
    RECALIBRATE_IMU = auto()
    CLEAR_LIDAR = auto()
    USE_BACKUP_SENSOR = auto()
    
    # Mission recovery
    ABORT_MISSION = auto()
    RETRY_DELIVERY = auto()
    
    # Communication recovery
    RECONNECT = auto()
    AUTONOMOUS_MODE = auto()
    
    # Manual actions
    MANUAL_INTERVENTION = auto()
    RESTART_SYSTEM = auto()


@dataclass
class RecoveryPlan:
    """A recovery plan for a specific fault."""
    fault_type: FaultType
    recovery_type: RecoveryType
    actions: List[RecoveryAction]
    priority: int  # Lower = higher priority
    description: str


class RecoveryManager:
    """
    Manages recovery actions for system faults.
    
    Provides recovery recommendations based on detected faults
    and system state.
    """
    
    # Recovery plans for each fault type
    RECOVERY_PLANS: Dict[FaultType, RecoveryPlan] = {
        # Locomotion
        FaultType.FALL: RecoveryPlan(
            FaultType.FALL,
            RecoveryType.MANUAL,
            [RecoveryAction.EMERGENCY_STOP, RecoveryAction.MANUAL_INTERVENTION],
            priority=1,
            description="Robot has fallen - manual intervention required"
        ),
        FaultType.TRIP: RecoveryPlan(
            FaultType.TRIP,
            RecoveryType.AUTOMATIC,
            [RecoveryAction.STOP_WALKING, RecoveryAction.SLOW_DOWN],
            priority=2,
            description="Trip detected - slowing down"
        ),
        FaultType.CONTACT_LOSS: RecoveryPlan(
            FaultType.CONTACT_LOSS,
            RecoveryType.AUTOMATIC,
            [RecoveryAction.SLOW_DOWN, RecoveryAction.RETURN_HOME],
            priority=2,
            description="Contact loss - reducing speed"
        ),
        FaultType.BALANCE_LOSS: RecoveryPlan(
            FaultType.BALANCE_LOSS,
            RecoveryType.AUTOMATIC,
            [RecoveryAction.STOP_WALKING, RecoveryAction.SLOW_DOWN],
            priority=2,
            description="Balance issue - stabilizing"
        ),
        
        # Battery
        FaultType.ANOMALOUS_DISCHARGE: RecoveryPlan(
            FaultType.ANOMALOUS_DISCHARGE,
            RecoveryType.AUTOMATIC,
            [RecoveryAction.REDUCE_POWER, RecoveryAction.GO_TO_CHARGER],
            priority=3,
            description="High discharge - returning to charger"
        ),
        FaultType.CHARGE_FAILURE: RecoveryPlan(
            FaultType.CHARGE_FAILURE,
            RecoveryType.MANUAL,
            [RecoveryAction.MANUAL_INTERVENTION],
            priority=3,
            description="Charger not working - check connection"
        ),
        FaultType.LOW_VOLTAGE: RecoveryPlan(
            FaultType.LOW_VOLTAGE,
            RecoveryType.AUTOMATIC,
            [RecoveryAction.EMERGENCY_STOP, RecoveryAction.GO_TO_CHARGER],
            priority=1,
            description="Critical battery - emergency return"
        ),
        
        # Sensors
        FaultType.IMU_DRIFT: RecoveryPlan(
            FaultType.IMU_DRIFT,
            RecoveryType.AUTOMATIC,
            [RecoveryAction.SLOW_DOWN, RecoveryAction.RECALIBRATE_IMU],
            priority=3,
            description="IMU drift detected - recalibrating"
        ),
        FaultType.LIDAR_OBSTRUCTION: RecoveryPlan(
            FaultType.LIDAR_OBSTRUCTION,
            RecoveryType.AUTOMATIC,
            [RecoveryAction.STOP_WALKING, RecoveryAction.CLEAR_LIDAR],
            priority=3,
            description="LiDAR blocked - stopping for safety"
        ),
        FaultType.SENSOR_TIMEOUT: RecoveryPlan(
            FaultType.SENSOR_TIMEOUT,
            RecoveryType.AUTOMATIC,
            [RecoveryAction.USE_BACKUP_SENSOR, RecoveryAction.SLOW_DOWN],
            priority=3,
            description="Sensor timeout - using backup"
        ),
        
        # Mission
        FaultType.WRONG_MEDICINE: RecoveryPlan(
            FaultType.WRONG_MEDICINE,
            RecoveryType.AUTOMATIC,
            [RecoveryAction.ABORT_MISSION, RecoveryAction.RETURN_HOME],
            priority=2,
            description="Wrong medicine - aborting delivery"
        ),
        FaultType.WRONG_BED: RecoveryPlan(
            FaultType.WRONG_BED,
            RecoveryType.AUTOMATIC,
            [RecoveryAction.RETRY_DELIVERY],
            priority=3,
            description="Wrong bed - retrying navigation"
        ),
        FaultType.MISSION_TIMEOUT: RecoveryPlan(
            FaultType.MISSION_TIMEOUT,
            RecoveryType.AUTOMATIC,
            [RecoveryAction.ABORT_MISSION, RecoveryAction.RETURN_HOME],
            priority=3,
            description="Mission timeout - returning home"
        ),
        
        # Communication
        FaultType.ROS2_DISCONNECT: RecoveryPlan(
            FaultType.ROS2_DISCONNECT,
            RecoveryType.AUTOMATIC,
            [RecoveryAction.RECONNECT, RecoveryAction.AUTONOMOUS_MODE],
            priority=3,
            description="ROS2 connection lost - attempting reconnect"
        ),
        FaultType.MESSAGE_TIMEOUT: RecoveryPlan(
            FaultType.MESSAGE_TIMEOUT,
            RecoveryType.AUTOMATIC,
            [RecoveryAction.RECONNECT],
            priority=4,
            description="Message timeout - reconnecting"
        ),
    }
    
    def __init__(self):
        """Initialize recovery manager."""
        self._active_recoveries: Dict[FaultType, RecoveryPlan] = {}
        self._recovery_history: List[tuple] = []  # (timestamp, fault_type, action)
        
        # Callbacks
        self._on_recovery_started: Optional[Callable] = None
        self._on_recovery_completed: Optional[Callable] = None
    
    def get_recovery_plan(self, fault_type: FaultType) -> Optional[RecoveryPlan]:
        """Get recovery plan for a fault type."""
        return self.RECOVERY_PLANS.get(fault_type)
    
    def get_recovery_actions(self, faults: List[Fault]) -> List[RecoveryAction]:
        """
        Get prioritized recovery actions for active faults.
        
        Args:
            faults: List of active faults
            
        Returns:
            Prioritized list of recovery actions
        """
        if not faults:
            return []
        
        # Get all applicable plans
        plans = []
        for fault in faults:
            if fault.fault_type in self.RECOVERY_PLANS:
                plans.append(self.RECOVERY_PLANS[fault.fault_type])
        
        # Sort by priority
        plans.sort(key=lambda p: p.priority)
        
        # Collect unique actions in priority order
        seen_actions = set()
        actions = []
        
        for plan in plans:
            for action in plan.actions:
                if action not in seen_actions:
                    seen_actions.add(action)
                    actions.append(action)
        
        return actions
    
    def start_recovery(self, fault_type: FaultType, timestamp: float):
        """Start recovery for a fault."""
        plan = self.get_recovery_plan(fault_type)
        if plan:
            self._active_recoveries[fault_type] = plan
            self._recovery_history.append((timestamp, fault_type, "started"))
            
            if self._on_recovery_started:
                self._on_recovery_started(fault_type, plan)
            
            print(f"[RECOVERY] Started: {plan.description}")
    
    def complete_recovery(self, fault_type: FaultType, timestamp: float):
        """Mark recovery as complete."""
        if fault_type in self._active_recoveries:
            del self._active_recoveries[fault_type]
            self._recovery_history.append((timestamp, fault_type, "completed"))
            
            if self._on_recovery_completed:
                self._on_recovery_completed(fault_type)
            
            print(f"[RECOVERY] Completed: {fault_type.name}")
    
    def requires_manual_intervention(self, faults: List[Fault]) -> bool:
        """Check if any active fault requires manual intervention."""
        for fault in faults:
            plan = self.get_recovery_plan(fault.fault_type)
            if plan and plan.recovery_type == RecoveryType.MANUAL:
                return True
        return False
    
    def get_manual_intervention_reasons(self, faults: List[Fault]) -> List[str]:
        """Get reasons for manual intervention."""
        reasons = []
        for fault in faults:
            plan = self.get_recovery_plan(fault.fault_type)
            if plan and plan.recovery_type == RecoveryType.MANUAL:
                reasons.append(plan.description)
        return reasons
    
    def set_callbacks(self, on_started: Callable, on_completed: Callable):
        """Set recovery event callbacks."""
        self._on_recovery_started = on_started
        self._on_recovery_completed = on_completed
    
    def get_status(self) -> Dict:
        """Get recovery manager status."""
        return {
            "active_recoveries": [
                f.name for f in self._active_recoveries.keys()
            ],
            "history_count": len(self._recovery_history),
        }
