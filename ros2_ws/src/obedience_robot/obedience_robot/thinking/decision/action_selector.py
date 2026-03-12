"""
Action Selector - Priority-Based Action Selection

Selects actions based on:
1. Current situation assessment
2. Active faults/alarms
3. Mission requirements
4. Energy constraints
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional, Dict
from ..common import FaultState


class ActionPriority(Enum):
    """Action priority levels (lower = higher priority)."""
    EMERGENCY = 1      # Safety critical
    HIGH = 2           # Fault response
    NORMAL = 3         # Mission tasks
    LOW = 4            # Background tasks
    IDLE = 5           # Nothing to do


class Action(Enum):
    """Available robot actions."""
    # Emergency
    EMERGENCY_STOP = auto()
    
    # Locomotion
    WALK_FORWARD = auto()
    TURN_LEFT = auto()
    TURN_RIGHT = auto()
    STAND_STILL = auto()
    SLOW_DOWN = auto()
    
    # Mission
    LOAD_MEDICINE = auto()
    DELIVER_MEDICINE = auto()
    RETURN_HOME = auto()
    GO_TO_CHARGER = auto()
    
    # Status
    WAIT = auto()
    IDLE = auto()


@dataclass
class ActionDecision:
    """Decision result."""
    action: Action
    priority: ActionPriority
    reason: str
    parameters: Dict = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class ActionSelector:
    """
    Priority-based action selector.
    
    Evaluates situation and selects appropriate action.
    """
    
    def __init__(self):
        """Initialize action selector."""
        self._current_decision: Optional[ActionDecision] = None
        self._decision_history: List[ActionDecision] = []
    
    def choose_action(self, situation: Dict) -> ActionDecision:
        """
        Choose action based on situation.
        
        Args:
            situation: Dictionary with:
                - health_status: Overall fault state
                - battery_level: 0.0 - 1.0
                - mission_active: bool
                - current_goal: goal type or None
                - at_target: bool
                
        Returns:
            ActionDecision with selected action
        """
        health = situation.get("health_status", FaultState.NONE)
        battery = situation.get("battery_level", 1.0)
        mission_active = situation.get("mission_active", False)
        current_goal = situation.get("current_goal", None)
        at_target = situation.get("at_target", False)
        
        # Priority 1: Emergency
        if health == FaultState.TRUE:
            decision = ActionDecision(
                action=Action.EMERGENCY_STOP,
                priority=ActionPriority.EMERGENCY,
                reason="Critical fault detected"
            )
        
        # Priority 2: Low battery
        elif battery < 0.10:
            decision = ActionDecision(
                action=Action.GO_TO_CHARGER,
                priority=ActionPriority.HIGH,
                reason=f"Critical battery: {battery*100:.0f}%"
            )
        
        # Priority 3: Suspect fault
        elif health == FaultState.SUSPECT:
            decision = ActionDecision(
                action=Action.SLOW_DOWN,
                priority=ActionPriority.HIGH,
                reason="Possible fault - reducing speed"
            )
        
        # Priority 4: Low battery warning
        elif battery < 0.20:
            decision = ActionDecision(
                action=Action.RETURN_HOME,
                priority=ActionPriority.NORMAL,
                reason=f"Low battery: {battery*100:.0f}%"
            )
        
        # Priority 5: Mission tasks
        elif mission_active and current_goal:
            if at_target:
                if current_goal == "load":
                    decision = ActionDecision(
                        action=Action.LOAD_MEDICINE,
                        priority=ActionPriority.NORMAL,
                        reason="At supply zone - loading"
                    )
                elif current_goal == "deliver":
                    decision = ActionDecision(
                        action=Action.DELIVER_MEDICINE,
                        priority=ActionPriority.NORMAL,
                        reason="At bed - delivering"
                    )
                else:
                    decision = ActionDecision(
                        action=Action.WAIT,
                        priority=ActionPriority.NORMAL,
                        reason="At target - waiting"
                    )
            else:
                decision = ActionDecision(
                    action=Action.WALK_FORWARD,
                    priority=ActionPriority.NORMAL,
                    reason=f"Navigating to {current_goal}"
                )
        
        # Priority 6: Idle
        else:
            decision = ActionDecision(
                action=Action.IDLE,
                priority=ActionPriority.IDLE,
                reason="No active mission"
            )
        
        self._current_decision = decision
        self._decision_history.append(decision)
        
        return decision
    
    @property
    def current_action(self) -> Optional[Action]:
        """Get currently selected action."""
        return self._current_decision.action if self._current_decision else None
    
    def get_action_parameters(self) -> Dict:
        """Get parameters for current action."""
        return self._current_decision.parameters if self._current_decision else {}
    
    def should_walk(self) -> bool:
        """Check if robot should be walking."""
        return self.current_action in [
            Action.WALK_FORWARD, 
            Action.RETURN_HOME, 
            Action.GO_TO_CHARGER
        ]
    
    def should_stop(self) -> bool:
        """Check if robot should stop."""
        return self.current_action in [
            Action.EMERGENCY_STOP,
            Action.STAND_STILL,
            Action.WAIT,
            Action.LOAD_MEDICINE,
            Action.DELIVER_MEDICINE,
            Action.IDLE
        ]
