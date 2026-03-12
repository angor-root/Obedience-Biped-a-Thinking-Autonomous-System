"""
Rules Engine - Situation Evaluation Based on Rules

Implements rule-based reasoning for:
- Safety constraints (laws)
- Operational rules
- Mission axioms
"""

from dataclasses import dataclass
from typing import List, Callable, Dict, Any, Optional
from ..common import FaultState


@dataclass
class Rule:
    """A single evaluation rule."""
    name: str
    condition: Callable[[Dict], bool]
    conclusion: str
    priority: int = 5  # Lower = higher priority
    category: str = "general"


@dataclass
class RuleResult:
    """Result of rule evaluation."""
    situation: str
    triggered_rules: List[str]
    recommendations: List[str]
    priority: int


class RulesEngine:
    """
    Rule-based reasoning engine.
    
    Evaluates situation against defined rules and provides
    conclusions and recommendations.
    """
    
    def __init__(self):
        """Initialize rules engine."""
        self.rules: List[Rule] = []
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default safety and operational rules."""
        # Safety Laws (highest priority)
        self.rules.extend([
            Rule(
                name="critical_battery",
                condition=lambda ctx: ctx.get("battery", 1.0) < 0.10,
                conclusion="CRITICAL: Battery critically low",
                priority=1,
                category="safety"
            ),
            Rule(
                name="fall_detected",
                condition=lambda ctx: ctx.get("contact_loss", False) and ctx.get("tilt", 0) > 0.5,
                conclusion="CRITICAL: Fall detected",
                priority=1,
                category="safety"
            ),
            Rule(
                name="sensor_failure",
                condition=lambda ctx: ctx.get("sensor_health", FaultState.FALSE) == FaultState.TRUE,
                conclusion="CRITICAL: Sensor failure",
                priority=1,
                category="safety"
            ),
        ])
        
        # Operational Rules
        self.rules.extend([
            Rule(
                name="low_battery",
                condition=lambda ctx: 0.10 <= ctx.get("battery", 1.0) < 0.20,
                conclusion="WARNING: Battery low - return to charger",
                priority=2,
                category="operational"
            ),
            Rule(
                name="high_tilt",
                condition=lambda ctx: ctx.get("tilt", 0) > 0.3,
                conclusion="WARNING: Excessive tilt - slow down",
                priority=2,
                category="operational"
            ),
            Rule(
                name="imu_drift",
                condition=lambda ctx: ctx.get("imu_fault", FaultState.FALSE) == FaultState.SUSPECT,
                conclusion="CAUTION: Possible IMU drift",
                priority=3,
                category="operational"
            ),
        ])
        
        # Mission Axioms
        self.rules.extend([
            Rule(
                name="mission_stalled",
                condition=lambda ctx: ctx.get("mission_time", 0) > 120 and ctx.get("progress", 0) < 0.5,
                conclusion="WARNING: Mission taking too long",
                priority=3,
                category="mission"
            ),
            Rule(
                name="off_course",
                condition=lambda ctx: ctx.get("distance_to_target", 0) > 5.0,
                conclusion="NOTICE: Far from target - verify navigation",
                priority=4,
                category="mission"
            ),
        ])
    
    def add_rule(self, rule: Rule):
        """Add a custom rule."""
        self.rules.append(rule)
    
    def evaluate(self, context: Dict[str, Any]) -> RuleResult:
        """
        Evaluate situation against all rules.
        
        Args:
            context: Dictionary with situation data:
                - battery: float (0.0 - 1.0)
                - tilt: float (radians)
                - contact_loss: bool
                - sensor_health: FaultState
                - imu_fault: FaultState
                - mission_time: float (seconds)
                - progress: float (0.0 - 1.0)
                - distance_to_target: float (meters)
                
        Returns:
            RuleResult with conclusions
        """
        triggered = []
        recommendations = []
        min_priority = 10
        
        # Sort rules by priority
        sorted_rules = sorted(self.rules, key=lambda r: r.priority)
        
        for rule in sorted_rules:
            try:
                if rule.condition(context):
                    triggered.append(rule.name)
                    recommendations.append(rule.conclusion)
                    min_priority = min(min_priority, rule.priority)
            except Exception:
                # Rule evaluation failed - skip
                pass
        
        # Determine overall situation
        if min_priority == 1:
            situation = "CRITICAL"
        elif min_priority == 2:
            situation = "WARNING"
        elif min_priority <= 4:
            situation = "CAUTION"
        else:
            situation = "NOMINAL"
        
        return RuleResult(
            situation=situation,
            triggered_rules=triggered,
            recommendations=recommendations,
            priority=min_priority
        )
    
    def evaluate_situation(self, energy: float, imu_tilt: float,
                           sensor_health: FaultState) -> str:
        """
        Simplified evaluation matching Thought System interface.
        
        Returns situation assessment string.
        """
        context = {
            "battery": energy,
            "tilt": imu_tilt,
            "sensor_health": sensor_health,
        }
        
        result = self.evaluate(context)
        
        if result.recommendations:
            return f"{result.situation}: {result.recommendations[0]}"
        return f"{result.situation}: All systems nominal"
