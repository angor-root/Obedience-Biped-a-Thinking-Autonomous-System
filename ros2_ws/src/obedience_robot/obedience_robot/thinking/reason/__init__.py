"""
Reason Subsystem - Rules and FMEA Evaluation

From the Thought System architecture:
- Laws Subsystem
- Rules Subsystem  
- Axioms Subsystem

evaluate_situation(self, energy, imu_tilt, sensor_health) -> str
"""

from .rules_engine import RulesEngine, Rule, RuleResult

__all__ = [
    'RulesEngine',
    'Rule',
    'RuleResult',
]
