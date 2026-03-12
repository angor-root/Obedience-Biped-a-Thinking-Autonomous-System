"""
Will Subsystem - Mission and Goal Management

From the Thought System architecture:
- Environment Subsystem
- Mission Subsystem
- Executive Subsystem
- Constellation Repair Subsystem

Manages high-level goals and mission state.
"""

from .mission_subsystem import MissionSubsystem, MissionGoal
from .executive_subsystem import ExecutiveSubsystem

__all__ = [
    'MissionSubsystem',
    'MissionGoal',
    'ExecutiveSubsystem',
]
