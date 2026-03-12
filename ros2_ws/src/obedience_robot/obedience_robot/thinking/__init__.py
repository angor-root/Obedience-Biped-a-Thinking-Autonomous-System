"""
Thinking System - High-Level Autonomy Module

Architecture based on cognitive systems with the following subsystems:
- Will: Mission management, execution, constellation repair
- Decision: Action selection, arbitration
- Reason: Rules engine, FMEA evaluation
- Understanding: Environment interpretation
- Sensory: Sensor processing (IMU, joints, contacts, LiDAR, battery)
- Presentation: Status reporting, alerts

This module integrates with ROS2 for high-level commands while
low-level control remains in the simulation loop.
"""

from .common import FaultState, SystemHealth, SensorStatus, LocomotionState, MissionPhase, FAULT_CHAINS

# Import submodules
from . import sensory
from . import will
from . import decision
from . import reason
from . import understanding
from . import presentation

__all__ = [
    # Common types
    'FaultState',
    'SystemHealth', 
    'SensorStatus',
    'LocomotionState',
    'MissionPhase',
    'FAULT_CHAINS',
    # Submodules
    'sensory',
    'will',
    'decision',
    'reason',
    'understanding',
    'presentation',
]
