"""
Core module - Data structures and classes for Obedience robot.

This module defines all the fundamental data types used throughout
the system for representing robot state, sensor data, and commands.
"""

from .robot_state import (
    RobotState,
    RobotConfig,
    JointState,
    LegState,
)
from .sensors import (
    IMUData,
    FootContact,
    ContactArray,
)
from .commands import (
    VelocityCommand,
    JointCommand,
    WalkingCommand,
)
