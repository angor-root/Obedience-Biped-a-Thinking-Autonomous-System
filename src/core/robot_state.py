"""
Robot State Data Structures.

Defines all classes for representing the complete state of the
Obedience bipedal robot, including proprioceptive data.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum, auto


class StanceLeg(Enum):
    """Which leg is currently in stance phase."""
    LEFT = auto()
    RIGHT = auto()
    DOUBLE = auto()  # Both feet on ground


@dataclass
class JointState:
    """
    Complete state of a single joint.
    
    Attributes:
        name: Joint identifier (e.g., "right_hip_z_j")
        position: Angular position in radians
        velocity: Angular velocity in rad/s
        acceleration: Angular acceleration in rad/s²
        torque_commanded: Torque applied by actuator (N·m)
        torque_external: Torque from external forces (N·m)
        torque_constraint: Torque from joint limits/contacts (N·m)
    """
    name: str
    position: float = 0.0
    velocity: float = 0.0
    acceleration: float = 0.0
    torque_commanded: float = 0.0
    torque_external: float = 0.0
    torque_constraint: float = 0.0
    
    @property
    def total_torque(self) -> float:
        """Total torque acting on joint."""
        return self.torque_commanded + self.torque_external + self.torque_constraint
    
    def is_at_limit(self, lower: float, upper: float, margin: float = 0.05) -> Tuple[bool, bool]:
        """Check if joint is near its limits."""
        at_lower = self.position <= lower + margin
        at_upper = self.position >= upper - margin
        return at_lower, at_upper


@dataclass
class LegState:
    """
    Complete state of one leg (5-DOF).
    
    Joint order: hip_yaw, hip_roll, hip_pitch, knee, ankle
    """
    side: str  # "left" or "right"
    
    # Individual joint states
    hip_yaw: JointState = field(default_factory=lambda: JointState("hip_yaw"))
    hip_roll: JointState = field(default_factory=lambda: JointState("hip_roll"))
    hip_pitch: JointState = field(default_factory=lambda: JointState("hip_pitch"))
    knee: JointState = field(default_factory=lambda: JointState("knee"))
    ankle: JointState = field(default_factory=lambda: JointState("ankle"))
    
    # End-effector (foot) state in torso frame
    foot_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    foot_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    foot_orientation: np.ndarray = field(default_factory=lambda: np.eye(3))
    
    # Contact state
    is_stance: bool = False
    contact_force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    contact_torque: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Jacobian matrix (4x5: [x, y, z, yaw] vs [hip_z, hip_y, hip, knee, ankle])
    jacobian: np.ndarray = field(default_factory=lambda: np.zeros((4, 5)))
    
    @property
    def joints(self) -> List[JointState]:
        """Return all joints in order."""
        return [self.hip_yaw, self.hip_roll, self.hip_pitch, self.knee, self.ankle]
    
    @property
    def q(self) -> np.ndarray:
        """Joint positions as array."""
        return np.array([j.position for j in self.joints])
    
    @property
    def dq(self) -> np.ndarray:
        """Joint velocities as array."""
        return np.array([j.velocity for j in self.joints])
    
    @property
    def tau(self) -> np.ndarray:
        """Joint torques as array."""
        return np.array([j.total_torque for j in self.joints])
    
    def ground_reaction_force(self) -> np.ndarray:
        """Get GRF if in contact, zeros otherwise."""
        return self.contact_force if self.is_stance else np.zeros(3)


@dataclass
class TorsoState:
    """
    State of the robot's torso (floating base).
    """
    # Position in world frame
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Orientation as quaternion [w, x, y, z]
    orientation_quat: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0]))
    
    # Orientation as rotation matrix
    orientation_mat: np.ndarray = field(default_factory=lambda: np.eye(3))
    
    # Linear velocity in world frame
    linear_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Angular velocity in body frame
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Linear acceleration in world frame
    linear_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Angular acceleration in body frame
    angular_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    @property
    def heading(self) -> float:
        """Yaw angle (heading) in world frame."""
        forward = self.orientation_mat[:, 0]  # Local X axis
        return np.arctan2(forward[1], forward[0])
    
    @property
    def pitch(self) -> float:
        """Pitch angle."""
        forward = self.orientation_mat[:, 0]
        return np.arcsin(-forward[2])
    
    @property
    def roll(self) -> float:
        """Roll angle."""
        R = self.orientation_mat
        return np.arctan2(R[2, 1], R[2, 2])


@dataclass
class CenterOfMass:
    """
    Center of mass state.
    """
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))  # World frame
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Position relative to stance foot
    position_stance_frame: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    @property
    def height(self) -> float:
        """CoM height above ground."""
        return self.position[2]


@dataclass
class RobotState:
    """
    Complete state of the Obedience robot.
    
    This is the primary data structure published by the MuJoCo bridge
    and consumed by all control nodes.
    """
    # Timestamp
    timestamp: float = 0.0
    
    # Torso (floating base)
    torso: TorsoState = field(default_factory=TorsoState)
    
    # Legs
    left_leg: LegState = field(default_factory=lambda: LegState(side="left"))
    right_leg: LegState = field(default_factory=lambda: LegState(side="right"))
    
    # Center of mass
    com: CenterOfMass = field(default_factory=CenterOfMass)
    
    # Walking phase
    stance_leg: StanceLeg = StanceLeg.DOUBLE
    gait_phase: float = 0.0  # 0-1 within current step
    
    # Emergency flags
    is_falling: bool = False
    is_stable: bool = True
    
    def get_leg(self, side: str) -> LegState:
        """Get leg by side name."""
        return self.left_leg if side == "left" else self.right_leg
    
    def get_stance_leg(self) -> Optional[LegState]:
        """Get the current stance leg, or None if double support."""
        if self.stance_leg == StanceLeg.LEFT:
            return self.left_leg
        elif self.stance_leg == StanceLeg.RIGHT:
            return self.right_leg
        return None
    
    def get_swing_leg(self) -> Optional[LegState]:
        """Get the current swing leg, or None if double support."""
        if self.stance_leg == StanceLeg.LEFT:
            return self.right_leg
        elif self.stance_leg == StanceLeg.RIGHT:
            return self.left_leg
        return None
    
    @property
    def joint_positions(self) -> Dict[str, float]:
        """All joint positions as dictionary."""
        joints = {}
        for leg in [self.left_leg, self.right_leg]:
            prefix = leg.side
            for joint in leg.joints:
                joints[f"{prefix}_{joint.name}"] = joint.position
        return joints
    
    @property  
    def joint_velocities(self) -> Dict[str, float]:
        """All joint velocities as dictionary."""
        joints = {}
        for leg in [self.left_leg, self.right_leg]:
            prefix = leg.side
            for joint in leg.joints:
                joints[f"{prefix}_{joint.name}"] = joint.velocity
        return joints
    
    @property
    def joint_torques(self) -> Dict[str, float]:
        """All joint torques as dictionary."""
        joints = {}
        for leg in [self.left_leg, self.right_leg]:
            prefix = leg.side
            for joint in leg.joints:
                joints[f"{prefix}_{joint.name}"] = joint.total_torque
        return joints


@dataclass
class RobotConfig:
    """
    Static configuration of the Obedience robot.
    
    Contains physical parameters, joint limits, and control parameters.
    """
    name: str = "obedience"
    
    # Physical parameters
    total_mass: float = 12.6  # kg
    torso_mass: float = 8.0   # kg
    leg_mass: float = 2.3     # kg per leg
    
    # Dimensions (m)
    torso_height: float = 0.6   # Height of torso capsule
    thigh_length: float = 0.6   # Hip to knee
    shin_length: float = 0.6    # Knee to ankle
    foot_length: float = 0.16   # Foot size
    hip_width: float = 0.1      # Distance between hip joints
    
    # Standing height (torso COM to ground)
    standing_height: float = 1.6
    
    # Joint names
    joint_names: List[str] = field(default_factory=lambda: [
        "left_hip_z_j", "left_hip_y_j", "left_hip", "left_knee", "left_foot_j",
        "right_hip_z_j", "right_hip_y_j", "right_hip", "right_knee", "right_foot_j"
    ])
    
    # Joint limits (rad)
    joint_limits: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "left_hip_z_j": (-0.35, 0.35),    # ±20°
        "left_hip_y_j": (-0.35, 0.35),    # ±20°
        "left_hip": (-1.05, 2.09),        # -60° to +120°
        "left_knee": (-2.09, -0.087),     # -120° to -5°
        "left_foot_j": (-1.05, 1.05),     # ±60°
        "right_hip_z_j": (-0.35, 0.35),
        "right_hip_y_j": (-0.35, 0.35),
        "right_hip": (-1.05, 1.05),
        "right_knee": (-2.09, -0.087),
        "right_foot_j": (-1.05, 1.05),
    })
    
    # Velocity limits (rad/s)
    velocity_limits: Dict[str, float] = field(default_factory=lambda: {
        "hip_z": 5.0,
        "hip_y": 5.0,
        "hip": 5.0,
        "knee": 5.0,
        "ankle": 5.0,
    })
    
    # Torque limits (N·m) - from actuator kv gains
    torque_limits: Dict[str, float] = field(default_factory=lambda: {
        "hip_z": 15.0,   # kv=15
        "hip_y": 30.0,   # kv=30
        "hip": 30.0,     # kv=30
        "knee": 30.0,    # kv=30
        "ankle": 0.1,    # kv=0.1 (passive)
    })
    
    # Control frequencies (Hz)
    control_frequency: float = 1000.0  # Low-level joint control
    state_publish_frequency: float = 500.0  # State publishing
    walking_update_frequency: float = 33.3  # ~30 Hz walking control
    
    @property
    def total_height(self) -> float:
        """Total robot height when standing."""
        return self.torso_height + self.thigh_length + self.shin_length
    
    def get_joint_limit(self, joint_name: str) -> Tuple[float, float]:
        """Get limits for a specific joint."""
        return self.joint_limits.get(joint_name, (-np.pi, np.pi))
