"""
Command Data Structures.

Defines all command types sent to the robot control system.
These are the inputs that controllers receive from higher-level
planning and navigation systems.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum, auto


class WalkingMode(Enum):
    """Walking controller operation modes."""
    IDLE = auto()           # Standing still
    WALKING = auto()        # Normal walking
    TURNING = auto()        # In-place turning
    STEPPING = auto()       # Precise stepping (e.g., onto platform)
    RECOVERY = auto()       # Balance recovery
    CROUCH = auto()         # Lowered stance


class GaitType(Enum):
    """Type of gait pattern."""
    WALK = auto()           # Normal walking
    TROT = auto()           # Faster, alternating legs
    CRAWL = auto()          # Slow, one leg at a time
    GALLOP = auto()         # Running (not implemented)


@dataclass
class VelocityCommand:
    """
    Velocity command for locomotion.
    
    Standard twist-like command for mobile robot navigation.
    Used by high-level planners to command robot motion.
    
    Attributes:
        timestamp: Command timestamp
        linear_x: Forward velocity (m/s), positive = forward
        linear_y: Lateral velocity (m/s), positive = left
        angular_z: Turn rate (rad/s), positive = counter-clockwise
    """
    timestamp: float = 0.0
    
    # Linear velocities (m/s)
    linear_x: float = 0.0   # Forward/backward
    linear_y: float = 0.0   # Left/right (strafe)
    linear_z: float = 0.0   # Up/down (not used in walking)
    
    # Angular velocities (rad/s)
    angular_x: float = 0.0  # Roll rate (not used)
    angular_y: float = 0.0  # Pitch rate (not used)
    angular_z: float = 0.0  # Yaw rate (turn)
    
    # Command validity
    timeout: float = 0.5    # Command expires after this time (s)
    priority: int = 0       # Higher priority overrides lower
    
    @classmethod
    def stop(cls) -> 'VelocityCommand':
        """Create a stop (zero velocity) command."""
        return cls()
    
    @classmethod
    def forward(cls, speed: float = 0.3) -> 'VelocityCommand':
        """Create forward walking command."""
        return cls(linear_x=speed)
    
    @classmethod  
    def turn(cls, rate: float = 0.5) -> 'VelocityCommand':
        """Create in-place turn command."""
        return cls(angular_z=rate)
    
    def is_zero(self, threshold: float = 0.01) -> bool:
        """Check if command is effectively zero."""
        return (abs(self.linear_x) < threshold and
                abs(self.linear_y) < threshold and
                abs(self.angular_z) < threshold)
    
    def is_expired(self, current_time: float) -> bool:
        """Check if command has expired."""
        return current_time - self.timestamp > self.timeout
    
    def clamp(self, max_linear: float = 0.5, max_angular: float = 1.0) -> 'VelocityCommand':
        """Clamp velocities to safe limits."""
        return VelocityCommand(
            timestamp=self.timestamp,
            linear_x=np.clip(self.linear_x, -max_linear, max_linear),
            linear_y=np.clip(self.linear_y, -max_linear * 0.3, max_linear * 0.3),
            angular_z=np.clip(self.angular_z, -max_angular, max_angular),
            timeout=self.timeout,
        )
    
    def interpolate(self, other: 'VelocityCommand', alpha: float) -> 'VelocityCommand':
        """Linearly interpolate between two commands."""
        return VelocityCommand(
            linear_x=self.linear_x * (1 - alpha) + other.linear_x * alpha,
            linear_y=self.linear_y * (1 - alpha) + other.linear_y * alpha,
            angular_z=self.angular_z * (1 - alpha) + other.angular_z * alpha,
        )


@dataclass
class JointCommand:
    """
    Command for a single joint.
    
    Supports multiple control modes: position, velocity, or torque.
    """
    joint_name: str
    
    # Control mode selection (only one should be set)
    position: Optional[float] = None      # Target position (rad)
    velocity: Optional[float] = None      # Target velocity (rad/s)
    torque: Optional[float] = None        # Target torque (N·m)
    
    # Feedforward terms
    position_gain: float = 100.0    # kp for position control
    velocity_gain: float = 10.0     # kd for velocity control
    feedforward_torque: float = 0.0 # Additional torque
    
    @property
    def is_position_control(self) -> bool:
        return self.position is not None
    
    @property
    def is_velocity_control(self) -> bool:
        return self.velocity is not None
    
    @property
    def is_torque_control(self) -> bool:
        return self.torque is not None


@dataclass
class JointTrajectoryPoint:
    """
    Single point in a joint trajectory.
    """
    time_from_start: float = 0.0  # seconds
    
    positions: Dict[str, float] = field(default_factory=dict)
    velocities: Dict[str, float] = field(default_factory=dict)
    accelerations: Dict[str, float] = field(default_factory=dict)
    effort: Dict[str, float] = field(default_factory=dict)


@dataclass
class JointTrajectory:
    """
    Full joint trajectory command.
    
    Specifies a time-parameterized sequence of joint states.
    """
    timestamp: float = 0.0
    joint_names: List[str] = field(default_factory=list)
    points: List[JointTrajectoryPoint] = field(default_factory=list)
    
    @property
    def duration(self) -> float:
        """Total trajectory duration."""
        if not self.points:
            return 0.0
        return self.points[-1].time_from_start
    
    def sample(self, time: float) -> Optional[JointTrajectoryPoint]:
        """
        Sample trajectory at given time.
        
        Returns interpolated point, or None if time is out of range.
        """
        if not self.points or time < 0 or time > self.duration:
            return None
        
        # Find surrounding points
        for i, point in enumerate(self.points):
            if point.time_from_start >= time:
                if i == 0:
                    return self.points[0]
                
                # Linear interpolation
                prev = self.points[i - 1]
                alpha = (time - prev.time_from_start) / (point.time_from_start - prev.time_from_start)
                
                interpolated = JointTrajectoryPoint(time_from_start=time)
                for name in self.joint_names:
                    if name in prev.positions and name in point.positions:
                        interpolated.positions[name] = (
                            prev.positions[name] * (1 - alpha) + 
                            point.positions[name] * alpha
                        )
                    if name in prev.velocities and name in point.velocities:
                        interpolated.velocities[name] = (
                            prev.velocities[name] * (1 - alpha) + 
                            point.velocities[name] * alpha
                        )
                return interpolated
        
        return self.points[-1]


@dataclass
class WalkingCommand:
    """
    High-level walking command.
    
    Combines velocity command with additional walking-specific parameters.
    """
    timestamp: float = 0.0
    
    # Velocity targets
    velocity: VelocityCommand = field(default_factory=VelocityCommand)
    
    # Walking mode
    mode: WalkingMode = WalkingMode.IDLE
    gait: GaitType = GaitType.WALK
    
    # Step parameters (overrides defaults if set)
    step_height: Optional[float] = None      # m
    step_duration: Optional[float] = None    # s
    target_com_height: Optional[float] = None  # m
    
    # Foot placement targets (for precise stepping)
    left_foot_target: Optional[np.ndarray] = None   # [x, y, z] world frame
    right_foot_target: Optional[np.ndarray] = None
    
    # Terrain adaptation
    terrain_type: str = "flat"  # "flat", "stairs", "slope", "rough"
    compliance: float = 0.5     # 0=stiff, 1=compliant
    
    @classmethod
    def walk_forward(cls, speed: float = 0.3) -> 'WalkingCommand':
        """Create simple forward walking command."""
        return cls(
            velocity=VelocityCommand.forward(speed),
            mode=WalkingMode.WALKING,
        )
    
    @classmethod
    def stop(cls) -> 'WalkingCommand':
        """Create stop command."""
        return cls(
            velocity=VelocityCommand.stop(),
            mode=WalkingMode.IDLE,
        )


@dataclass
class FootPlacementCommand:
    """
    Explicit foot placement command.
    
    Used for precise stepping tasks like climbing stairs
    or stepping onto specific locations.
    """
    timestamp: float = 0.0
    
    # Target foot position in world frame
    foot_side: str = "left"
    target_position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    target_orientation: float = 0.0  # Yaw angle
    
    # Timing
    lift_time: float = 0.0      # When to lift foot
    land_time: float = 0.4      # When to land
    settle_time: float = 0.2    # Stabilization after landing
    
    # Trajectory parameters
    swing_height: float = 0.15   # Max height during swing
    approach_angle: float = 0.0  # Angle of approach (for stairs)
    
    # Safety
    force_limit: float = 100.0   # Max landing force (N)
    position_tolerance: float = 0.05  # Acceptable position error (m)


@dataclass
class BalanceCommand:
    """
    Balance/recovery command.
    
    Specifies desired body pose for balance control.
    """
    timestamp: float = 0.0
    
    # Desired COM position relative to support polygon
    com_offset: np.ndarray = field(default_factory=lambda: np.zeros(2))  # [x, y]
    
    # Desired body orientation (Euler angles)
    roll: float = 0.0   # rad
    pitch: float = 0.0  # rad
    yaw: float = 0.0    # rad
    
    # Desired COM height
    height: float = 1.45  # m
    
    # Recovery mode
    is_recovery: bool = False
    capture_point_target: Optional[np.ndarray] = None  # Override for recovery


@dataclass
class EmergencyCommand:
    """
    Emergency stop/recovery command.
    """
    timestamp: float = 0.0
    
    # Command type
    stop: bool = False           # Immediate stop
    freeze: bool = False         # Hold current position
    crouch: bool = False         # Lower to safe crouch
    shutdown: bool = False       # Power down actuators
    
    # Recovery
    attempt_recovery: bool = False
    recovery_timeout: float = 5.0  # Max time to try recovery
    
    reason: str = ""  # Why emergency was triggered
