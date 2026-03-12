"""
Sensor Data Structures.

Defines classes for IMU, force sensors, and contact detection.
These represent the exteroceptive and additional proprioceptive
data available from MuJoCo simulation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum, auto


class ContactState(Enum):
    """Foot contact state classification."""
    NO_CONTACT = auto()      # Foot in air
    HEEL_CONTACT = auto()    # Only heel touching
    TOE_CONTACT = auto()     # Only toe touching
    FULL_CONTACT = auto()    # Full foot on ground
    EDGE_CONTACT = auto()    # Partial/unstable contact
    SLIPPING = auto()        # Contact but sliding


@dataclass
class IMUData:
    """
    Inertial Measurement Unit data.
    
    Simulates a 6-axis IMU mounted on the robot's torso.
    In MuJoCo, this data comes from body accelerations and velocities.
    
    Attributes:
        timestamp: Time of measurement
        linear_acceleration: Accelerometer reading [ax, ay, az] in body frame (m/s²)
        angular_velocity: Gyroscope reading [wx, wy, wz] in body frame (rad/s)
        orientation: Estimated orientation as quaternion [w, x, y, z] (optional)
    """
    timestamp: float = 0.0
    
    # Raw sensor data
    linear_acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Filtered/estimated orientation (from complementary or Kalman filter)
    orientation: Optional[np.ndarray] = None  # [w, x, y, z]
    
    # Sensor noise characteristics (for Kalman filter)
    acc_noise_std: float = 0.1        # m/s² standard deviation
    gyro_noise_std: float = 0.01      # rad/s standard deviation
    acc_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Covariance matrices
    acc_covariance: np.ndarray = field(default_factory=lambda: np.eye(3) * 0.01)
    gyro_covariance: np.ndarray = field(default_factory=lambda: np.eye(3) * 0.001)
    
    def apply_noise(self, acc_noise: np.ndarray, gyro_noise: np.ndarray) -> 'IMUData':
        """Apply simulated sensor noise for realism."""
        return IMUData(
            timestamp=self.timestamp,
            linear_acceleration=self.linear_acceleration + acc_noise,
            angular_velocity=self.angular_velocity + gyro_noise,
            orientation=self.orientation,
            acc_noise_std=self.acc_noise_std,
            gyro_noise_std=self.gyro_noise_std,
        )
    
    @property
    def is_upright(self, threshold: float = 0.5) -> bool:
        """Check if Z-axis acceleration indicates upright orientation."""
        # When upright, Z acceleration should be close to -g
        return abs(self.linear_acceleration[2] + 9.81) < threshold


@dataclass
class FootContact:
    """
    Contact sensor data for one foot.
    
    Combines force sensing with contact geometry detection
    from MuJoCo's contact solver.
    
    Attributes:
        side: "left" or "right"
        timestamp: Time of measurement
        is_contact: Whether foot is touching ground
        contact_points: Number of contact points
        force: Contact force in world frame [fx, fy, fz] (N)
        torque: Contact torque in foot frame [tx, ty, tz] (N·m)
        cop_position: Center of pressure in foot frame [x, y] (m)
    """
    side: str
    timestamp: float = 0.0
    
    # Contact detection
    is_contact: bool = False
    contact_state: ContactState = ContactState.NO_CONTACT
    contact_points: int = 0
    
    # Force/torque sensing
    force: np.ndarray = field(default_factory=lambda: np.zeros(3))      # World frame
    torque: np.ndarray = field(default_factory=lambda: np.zeros(3))     # Foot frame
    force_local: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Foot frame
    
    # Center of pressure (relative to foot center)
    cop_position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    
    # Contact geometry
    contact_normal: np.ndarray = field(default_factory=lambda: np.array([0, 0, 1]))
    penetration_depth: float = 0.0
    
    @property
    def normal_force(self) -> float:
        """Vertical (normal) component of contact force."""
        return max(0.0, self.force[2])
    
    @property
    def tangential_force(self) -> float:
        """Horizontal (tangential) component of contact force."""
        return np.sqrt(self.force[0]**2 + self.force[1]**2)
    
    @property
    def friction_ratio(self) -> float:
        """Ratio of tangential to normal force (for slip detection)."""
        if self.normal_force < 1.0:  # Avoid division by small numbers
            return 0.0
        return self.tangential_force / self.normal_force
    
    def is_slipping(self, friction_coeff: float = 0.8) -> bool:
        """
        Detect if foot is slipping.
        
        Uses Coulomb friction model: |F_t| > μ * F_n indicates slip.
        """
        return self.is_contact and self.friction_ratio > friction_coeff
    
    def is_stable_contact(self, min_force: float = 10.0) -> bool:
        """Check if contact is stable (sufficient normal force, not slipping)."""
        return (self.is_contact and 
                self.normal_force > min_force and 
                not self.is_slipping())


@dataclass
class ContactArray:
    """
    Collection of all foot contacts.
    
    Provides convenience methods for analyzing overall contact state.
    """
    timestamp: float = 0.0
    left_foot: FootContact = field(default_factory=lambda: FootContact(side="left"))
    right_foot: FootContact = field(default_factory=lambda: FootContact(side="right"))
    
    @property
    def is_double_support(self) -> bool:
        """Both feet in contact."""
        return self.left_foot.is_contact and self.right_foot.is_contact
    
    @property
    def is_single_support(self) -> bool:
        """Exactly one foot in contact."""
        return self.left_foot.is_contact != self.right_foot.is_contact
    
    @property
    def is_flight(self) -> bool:
        """No feet in contact (should not happen in walking)."""
        return not self.left_foot.is_contact and not self.right_foot.is_contact
    
    @property
    def stance_side(self) -> Optional[str]:
        """Return which side is in stance, or None if double/flight."""
        if self.is_single_support:
            return "left" if self.left_foot.is_contact else "right"
        return None
    
    @property
    def total_vertical_force(self) -> float:
        """Sum of vertical GRF from both feet."""
        return self.left_foot.normal_force + self.right_foot.normal_force
    
    def get_foot(self, side: str) -> FootContact:
        """Get foot contact by side name."""
        return self.left_foot if side == "left" else self.right_foot
    
    def get_cop_global(self, left_foot_pos: np.ndarray, right_foot_pos: np.ndarray) -> np.ndarray:
        """
        Compute global center of pressure from both feet.
        
        Returns weighted average of foot positions based on normal forces.
        """
        total_force = self.total_vertical_force
        if total_force < 1.0:
            return np.zeros(3)
        
        cop = (self.left_foot.normal_force * left_foot_pos + 
               self.right_foot.normal_force * right_foot_pos) / total_force
        return cop


@dataclass
class JointTorqueSensor:
    """
    Torque sensor data for actuated joints.
    
    MuJoCo provides actual torques applied by actuators,
    as well as forces from constraints and external contacts.
    """
    joint_name: str
    timestamp: float = 0.0
    
    # Torques (N·m)
    commanded_torque: float = 0.0    # What we asked for
    actual_torque: float = 0.0       # What actuator applied
    external_torque: float = 0.0     # From contacts/collisions
    constraint_torque: float = 0.0   # From joint limits
    
    # Motor state
    motor_velocity: float = 0.0      # rad/s
    motor_current: float = 0.0       # A (estimated from torque)
    motor_temperature: float = 25.0  # °C (simulated)
    
    @property
    def total_torque(self) -> float:
        """Net torque on joint."""
        return self.actual_torque + self.external_torque + self.constraint_torque
    
    @property
    def tracking_error(self) -> float:
        """Difference between commanded and actual torque."""
        return self.commanded_torque - self.actual_torque
    
    def estimate_power(self) -> float:
        """Estimate mechanical power: P = τ * ω."""
        return abs(self.actual_torque * self.motor_velocity)


@dataclass
class ProprioceptionData:
    """
    Complete proprioceptive data snapshot.
    
    Combines all internal sensing for state estimation.
    """
    timestamp: float = 0.0
    
    # IMU
    imu: IMUData = field(default_factory=IMUData)
    
    # Contacts
    contacts: ContactArray = field(default_factory=ContactArray)
    
    # Joint torques (for each actuated joint)
    joint_torques: List[JointTorqueSensor] = field(default_factory=list)
    
    def get_joint_torque(self, joint_name: str) -> Optional[JointTorqueSensor]:
        """Get torque sensor for specific joint."""
        for jt in self.joint_torques:
            if jt.joint_name == joint_name:
                return jt
        return None
    
    @property
    def total_power_consumption(self) -> float:
        """Sum of power across all joints."""
        return sum(jt.estimate_power() for jt in self.joint_torques)
