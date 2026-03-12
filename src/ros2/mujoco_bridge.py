"""
MuJoCo-ROS2 Bridge Node.

This module provides the core bridge between MuJoCo simulation
and ROS2 ecosystem. It runs the physics simulation and publishes
robot state while subscribing to control commands.

Architecture:
    MuJoCo Simulation ←→ Bridge Node ←→ ROS2 Topics/Services

The bridge operates at high frequency (1kHz) to match MuJoCo's
timestep, publishing sensor data and applying control commands
each simulation step.

Usage:
    ros2 run obedience mujoco_bridge --ros-args -p scene:=hospital_scene.xml
"""

import os
import sys
import time
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import mujoco.viewer

from core.robot_state import (
    RobotState, RobotConfig, JointState, LegState, 
    TorsoState, CenterOfMass, StanceLeg
)
from core.sensors import (
    IMUData, FootContact, ContactArray, ContactState
)
from core.commands import VelocityCommand, WalkingCommand


# =============================================================================
# MuJoCo Data Extraction Utilities
# =============================================================================

def get_body_state(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> dict:
    """
    Extract complete state of a body from MuJoCo.
    
    Returns:
        dict with position, orientation, velocity, acceleration
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    
    return {
        "position": data.xpos[body_id].copy(),
        "orientation_mat": data.xmat[body_id].reshape(3, 3).copy(),
        "orientation_quat": data.xquat[body_id].copy(),
        "linear_velocity": data.cvel[body_id, 3:6].copy(),
        "angular_velocity": data.cvel[body_id, 0:3].copy(),
        "linear_acceleration": data.cacc[body_id, 3:6].copy(),
        "angular_acceleration": data.cacc[body_id, 0:3].copy(),
    }


def get_joint_state(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str) -> JointState:
    """
    Extract complete joint state from MuJoCo.
    """
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    qpos_addr = model.jnt_qposadr[jid]
    qvel_addr = model.jnt_dofadr[jid]
    
    return JointState(
        name=joint_name,
        position=data.qpos[qpos_addr],
        velocity=data.qvel[qvel_addr],
        acceleration=data.qacc[qvel_addr],
        torque_commanded=data.qfrc_actuator[qvel_addr],
        torque_external=data.qfrc_passive[qvel_addr],
        torque_constraint=data.qfrc_constraint[qvel_addr],
    )


def get_contacts_for_geom(model: mujoco.MjModel, data: mujoco.MjData, 
                          geom_name: str, ground_geoms: list) -> FootContact:
    """
    Extract contact information for a specific geom (foot).
    """
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    ground_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g) 
                  for g in ground_geoms if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g) >= 0]
    
    side = "left" if "left" in geom_name else "right"
    contact = FootContact(side=side, timestamp=data.time)
    
    total_force = np.zeros(3)
    contact_count = 0
    
    for i in range(data.ncon):
        c = data.contact[i]
        if (c.geom1 == geom_id and c.geom2 in ground_ids) or \
           (c.geom2 == geom_id and c.geom1 in ground_ids):
            
            contact_count += 1
            
            # Get contact force using mj_contactForce
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            
            # Accumulate forces (force[:3] is normal+tangent, force[3:] is torque)
            # Transform from contact frame to world frame
            contact_frame = c.frame.reshape(3, 3)
            world_force = contact_frame @ force[:3]
            total_force += world_force
    
    contact.is_contact = contact_count > 0
    contact.contact_points = contact_count
    contact.force = total_force
    
    if contact.is_contact:
        if contact.normal_force > 50:
            contact.contact_state = ContactState.FULL_CONTACT
        else:
            contact.contact_state = ContactState.EDGE_CONTACT
    else:
        contact.contact_state = ContactState.NO_CONTACT
    
    return contact


def get_imu_data(model: mujoco.MjModel, data: mujoco.MjData, 
                 body_name: str = "torso") -> IMUData:
    """
    Extract IMU-like data from torso body.
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    
    # Get body frame rotation
    R = data.xmat[body_id].reshape(3, 3)
    
    # Linear acceleration in body frame
    world_acc = data.cacc[body_id, 3:6].copy()
    # Add gravity (in world frame, then transform to body)
    world_acc[2] += 9.81  # Remove gravity compensation to get accelerometer reading
    body_acc = R.T @ world_acc
    
    # Angular velocity already in body frame from MuJoCo
    body_omega = data.cvel[body_id, 0:3].copy()
    
    return IMUData(
        timestamp=data.time,
        linear_acceleration=body_acc,
        angular_velocity=body_omega,
        orientation=data.xquat[body_id].copy(),
    )


# =============================================================================
# MuJoCo Bridge Class
# =============================================================================

@dataclass
class BridgeConfig:
    """Configuration for MuJoCo-ROS2 bridge."""
    # Scene file
    scene_xml: str = "hospital_scene.xml"
    
    # Simulation parameters
    realtime_factor: float = 1.0  # 1.0 = real-time, 0 = as fast as possible
    
    # Publishing rates (Hz)
    state_publish_rate: float = 500.0
    imu_publish_rate: float = 1000.0
    contact_publish_rate: float = 500.0
    
    # Robot configuration
    robot_config: RobotConfig = field(default_factory=RobotConfig)
    
    # Geom names for contact detection
    ground_geoms: list = field(default_factory=lambda: ["ground"])
    left_foot_geoms: list = field(default_factory=lambda: ["left_foot_geom", "left_shin_geom"])
    right_foot_geoms: list = field(default_factory=lambda: ["right_foot_geom", "right_shin_geom"])


class MuJoCoBridge:
    """
    Bridge between MuJoCo simulation and ROS2.
    
    This class can be used standalone (without ROS2) for testing,
    or integrated with ROS2 nodes.
    
    Standalone Usage:
        bridge = MuJoCoBridge(config)
        bridge.start()
        
        # In another thread or callback:
        state = bridge.get_robot_state()
        bridge.set_velocity_command(cmd)
        
    ROS2 Integration:
        The bridge provides callbacks that can be connected to
        ROS2 publishers/subscribers.
    """
    
    def __init__(self, config: Optional[BridgeConfig] = None):
        self.config = config or BridgeConfig()
        
        # MuJoCo objects
        self.model: Optional[mujoco.MjModel] = None
        self.data: Optional[mujoco.MjData] = None
        self.viewer: Optional[mujoco.viewer.Handle] = None
        
        # State cache
        self._robot_state = RobotState()
        self._imu_data = IMUData()
        self._contacts = ContactArray()
        
        # Command buffer
        self._velocity_cmd = VelocityCommand()
        self._joint_velocities: Dict[str, float] = {}
        
        # Callbacks (for ROS2 integration)
        self._state_callback: Optional[Callable[[RobotState], None]] = None
        self._imu_callback: Optional[Callable[[IMUData], None]] = None
        self._contact_callback: Optional[Callable[[ContactArray], None]] = None
        
        # Control
        self._running = False
        self._lock = threading.Lock()
        
        # Timing
        self._sim_time = 0.0
        self._last_state_publish = 0.0
        self._last_imu_publish = 0.0
        self._last_contact_publish = 0.0
        
        # Actuator mapping
        self._ctrl_idx = {
            "right_hip_z": 0, "right_hip_y": 1, "right_hip": 2,
            "right_knee": 3, "right_foot": 4,
            "left_hip_z": 5, "left_hip_y": 6, "left_hip": 7,
            "left_knee": 8, "left_foot": 9
        }
    
    def load_scene(self, xml_path: str):
        """Load MuJoCo scene from XML file."""
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Initialize robot pose (slightly bent knees for stability)
        for knee in ["right_knee", "left_knee"]:
            try:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, knee)
                self.data.qpos[self.model.jnt_qposadr[jid]] = np.deg2rad(-15)
            except:
                pass
        
        mujoco.mj_forward(self.model, self.data)
        print(f"[Bridge] Loaded scene: {xml_path}")
        print(f"[Bridge] Timestep: {self.model.opt.timestep}s")
    
    def set_callbacks(self,
                      state_cb: Optional[Callable[[RobotState], None]] = None,
                      imu_cb: Optional[Callable[[IMUData], None]] = None,
                      contact_cb: Optional[Callable[[ContactArray], None]] = None):
        """Set callbacks for data publishing (ROS2 integration)."""
        self._state_callback = state_cb
        self._imu_callback = imu_cb
        self._contact_callback = contact_cb
    
    def set_velocity_command(self, cmd: VelocityCommand):
        """Set velocity command from navigation."""
        with self._lock:
            self._velocity_cmd = cmd
    
    def set_joint_velocities(self, velocities: Dict[str, float]):
        """Set direct joint velocity commands from walking controller."""
        with self._lock:
            self._joint_velocities = velocities.copy()
    
    def get_robot_state(self) -> RobotState:
        """Get latest robot state."""
        with self._lock:
            return self._robot_state
    
    def get_imu_data(self) -> IMUData:
        """Get latest IMU data."""
        with self._lock:
            return self._imu_data
    
    def get_contacts(self) -> ContactArray:
        """Get latest contact data."""
        with self._lock:
            return self._contacts
    
    def _update_robot_state(self):
        """Extract complete robot state from MuJoCo."""
        if self.model is None or self.data is None:
            return
        
        state = RobotState(timestamp=self.data.time)
        
        # Torso state
        torso_data = get_body_state(self.model, self.data, "torso")
        state.torso = TorsoState(
            position=torso_data["position"],
            orientation_quat=torso_data["orientation_quat"],
            orientation_mat=torso_data["orientation_mat"],
            linear_velocity=torso_data["linear_velocity"],
            angular_velocity=torso_data["angular_velocity"],
            linear_acceleration=torso_data["linear_acceleration"],
            angular_acceleration=torso_data["angular_acceleration"],
        )
        
        # Left leg joints
        state.left_leg = LegState(side="left")
        state.left_leg.hip_yaw = get_joint_state(self.model, self.data, "left_hip_z_j")
        state.left_leg.hip_roll = get_joint_state(self.model, self.data, "left_hip_y_j")
        state.left_leg.hip_pitch = get_joint_state(self.model, self.data, "left_hip")
        state.left_leg.knee = get_joint_state(self.model, self.data, "left_knee")
        state.left_leg.ankle = get_joint_state(self.model, self.data, "left_foot_j")
        
        # Right leg joints
        state.right_leg = LegState(side="right")
        state.right_leg.hip_yaw = get_joint_state(self.model, self.data, "right_hip_z_j")
        state.right_leg.hip_roll = get_joint_state(self.model, self.data, "right_hip_y_j")
        state.right_leg.hip_pitch = get_joint_state(self.model, self.data, "right_hip")
        state.right_leg.knee = get_joint_state(self.model, self.data, "right_knee")
        state.right_leg.ankle = get_joint_state(self.model, self.data, "right_foot_j")
        
        # Contact state determines stance leg
        left_contact = any(get_contacts_for_geom(
            self.model, self.data, g, self.config.ground_geoms
        ).is_contact for g in self.config.left_foot_geoms 
        if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) >= 0)
        
        right_contact = any(get_contacts_for_geom(
            self.model, self.data, g, self.config.ground_geoms
        ).is_contact for g in self.config.right_foot_geoms
        if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) >= 0)
        
        state.left_leg.is_stance = left_contact
        state.right_leg.is_stance = right_contact
        
        if left_contact and right_contact:
            state.stance_leg = StanceLeg.DOUBLE
        elif left_contact:
            state.stance_leg = StanceLeg.LEFT
        elif right_contact:
            state.stance_leg = StanceLeg.RIGHT
        else:
            state.stance_leg = StanceLeg.DOUBLE  # Default to double if neither
        
        with self._lock:
            self._robot_state = state
    
    def _update_sensors(self):
        """Update IMU and contact sensor data."""
        if self.model is None or self.data is None:
            return
        
        # IMU
        imu = get_imu_data(self.model, self.data, "torso")
        
        # Contacts
        contacts = ContactArray(timestamp=self.data.time)
        
        # Try multiple foot geoms for contact detection
        for geom_name in self.config.left_foot_geoms:
            try:
                foot_contact = get_contacts_for_geom(
                    self.model, self.data, geom_name, self.config.ground_geoms
                )
                if foot_contact.is_contact:
                    contacts.left_foot = foot_contact
                    break
            except:
                pass
        
        for geom_name in self.config.right_foot_geoms:
            try:
                foot_contact = get_contacts_for_geom(
                    self.model, self.data, geom_name, self.config.ground_geoms
                )
                if foot_contact.is_contact:
                    contacts.right_foot = foot_contact
                    break
            except:
                pass
        
        with self._lock:
            self._imu_data = imu
            self._contacts = contacts
    
    def _apply_control(self):
        """Apply control commands to MuJoCo actuators."""
        if self.model is None or self.data is None:
            return
        
        with self._lock:
            joint_vels = self._joint_velocities.copy()
        
        # Apply joint velocity commands
        for name, vel in joint_vels.items():
            ctrl_name = name.replace("_j", "").replace("left_hip_z", "left_hip_z") \
                           .replace("right_hip_z", "right_hip_z")
            
            # Map to actuator index
            if "right_hip_z" in name:
                idx = self._ctrl_idx["right_hip_z"]
            elif "right_hip_y" in name:
                idx = self._ctrl_idx["right_hip_y"]
            elif "right_hip" in name and "z" not in name and "y" not in name:
                idx = self._ctrl_idx["right_hip"]
            elif "right_knee" in name:
                idx = self._ctrl_idx["right_knee"]
            elif "left_hip_z" in name:
                idx = self._ctrl_idx["left_hip_z"]
            elif "left_hip_y" in name:
                idx = self._ctrl_idx["left_hip_y"]
            elif "left_hip" in name and "z" not in name and "y" not in name:
                idx = self._ctrl_idx["left_hip"]
            elif "left_knee" in name:
                idx = self._ctrl_idx["left_knee"]
            else:
                continue
            
            self.data.ctrl[idx] = np.clip(vel, -5, 5)
    
    def _check_publish(self):
        """Check if it's time to publish data and invoke callbacks."""
        current_time = self.data.time if self.data else 0.0
        
        # State publishing
        state_period = 1.0 / self.config.state_publish_rate
        if current_time - self._last_state_publish >= state_period:
            self._last_state_publish = current_time
            if self._state_callback:
                self._state_callback(self._robot_state)
        
        # IMU publishing
        imu_period = 1.0 / self.config.imu_publish_rate
        if current_time - self._last_imu_publish >= imu_period:
            self._last_imu_publish = current_time
            if self._imu_callback:
                self._imu_callback(self._imu_data)
        
        # Contact publishing
        contact_period = 1.0 / self.config.contact_publish_rate
        if current_time - self._last_contact_publish >= contact_period:
            self._last_contact_publish = current_time
            if self._contact_callback:
                self._contact_callback(self._contacts)
    
    def step(self):
        """Execute one simulation step."""
        if self.model is None or self.data is None:
            return
        
        # Apply control
        self._apply_control()
        
        # Step physics
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        
        # Update state
        self._update_robot_state()
        self._update_sensors()
        
        # Check callbacks
        self._check_publish()
        
        self._sim_time = self.data.time
    
    def run_with_viewer(self, duration: float = float('inf')):
        """
        Run simulation with MuJoCo viewer.
        
        Args:
            duration: How long to run (seconds), inf for indefinite
        """
        if self.model is None:
            raise RuntimeError("No scene loaded. Call load_scene() first.")
        
        self._running = True
        dt = self.model.opt.timestep
        start_time = time.time()
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            self.viewer = viewer
            
            while viewer.is_running() and self._running:
                if self._sim_time >= duration:
                    break
                
                self.step()
                viewer.sync()
                
                # Real-time pacing
                if self.config.realtime_factor > 0:
                    elapsed_real = time.time() - start_time
                    elapsed_sim = self._sim_time
                    target_real = elapsed_sim / self.config.realtime_factor
                    
                    if elapsed_real < target_real:
                        time.sleep(target_real - elapsed_real)
        
        self._running = False
        self.viewer = None
    
    def stop(self):
        """Stop simulation."""
        self._running = False


# =============================================================================
# Standalone Test
# =============================================================================

def main():
    """Test the bridge standalone (without ROS2)."""
    import os
    
    # Find project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    xml_path = os.path.join(project_root, "models/xml/hospital_scene.xml")
    
    print("=" * 60)
    print("MuJoCo-ROS2 Bridge - Standalone Test")
    print("=" * 60)
    
    # Create bridge
    config = BridgeConfig()
    bridge = MuJoCoBridge(config)
    
    # Define callbacks to print data
    def on_state(state: RobotState):
        print(f"\r[State] t={state.timestamp:.2f}s | "
              f"pos=({state.torso.position[0]:.2f}, {state.torso.position[1]:.2f}) | "
              f"stance={state.stance_leg.name}", end="")
    
    bridge.set_callbacks(state_cb=on_state)
    
    # Load and run
    bridge.load_scene(xml_path)
    
    print("\n[Bridge] Starting simulation...")
    print("[Bridge] Press Ctrl+C to stop\n")
    
    try:
        bridge.run_with_viewer(duration=30.0)
    except KeyboardInterrupt:
        print("\n[Bridge] Stopped by user")
    
    print("\n[Bridge] Done")


if __name__ == "__main__":
    main()
