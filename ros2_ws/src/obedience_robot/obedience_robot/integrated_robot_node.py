#!/usr/bin/env python3
"""
Integrated Robot Node - MuJoCo + Walking Controller + ROS2 Interface

This node runs the complete robot simulation with integrated walking control.
The walking controller runs at high frequency (~500Hz) inside the simulation loop,
while ROS2 is only used for high-level commands and telemetry.

Architecture:
    ┌─────────────────────────────────────────┐
    │         HIGH LEVEL (ROS2)               │
    │  cmd_vel → Navigation commands          │
    │  telemetry ← joint_states, imu, odom    │
    └──────────────────┬──────────────────────┘
                       │
    ┌──────────────────▼──────────────────────┐
    │         LOW LEVEL (Internal)            │
    │  Walking Controller + MuJoCo Sim        │
    │           (~500 Hz loop)                │
    └─────────────────────────────────────────┘

Subscribed Topics:
    /cmd_vel (geometry_msgs/Twist): Velocity command (vx, vy, omega_z)

Published Topics:
    /joint_states (sensor_msgs/JointState): Joint positions, velocities, efforts
    /imu/data (sensor_msgs/Imu): IMU orientation, angular velocity, acceleration
    /odom (nav_msgs/Odometry): Robot odometry

Parameters:
    scene_xml (str): Path to MuJoCo scene XML
    use_viewer (bool): Launch MuJoCo viewer (default: true)
    publish_rate (float): Telemetry publish rate Hz (default: 50)
"""

import os
import sys
import copy
import json
import threading
import numpy as np

# Add project paths BEFORE other imports
# Find project root from ros2_ws location
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Handle both source and install locations
if 'ros2_ws/src' in _SCRIPT_DIR:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR))))
elif 'ros2_ws/build' in _SCRIPT_DIR:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR))))
elif 'ros2_ws/install' in _SCRIPT_DIR:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_SCRIPT_DIR))))
else:
    # Fallback: assume standard location
    PROJECT_ROOT = '/root/Obedience-Biped-a-Thinking-Autonomous-System'

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Header, Float32, String
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist, Vector3, Quaternion, Point, PoseWithCovariance, TwistWithCovariance
from nav_msgs.msg import Odometry

import mujoco
import mujoco.viewer

from walking.utils import (
    geoms_contacting_geoms,
    capsule_end_frame_world,
    world_p_to_frame,
    torso_state_in_stance_frame,
)
from walking.jacobian import get_pos_3d_jacobians


# ============================================================================
# Walking Controller Parameters
# ============================================================================
K_XI = -1.0       # Capture point tuning gain
K_PZ = 5          # Height control gain
K_PT = -15        # Orientation control gain
DY_ROCKING = 0.1  # Lateral rocking amplitude
G = 9.81          # Gravity
T_DES = 0.4       # Step period

FOOT_BODIES = {"Right": "right_shin", "Left": "left_shin"}
GROUND_GEOMS = ["ground"]

CTRL_IDX = {
    "right_hip_z": 0, "right_hip_y": 1, "right_hip": 2,
    "right_knee": 3, "right_foot": 4,
    "left_hip_z": 5, "left_hip_y": 6, "left_hip": 7,
    "left_knee": 8, "left_foot": 9
}

JOINT_NAMES = [
    "left_hip_z_j", "left_hip_y_j", "left_hip", "left_knee", "left_foot_j",
    "right_hip_z_j", "right_hip_y_j", "right_hip", "right_knee", "right_foot_j"
]


# ============================================================================
# Walking Controller Functions
# ============================================================================
def compute_capture_point_x(x_com, dx_com, ddx_com, z_com, dx_des):
    z_com = np.clip(z_com, 0.1, 1000)
    omega_n = np.sqrt(G / z_com)
    xi = x_com + dx_com / omega_n
    dxi = dx_com + ddx_com / omega_n
    xi_des = x_com + dx_des / omega_n
    return xi - dxi / omega_n - K_XI * (xi - xi_des)


def compute_capture_point_y(y_com, dy_com, ddy_com, z_com, dy_des):
    z_com = np.clip(z_com, 0.1, 1000)
    omega_n = np.sqrt(G / z_com)
    xi = y_com + dy_com / omega_n
    dxi = dy_com + ddy_com / omega_n
    xi_des = y_com + dy_des / omega_n
    return xi - dxi / omega_n - K_XI * (xi - xi_des)


def compute_swing_height(z_target, h_target, h_0, t_start, t_now, e_step_down):
    delta_t = t_now - t_start
    tau = delta_t / T_DES + e_step_down
    tau_clipped = np.clip(tau, 0.0, 1.0)
    h_m = max(1.1 * h_0, h_target)
    if tau_clipped < 0.5:
        z_swing = (h_m - h_0) * np.sin(np.pi * tau_clipped) + h_0
    else:
        z_swing = h_m * np.sin(np.pi * tau_clipped)
    z_offset = -tau + 1 if tau > 1 else 0.0
    return z_swing + z_offset, z_target + z_offset


def rotation_error(R, R_d):
    skew = 0.5 * (np.matmul(R_d.T, R) - np.matmul(R.T, R_d))
    return np.array([skew[2, 1], skew[0, 2], skew[1, 0]])


def stance_foot_velocity(p_com, z_com, R_com, R_com_des):
    vec_h = -p_com / np.linalg.norm(p_com)
    vec_theta = np.array([-vec_h[2], 0, vec_h[0]])
    vec_phi = np.array([0, -1, 0])
    ez = p_com[2] - z_com
    v_z = K_PZ * ez * vec_h
    er = rotation_error(R_com, R_com_des)
    v_theta = K_PT * er[1] * vec_theta
    v_phi = K_PT * er[0] * vec_phi
    return -v_z + v_theta + v_phi


def swing_leg_controller(p_des, p_current):
    error = p_des - p_current
    error[1] *= 0.8
    return np.clip(10 * error, -10, 10)


def turn_controller(q_l, q_r, dz_omega, t, stance_leg):
    delta_omega = dz_omega * T_DES
    t = np.clip(t, 0, T_DES)
    if stance_leg == "Right":
        offset = min(delta_omega / 2, q_r) if delta_omega > 0 else max(delta_omega / 2, q_r)
        q_r_des = offset - delta_omega * (t / T_DES)
        q_l_des = -delta_omega / 2 + delta_omega * (t / T_DES)
        return q_r_des - q_r, q_l_des - q_l
    else:
        offset = min(delta_omega / 2, q_l) if delta_omega > 0 else max(delta_omega / 2, q_l)
        q_l_des = offset - delta_omega * (t / T_DES)
        q_r_des = -delta_omega / 2 + delta_omega * (t / T_DES)
        return q_l_des - q_l, q_r_des - q_r


class WalkingController:
    """Capture Point Walking Controller."""
    
    def __init__(self, model, data, t0):
        self.model = model
        self.data = data
        self.stance_side = "Right"
        self.swing_side = "Left"
        self.z_target = 1.45
        self.h_target = 0.2
        self.R_target = np.eye(3)
        self.min_step_time = 0.2
        self.pre_step_time = 0.0
        self.contact_lifted = False
        self.switch_props = {"t0": None, "p_foot_0": None, "p_com_0": None}
        self._initialize(t0)

    def _get_pose(self):
        p_stance_w, R_stance_w = capsule_end_frame_world(
            self.model, self.data, FOOT_BODIES[self.stance_side]
        )
        p_swing_w, _ = capsule_end_frame_world(
            self.model, self.data, FOOT_BODIES[self.swing_side]
        )
        p_swing = world_p_to_frame(p_swing_w, p_stance_w, R_stance_w)
        torso_state = torso_state_in_stance_frame(
            self.model, self.data, p_stance_w, R_stance_w
        )
        torso_state["position"][0] -= 0.05
        return p_swing, torso_state

    def _initialize(self, t0, dx_des=0.0):
        p_swing, torso_state = self._get_pose()
        self.switch_props["p_foot_0"] = copy.deepcopy(p_swing)
        self.switch_props["p_com_0"] = copy.deepcopy(torso_state["position"])
        self.switch_props["t0"] = t0

    def _switch_leg(self, contact_l, contact_r, t):
        contact = contact_l if self.swing_side == "Left" else contact_r
        if not contact:
            self.contact_lifted = True
        min_time = t > self.pre_step_time + self.min_step_time
        if contact and min_time:
            self.contact_lifted = False
            self.stance_side, self.swing_side = self.swing_side, self.stance_side
            self.pre_step_time = t
            return True
        return False

    def step(self, contact_l, contact_r, q, dx_des, dy_des, dz_omega, t):
        switch = self._switch_leg(contact_l, contact_r, t)
        p_swing, torso_state = self._get_pose()
        
        p_x = compute_capture_point_x(
            torso_state["position"][0], torso_state["velocity"][0],
            torso_state["acceleration"][0], torso_state["position"][2], dx_des
        )
        
        dy_rock = -DY_ROCKING if self.stance_side == "Right" else DY_ROCKING
        p_y = compute_capture_point_y(
            torso_state["position"][1], torso_state["velocity"][1],
            torso_state["acceleration"][1], torso_state["position"][2], dy_des + dy_rock
        )
        
        e_step_down = 0
        if self.stance_side == "Right":
            if p_y < 0.0:
                e_step_down += -2 * p_y
            elif p_y > 0.5:
                e_step_down += 2 * (p_y - 0.5)
            p_y = np.clip(p_y, 0.15, 0.6)
        else:
            if p_y > 0.0:
                e_step_down += 2 * p_y
            elif p_y < -0.5:
                e_step_down += -2 * (p_y + 0.5)
            p_y = np.clip(p_y, -0.6, -0.15)
        
        if abs(p_x) > 0.5:
            e_step_down += 2 * (abs(p_x) - 0.5)
        p_x = np.clip(p_x, -0.6, 0.6)
        
        if switch:
            self.switch_props["p_foot_0"] = copy.deepcopy(p_swing)
            self.switch_props["p_com_0"] = copy.deepcopy(torso_state["position"])
            self.switch_props["t0"] = t
        
        z_swing, z_com = compute_swing_height(
            self.z_target, self.h_target, self.switch_props["p_foot_0"][2],
            self.switch_props["t0"], t, e_step_down
        )
        
        vel_stance = stance_foot_velocity(
            torso_state["position"], z_com, torso_state["orientation"], self.R_target
        )
        vel_swing = swing_leg_controller(np.array([p_x, p_y, z_swing]), p_swing)
        
        dqz_stance, dqz_swing = turn_controller(
            q[0], q[5], dz_omega, t - self.switch_props["t0"], self.stance_side
        )
        
        Jr, Jl = get_pos_3d_jacobians(q)
        J_stance, J_swing = (Jr, Jl) if self.stance_side == "Right" else (Jl, Jr)
        
        R_t = torso_state["orientation"]
        v_stance = R_t.T @ vel_stance.reshape(3, 1)
        vel_vec_stance = np.array([[v_stance[0, 0]], [v_stance[1, 0]], [v_stance[2, 0]], [5 * dqz_stance]])
        dq_stance = np.linalg.pinv(J_stance) @ vel_vec_stance
        
        v_swing = R_t.T @ vel_swing.reshape(3, 1)
        vel_vec_swing = np.array([[v_swing[0, 0]], [v_swing[1, 0]], [v_swing[2, 0]], [5 * dqz_swing]])
        dq_swing = np.linalg.pinv(J_swing) @ vel_vec_swing
        
        if self.stance_side == "Right":
            return {
                "right_hip_z": dq_stance[0, 0], "right_hip_y": dq_stance[1, 0],
                "right_hip": dq_stance[2, 0], "right_knee": dq_stance[3, 0],
                "left_hip_z": dq_swing[0, 0], "left_hip_y": dq_swing[1, 0],
                "left_hip": dq_swing[2, 0], "left_knee": dq_swing[3, 0]
            }
        else:
            return {
                "left_hip_z": dq_stance[0, 0], "left_hip_y": dq_stance[1, 0],
                "left_hip": dq_stance[2, 0], "left_knee": dq_stance[3, 0],
                "right_hip_z": dq_swing[0, 0], "right_hip_y": dq_swing[1, 0],
                "right_hip": dq_swing[2, 0], "right_knee": dq_swing[3, 0]
            }


# ============================================================================
# Integrated ROS2 Node
# ============================================================================
class IntegratedRobotNode(Node):
    """
    Integrated Robot Node with internal walking controller.
    
    This node runs the full simulation loop with walking control at high frequency,
    while exposing ROS2 interfaces for high-level commands and telemetry.
    """
    
    def __init__(self):
        super().__init__('obedience_robot')
        
        # Parameters
        self.declare_parameter('scene_xml', 'hospital_scene.xml')
        self.declare_parameter('use_viewer', True)
        self.declare_parameter('publish_rate', 50.0)
        
        scene_xml = self.get_parameter('scene_xml').value
        self.use_viewer = self.get_parameter('use_viewer').value
        self.publish_rate = self.get_parameter('publish_rate').value
        
        # Load MuJoCo model
        self._load_scene(scene_xml)
        
        # Velocity command from ROS2 (protected by lock)
        self.cmd_lock = threading.Lock()
        self.cmd_vel = {'vx': 0.0, 'vy': 0.0, 'omega': 0.0}
        
        # Navigation target (protected by lock)
        self.nav_lock = threading.Lock()
        self.nav_target = None  # [x, y] or None
        
        # QoS for real-time
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Publishers (telemetry)
        self.joint_pub = self.create_publisher(JointState, '/joint_states', qos)
        self.imu_pub = self.create_publisher(Imu, '/imu', qos)  # Changed from /imu/data
        self.odom_pub = self.create_publisher(Odometry, '/odom', qos)
        
        # Additional publishers for thinking/health nodes
        self.battery_pub = self.create_publisher(Float32, '/battery_level', qos)
        self.contact_pub = self.create_publisher(String, '/contacts', qos)
        self.position_pub = self.create_publisher(Point, '/robot_position', qos)
        
        # Battery simulation state
        self._battery_level = 100.0  # Start at 100%
        self._battery_drain_rate = 0.001  # % per second while walking
        
        # Fault injection state
        self._external_force = np.zeros(6)  # [torque_x, torque_y, torque_z, force_x, force_y, force_z]
        self._force_duration = 0.0
        self._motor_fault = False
        self._sensor_fault = False
        
        # Subscribers
        self.cmd_sub = self.create_subscription(
            Twist, '/cmd_vel', self._cmd_vel_callback, qos)
        # Navigation target subscriber (Point message: x, y)
        self.nav_sub = self.create_subscription(
            Point, '/nav_target', self._nav_target_callback, qos)
        # Fault injection subscriber
        self.fault_sub = self.create_subscription(
            String, '/robot_perturbation', self._perturbation_callback, qos)
        
        # Static equilibrium command (for charging)
        from std_msgs.msg import Bool
        self.equilibrium_sub = self.create_subscription(
            Bool, '/static_equilibrium', self._equilibrium_callback, qos)
        self._static_equilibrium_mode = False
        self._charging_at_station = False
        
        # Battery injection subscriber (for fault injection from GUI)
        self.battery_set_sub = self.create_subscription(
            Float32, '/set_battery', self._set_battery_callback, qos)
        
        # Simulation state
        self.running = True
        self.sim_time = 0.0
        
        self.get_logger().info('Integrated Robot Node initialized')
        self.get_logger().info(f'Scene: {scene_xml}')
        self.get_logger().info(f'Viewer: {self.use_viewer}')
        self.get_logger().info(f'Telemetry rate: {self.publish_rate} Hz')
        self.get_logger().info('Nav target: publish to /nav_target (Point msg)')
        self.get_logger().info('Perturbation: /robot_perturbation (JSON)')
    
    def _perturbation_callback(self, msg: String):
        """Handle fault injection perturbations.
        
        Perturbations are designed to INTERFERE, not cause complete falls.
        The robot should recover from these disturbances.
        """
        try:
            data = json.loads(msg.data)
            fault_type = data.get('type', '')
            
            if fault_type == 'push':
                # Apply external push to torso (moderate)
                fx = data.get('force_x', 0.0)
                fy = data.get('force_y', 0.0)
                fz = data.get('force_z', 0.0)
                # Clamp forces to reasonable levels
                fx = np.clip(fx, -80, 80)
                fy = np.clip(fy, -80, 80)
                fz = np.clip(fz, -50, 50)
                self._external_force = np.array([0, 0, 0, fx, fy, fz])
                self._force_duration = min(data.get('duration', 0.2), 0.3)
                self.get_logger().warn(f'PERTURBATION: Push [{fx}, {fy}, {fz}]N for {self._force_duration}s')
                
            elif fault_type == 'trip':
                # Moderate lateral push - robot should recover
                self._external_force = np.array([0, 0, 0, 50.0, 30.0, 0])
                self._force_duration = 0.15
                self.get_logger().warn('PERTURBATION: Trip (recoverable)')
                
            elif fault_type == 'stumble':
                # Light perturbation - slight balance challenge
                self._external_force = np.array([0, 0, 0, 25.0, 15.0, 0])
                self._force_duration = 0.1
                self.get_logger().warn('PERTURBATION: Stumble (light)')
                
            elif fault_type == 'wind':
                # Sustained light force like wind
                self._external_force = np.array([0, 0, 0, 15.0, 10.0, 0])
                self._force_duration = 1.0
                self.get_logger().info('PERTURBATION: Wind gust')
                
            elif fault_type == 'fall':
                # Strong push but still potentially recoverable
                self._external_force = np.array([0, 0, 0, 80.0, 0, -30])
                self._force_duration = 0.25
                self._motor_fault = False  # Don't disable motors
                self.get_logger().error('PERTURBATION: Strong push (potential fall)')
                
            elif fault_type == 'critical_fall':
                # Only this should cause actual fall - very strong
                self._external_force = np.array([0, 0, 0, 200.0, 0, -80])
                self._force_duration = 0.4
                self._motor_fault = True
                self.get_logger().error('PERTURBATION: CRITICAL FALL induced!')
                
            elif fault_type == 'motor_fault':
                self._motor_fault = data.get('enabled', True)
                self.get_logger().warn(f'MOTOR FAULT: {"ON" if self._motor_fault else "OFF"}')
                
            elif fault_type == 'sensor_fault':
                self._sensor_fault = data.get('enabled', True)
                self.get_logger().warn(f'SENSOR FAULT: {"ON" if self._sensor_fault else "OFF"}')
                
            elif fault_type == 'clear':
                self._external_force = np.zeros(6)
                self._force_duration = 0.0
                self._motor_fault = False
                self._sensor_fault = False
                self.get_logger().info('PERTURBATIONS CLEARED')
                
        except json.JSONDecodeError:
            self.get_logger().error('Invalid perturbation JSON')
    
    def _equilibrium_callback(self, msg):
        """Handle static equilibrium command."""
        if msg.data and not self._static_equilibrium_mode:
            self._static_equilibrium_mode = True
            self._charging_at_station = True
            # Clear navigation
            with self.nav_lock:
                self.nav_target = None
            with self.cmd_lock:
                self.cmd_vel = {'vx': 0.0, 'vy': 0.0, 'omega': 0.0}
            self.get_logger().info('=' * 50)
            self.get_logger().info('  STATIC EQUILIBRIUM MODE ACTIVATED')
            self.get_logger().info('  Robot maintaining stationary balance')
            self.get_logger().info('  Charging in progress...')
            self.get_logger().info('=' * 50)
        elif not msg.data and self._static_equilibrium_mode:
            self._static_equilibrium_mode = False
            self._charging_at_station = False
            self.get_logger().info('Static equilibrium mode deactivated')
    
    def _set_battery_callback(self, msg: Float32):
        """Handle direct battery level injection."""
        old_level = self._battery_level
        self._battery_level = max(0.0, min(100.0, msg.data))
        self.get_logger().warn(f'BATTERY INJECTION: {old_level:.1f}% -> {self._battery_level:.1f}%')
    
    def _load_scene(self, scene_xml):
        """Load MuJoCo scene."""
        # Try multiple paths
        paths_to_try = [
            scene_xml,
            os.path.join(PROJECT_ROOT, 'models/xml', scene_xml),
            os.path.join(PROJECT_ROOT, scene_xml),
        ]
        
        for path in paths_to_try:
            if os.path.exists(path):
                self.model = mujoco.MjModel.from_xml_path(path)
                self.data = mujoco.MjData(self.model)
                self.get_logger().info(f'Loaded: {path}')
                return
        
        raise FileNotFoundError(f'Scene not found: {scene_xml}')
    
    def _cmd_vel_callback(self, msg: Twist):
        """Handle velocity commands from high-level navigation."""
        with self.cmd_lock:
            self.cmd_vel['vx'] = msg.linear.x
            self.cmd_vel['vy'] = msg.linear.y
            self.cmd_vel['omega'] = msg.angular.z
        # Clear navigation target when manual cmd_vel is received
        with self.nav_lock:
            self.nav_target = None
    
    def _nav_target_callback(self, msg):
        """Handle navigation target (Point message)."""
        with self.nav_lock:
            self.nav_target = np.array([msg.x, msg.y])
            self.get_logger().info(f'Navigation target: ({msg.x:.2f}, {msg.y:.2f})')
    
    def _compute_nav_velocity(self, robot_pos, robot_heading, target):
        """Compute velocity commands to navigate to target (like mission planner)."""
        delta = target - robot_pos[:2]
        distance = np.linalg.norm(delta)
        
        # Check if arrived
        if distance < 0.4:
            with self.nav_lock:
                self.nav_target = None
            self.get_logger().info('Navigation target reached!')
            return 0.0, 0.0, 0.0
        
        # Calculate required heading
        target_heading = np.arctan2(delta[1], delta[0])
        
        # Heading error (normalize to [-pi, pi])
        heading_error = target_heading - robot_heading
        heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))
        
        # Turn rate (proportional control)
        turn_rate = np.clip(1.5 * heading_error, -0.5, 0.5)
        
        # Forward velocity (reduce when turning, increase when aligned)
        forward_vel = 0.15 * (1.0 - abs(heading_error) / np.pi)
        forward_vel = np.clip(forward_vel, 0.03, 0.15)
        
        return forward_vel, 0.0, turn_rate
    
    def _get_robot_heading(self):
        """Get robot heading angle from torso orientation."""
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        R = self.data.xmat[torso_id].reshape(3, 3)
        forward = R[:, 0]  # Local +X is forward
        return np.arctan2(forward[1], forward[0])
    
    def _get_cmd_vel(self):
        """Get current velocity command. Uses nav target if set, else manual cmd_vel."""
        # Velocity limits
        MAX_VX = 0.15
        MAX_VY = 0.05
        MAX_OMEGA = 0.5
        
        # Check for navigation target first
        with self.nav_lock:
            nav_target = self.nav_target.copy() if self.nav_target is not None else None
        
        if nav_target is not None:
            # Automatic navigation mode
            torso = self._get_torso_state()
            robot_pos = torso['position']
            robot_heading = self._get_robot_heading()
            vx, vy, omega = self._compute_nav_velocity(robot_pos, robot_heading, nav_target)
            return vx, vy, omega
        
        # Manual cmd_vel mode
        with self.cmd_lock:
            vx = np.clip(self.cmd_vel['vx'], -MAX_VX, MAX_VX)
            vy = np.clip(self.cmd_vel['vy'], -MAX_VY, MAX_VY)
            omega = np.clip(self.cmd_vel['omega'], -MAX_OMEGA, MAX_OMEGA)
            return vx, vy, omega
    
    def _get_joint_angle(self, joint_name):
        """Get joint angle by name."""
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        return self.data.qpos[self.model.jnt_qposadr[jid]]
    
    def _get_torso_state(self):
        """Get torso position and orientation."""
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        pos = self.data.xpos[torso_id].copy()
        quat = self.data.xquat[torso_id].copy()
        vel = self.data.cvel[torso_id].copy()  # [angular, linear]
        R = self.data.xmat[torso_id].reshape(3, 3)
        return {
            'position': pos,
            'quaternion': quat,
            'rotation': R,
            'linear_vel': vel[3:6],
            'angular_vel': vel[0:3]
        }
    
    def _check_foot_contact(self, foot: str) -> bool:
        """Check if specified foot is in contact with ground.
        
        Args:
            foot: 'left' or 'right'
            
        Returns:
            True if foot is contacting ground
        """
        geom_name = f"{foot}_foot_geom"
        contacts = geoms_contacting_geoms(
            self.model, self.data,
            [geom_name],
            GROUND_GEOMS
        )
        return contacts.get(geom_name, False)
    
    def _publish_telemetry(self):
        """Publish sensor data to ROS2."""
        now = self.get_clock().now().to_msg()
        
        # Joint states
        joint_msg = JointState()
        joint_msg.header = Header(stamp=now, frame_id='base_link')
        joint_msg.name = JOINT_NAMES
        
        positions = []
        velocities = []
        efforts = []
        for name in JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_idx = self.model.jnt_qposadr[jid]
            qvel_idx = self.model.jnt_dofadr[jid]
            positions.append(float(self.data.qpos[qpos_idx]))
            velocities.append(float(self.data.qvel[qvel_idx]))
            # Effort from actuator
            ctrl_name = name.replace('_j', '') if name.endswith('_j') else name
            if ctrl_name in CTRL_IDX:
                efforts.append(float(self.data.ctrl[CTRL_IDX[ctrl_name]]))
            else:
                efforts.append(0.0)
        
        joint_msg.position = positions
        joint_msg.velocity = velocities
        joint_msg.effort = efforts
        self.joint_pub.publish(joint_msg)
        
        # IMU
        torso = self._get_torso_state()
        imu_msg = Imu()
        imu_msg.header = Header(stamp=now, frame_id='imu_link')
        
        q = torso['quaternion']
        imu_msg.orientation = Quaternion(x=q[1], y=q[2], z=q[3], w=q[0])
        
        ang_vel = torso['angular_vel']
        imu_msg.angular_velocity = Vector3(x=ang_vel[0], y=ang_vel[1], z=ang_vel[2])
        
        # Accelerometer (linear acceleration + gravity in body frame)
        R = torso['rotation']
        gravity_world = np.array([0, 0, 9.81])
        gravity_body = R.T @ gravity_world
        imu_msg.linear_acceleration = Vector3(x=gravity_body[0], y=gravity_body[1], z=gravity_body[2])
        
        imu_msg.orientation_covariance = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]
        imu_msg.angular_velocity_covariance = [0.001, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.001]
        imu_msg.linear_acceleration_covariance = [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]
        self.imu_pub.publish(imu_msg)
        
        # Odometry
        odom_msg = Odometry()
        odom_msg.header = Header(stamp=now, frame_id='odom')
        odom_msg.child_frame_id = 'base_link'
        
        pos = torso['position']
        odom_msg.pose.pose.position.x = pos[0]
        odom_msg.pose.pose.position.y = pos[1]
        odom_msg.pose.pose.position.z = pos[2]
        odom_msg.pose.pose.orientation = Quaternion(x=q[1], y=q[2], z=q[3], w=q[0])
        
        lin_vel = torso['linear_vel']
        odom_msg.twist.twist.linear = Vector3(x=lin_vel[0], y=lin_vel[1], z=lin_vel[2])
        odom_msg.twist.twist.angular = Vector3(x=ang_vel[0], y=ang_vel[1], z=ang_vel[2])
        self.odom_pub.publish(odom_msg)
        
        # Robot Position (for thinking/health nodes)
        pos_msg = Point()
        pos_msg.x = pos[0]
        pos_msg.y = pos[1]
        pos_msg.z = pos[2]  # Height - used for fall detection
        self.position_pub.publish(pos_msg)
        
        # Battery Level (simulated)
        battery_msg = Float32()
        battery_msg.data = self._battery_level
        self.battery_pub.publish(battery_msg)
        
        # Contact sensors
        left_contact = self._check_foot_contact('left')
        right_contact = self._check_foot_contact('right')
        contact_msg = String()
        contact_msg.data = json.dumps({
            'left_foot': left_contact,
            'right_foot': right_contact
        })
        self.contact_pub.publish(contact_msg)
    
    def run(self):
        """Main simulation loop with integrated walking control."""
        dt = self.model.opt.timestep
        
        # Initialize robot pose (bent knees)
        for knee in ["right_knee", "left_knee"]:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, knee)
            self.data.qpos[self.model.jnt_qposadr[jid]] = np.deg2rad(-15)
        
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        
        # Initialize walking controller
        walking = WalkingController(self.model, self.data, self.sim_time)
        dq_cmd = {}
        
        # Telemetry timing
        publish_period = 1.0 / self.publish_rate
        last_publish = 0.0
        
        # Control timing (every 30 simulation steps)
        ctrl_decimation = 30
        step_count = 0
        
        def simulation_loop(viewer=None):
            nonlocal dq_cmd, last_publish, step_count
            
            while self.running and rclpy.ok():
                if viewer is not None and not viewer.is_running():
                    break
                
                # Simulation step
                self.sim_time += dt
                mujoco.mj_step(self.model, self.data)
                mujoco.mj_forward(self.model, self.data)
                step_count += 1
                
                # Get velocity command from ROS2
                # In static equilibrium mode, force zero velocity
                if self._static_equilibrium_mode:
                    dx_cmd, dy_cmd, dz_cmd = 0.0, 0.0, 0.0
                    # Also simulate charging (very slow: 0.0001% per step @ 200Hz = ~0.02%/sec = ~100% in ~80min)
                    if self._charging_at_station:
                        self._battery_level = min(100.0, self._battery_level + 0.0001)
                else:
                    dx_cmd, dy_cmd, dz_cmd = self._get_cmd_vel()
                
                # Get foot contacts
                contact = geoms_contacting_geoms(
                    self.model, self.data,
                    ["left_shin_geom", "right_shin_geom", "right_foot_geom", "left_foot_geom"],
                    GROUND_GEOMS
                )
                
                # Get joint angles
                q = np.array([
                    self._get_joint_angle("left_hip_z_j"),
                    self._get_joint_angle("left_hip_y_j"),
                    self._get_joint_angle("left_hip"),
                    self._get_joint_angle("left_knee"),
                    self._get_joint_angle("left_foot_j"),
                    self._get_joint_angle("right_hip_z_j"),
                    self._get_joint_angle("right_hip_y_j"),
                    self._get_joint_angle("right_hip"),
                    self._get_joint_angle("right_knee"),
                    self._get_joint_angle("right_foot_j"),
                ])
                
                # Update walking controller (decimated)
                if step_count % ctrl_decimation == 0:
                    dq_cmd = walking.step(
                        contact["left_foot_geom"],
                        contact["right_foot_geom"],
                        q, dx_cmd, dy_cmd, dz_cmd, self.sim_time
                    )
                    if viewer is not None:
                        viewer.sync()
                
                # Apply joint commands
                if dq_cmd:
                    # Check for motor fault
                    if self._motor_fault:
                        # Motors disabled - no control
                        for key in CTRL_IDX:
                            self.data.ctrl[CTRL_IDX[key]] = 0.0
                    else:
                        self.data.ctrl[CTRL_IDX["right_hip_z"]] = np.clip(dq_cmd["right_hip_z"], -5, 5)
                        self.data.ctrl[CTRL_IDX["right_hip_y"]] = np.clip(dq_cmd["right_hip_y"], -5, 5)
                        self.data.ctrl[CTRL_IDX["right_hip"]] = np.clip(dq_cmd["right_hip"], -5, 5)
                        self.data.ctrl[CTRL_IDX["right_knee"]] = np.clip(dq_cmd["right_knee"], -5, 5)
                        self.data.ctrl[CTRL_IDX["right_foot"]] = 0.0
                        self.data.ctrl[CTRL_IDX["left_hip_z"]] = np.clip(dq_cmd["left_hip_z"], -5, 5)
                        self.data.ctrl[CTRL_IDX["left_hip_y"]] = np.clip(dq_cmd["left_hip_y"], -5, 5)
                        self.data.ctrl[CTRL_IDX["left_hip"]] = np.clip(dq_cmd["left_hip"], -5, 5)
                        self.data.ctrl[CTRL_IDX["left_knee"]] = np.clip(dq_cmd["left_knee"], -5, 5)
                        self.data.ctrl[CTRL_IDX["left_foot"]] = 0.0
                
                # Apply external forces (fault injection)
                torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
                if self._force_duration > 0:
                    self.data.xfrc_applied[torso_id] = self._external_force
                    self._force_duration -= dt
                    # Log force application every 0.1s
                    if int(self._force_duration * 10) % 1 == 0:
                        self.get_logger().warn(f'APPLYING FORCE: {self._external_force[3:]}, dur={self._force_duration:.2f}s')
                    if self._force_duration <= 0:
                        self._external_force = np.zeros(6)
                        self._motor_fault = False  # Clear motor fault after push
                        self.get_logger().info('Perturbation force ended')
                else:
                    self.data.xfrc_applied[torso_id] = np.zeros(6)
                
                # Publish telemetry (decimated)
                if self.sim_time - last_publish >= publish_period:
                    self._publish_telemetry()
                    last_publish = self.sim_time
                    # Simulate battery drain
                    if dx_cmd != 0 or dy_cmd != 0 or dz_cmd != 0:
                        self._battery_level -= self._battery_drain_rate
                    self._battery_level = max(0.0, self._battery_level)
                    # Process ROS2 callbacks
                    rclpy.spin_once(self, timeout_sec=0)
        
        # Run with or without viewer
        if self.use_viewer:
            self.get_logger().info('Starting simulation with viewer...')
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                simulation_loop(viewer)
        else:
            self.get_logger().info('Starting simulation (headless)...')
            simulation_loop(None)
        
        self.get_logger().info('Simulation stopped')
    
    def stop(self):
        """Stop the simulation."""
        self.running = False


def main(args=None):
    rclpy.init(args=args)
    
    node = IntegratedRobotNode()
    
    try:
        node.run()
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted')
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
