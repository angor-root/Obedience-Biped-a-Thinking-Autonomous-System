"""
Supply-to-Bed Delivery Mission Test

Obedience bipedal robot medicine delivery system.

Mission:
    1. Go to Supply Zone (red) - load medicine (3s)
    2. Navigate to Bed 1 - deliver medicine (2s)
    3. Return to Supply Zone - load medicine (3s)
    4. Navigate to Bed 2 - deliver medicine (2s)
    5. Return to Supply Zone - load medicine (3s)
    6. Navigate to Bed 3 - deliver medicine (2s)
    7. Return to charging station

Zone Locations:
    - Supply Zone: (2, 0) - Red platform
    - Bed 1: (-3, -1)
    - Bed 2: (-5, -1)
    - Bed 3: (-5, 1)
    - Charging: (3.5, -4.5) - Southeast corner
"""

import sys
import os
import copy
import time
import numpy as np
import mujoco
import mujoco.viewer
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, List, Tuple

# Add src to path
_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from walking.utils import (
    geoms_contacting_geoms,
    capsule_end_frame_world,
    world_p_to_frame,
    torso_state_in_stance_frame,
)
from walking.jacobian import get_pos_3d_jacobians
from battery.battery_model import BatteryModel, BatteryConfig, RobotState


# ============================================================================
# WALKING CONTROLLER (copied from test_hospital_mission.py)
# ============================================================================

K_XI = -1.0
K_PZ = 5
K_PT = -15
DY_ROCKING = 0.1
G = 9.81
T_DES = 0.4

FOOT_BODIES = {"Right": "right_shin", "Left": "left_shin"}
GROUND_GEOMS = ["ground"]

CTRL_IDX = {
    "right_hip_z": 0, "right_hip_y": 1, "right_hip": 2,
    "right_knee": 3, "right_foot": 4,
    "left_hip_z": 5, "left_hip_y": 6, "left_hip": 7,
    "left_knee": 8, "left_foot": 9
}


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
    """Walking controller with capture point algorithm."""
    
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
# SUPPLY DELIVERY MISSION PLANNER
# ============================================================================

class SupplyMissionState(Enum):
    """Mission states for supply-to-bed delivery."""
    IDLE = auto()
    NAVIGATING_TO_SUPPLY = auto()
    LOADING_MEDICINE = auto()
    NAVIGATING_TO_BED = auto()
    DELIVERING_MEDICINE = auto()
    RETURNING_HOME = auto()
    COMPLETED = auto()


@dataclass
class SupplyWaypoint:
    """Waypoint definition."""
    name: str
    position: np.ndarray
    wait_time: float  # seconds to wait at this point


class SupplyMissionPlanner:
    """
    Mission planner for supply-to-bed delivery.
    
    Mission sequence:
        Supply → Bed1 → Supply → Bed2 → Supply → Bed3 → Charging
    """
    
    # Zone positions from hospital_scene.xml
    SUPPLY_ZONE = np.array([2.0, 0.0])
    BED_1 = np.array([-3.0, -1.0])
    BED_2 = np.array([-5.0, -1.0])
    BED_3 = np.array([-5.0, 1.0])
    CHARGING = np.array([3.5, -4.5])  # Southeast corner
    
    # Timing
    LOAD_TIME = 3.0   # seconds to load medicine at supply
    DELIVER_TIME = 2.0  # seconds to deliver at bed
    
    # Navigation
    WAYPOINT_TOLERANCE = 0.4  # meters
    MAX_SPEED = 0.15  # m/s forward
    TURN_GAIN = 1.5  # rad/s per rad error
    
    def __init__(self):
        """Initialize mission planner."""
        self.state = SupplyMissionState.IDLE
        self.current_target_idx = 0
        self.wait_timer = 0.0
        self.deliveries_completed = 0
        self.mission_start_time = 0.0
        
        # Build mission sequence
        self.mission_sequence: List[SupplyWaypoint] = [
            SupplyWaypoint("supply_1", self.SUPPLY_ZONE, self.LOAD_TIME),
            SupplyWaypoint("bed_1", self.BED_1, self.DELIVER_TIME),
            SupplyWaypoint("supply_2", self.SUPPLY_ZONE, self.LOAD_TIME),
            SupplyWaypoint("bed_2", self.BED_2, self.DELIVER_TIME),
            SupplyWaypoint("supply_3", self.SUPPLY_ZONE, self.LOAD_TIME),
            SupplyWaypoint("bed_3", self.BED_3, self.DELIVER_TIME),
            SupplyWaypoint("home", self.CHARGING, 0.0),
        ]
    
    @property
    def current_target(self) -> Optional[SupplyWaypoint]:
        """Get current target waypoint."""
        if self.current_target_idx < len(self.mission_sequence):
            return self.mission_sequence[self.current_target_idx]
        return None
    
    def start_mission(self, current_time: float):
        """Start the supply delivery mission."""
        self.state = SupplyMissionState.NAVIGATING_TO_SUPPLY
        self.current_target_idx = 0
        self.wait_timer = 0.0
        self.deliveries_completed = 0
        self.mission_start_time = current_time
        
        print("\n" + "=" * 60)
        print("SUPPLY DELIVERY MISSION STARTED")
        print("=" * 60)
        print("Route: Supply → Bed1 → Supply → Bed2 → Supply → Bed3 → Home")
        print(f"Load time: {self.LOAD_TIME}s | Deliver time: {self.DELIVER_TIME}s")
        print("=" * 60 + "\n")
    
    def update(self, robot_pos: np.ndarray, current_time: float, dt: float) -> Tuple[np.ndarray, float, float]:
        """
        Update mission state and return velocity commands.
        
        Returns:
            Tuple of (target_position, dx_cmd, dz_omega_cmd)
        """
        if self.state == SupplyMissionState.IDLE:
            return np.zeros(2), 0.0, 0.0
        
        if self.state == SupplyMissionState.COMPLETED:
            return np.zeros(2), 0.0, 0.0
        
        target = self.current_target
        if target is None:
            self.state = SupplyMissionState.COMPLETED
            return np.zeros(2), 0.0, 0.0
        
        # Distance to target
        robot_2d = robot_pos[:2]
        distance = np.linalg.norm(target.position - robot_2d)
        
        # Check if arrived
        if distance < self.WAYPOINT_TOLERANCE:
            return self._handle_arrival(target, current_time, dt)
        
        # Return target position for navigation
        return target.position, 0.0, 0.0
    
    def _handle_arrival(self, target: SupplyWaypoint, current_time: float, dt: float) -> Tuple[np.ndarray, float, float]:
        """Handle arrival at waypoint."""
        waiting_states = {
            SupplyMissionState.NAVIGATING_TO_SUPPLY: SupplyMissionState.LOADING_MEDICINE,
            SupplyMissionState.NAVIGATING_TO_BED: SupplyMissionState.DELIVERING_MEDICINE,
        }
        
        # Start waiting if just arrived
        if self.state in waiting_states:
            new_state = waiting_states[self.state]
            self.state = new_state
            self.wait_timer = 0.0
            
            if new_state == SupplyMissionState.LOADING_MEDICINE:
                print(f"[SUPPLY] Loading medicine... ({target.wait_time}s)")
            else:
                print(f"[DELIVER] Delivering medicine to {target.name}... ({target.wait_time}s)")
        
        # Update wait timer
        if self.state in [SupplyMissionState.LOADING_MEDICINE, SupplyMissionState.DELIVERING_MEDICINE]:
            self.wait_timer += dt
            
            if self.wait_timer >= target.wait_time:
                # Done waiting, move to next waypoint
                if self.state == SupplyMissionState.DELIVERING_MEDICINE:
                    self.deliveries_completed += 1
                    print(f"[DELIVER] Delivery {self.deliveries_completed}/3 completed!")
                else:
                    print(f"[SUPPLY] Medicine loaded!")
                
                self.current_target_idx += 1
                self.wait_timer = 0.0
                
                # Determine next state
                next_target = self.current_target
                if next_target is None:
                    self.state = SupplyMissionState.COMPLETED
                elif next_target.name.startswith("supply"):
                    self.state = SupplyMissionState.NAVIGATING_TO_SUPPLY
                elif next_target.name.startswith("bed"):
                    self.state = SupplyMissionState.NAVIGATING_TO_BED
                elif next_target.name == "home":
                    self.state = SupplyMissionState.RETURNING_HOME
        
        # Also handle returning home state
        if self.state == SupplyMissionState.RETURNING_HOME:
            distance = np.linalg.norm(target.position - np.zeros(2))
            if distance < self.WAYPOINT_TOLERANCE:
                self.state = SupplyMissionState.COMPLETED
        
        return target.position, 0.0, 0.0
    
    def compute_velocity_command(self, robot_pos: np.ndarray, robot_heading: float, 
                                  target_pos: np.ndarray) -> Tuple[float, float, float]:
        """
        Compute velocity commands for navigation.
        
        Returns:
            (dx_cmd, dy_cmd, dz_omega_cmd)
        """
        # If waiting, don't move
        if self.state in [SupplyMissionState.LOADING_MEDICINE, 
                          SupplyMissionState.DELIVERING_MEDICINE,
                          SupplyMissionState.COMPLETED,
                          SupplyMissionState.IDLE]:
            return 0.0, 0.0, 0.0
        
        # Compute direction to target
        robot_2d = robot_pos[:2]
        to_target = target_pos - robot_2d
        distance = np.linalg.norm(to_target)
        
        if distance < 0.01:
            return 0.0, 0.0, 0.0
        
        # Target heading
        target_heading = np.arctan2(to_target[1], to_target[0])
        
        # Heading error (normalized to [-pi, pi])
        heading_error = target_heading - robot_heading
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
        
        # Turn first, then walk
        dz_omega = np.clip(self.TURN_GAIN * heading_error, -0.5, 0.5)
        
        # Only walk forward if roughly aligned
        if abs(heading_error) < 0.3:  # ~17 degrees
            dx_cmd = min(self.MAX_SPEED, 0.5 * distance)
        else:
            dx_cmd = 0.0
        
        return dx_cmd, 0.0, dz_omega
    
    def get_status(self) -> dict:
        """Get mission status."""
        return {
            "state": self.state.name,
            "target": self.current_target.name if self.current_target else "None",
            "deliveries": f"{self.deliveries_completed}/3",
            "wait_timer": f"{self.wait_timer:.1f}s" if self.wait_timer > 0 else "-",
        }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_joint_angle(model, data, joint_name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return data.qpos[model.jnt_qposadr[jid]]


def get_robot_position(model, data):
    """Get robot torso position in world frame."""
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    return data.xpos[torso_id].copy()


def get_robot_heading(model, data):
    """Get robot heading angle from torso orientation."""
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    R = data.xmat[torso_id].reshape(3, 3)
    forward = R[:, 0]
    return np.arctan2(forward[1], forward[0])


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run the supply-to-bed delivery mission test."""
    print("=" * 60)
    print("OBEDIENCE - Supply-to-Bed Delivery Mission Test")
    print("=" * 60)
    
    # Load hospital scene
    xml_path = os.path.join(_ROOT, "models/xml/hospital_scene.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Initialize robot pose
    for knee in ["right_knee", "left_knee"]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, knee)
        data.qpos[model.jnt_qposadr[jid]] = np.deg2rad(-15)
    
    mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    
    dt = model.opt.timestep
    t = 0.0
    
    # Initialize battery
    battery_config = BatteryConfig(
        capacity_wh=50.0,
        power_walking=40.0,
        power_idle=5.0,
        power_delivering=8.0,
        low_threshold=0.20,
        critical_threshold=0.10,
    )
    battery = BatteryModel(battery_config, initial_soc=1.0)
    
    # Initialize mission planner
    mission = SupplyMissionPlanner()
    
    # Status display
    status_interval = 2.0
    last_status_time = 0.0
    
    print(f"\n[INIT] Battery: {battery.soc_percent:.1f}%")
    print(f"[INIT] Robot spawns at origin (0, 0)")
    print(f"[INIT] Supply zone at (2, 0) - RED")
    print(f"[INIT] Beds at (-3,-1), (-5,-1), (-5,1)")
    print(f"[INIT] Charging station at (3.5, -4.5) - GREEN")
    print("\n" + "-" * 60)
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        walking = WalkingController(model, data, t)
        dq_cmd = {}
        
        mission_started = False
        
        while viewer.is_running():
            t += dt
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            
            # Get robot state
            robot_pos = get_robot_position(model, data)
            robot_heading = get_robot_heading(model, data)
            
            # Start mission after stabilization
            if not mission_started and t > 1.0:
                mission.start_mission(t)
                mission_started = True
            
            # Update mission and get commands
            dx_cmd, dy_cmd, dz_cmd = 0.0, 0.0, 0.0
            
            if mission_started:
                target_pos, _, _ = mission.update(robot_pos, t, dt)
                
                # Compute velocity command
                if mission.state not in [SupplyMissionState.IDLE, 
                                         SupplyMissionState.COMPLETED,
                                         SupplyMissionState.LOADING_MEDICINE,
                                         SupplyMissionState.DELIVERING_MEDICINE]:
                    dx_cmd, dy_cmd, dz_cmd = mission.compute_velocity_command(
                        robot_pos, robot_heading, target_pos
                    )
                    battery.set_state(RobotState.WALKING)
                elif mission.state in [SupplyMissionState.LOADING_MEDICINE,
                                       SupplyMissionState.DELIVERING_MEDICINE]:
                    battery.set_state(RobotState.DELIVERING)
                else:
                    battery.set_state(RobotState.IDLE)
            
            # Update battery
            battery.update(dt, t)
            
            # Get foot contacts
            contact = geoms_contacting_geoms(
                model, data,
                ["left_shin_geom", "right_shin_geom", "right_foot_geom", "left_foot_geom"],
                GROUND_GEOMS
            )
            
            # Get joint angles
            q = np.array([
                get_joint_angle(model, data, "left_hip_z_j"),
                get_joint_angle(model, data, "left_hip_y_j"),
                get_joint_angle(model, data, "left_hip"),
                get_joint_angle(model, data, "left_knee"),
                get_joint_angle(model, data, "left_foot_j"),
                get_joint_angle(model, data, "right_hip_z_j"),
                get_joint_angle(model, data, "right_hip_y_j"),
                get_joint_angle(model, data, "right_hip"),
                get_joint_angle(model, data, "right_knee"),
                get_joint_angle(model, data, "right_foot_j"),
            ])
            
            # Update walking controller
            if int(t / dt) % 30 == 0:
                viewer.user_scn.ngeom = 0
                dq_cmd = walking.step(
                    contact["left_foot_geom"],
                    contact["right_foot_geom"],
                    q, dx_cmd, dy_cmd, dz_cmd, t
                )
                viewer.sync()
            
            # Apply joint commands
            if dq_cmd:
                data.ctrl[CTRL_IDX["right_hip_z"]] = np.clip(dq_cmd["right_hip_z"], -5, 5)
                data.ctrl[CTRL_IDX["right_hip_y"]] = np.clip(dq_cmd["right_hip_y"], -5, 5)
                data.ctrl[CTRL_IDX["right_hip"]] = np.clip(dq_cmd["right_hip"], -5, 5)
                data.ctrl[CTRL_IDX["right_knee"]] = np.clip(dq_cmd["right_knee"], -5, 5)
                data.ctrl[CTRL_IDX["right_foot"]] = 0.0
                data.ctrl[CTRL_IDX["left_hip_z"]] = np.clip(dq_cmd["left_hip_z"], -5, 5)
                data.ctrl[CTRL_IDX["left_hip_y"]] = np.clip(dq_cmd["left_hip_y"], -5, 5)
                data.ctrl[CTRL_IDX["left_hip"]] = np.clip(dq_cmd["left_hip"], -5, 5)
                data.ctrl[CTRL_IDX["left_knee"]] = np.clip(dq_cmd["left_knee"], -5, 5)
                data.ctrl[CTRL_IDX["left_foot"]] = 0.0
            
            # Print status periodically
            if t - last_status_time >= status_interval:
                last_status_time = t
                status = mission.get_status()
                print(f"[t={t:6.1f}s] Battery: {battery.soc_percent:5.1f}% | "
                      f"State: {status['state']:20s} | "
                      f"Target: {status['target']:10s} | "
                      f"Deliveries: {status['deliveries']} | "
                      f"Pos: ({robot_pos[0]:5.2f}, {robot_pos[1]:5.2f})")
            
            # Check mission completion
            if mission.state == SupplyMissionState.COMPLETED:
                elapsed = t - mission.mission_start_time
                print("\n" + "=" * 60)
                print("MISSION COMPLETED SUCCESSFULLY!")
                print(f"Total Deliveries: {mission.deliveries_completed}")
                print(f"Final Battery: {battery.soc_percent:.1f}%")
                print(f"Total Time: {elapsed:.1f}s")
                print("=" * 60)
                
                # Keep viewer open
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.1)
                break
            
            time.sleep(max(0.0, dt - 0.0001))


if __name__ == "__main__":
    main()
