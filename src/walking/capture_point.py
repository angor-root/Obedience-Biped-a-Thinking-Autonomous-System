"""
Capture Point Walking Controller for Bipedal Robot.

This module implements the capture point walking control algorithm for
dynamically stable bipedal locomotion. Based on inverted pendulum dynamics,
it computes optimal foot placements to maintain balance during walking.

Reference:
    - Pratt et al., "Capture Point: A Step toward Humanoid Push Recovery"
    - The5439Workshop: https://github.com/The5439Workshop/Bipedal_walking_capture_point
"""

import copy
import time
import os
import numpy as np
import mujoco
import mujoco.viewer

from utils import (
    geoms_contacting_geoms,
    capsule_end_frame_world,
    world_p_to_frame,
    torso_state_in_stance_frame,
    draw_frame,
    KeyboardController,
)
from jacobian import get_pos_3d_jacobians


# Control parameters
K_XI = -1.0      # Capture point tuning gain
K_PZ = 5         # Height control gain
K_PT = -8        # Orientation control gain
DY_ROCKING = 0.1 # Lateral rocking amplitude
G = 9.81         # Gravity

# Walking parameters
MIN_STEP = 0.10
T_DES = 0.4      # Desired step period

# Robot configuration
FOOT_BODIES = {"Right": "right_shin", "Left": "left_shin"}
GROUND_GEOMS = ["ground", "obstacle_box", "obstacle_box_2"]

# Actuator indices
CTRL_IDX = {
    "right_hip_y": 0, "right_hip": 1, "right_knee": 2, "right_foot": 3,
    "left_hip_y": 4, "left_hip": 5, "left_knee": 6, "left_foot": 7
}


def compute_capture_point_x(x_com, dx_com, ddx_com, z_com, dx_des):
    """
    Compute forward capture point foot placement.
    
    Args:
        x_com: CoM position in x-axis (stance frame)
        dx_com: CoM velocity in x-axis
        ddx_com: CoM acceleration in x-axis
        z_com: CoM height above stance foot
        dx_des: Desired forward velocity
        
    Returns:
        p: Target foot placement in x-axis
    """
    z_com = np.clip(z_com, 0.1, 1000)
    omega_n = np.sqrt(G / z_com)
    xi = x_com + dx_com / omega_n
    dxi = dx_com + ddx_com / omega_n
    xi_des = x_com + dx_des / omega_n
    p = xi - dxi / omega_n - K_XI * (xi - xi_des)
    return p


def compute_capture_point_y(y_com, dy_com, ddy_com, z_com, dy_des):
    """
    Compute lateral capture point foot placement.
    
    Args:
        y_com: CoM position in y-axis (stance frame)
        dy_com: CoM velocity in y-axis
        ddy_com: CoM acceleration in y-axis
        z_com: CoM height above stance foot
        dy_des: Desired lateral velocity
        
    Returns:
        p: Target foot placement in y-axis
    """
    z_com = np.clip(z_com, 0.1, 1000)
    omega_n = np.sqrt(G / z_com)
    xi = y_com + dy_com / omega_n
    dxi = dy_com + ddy_com / omega_n
    xi_des = y_com + dy_des / omega_n
    p = xi - dxi / omega_n - K_XI * (xi - xi_des)
    return p


def compute_swing_height(z_target, h_target, h_0, t_start, t_now, e_step_down):
    """
    Compute swing foot height trajectory.
    
    Args:
        z_target: Target CoM height
        h_target: Target step height
        h_0: Swing foot height at leg switch
        t_start: Time at leg switch
        t_now: Current time
        e_step_down: Step down error term
        
    Returns:
        z_swing: Current swing foot height target
        z_com: Target CoM height
    """
    delta_t = t_now - t_start
    tau = delta_t / T_DES + e_step_down
    tau_clipped = np.clip(tau, 0.0, 1.0)
    
    h_m = max(1.1 * h_0, h_target)
    
    if tau_clipped < 0.5:
        z_swing = (h_m - h_0) * np.sin(np.pi * tau_clipped) + h_0
    else:
        z_swing = h_m * np.sin(np.pi * tau_clipped)
    
    z_offset = 0.0
    if tau > 1:
        z_offset = -tau + 1
    
    return z_swing + z_offset, z_target + z_offset


def rotation_error(R, R_d):
    """Compute orientation error from rotation matrices."""
    skew = 0.5 * (np.matmul(R_d.T, R) - np.matmul(R.T, R_d))
    return np.array([skew[2, 1], skew[0, 2], skew[1, 0]])


def stance_foot_velocity(p_com, z_com, R_com, R_com_des):
    """
    Generate stance foot velocity for height and orientation regulation.
    
    Args:
        p_com: CoM position in foot frame
        z_com: Target height
        R_com: Current orientation in foot frame
        R_com_des: Desired orientation
        
    Returns:
        Velocity vector for stance foot
    """
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
    """Compute swing leg velocity command."""
    error = p_des - p_current
    error[1] *= 0.8
    return np.clip(10 * error, -10, 10)


class WalkingController:
    """Capture point based walking controller."""
    
    def __init__(self, model, data, viewer, t0):
        self.model = model
        self.data = data
        self.viewer = viewer
        
        self.stance_side = "Right"
        self.swing_side = "Left"
        
        self.z_target = 1.45
        self.h_target = 0.2
        self.R_target = np.eye(3)
        
        self.min_step_time = 0.2
        self.pre_step_time = 0.0
        self.contact_lifted = False
        
        self.switch_props = {
            "t0": None,
            "p_foot_0": None,
            "p_com_0": None,
        }
        
        self._initialize(t0)
    
    def _get_pose(self):
        """Get current foot and torso poses."""
        p_stance_w, R_stance_w = capsule_end_frame_world(
            self.model, self.data, FOOT_BODIES[self.stance_side]
        )
        draw_frame(self.viewer, p_stance_w, R_stance_w, size=0.2)
        
        p_swing_w, _ = capsule_end_frame_world(
            self.model, self.data, FOOT_BODIES[self.swing_side]
        )
        p_swing = world_p_to_frame(p_swing_w, p_stance_w, R_stance_w)
        
        torso_state = torso_state_in_stance_frame(
            self.model, self.data, p_stance_w, R_stance_w
        )
        draw_frame(self.viewer, torso_state["position_w"], 
                   torso_state["orientation_w"], size=0.5)
        
        torso_state["position"][0] -= 0.05
        return p_swing, torso_state
    
    def _initialize(self, t0, dx_des=0.0):
        """Initialize controller state."""
        p_swing, torso_state = self._get_pose()
        
        self.switch_props["p_foot_0"] = copy.deepcopy(p_swing)
        self.switch_props["p_com_0"] = copy.deepcopy(torso_state["position"])
        self.switch_props["t0"] = t0
    
    def _switch_leg(self, contact_l, contact_r, t):
        """Handle stance/swing leg switching."""
        if self.swing_side == "Left":
            contact = contact_l
        else:
            contact = contact_r
        
        if not contact:
            self.contact_lifted = True
        
        min_time = t > self.pre_step_time + self.min_step_time
        
        if contact and min_time:
            self.contact_lifted = False
            self.stance_side, self.swing_side = self.swing_side, self.stance_side
            self.pre_step_time = t
            return True
        return False
    
    def step(self, contact_l, contact_r, q, dx_des, dy_des, t):
        """
        Execute one control step.
        
        Args:
            contact_l: Left foot contact state
            contact_r: Right foot contact state
            q: Joint positions
            dx_des: Desired forward velocity
            dy_des: Desired lateral velocity
            t: Current time
            
        Returns:
            Dictionary of joint velocity commands
        """
        switch = self._switch_leg(contact_l, contact_r, t)
        p_swing, torso_state = self._get_pose()
        
        # Compute capture points
        p_x = compute_capture_point_x(
            torso_state["position"][0],
            torso_state["velocity"][0],
            torso_state["acceleration"][0],
            torso_state["position"][2],
            dx_des
        )
        
        dy_rock = -DY_ROCKING if self.stance_side == "Right" else DY_ROCKING
        p_y = compute_capture_point_y(
            torso_state["position"][1],
            torso_state["velocity"][1],
            torso_state["acceleration"][1],
            torso_state["position"][2],
            dy_des + dy_rock
        )
        
        # Compute step down error
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
        
        # Update switch properties
        if switch:
            self.switch_props["p_foot_0"] = copy.deepcopy(p_swing)
            self.switch_props["p_com_0"] = copy.deepcopy(torso_state["position"])
            self.switch_props["t0"] = t
        
        # Compute swing foot height
        z_swing, z_com = compute_swing_height(
            self.z_target,
            self.h_target,
            self.switch_props["p_foot_0"][2],
            self.switch_props["t0"],
            t,
            e_step_down
        )
        
        # Compute stance foot velocity
        vel_stance = stance_foot_velocity(
            torso_state["position"],
            z_com,
            torso_state["orientation"],
            self.R_target
        )
        
        # Compute swing foot velocity
        vel_swing = swing_leg_controller(
            np.array([p_x, p_y, z_swing]),
            p_swing
        )
        
        # Transform to joint velocities using Jacobians
        Jr, Jl = get_pos_3d_jacobians(q)
        
        if self.stance_side == "Right":
            J_stance, J_swing = Jr, Jl
        else:
            J_stance, J_swing = Jl, Jr
        
        R_t = torso_state["orientation"]
        
        v_stance = R_t.T @ vel_stance.reshape(3, 1)
        dq_stance = np.linalg.pinv(J_stance) @ v_stance
        
        v_swing = R_t.T @ vel_swing.reshape(3, 1)
        dq_swing = np.linalg.pinv(J_swing) @ v_swing
        
        # Map to joint commands
        if self.stance_side == "Right":
            return {
                "right_hip_y": dq_stance[0, 0],
                "right_hip": dq_stance[1, 0],
                "right_knee": dq_stance[2, 0],
                "left_hip_y": dq_swing[0, 0],
                "left_hip": dq_swing[1, 0],
                "left_knee": dq_swing[2, 0]
            }
        else:
            return {
                "left_hip_y": dq_stance[0, 0],
                "left_hip": dq_stance[1, 0],
                "left_knee": dq_stance[2, 0],
                "right_hip_y": dq_swing[0, 0],
                "right_hip": dq_swing[1, 0],
                "right_knee": dq_swing[2, 0]
            }


def get_joint_angle(model, data, joint_name):
    """Get joint angle by name."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    return data.qpos[model.jnt_qposadr[jid]]


def main():
    """Main simulation loop."""
    xml_path = os.path.join(os.path.dirname(__file__), "../../models/xml/biped_3d_feet.xml")
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    
    # Initialize knee angles
    for knee in ["right_knee", "left_knee"]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, knee)
        data.qpos[model.jnt_qposadr[jid]] = np.deg2rad(-15)
    
    mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    
    dt = model.opt.timestep
    t = 0
    
    controller = KeyboardController(v_step=0.6, v_side=0.15, yaw_step=1.0)
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        walking = WalkingController(model, data, viewer, t)
        dq_cmd = {}
        
        while viewer.is_running():
            t += dt
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            
            # Get foot contacts
            contact = geoms_contacting_geoms(
                model, data,
                ["left_shin_geom", "right_shin_geom", 
                 "right_foot_geom", "left_foot_geom"],
                GROUND_GEOMS
            )
            
            # Get joint angles
            q = np.array([
                get_joint_angle(model, data, "left_hip_y_j"),
                get_joint_angle(model, data, "left_hip"),
                get_joint_angle(model, data, "left_knee"),
                get_joint_angle(model, data, "right_hip_y_j"),
                get_joint_angle(model, data, "right_hip"),
                get_joint_angle(model, data, "right_knee")
            ])
            
            # Update controller at reduced rate
            if int(t / dt) % 30 == 0:
                dx, dy, _ = controller.get_cmd()
                viewer.user_scn.ngeom = 0
                
                dq_cmd = walking.step(
                    contact["left_foot_geom"],
                    contact["right_foot_geom"],
                    q, -dx, dy, t
                )
                viewer.sync()
            
            # Apply joint commands
            if dq_cmd:
                data.ctrl[CTRL_IDX["right_hip"]] = np.clip(dq_cmd["right_hip"], -5, 5)
                data.ctrl[CTRL_IDX["right_knee"]] = np.clip(dq_cmd["right_knee"], -5, 5)
                data.ctrl[CTRL_IDX["left_hip"]] = np.clip(dq_cmd["left_hip"], -5, 5)
                data.ctrl[CTRL_IDX["left_knee"]] = np.clip(dq_cmd["left_knee"], -5, 5)
                data.ctrl[CTRL_IDX["right_hip_y"]] = np.clip(dq_cmd["right_hip_y"], -5, 5)
                data.ctrl[CTRL_IDX["left_hip_y"]] = np.clip(dq_cmd["left_hip_y"], -5, 5)
            
            time.sleep(max(0.0, dt - 0.0001))


if __name__ == "__main__":
    main()
