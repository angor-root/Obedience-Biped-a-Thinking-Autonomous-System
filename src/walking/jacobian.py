"""
Kinematic Jacobian computation using Pinocchio.

This module computes leg Jacobians for inverse kinematics control
of the bipedal robot's foot positions. Supports 5-DOF leg model.
"""

import pinocchio as pin
import numpy as np
import os

_URDF_PATH = os.path.join(os.path.dirname(__file__), "../../urdf/biped_3d_5dof_leg.urdf")


def _initialize_model():
    """Initialize Pinocchio model using foot frames from URDF."""
    model = pin.buildModelFromUrdf(_URDF_PATH)
    data = model.createData()
    
    left_foot_fid = model.getFrameId("left_foot")
    right_foot_fid = model.getFrameId("right_foot")
    
    return model, data, right_foot_fid, left_foot_fid


_model, _data, _right_frame, _left_frame = _initialize_model()


def _joint_velocity_indices(model, joint_name):
    """Get velocity indices for a joint in the velocity vector."""
    jid = model.getJointId(joint_name)
    start = model.idx_vs[jid]
    nv = model.joints[jid].nv
    return list(range(start, start + nv))


def get_pos_3d_jacobians(q):
    """
    Compute position + yaw Jacobians for both feet (5-DOF legs).
    
    Args:
        q: Joint configuration vector (10 elements)
           [left_hip_yaw, left_hip_roll, left_hip_pitch, left_knee, left_ankle,
            right_hip_yaw, right_hip_roll, right_hip_pitch, right_knee, right_ankle]
    
    Returns:
        J_r: Right foot Jacobian (4x5) - [x, y, z, yaw]
        J_l: Left foot Jacobian (4x5) - [x, y, z, yaw]
    """
    pin.forwardKinematics(_model, _data, q)
    pin.updateFramePlacements(_model, _data)

    Jw_right = pin.computeFrameJacobian(
        _model, _data, q, _right_frame, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )
    Jw_left = pin.computeFrameJacobian(
        _model, _data, q, _left_frame, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
    )

    # Extract position (x,y,z) and yaw rotation (index 5)
    Jr_lin = Jw_right[[0, 1, 2, 5], :]
    Jl_lin = Jw_left[[0, 1, 2, 5], :]

    right_cols = (
        _joint_velocity_indices(_model, "right_hip_yaw") +
        _joint_velocity_indices(_model, "right_hip_roll") +
        _joint_velocity_indices(_model, "right_hip_pitch") +
        _joint_velocity_indices(_model, "right_knee") +
        _joint_velocity_indices(_model, "right_ankle")
    )
    left_cols = (
        _joint_velocity_indices(_model, "left_hip_yaw") +
        _joint_velocity_indices(_model, "left_hip_roll") +
        _joint_velocity_indices(_model, "left_hip_pitch") +
        _joint_velocity_indices(_model, "left_knee") +
        _joint_velocity_indices(_model, "left_ankle")
    )
    
    J_r = Jr_lin[:, right_cols]
    J_l = Jl_lin[:, left_cols]

    return J_r, J_l
