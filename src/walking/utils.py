"""
Utility functions for bipedal walking control.

This module provides helper functions for contact detection, frame transformations,
and robot state extraction used by the capture point walking controller.
"""

import mujoco
import numpy as np


def bodies_contacting_objects(model, data, bodies, targets):
    """Check if specified bodies are in contact with target objects."""
    body_ids = set(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in bodies
    )
    target_ids = set(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, t) for t in targets
    )
    contact = {b: False for b in bodies}

    for i in range(data.ncon):
        c = data.contact[i]
        b1 = model.geom_bodyid[c.geom1]
        b2 = model.geom_bodyid[c.geom2]

        if b1 in body_ids and b2 in target_ids:
            contact[model.body(b1).name] = True
        elif b2 in body_ids and b1 in target_ids:
            contact[model.body(b2).name] = True

    return contact


def geoms_contacting_geoms(model, data, source_geoms, target_geoms):
    """Check if source geometries are in contact with target geometries."""
    source_ids = set(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in source_geoms
    )
    target_ids = set(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in target_geoms
    )
    contact = {g: False for g in source_geoms}

    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 in source_ids and c.geom2 in target_ids:
            contact[model.geom(c.geom1).name] = True
        elif c.geom2 in source_ids and c.geom1 in target_ids:
            contact[model.geom(c.geom2).name] = True

    return contact


def capsule_end_frame_world(model, data, body_name, torso_name="torso"):
    """
    Compute the position and orientation of a capsule end in world frame.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        body_name: Name of the capsule body
        torso_name: Name of the torso body for orientation reference
        
    Returns:
        p_end: Position of capsule end in world frame (3,)
        R: Orientation frame with z=up, x=torso forward projected to ground (3,3)
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, torso_name)

    p_body = data.xpos[body_id]
    R_body = data.xmat[body_id].reshape(3, 3)

    body_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body_id]
    g = body_geoms[0]
    half_length = model.geom_size[g][1]

    body_axis_world = R_body[:, 2]
    p_end = p_body - body_axis_world * (2 * half_length)

    R_torso = data.xmat[torso_id].reshape(3, 3)
    x_torso = R_torso[:, 0]
    z = np.array([0.0, 0.0, 1.0])
    x = x_torso - np.dot(x_torso, z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack((x, y, z))

    return p_end, R


def world_p_to_frame(p_world, p_frame, R_frame):
    """Transform a world point to a local frame."""
    return R_frame.T @ (p_world - p_frame)


def torso_state_in_stance_frame(model, data, p_c, R_c, torso_name="torso"):
    """
    Get torso state (position, velocity, acceleration, orientation) in stance frame.
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        p_c: Stance frame origin in world
        R_c: Stance frame rotation matrix
        torso_name: Name of torso body
        
    Returns:
        Dictionary with position, velocity, acceleration, and orientation in stance frame
    """
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, torso_name)

    p_t = data.xpos[torso_id]
    R_t = data.xmat[torso_id].reshape(3, 3)

    v_6d = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, torso_id, v_6d, 0)
    w_t = v_6d[:3]
    v_t = v_6d[3:]

    a_6d = np.zeros(6)
    mujoco.mj_objectAcceleration(model, data, mujoco.mjtObj.mjOBJ_BODY, torso_id, a_6d, 0)
    alpha_t = a_6d[:3]
    a_t = a_6d[3:]

    p = R_c.T @ (p_t - p_c)
    v = R_c.T @ v_t
    a = R_c.T @ a_t
    R = R_c.T @ R_t
    w = R_c.T @ w_t
    alpha = R_c.T @ alpha_t

    return {
        "position": p,
        "position_w": p_t,
        "velocity": v,
        "acceleration": a,
        "orientation": R,
        "orientation_w": R_t,
        "angular_velocity": w,
        "angular_acceleration": alpha
    }


def draw_frame(viewer, pos, mat, size=0.1, rgba_alpha=1.0):
    """Draw a coordinate frame visualization."""
    mat = np.asarray(mat).reshape(3, 3)
    colors = [
        [1, 0, 0, rgba_alpha],
        [0, 1, 0, rgba_alpha],
        [0, 0, 1, rgba_alpha]
    ]
    
    for i in range(3):
        axis_dir = mat[:, i]
        start = pos
        end = pos + axis_dir * size
        
        mujoco.mjv_connector(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            width=size * 0.05,
            from_=start,
            to=end
        )
        viewer.user_scn.geoms[viewer.user_scn.ngeom].rgba = colors[i]
        viewer.user_scn.ngeom += 1


class KeyboardController:
    """Keyboard-based velocity command interface with smoothing."""
    
    def __init__(self, v_step=0.5, v_side=0.1, yaw_step=0.8, alpha=0.01):
        # Import pynput only when needed
        from pynput import keyboard
        self._keyboard = keyboard
        
        self.v_step = v_step
        self.v_side = v_side
        self.yaw_step = yaw_step
        self.alpha = alpha
        
        self.current_dx = 0.0
        self.current_dy = 0.0
        self.current_yaw = 0.0
        
        self.pressed_keys = set()
        self.key_map = {
            keyboard.Key.up: 'up',
            keyboard.Key.down: 'down',
            keyboard.Key.left: 'left',
            keyboard.Key.right: 'right',
        }
        
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self.listener.start()

    def _on_press(self, key):
        if key in self.key_map:
            self.pressed_keys.add(self.key_map[key])
        try:
            if hasattr(key, 'char') and key.char is not None:
                self.pressed_keys.add(key.char.lower())
        except AttributeError:
            pass

    def _on_release(self, key):
        if key in self.key_map:
            self.pressed_keys.discard(self.key_map[key])
        try:
            if hasattr(key, 'char') and key.char is not None:
                self.pressed_keys.discard(key.char.lower())
        except AttributeError:
            pass
    
    def record_toggle(self):
        return None

    def get_cmd(self):
        """Get smoothed velocity commands based on keyboard input."""
        target_dx = 0.0
        target_dy = 0.0
        target_yaw = 0.0
        
        if 'up' in self.pressed_keys:
            target_dx += self.v_step
        if 'down' in self.pressed_keys:
            target_dx -= self.v_step
        if 'left' in self.pressed_keys:
            target_dy += self.v_side
        if 'right' in self.pressed_keys:
            target_dy -= self.v_side
        if 'q' in self.pressed_keys:
            target_yaw += self.yaw_step
        if 'e' in self.pressed_keys:
            target_yaw -= self.yaw_step
        
        self.current_dx += self.alpha * (target_dx - self.current_dx)
        self.current_dy += self.alpha * (target_dy - self.current_dy)
        self.current_yaw += self.alpha * (target_yaw - self.current_yaw)
        
        if abs(self.current_dx) < 0.001:
            self.current_dx = 0.0
        if abs(self.current_dy) < 0.001:
            self.current_dy = 0.0
        if abs(self.current_yaw) < 0.001:
            self.current_yaw = 0.0
        
        return self.current_dx, self.current_dy, self.current_yaw
