#!/usr/bin/env python3
"""
NASA-Style Mission Control GUI - Professional Edition
OBEDIENCE Bipedal Robot System

Features:
- Schematic Digital Twin (professional 2D representation)
- Mission Map with trajectory
- Direct fault injection buttons (no sliders)
- ROS2 Watchdog with disconnect simulation
- FMEA Panel
- Event Log

Based on NASA Mission Control standards
"""

import sys
import os
import json
import time
import threading
import math
from datetime import datetime
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import ttk
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import String, Float32, Bool
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Point

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from health.fault_detector import FaultType

# =============================================================================
# COLORS - NASA Dark Theme
# =============================================================================
C = {
    'bg': '#0a0a0f',
    'panel': '#12121a',
    'widget': '#1a1a25',
    'border': '#2a2a3a',
    'accent': '#00ff88',
    'warn': '#ffaa00',
    'danger': '#ff3333',
    'critical': '#ff0000',
    'text': '#e0e0e0',
    'dim': '#606070',
    'ok': '#00cc66',
    'robot': '#00ffaa',
    'target': '#ffff00',
    'path': '#0088ff',
    'bed': '#4466aa',
    'supply': '#aa4444',
    'charge': '#44aa44',
}

# Hospital bounds
MAP = {'x_min': -8, 'x_max': 5, 'y_min': -6, 'y_max': 6}

WAYPOINTS = {
    'supply': (2.0, 0.0),
    'bed_1': (-3.0, -1.3),
    'bed_2': (-5.0, -1.3),
    'bed_3': (-5.0, 1.3),
    'charging': (3.5, -4.5),
}


# =============================================================================
# DATA
# =============================================================================
@dataclass
class State:
    pos: Tuple[float, float] = (0.0, 0.0)
    height: float = 0.68
    quat: Tuple[float, float, float, float] = (0, 0, 0, 1)
    battery: float = 100.0
    mission_active: bool = False
    current_wp: str = ""
    trajectory: List[Tuple[float, float]] = field(default_factory=list)
    ros_connected: bool = True
    emergency: bool = False
    faults: List[str] = field(default_factory=list)
    # Component states (OK, WARN, FAIL)
    left_leg: str = "OK"
    right_leg: str = "OK"
    torso: str = "OK"
    imu: str = "OK"
    comms: str = "OK"


# =============================================================================
# ROS2 NODE
# =============================================================================
class ControlNode(Node):
    def __init__(self, callback):
        super().__init__('mission_control')
        self.cb = callback
        
        qos_fast = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        qos_rel = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        
        # Subscribers
        self.create_subscription(Point, '/robot_position', self._pos_cb, qos_fast)
        self.create_subscription(Imu, '/imu', self._imu_cb, qos_fast)
        self.create_subscription(Float32, '/battery_level', self._bat_cb, qos_fast)
        self.create_subscription(String, '/health_status', self._health_cb, qos_rel)
        self.create_subscription(String, '/mission_status', self._mission_cb, qos_fast)
        self.create_subscription(String, '/emergency_status', self._emerg_cb, qos_rel)
        
        # Publishers
        self.inject_pub = self.create_publisher(String, '/inject_fault', qos_rel)
        self.clear_pub = self.create_publisher(String, '/clear_fault', qos_rel)
        self.perturb_pub = self.create_publisher(String, '/robot_perturbation', qos_fast)
        self.battery_set_pub = self.create_publisher(Float32, '/set_battery', qos_rel)
        self.watchdog_pub = self.create_publisher(Bool, '/simulate_disconnect', qos_rel)
        
        # Watchdog
        self._last_msg = time.time()
        self.create_timer(1.0, self._watchdog)
        
        self.get_logger().info('Mission Control Node OK')
        
    def _pos_cb(self, msg):
        self._last_msg = time.time()
        self.cb('pos', (msg.x, msg.y, msg.z))
        
    def _imu_cb(self, msg):
        self._last_msg = time.time()
        self.cb('imu', (msg.orientation.x, msg.orientation.y, 
                       msg.orientation.z, msg.orientation.w))
        
    def _bat_cb(self, msg):
        self._last_msg = time.time()
        self.cb('battery', msg.data)
        
    def _health_cb(self, msg):
        self._last_msg = time.time()
        try:
            self.cb('health', json.loads(msg.data))
        except: pass
        
    def _mission_cb(self, msg):
        self._last_msg = time.time()
        try:
            self.cb('mission', json.loads(msg.data))
        except: pass
        
    def _emerg_cb(self, msg):
        try:
            self.cb('emergency', json.loads(msg.data))
        except: pass
        
    def _watchdog(self):
        elapsed = time.time() - self._last_msg
        self.cb('watchdog', elapsed < 5.0)
        
    # === INJECTION COMMANDS ===
    def inject_battery_critical(self):
        """Set battery to 21% (critical low)"""
        msg = Float32()
        msg.data = 21.0
        self.battery_set_pub.publish(msg)
        self.get_logger().warn('INJECT: Battery -> 21%')
        
    def inject_ros_disconnect(self, duration=10.0):
        """Simulate ROS2 disconnect for N seconds"""
        msg = Bool()
        msg.data = True
        self.watchdog_pub.publish(msg)
        self.get_logger().error(f'INJECT: ROS2 DISCONNECT ({duration}s)')
        
    def inject_fall(self):
        """Inject fall perturbation"""
        msg = String()
        msg.data = json.dumps({'type': 'fall'})
        self.perturb_pub.publish(msg)
        self.get_logger().error('INJECT: FALL')
        
    def inject_trip(self):
        """Inject trip perturbation"""
        msg = String()
        msg.data = json.dumps({'type': 'trip'})
        self.perturb_pub.publish(msg)
        self.get_logger().warn('INJECT: TRIP')
        
    def inject_push(self):
        """Inject strong lateral push"""
        msg = String()
        msg.data = json.dumps({
            'type': 'push',
            'force_x': 80.0,
            'force_y': 40.0,
            'force_z': 0,
            'duration': 0.25
        })
        self.perturb_pub.publish(msg)
        self.get_logger().warn('INJECT: PUSH')
        
    def inject_motor_fault(self):
        """Inject motor failure"""
        msg = String()
        msg.data = json.dumps({'type': 'motor_fault', 'enabled': True})
        self.perturb_pub.publish(msg)
        self.get_logger().error('INJECT: MOTOR FAULT')
    
    def inject_stumble(self):
        """Inject light stumble (recoverable)"""
        msg = String()
        msg.data = json.dumps({'type': 'stumble'})
        self.perturb_pub.publish(msg)
        self.get_logger().info('INJECT: STUMBLE (light)')
    
    def inject_wind(self):
        """Inject wind gust (sustained light force)"""
        msg = String()
        msg.data = json.dumps({'type': 'wind'})
        self.perturb_pub.publish(msg)
        self.get_logger().info('INJECT: WIND GUST')
    
    def inject_critical_fall(self):
        """Inject critical fall (severe)"""
        msg = String()
        msg.data = json.dumps({'type': 'critical_fall'})
        self.perturb_pub.publish(msg)
        self.get_logger().error('INJECT: CRITICAL FALL')
    
    def restore_authority(self):
        """Restore ROS2 authority after watchdog trip"""
        msg = Bool()
        msg.data = True
        # Need a publisher for this (VOLATILE to match subscriber)
        if not hasattr(self, 'authority_pub'):
            qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE,
                           durability=DurabilityPolicy.VOLATILE)
            self.authority_pub = self.create_publisher(Bool, '/restore_ros_authority', qos)
        self.authority_pub.publish(msg)
        self.get_logger().info('RESTORE: ROS2 Authority')
        
    def clear_all(self):
        """Clear all faults"""
        msg = String()
        msg.data = json.dumps({'type': 'clear'})
        self.perturb_pub.publish(msg)
        
        msg2 = String()
        msg2.data = json.dumps({'fault_type': 'all'})
        self.clear_pub.publish(msg2)
        self.get_logger().info('CLEAR ALL FAULTS')


# =============================================================================
# GUI
# =============================================================================
class MissionControlGUI:
    def __init__(self):
        self.state = State()
        self.start_time = time.time()
        self.node: Optional[ControlNode] = None
        self.fault_log = []  # Fault history with format: ID,Time,System,Sensor,Type,Sev,MsgType,Msg
        self._fault_id = 0
        
        # Window - Made larger
        self.root = tk.Tk()
        self.root.title("◆ OBEDIENCE MISSION CONTROL ◆")
        self.root.geometry("1500x900")
        self.root.configure(bg=C['bg'])
        
        self._build()
        self.root.after(100, self._update)
        
    def _build(self):
        # Main
        main = tk.Frame(self.root, bg=C['bg'])
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Top status bar
        self._build_status(main)
        
        # Content: 3 columns
        content = tk.Frame(main, bg=C['bg'])
        content.pack(fill=tk.BOTH, expand=True, pady=8)
        
        # Left: Schematic + Map
        left = tk.Frame(content, bg=C['bg'], width=500)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        self._build_schematic(left)
        self._build_map(left)
        
        # Center: Battery + Log
        center = tk.Frame(content, bg=C['bg'], width=350)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        self._build_battery_panel(center)
        self._build_log(center)
        
        # Right: Fault Injection + FMEA
        right = tk.Frame(content, bg=C['bg'], width=400)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        self._build_injection(right)
        self._build_fmea(right)
        
    def _build_status(self, parent):
        bar = tk.Frame(parent, bg=C['panel'], height=45)
        bar.pack(fill=tk.X)
        bar.pack_propagate(False)
        
        # Title
        tk.Label(bar, text="◆ OBEDIENCE MISSION CONTROL ◆",
                bg=C['panel'], fg=C['accent'],
                font=('Consolas', 14, 'bold')).pack(side=tk.LEFT, padx=15)
        
        # ROS2 Status
        self.ros_lbl = tk.Label(bar, text="● ROS2: ---",
                               bg=C['panel'], fg=C['dim'],
                               font=('Consolas', 11))
        self.ros_lbl.pack(side=tk.LEFT, padx=20)
        
        # Watchdog
        self.wd_lbl = tk.Label(bar, text="WD: ---",
                              bg=C['panel'], fg=C['dim'],
                              font=('Consolas', 10))
        self.wd_lbl.pack(side=tk.LEFT, padx=10)
        
        # Mission
        self.mission_lbl = tk.Label(bar, text="MISSION: IDLE",
                                   bg=C['panel'], fg=C['dim'],
                                   font=('Consolas', 11))
        self.mission_lbl.pack(side=tk.LEFT, padx=20)
        
        # MET
        self.met_lbl = tk.Label(bar, text="MET: 00:00:00",
                               bg=C['panel'], fg=C['text'],
                               font=('Consolas', 11))
        self.met_lbl.pack(side=tk.RIGHT, padx=15)
        
    def _build_schematic(self, parent):
        """Professional schematic diagram of robot"""
        frame = tk.LabelFrame(parent, text=" ROBOT SCHEMATIC ",
                             bg=C['panel'], fg=C['accent'],
                             font=('Consolas', 10, 'bold'))
        frame.pack(fill=tk.X, pady=4)
        
        self.schem_canvas = tk.Canvas(frame, width=480, height=220,
                                     bg=C['widget'], highlightthickness=1,
                                     highlightbackground=C['border'])
        self.schem_canvas.pack(padx=5, pady=5)
        self._draw_schematic()
        
    def _draw_schematic(self):
        """Draw professional 2D schematic"""
        c = self.schem_canvas
        c.delete('all')
        
        cx, cy = 240, 110
        
        # Grid lines (subtle)
        for x in range(0, 481, 40):
            c.create_line(x, 0, x, 220, fill='#1a1a25', dash=(1,3))
        for y in range(0, 221, 40):
            c.create_line(0, y, 480, y, fill='#1a1a25', dash=(1,3))
        
        # Component colors based on state
        def get_color(status):
            return {'OK': C['ok'], 'WARN': C['warn'], 'FAIL': C['danger']}.get(status, C['dim'])
        
        torso_c = get_color(self.state.torso)
        left_c = get_color(self.state.left_leg)
        right_c = get_color(self.state.right_leg)
        imu_c = get_color(self.state.imu)
        comm_c = get_color(self.state.comms)
        
        # === TORSO (rectangular body) ===
        c.create_rectangle(cx-35, cy-50, cx+35, cy+20,
                          outline=torso_c, width=2, fill=C['widget'])
        c.create_text(cx, cy-15, text="TORSO", fill=torso_c, font=('Consolas', 8))
        
        # === IMU (small box on torso) ===
        c.create_rectangle(cx-15, cy-45, cx+15, cy-30,
                          outline=imu_c, width=2, fill=C['widget'])
        c.create_text(cx, cy-37, text="IMU", fill=imu_c, font=('Consolas', 7))
        
        # === COMMS (antenna) ===
        c.create_line(cx+25, cy-50, cx+25, cy-70, fill=comm_c, width=2)
        c.create_oval(cx+20, cy-75, cx+30, cy-70, outline=comm_c, fill=C['widget'])
        c.create_text(cx+45, cy-65, text="COMM", fill=comm_c, font=('Consolas', 7), anchor='w')
        
        # === LEFT LEG ===
        # Hip joint
        c.create_oval(cx-40, cy+15, cx-20, cy+35, outline=left_c, width=2)
        c.create_text(cx-30, cy+25, text="H", fill=left_c, font=('Consolas', 7))
        # Thigh
        c.create_line(cx-30, cy+35, cx-35, cy+80, fill=left_c, width=3)
        # Knee joint
        c.create_oval(cx-45, cy+75, cx-25, cy+95, outline=left_c, width=2)
        c.create_text(cx-35, cy+85, text="K", fill=left_c, font=('Consolas', 7))
        # Shin
        c.create_line(cx-35, cy+95, cx-35, cy+140, fill=left_c, width=3)
        # Foot
        c.create_rectangle(cx-50, cy+135, cx-20, cy+150, outline=left_c, width=2)
        c.create_text(cx-35, cy+142, text="F", fill=left_c, font=('Consolas', 7))
        
        # === RIGHT LEG ===
        c.create_oval(cx+20, cy+15, cx+40, cy+35, outline=right_c, width=2)
        c.create_text(cx+30, cy+25, text="H", fill=right_c, font=('Consolas', 7))
        c.create_line(cx+30, cy+35, cx+35, cy+80, fill=right_c, width=3)
        c.create_oval(cx+25, cy+75, cx+45, cy+95, outline=right_c, width=2)
        c.create_text(cx+35, cy+85, text="K", fill=right_c, font=('Consolas', 7))
        c.create_line(cx+35, cy+95, cx+35, cy+140, fill=right_c, width=3)
        c.create_rectangle(cx+20, cy+135, cx+50, cy+150, outline=right_c, width=2)
        c.create_text(cx+35, cy+142, text="F", fill=right_c, font=('Consolas', 7))
        
        # === Labels ===
        c.create_text(cx-70, cy+90, text="LEFT", fill=left_c, 
                     font=('Consolas', 9), anchor='e')
        c.create_text(cx+70, cy+90, text="RIGHT", fill=right_c, 
                     font=('Consolas', 9), anchor='w')
        
        # === STATUS PANEL (right side) ===
        sx = 380
        c.create_rectangle(sx-10, 15, sx+90, 205, outline=C['border'], fill=C['panel'])
        c.create_text(sx+40, 25, text="STATUS", fill=C['text'], font=('Consolas', 9, 'bold'))
        
        items = [
            ("TORSO", self.state.torso),
            ("IMU", self.state.imu),
            ("COMM", self.state.comms),
            ("L-LEG", self.state.left_leg),
            ("R-LEG", self.state.right_leg),
        ]
        for i, (name, status) in enumerate(items):
            y = 50 + i*30
            color = get_color(status)
            c.create_text(sx, y, text=name, fill=C['dim'], font=('Consolas', 8), anchor='w')
            c.create_text(sx+70, y, text=status, fill=color, font=('Consolas', 8, 'bold'), anchor='e')
        
        # === HEIGHT indicator ===
        h = self.state.height
        h_color = C['ok'] if h > 0.5 else (C['warn'] if h > 0.3 else C['danger'])
        c.create_text(sx+40, 190, text=f"H: {h:.2f}m", fill=h_color, font=('Consolas', 9))
        
    def _build_map(self, parent):
        frame = tk.LabelFrame(parent, text=" MISSION MAP ",
                             bg=C['panel'], fg=C['accent'],
                             font=('Consolas', 10, 'bold'))
        frame.pack(fill=tk.BOTH, expand=True, pady=4)
        
        self.map_canvas = tk.Canvas(frame, width=480, height=280,
                                   bg=C['widget'], highlightthickness=1,
                                   highlightbackground=C['border'])
        self.map_canvas.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        self._draw_map()
        
    def _draw_map(self):
        c = self.map_canvas
        c.delete('all')
        
        w, h = c.winfo_width() or 480, c.winfo_height() or 280
        
        def w2c(x, y):
            cx = (x - MAP['x_min']) / (MAP['x_max'] - MAP['x_min']) * w * 0.9 + w*0.05
            cy = h - ((y - MAP['y_min']) / (MAP['y_max'] - MAP['y_min']) * h * 0.9 + h*0.05)
            return cx, cy
        
        # Grid
        for x in range(-8, 6, 2):
            px, _ = w2c(x, 0)
            c.create_line(px, 0, px, h, fill='#1a1a25', dash=(1,3))
            c.create_text(px, h-8, text=str(x), fill=C['dim'], font=('Consolas', 7))
        for y in range(-6, 7, 2):
            _, py = w2c(0, y)
            c.create_line(0, py, w, py, fill='#1a1a25', dash=(1,3))
        
        # Beds
        for name, pos in [('BED1', (-3,-2.5)), ('BED2', (-5,-2.5)), ('BED3', (-5,2.5))]:
            bx, by = w2c(pos[0], pos[1])
            c.create_rectangle(bx-20, by-12, bx+20, by+12,
                              fill=C['bed'], outline=C['border'])
            c.create_text(bx, by, text=name, fill=C['text'], font=('Consolas', 7))
        
        # Waypoints
        wp_colors = {'supply': C['supply'], 'charging': C['charge']}
        for name, pos in WAYPOINTS.items():
            px, py = w2c(pos[0], pos[1])
            color = wp_colors.get(name, C['accent'])
            
            is_current = (name == self.state.current_wp)
            if is_current:
                c.create_oval(px-15, py-15, px+15, py+15, outline=C['target'], width=2)
            
            c.create_oval(px-8, py-8, px+8, py+8, fill=color, outline=C['border'])
            c.create_text(px, py-18, text=name.upper(), fill=C['text'], font=('Consolas', 7))
        
        # Trajectory
        if len(self.state.trajectory) > 1:
            pts = []
            for x, y in self.state.trajectory[-150:]:
                pts.extend(w2c(x, y))
            if len(pts) >= 4:
                c.create_line(pts, fill=C['path'], width=2, smooth=True)
        
        # Robot
        rx, ry = w2c(self.state.pos[0], self.state.pos[1])
        q = self.state.quat
        heading = math.atan2(2*(q[3]*q[2] + q[0]*q[1]), 1-2*(q[1]**2 + q[2]**2))
        
        # Robot triangle (direction indicator)
        size = 12
        pts = [
            rx + size*math.cos(heading), ry - size*math.sin(heading),
            rx + size*0.6*math.cos(heading+2.5), ry - size*0.6*math.sin(heading+2.5),
            rx + size*0.6*math.cos(heading-2.5), ry - size*0.6*math.sin(heading-2.5),
        ]
        c.create_polygon(pts, fill=C['robot'], outline=C['text'])
        
        # Position text
        c.create_text(10, 15, text=f"X: {self.state.pos[0]:.2f}m",
                     fill=C['text'], font=('Consolas', 9), anchor='w')
        c.create_text(10, 30, text=f"Y: {self.state.pos[1]:.2f}m",
                     fill=C['text'], font=('Consolas', 9), anchor='w')
        
    def _build_battery_panel(self, parent):
        frame = tk.LabelFrame(parent, text=" POWER SYSTEM ",
                             bg=C['panel'], fg=C['accent'],
                             font=('Consolas', 10, 'bold'))
        frame.pack(fill=tk.X, pady=4)
        
        self.bat_canvas = tk.Canvas(frame, width=330, height=100,
                                   bg=C['widget'], highlightthickness=1,
                                   highlightbackground=C['border'])
        self.bat_canvas.pack(padx=5, pady=5)
        self._draw_battery()
        
    def _draw_battery(self):
        c = self.bat_canvas
        c.delete('all')
        
        bat = self.state.battery
        
        # Color based on level
        if bat > 50:
            color = C['ok']
        elif bat > 20:
            color = C['warn']
        else:
            color = C['danger']
        
        # Battery outline
        c.create_rectangle(20, 30, 280, 70, outline=C['border'], width=2)
        c.create_rectangle(280, 40, 290, 60, fill=C['border'])  # Terminal
        
        # Fill
        fill_w = (bat / 100) * 256
        c.create_rectangle(22, 32, 22 + fill_w, 68, fill=color, outline='')
        
        # Percentage text
        c.create_text(150, 50, text=f"{bat:.1f}%",
                     fill=C['text'], font=('Consolas', 16, 'bold'))
        
        # Status text
        status = "NOMINAL" if bat > 50 else ("LOW" if bat > 20 else "CRITICAL")
        c.create_text(150, 85, text=f"STATUS: {status}",
                     fill=color, font=('Consolas', 10))
        
        # Thresholds
        for thresh in [20, 50]:
            x = 22 + (thresh / 100) * 256
            c.create_line(x, 25, x, 75, fill=C['dim'], dash=(2,2))
        
    def _build_log(self, parent):
        frame = tk.LabelFrame(parent, text=" EVENT LOG ",
                             bg=C['panel'], fg=C['accent'],
                             font=('Consolas', 10, 'bold'))
        frame.pack(fill=tk.BOTH, expand=True, pady=4)
        
        self.log_text = tk.Text(frame, height=15, width=40,
                               bg=C['widget'], fg=C['text'],
                               font=('Consolas', 9), wrap=tk.WORD,
                               highlightthickness=1,
                               highlightbackground=C['border'])
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.log_text.tag_configure('INFO', foreground=C['text'])
        self.log_text.tag_configure('WARN', foreground=C['warn'])
        self.log_text.tag_configure('ERROR', foreground=C['danger'])
        self.log_text.tag_configure('OK', foreground=C['ok'])
        
        self._log("SYSTEM", "INFO", "Mission Control initialized")
        
    def _log(self, sys, sev, msg):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{ts}] [{sys}] {msg}\n", sev)
        self.log_text.see(tk.END)
        # Keep max 100 lines
        if int(self.log_text.index('end-1c').split('.')[0]) > 100:
            self.log_text.delete('1.0', '2.0')
        
    def _build_injection(self, parent):
        frame = tk.LabelFrame(parent, text=" FAULT INJECTION ",
                             bg=C['panel'], fg=C['danger'],
                             font=('Consolas', 10, 'bold'))
        frame.pack(fill=tk.X, pady=4)
        
        inner = tk.Frame(frame, bg=C['panel'])
        inner.pack(padx=10, pady=10)
        
        # Row 1: Light perturbations (recoverable)
        row1 = tk.Frame(inner, bg=C['panel'])
        row1.pack(fill=tk.X, pady=3)
        
        tk.Label(row1, text="LIGHT:", bg=C['panel'], fg=C['ok'],
                font=('Consolas', 9)).pack(side=tk.LEFT, padx=5)
        
        tk.Button(row1, text="~ STUMBLE", bg='#224422', fg=C['text'],
                 font=('Consolas', 9), width=10,
                 command=self._inject_stumble).pack(side=tk.LEFT, padx=2)
        
        tk.Button(row1, text="≋ WIND", bg='#223344', fg=C['text'],
                 font=('Consolas', 9), width=8,
                 command=self._inject_wind).pack(side=tk.LEFT, padx=2)
        
        tk.Button(row1, text="→ PUSH", bg='#334455', fg=C['text'],
                 font=('Consolas', 9), width=8,
                 command=self._inject_push).pack(side=tk.LEFT, padx=2)
        
        # Row 2: Medium perturbations
        row2 = tk.Frame(inner, bg=C['panel'])
        row2.pack(fill=tk.X, pady=3)
        
        tk.Label(row2, text="MEDIUM:", bg=C['panel'], fg=C['warn'],
                font=('Consolas', 9)).pack(side=tk.LEFT, padx=5)
        
        tk.Button(row2, text="↯ TRIP", bg='#553300', fg=C['text'],
                 font=('Consolas', 9, 'bold'), width=8,
                 command=self._inject_trip).pack(side=tk.LEFT, padx=2)
        
        tk.Button(row2, text="⚡ FALL", bg='#553322', fg=C['text'],
                 font=('Consolas', 9, 'bold'), width=8,
                 command=self._inject_fall).pack(side=tk.LEFT, padx=2)
        
        tk.Button(row2, text="🔋 BAT→21%", bg='#554400', fg=C['text'],
                 font=('Consolas', 9, 'bold'), width=10,
                 command=self._inject_battery).pack(side=tk.LEFT, padx=2)
        
        # Row 3: Critical/System faults
        row3 = tk.Frame(inner, bg=C['panel'])
        row3.pack(fill=tk.X, pady=3)
        
        tk.Label(row3, text="CRITICAL:", bg=C['panel'], fg=C['danger'],
                font=('Consolas', 9)).pack(side=tk.LEFT, padx=5)
        
        tk.Button(row3, text="☠ CRITICAL FALL", bg='#550000', fg=C['text'],
                 font=('Consolas', 9, 'bold'), width=14,
                 command=self._inject_critical_fall).pack(side=tk.LEFT, padx=2)
        
        tk.Button(row3, text="⚙ MOTOR", bg='#553355', fg=C['text'],
                 font=('Consolas', 9, 'bold'), width=8,
                 command=self._inject_motor).pack(side=tk.LEFT, padx=2)
        
        # Row 4: Communication/Watchdog
        row4 = tk.Frame(inner, bg=C['panel'])
        row4.pack(fill=tk.X, pady=3)
        
        tk.Label(row4, text="COMMS:", bg=C['panel'], fg=C['dim'],
                font=('Consolas', 9)).pack(side=tk.LEFT, padx=5)
        
        tk.Button(row4, text="📡 ROS2 DISCONNECT", bg='#550055', fg=C['text'],
                 font=('Consolas', 9, 'bold'), width=18,
                 command=self._inject_disconnect).pack(side=tk.LEFT, padx=2)
        
        # Row 5: Recovery actions
        row5 = tk.Frame(inner, bg=C['panel'])
        row5.pack(fill=tk.X, pady=8)
        
        tk.Button(row5, text="↻ RESTORE AUTHORITY", bg='#224466', fg=C['text'],
                 font=('Consolas', 10, 'bold'), width=20,
                 command=self._restore_authority).pack(side=tk.LEFT, padx=3)
        
        tk.Button(row5, text="✓ CLEAR ALL", bg=C['ok'], fg=C['bg'],
                 font=('Consolas', 10, 'bold'), width=12,
                 command=self._clear_all).pack(side=tk.LEFT, padx=3)
    
    def _inject_stumble(self):
        if self.node:
            self.node.inject_stumble()
            self._add_fault("PERTURBATION", "TORSO", "STUMBLE", "LOW", "INFO", "Light stumble injected")
            
    def _inject_wind(self):
        if self.node:
            self.node.inject_wind()
            self._add_fault("PERTURBATION", "TORSO", "WIND", "LOW", "INFO", "Wind gust injected")
        
    def _inject_fall(self):
        if self.node:
            self.node.inject_fall()
            self._add_fault("PERTURBATION", "TORSO", "FALL", "MEDIUM", "WARN", "Fall perturbation injected")
            self.state.torso = "WARN"
            
    def _inject_critical_fall(self):
        if self.node:
            self.node.inject_critical_fall()
            self._add_fault("PERTURBATION", "TORSO", "CRITICAL_FALL", "CRITICAL", "ERROR", "Critical fall induced!")
            self.state.torso = "FAIL"
            
    def _inject_trip(self):
        if self.node:
            self.node.inject_trip()
            self._add_fault("PERTURBATION", "LEGS", "TRIP", "MEDIUM", "WARN", "Trip perturbation injected")
            
    def _inject_push(self):
        if self.node:
            self.node.inject_push()
            self._add_fault("PERTURBATION", "TORSO", "PUSH", "LOW", "INFO", "Push perturbation injected")
            
    def _inject_battery(self):
        if self.node:
            self.node.inject_battery_critical()
            self.state.battery = 21.0
            self._add_fault("POWER", "BATTERY", "LOW_VOLTAGE", "HIGH", "WARN", "Battery set to 21%")
            
    def _inject_motor(self):
        if self.node:
            self.node.inject_motor_fault()
            self._add_fault("ACTUATOR", "MOTORS", "MOTOR_FAULT", "CRITICAL", "ERROR", "Motor fault injected")
            self.state.left_leg = "FAIL"
            self.state.right_leg = "FAIL"
            
    def _inject_disconnect(self):
        if self.node:
            self.node.inject_ros_disconnect()
            self._add_fault("COMMS", "ROS2", "DISCONNECT", "HIGH", "ERROR", "ROS2 disconnect simulated (10s)")
            self.state.comms = "FAIL"
    
    def _restore_authority(self):
        if self.node:
            self.node.restore_authority()
            self._add_fault("COMMS", "ROS2", "AUTHORITY_RESTORE", "INFO", "INFO", "Manual authority restoration requested")
            self.state.comms = "OK"
            
    def _clear_all(self):
        if self.node:
            self.node.clear_all()
            self._add_fault("SYSTEM", "ALL", "CLEAR", "INFO", "INFO", "All faults cleared")
            self.state.torso = "OK"
            self.state.left_leg = "OK"
            self.state.right_leg = "OK"
            self.state.imu = "OK"
            self.state.comms = "OK"
    
    def _add_fault(self, system: str, sensor: str, fault_type: str, 
                   severity: str, msg_type: str, message: str):
        """Add fault with standard format: ID, Time, System, Sensor, Type, Sev, MsgType, Msg"""
        self._fault_id += 1
        entry = {
            'id': self._fault_id,
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'system': system,
            'sensor': sensor,
            'type': fault_type,
            'severity': severity,
            'msg_type': msg_type,
            'message': message
        }
        self.fault_log.append(entry)
        
        # Update log display
        sev_tag = {'CRITICAL': 'ERROR', 'HIGH': 'ERROR', 'MEDIUM': 'WARN', 'LOW': 'INFO', 'INFO': 'OK'}.get(severity, 'INFO')
        self._log(system, sev_tag, f"[{fault_type}] {message}")
        
        # Keep only last 100
        if len(self.fault_log) > 100:
            self.fault_log = self.fault_log[-100:]
        
    def _build_fmea(self, parent):
        frame = tk.LabelFrame(parent, text=" FAULT LOG (ID|TIME|SYS|SENSOR|TYPE|SEV|MSG) ",
                             bg=C['panel'], fg=C['accent'],
                             font=('Consolas', 10, 'bold'))
        frame.pack(fill=tk.BOTH, expand=True, pady=4)
        
        # Full fault table with new format
        cols = ('id', 'time', 'system', 'sensor', 'type', 'severity', 'message')
        self.fmea_tree = ttk.Treeview(frame, columns=cols, show='headings', height=12)
        
        # Configure columns
        self.fmea_tree.heading('id', text='ID')
        self.fmea_tree.heading('time', text='TIME')
        self.fmea_tree.heading('system', text='SYSTEM')
        self.fmea_tree.heading('sensor', text='SENSOR')
        self.fmea_tree.heading('type', text='TYPE')
        self.fmea_tree.heading('severity', text='SEV')
        self.fmea_tree.heading('message', text='MESSAGE')
        
        self.fmea_tree.column('id', width=40, anchor='center')
        self.fmea_tree.column('time', width=70, anchor='center')
        self.fmea_tree.column('system', width=80, anchor='center')
        self.fmea_tree.column('sensor', width=70, anchor='center')
        self.fmea_tree.column('type', width=100, anchor='center')
        self.fmea_tree.column('severity', width=60, anchor='center')
        self.fmea_tree.column('message', width=200, anchor='w')
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(frame, orient='vertical', command=self.fmea_tree.yview)
        self.fmea_tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.fmea_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Style
        style = ttk.Style()
        style.configure('Treeview', background=C['widget'], foreground=C['text'],
                       fieldbackground=C['widget'], rowheight=20)
        style.configure('Treeview.Heading', background=C['panel'], foreground=C['accent'])
        
        self._update_fmea()
        
    def _update_fmea(self):
        """Update fault log display from fault_log list."""
        # Clear existing
        for item in self.fmea_tree.get_children():
            self.fmea_tree.delete(item)
        
        # Add recent faults (newest first for visibility)
        for entry in reversed(self.fault_log[-20:]):
            values = (
                entry.get('id', 0),
                entry.get('timestamp', ''),
                entry.get('system', ''),
                entry.get('sensor', ''),
                entry.get('type', ''),
                entry.get('severity', ''),
                entry.get('message', '')[:50]  # Truncate long messages
            )
            self.fmea_tree.insert('', 'end', values=values)
            
    # === DATA CALLBACK ===
    def _handle_data(self, dtype, data):
        if dtype == 'pos':
            self.state.pos = (data[0], data[1])
            self.state.height = data[2]
            self.state.trajectory.append((data[0], data[1]))
            if len(self.state.trajectory) > 300:
                self.state.trajectory = self.state.trajectory[-300:]
                
        elif dtype == 'imu':
            self.state.quat = data
            
        elif dtype == 'battery':
            self.state.battery = data
            
        elif dtype == 'health':
            self.state.faults = data.get('active_faults', [])
            
        elif dtype == 'mission':
            self.state.mission_active = data.get('active', False)
            wp = data.get('current_waypoint')
            if wp and isinstance(wp, dict):
                self.state.current_wp = wp.get('name', '')
                
        elif dtype == 'emergency':
            if data.get('emergency'):
                self.state.emergency = True
                self._log("EMERGENCY", "ERROR", data.get('reason', 'Unknown'))
                
        elif dtype == 'watchdog':
            self.state.ros_connected = data
            if not data and self.state.comms != "FAIL":
                self.state.comms = "FAIL"
                self._log("WATCHDOG", "ERROR", "ROS2 connection lost!")
            elif data and self.state.comms == "FAIL":
                self.state.comms = "OK"
                self._log("WATCHDOG", "OK", "ROS2 connection restored")
                
    # === UPDATE LOOP ===
    def _update(self):
        # Status bar
        if self.state.ros_connected:
            self.ros_lbl.configure(text="● ROS2: CONNECTED", fg=C['ok'])
            self.wd_lbl.configure(text="WD: OK", fg=C['ok'])
        else:
            self.ros_lbl.configure(text="● ROS2: DISCONNECTED", fg=C['danger'])
            self.wd_lbl.configure(text="WD: TIMEOUT", fg=C['danger'])
            
        if self.state.mission_active:
            self.mission_lbl.configure(text=f"MISSION: {self.state.current_wp}",
                                      fg=C['ok'])
        else:
            self.mission_lbl.configure(text="MISSION: IDLE", fg=C['dim'])
            
        # MET
        met = int(time.time() - self.start_time)
        h, r = divmod(met, 3600)
        m, s = divmod(r, 60)
        self.met_lbl.configure(text=f"MET: {h:02d}:{m:02d}:{s:02d}")
        
        # Update displays
        self._draw_schematic()
        self._draw_map()
        self._draw_battery()
        self._update_fmea()
        
        self.root.after(100, self._update)
        
    def set_node(self, node):
        self.node = node
        self._log("SYSTEM", "OK", "ROS2 node connected")
        
    def run(self):
        self.root.mainloop()


# =============================================================================
# MAIN
# =============================================================================
def main(args=None):
    rclpy.init(args=args)
    
    gui = MissionControlGUI()
    node = ControlNode(gui._handle_data)
    gui.set_node(node)
    
    def spin():
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            
    t = threading.Thread(target=spin, daemon=True)
    t.start()
    
    try:
        gui.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
