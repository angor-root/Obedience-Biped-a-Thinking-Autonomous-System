#!/usr/bin/env python3
"""
Thinking Node - High-Level Autonomy ROS2 Node

This node integrates all thinking subsystems:
- Sensory: Process sensor data from MuJoCo bridge
- Will: Mission planning and executive control
- Decision: Action selection based on rules
- Reason: FMEA-based rules evaluation
- Understanding: Environment interpretation
- Presentation: Status reporting

Communication:
- Subscribes to sensor topics from mujoco_bridge
- Publishes navigation targets and status
- Coordinates with health_node for fault awareness
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import numpy as np
import time
from datetime import datetime
from typing import Optional, List, Dict
import json

# ROS2 message types
from std_msgs.msg import String, Float32, Bool
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Point, PoseStamped

# Thinking subsystems
from .thinking.common import FaultState, SystemHealth, MissionPhase, LocomotionState
from .thinking.sensory import (
    IMUProcessor, JointStateProcessor, ContactProcessor,
    LiDARProcessor, BatteryMonitor
)
from .thinking.will import MissionSubsystem, ExecutiveSubsystem
from .thinking.decision import ActionSelector
from .thinking.reason import RulesEngine
from .thinking.understanding import EnvironmentInterpreter
from .thinking.presentation import StatusReporter
from .thinking.knowledge import LearningEngine, KnowledgeBase


class ThinkingNode(Node):
    """
    Main Thinking System ROS2 Node.
    
    Orchestrates high-level autonomy for the bipedal robot:
    1. Receives sensor data from simulation
    2. Processes through sensory subsystem
    3. Evaluates rules and selects actions
    4. Commands navigation targets
    5. Reports status
    """
    
    def __init__(self):
        super().__init__('thinking_node')
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("  THINKING NODE - High-Level Autonomy System")
        self.get_logger().info("=" * 60)
        
        # Declare parameters
        self.declare_parameter('update_rate', 10.0)  # Hz
        self.declare_parameter('mission_file', '')
        self.declare_parameter('auto_start_mission', True)
        
        update_rate = self.get_parameter('update_rate').value
        
        # Initialize subsystems
        self._init_sensory_subsystem()
        self._init_cognitive_subsystems()
        
        # QoS profile for reliable communication
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )
        
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=1
        )
        
        # === SUBSCRIBERS (from MuJoCo bridge) ===
        self.imu_sub = self.create_subscription(
            Imu, '/imu', self._imu_callback, qos_sensor)
        
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_state_callback, qos_sensor)
        
        self.contact_sub = self.create_subscription(
            String, '/contacts', self._contact_callback, qos_sensor)
        
        self.battery_sub = self.create_subscription(
            Float32, '/battery_level', self._battery_callback, qos_sensor)
        
        # Health status from health_node
        self.health_sub = self.create_subscription(
            String, '/health_status', self._health_callback, qos_reliable)
        
        # Mission commands
        self.mission_cmd_sub = self.create_subscription(
            String, '/mission_command', self._mission_command_callback, qos_reliable)
        
        # === PUBLISHERS ===
        # Navigation target for walking controller
        self.nav_target_pub = self.create_publisher(
            Point, '/nav_target', qos_reliable)
        
        # Mission status
        self.mission_status_pub = self.create_publisher(
            String, '/mission_status', qos_reliable)
        
        # System status (for presentation)
        self.system_status_pub = self.create_publisher(
            String, '/system_status', qos_reliable)
        
        # Current action
        self.action_pub = self.create_publisher(
            String, '/current_action', qos_reliable)
        
        # Static equilibrium command to robot
        self.equilibrium_pub = self.create_publisher(
            Bool, '/static_equilibrium', qos_reliable)
        
        # Emergency status
        self.emergency_pub = self.create_publisher(
            String, '/emergency_status', qos_reliable)
        
        # Simulated disconnect subscriber (for testing watchdog)
        self.disconnect_sub = self.create_subscription(
            Bool, '/simulate_disconnect', self._simulate_disconnect_callback, qos_reliable)
        self._simulated_disconnect = False
        self._disconnect_end_time = 0.0
        
        # Robot position (from simulation)
        self.position_sub = self.create_subscription(
            Point, '/robot_position', self._position_callback, qos_sensor)
        
        # === TIMERS ===
        self.update_timer = self.create_timer(
            1.0 / update_rate, self._update_callback)
        
        self.status_timer = self.create_timer(
            1.0, self._publish_status)  # 1 Hz status updates
        
        # State
        self.robot_position = np.array([0.0, 0.0])
        self.system_health = SystemHealth()
        self.is_mission_active = False
        self.current_target: Optional[np.ndarray] = None
        
        # Mission waypoints (simple list-based management)
        self._waypoints = []
        self._waypoint_index = 0
        self._current_waypoint = None
        
        # Sensor data storage (populated by callbacks)
        self._imu_data = {}
        self._joint_data = {}
        self._contact_data = {'left': False, 'right': False}
        self._battery_level = 100.0
        self._robot_height = 0.68
        
        # ROS2 connection monitoring (Watchdog)
        self._last_sensor_time = time.time()
        self._ros_connected = True
        self._ros_authority = True  # Can accept remote commands
        self._connection_timeout = 5.0  # seconds
        self._emergency_mode = False
        self._static_equilibrium = False
        self._charging_mode = False
        
        # Knowledge/Learning System
        self._knowledge_base = []
        self._fault_history = []
        self._fault_id_counter = 0
        self._init_knowledge_system()
        
        # Authority restoration subscriber (use VOLATILE to match GUI publisher)
        qos_authority = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        self.authority_sub = self.create_subscription(
            Bool, '/restore_ros_authority', self._restore_authority_callback, qos_authority)
        
        self.get_logger().info("Thinking Node initialized")
        self.get_logger().info(f"  Update rate: {update_rate} Hz")
        self.get_logger().info("  Knowledge system enabled")
        self.get_logger().info("  Waiting for sensor data...")
        
        # Auto-start mission if configured
        if self.get_parameter('auto_start_mission').value:
            self.get_logger().info("  Auto-start enabled, waiting 5s...")
            self._auto_start_timer = self.create_timer(5.0, self._auto_start_mission_once)
    
    # =========================================================================
    # KNOWLEDGE/LEARNING SYSTEM
    # =========================================================================
    
    def _init_knowledge_system(self):
        """Initialize the knowledge/learning system."""
        self._knowledge_file = '/tmp/obedience_knowledge.json'
        self._load_knowledge()
        self.get_logger().info("Knowledge system initialized")
    
    def _load_knowledge(self):
        """Load knowledge base from file."""
        import os
        if os.path.exists(self._knowledge_file):
            try:
                with open(self._knowledge_file, 'r') as f:
                    data = json.load(f)
                    self._knowledge_base = data.get('knowledge', [])
                    self._fault_history = data.get('faults', [])
                    self._fault_id_counter = data.get('fault_id', 0)
                self.get_logger().info(f"  Loaded {len(self._knowledge_base)} knowledge entries")
                self.get_logger().info(f"  Loaded {len(self._fault_history)} fault records")
            except Exception as e:
                self.get_logger().warn(f"Could not load knowledge: {e}")
    
    def _save_knowledge(self):
        """Save knowledge base to file."""
        try:
            with open(self._knowledge_file, 'w') as f:
                json.dump({
                    'knowledge': self._knowledge_base[-100:],  # Keep last 100
                    'faults': self._fault_history[-500:],  # Keep last 500
                    'fault_id': self._fault_id_counter
                }, f, indent=2)
        except Exception as e:
            self.get_logger().error(f"Could not save knowledge: {e}")
    
    def _log_to_knowledge(self, entry: dict):
        """Add entry to knowledge base (now uses LearningEngine)."""
        # Use new knowledge system if available
        if hasattr(self, 'learning'):
            self.learning.kb.record_experience(
                event_type=entry.get('event', 'GENERAL'),
                system=entry.get('system', 'GENERAL'),
                sensor=entry.get('sensor', 'GENERAL'),
                severity=entry.get('severity', 'INFO'),
                message=entry.get('message', str(entry)),
                context=entry,
                outcome=entry.get('outcome', 'ONGOING')
            )
        else:
            # Fallback to old system
            entry['id'] = len(self._knowledge_base)
            entry['learned_at'] = time.time()
            self._knowledge_base.append(entry)
            if len(self._knowledge_base) % 10 == 0:
                self._save_knowledge()
    
    def _learn_from_entry(self, entry: dict):
        """Extract learnable patterns from knowledge entry."""
        event = entry.get('event', '')
        
        if event == 'AUTONOMOUS_DECISION':
            # Learn from autonomous decisions
            self.get_logger().debug(f"Learning from: {entry.get('decision')} at criticality {entry.get('criticality')}")
        
        elif event == 'FAULT_DETECTED':
            # Learn fault patterns  
            fault_type = entry.get('fault_type', '')
            if fault_type:
                # Check for recurring patterns
                recent_same_faults = [f for f in self._fault_history[-20:] 
                                     if f.get('type') == fault_type]
                if len(recent_same_faults) >= 3:
                    self.get_logger().warn(f"PATTERN: {fault_type} occurring frequently ({len(recent_same_faults)} times recently)")
    
    def _log_fault(self, system: str, sensor: str, fault_type: str, 
                   severity: str, msg_type: str, message: str):
        """
        Log fault with standard format:
        ID, Timestamp, System, Sensor, Type, Severity, MsgType, Message
        """
        self._fault_id_counter += 1
        fault_entry = {
            'id': self._fault_id_counter,
            'timestamp': datetime.now().isoformat(),
            'time_epoch': time.time(),
            'system': system,
            'sensor': sensor,
            'type': fault_type,
            'severity': severity,
            'msg_type': msg_type,
            'message': message
        }
        self._fault_history.append(fault_entry)
        
        # Log to knowledge for learning
        self._log_to_knowledge({
            'event': 'FAULT_DETECTED',
            'fault_type': fault_type,
            **fault_entry
        })
        
        # Print formatted log
        self.get_logger().warn(
            f"FAULT[{fault_entry['id']}] {fault_entry['timestamp']} | "
            f"{system}/{sensor} | {fault_type} | {severity} | {msg_type}: {message}"
        )
        
        return fault_entry
    
    def _restore_authority_callback(self, msg: Bool):
        """Handle manual authority restoration."""
        if msg.data:
            if not self._ros_authority:
                self._ros_authority = True
                self._ros_connected = True
                self.get_logger().info("=" * 50)
                self.get_logger().info("  ROS2 AUTHORITY MANUALLY RESTORED")
                self.get_logger().info("  Robot accepting remote commands again")
                self.get_logger().info("=" * 50)
                
                # Log to knowledge
                self._log_to_knowledge({
                    'event': 'AUTHORITY_RESTORED',
                    'manual': True,
                    'timestamp': time.time()
                })
            else:
                self.get_logger().info("ROS2 authority already active")
    
    def _init_sensory_subsystem(self):
        """Initialize all sensor processors."""
        self.get_logger().info("Initializing Sensory Subsystem...")
        
        self.imu_processor = IMUProcessor()
        self.joint_processor = JointStateProcessor()
        self.contact_processor = ContactProcessor()
        self.lidar_processor = LiDARProcessor()
        self.battery_monitor = BatteryMonitor()
        
        self.get_logger().info("  ✓ IMU Processor")
        self.get_logger().info("  ✓ Joint State Processor")
        self.get_logger().info("  ✓ Contact Processor")
        self.get_logger().info("  ✓ LiDAR Processor")
        self.get_logger().info("  ✓ Battery Monitor")
    
    def _init_cognitive_subsystems(self):
        """Initialize cognitive processing subsystems."""
        self.get_logger().info("Initializing Cognitive Subsystems...")
        
        # Knowledge - Learning and memory (FIRST - used by others)
        self.knowledge = KnowledgeBase()
        self.learning = LearningEngine(self.knowledge)
        self.get_logger().info("  ✓ Knowledge Base (persistent memory)")
        self.get_logger().info("  ✓ Learning Engine (experiential learning)")
        
        # Understanding - Environment interpretation
        self.environment = EnvironmentInterpreter()
        self.get_logger().info("  ✓ Environment Interpreter")
        
        # Reason - Rules engine
        self.rules_engine = RulesEngine()
        self.get_logger().info("  ✓ Rules Engine (FMEA)")
        
        # Decision - Action selector
        self.action_selector = ActionSelector()
        self.get_logger().info("  ✓ Action Selector")
        
        # Will - Mission management
        self.mission = MissionSubsystem()
        self.executive = ExecutiveSubsystem()
        self.get_logger().info("  ✓ Mission Subsystem")
        self.get_logger().info("  ✓ Executive Subsystem")
        
        # Presentation - Status reporting
        self.status_reporter = StatusReporter()
        self.get_logger().info("  ✓ Status Reporter")
    
    # === SENSOR CALLBACKS ===
    
    def _imu_callback(self, msg: Imu):
        """Process IMU data."""
        self._last_sensor_time = time.time()  # Update connection timestamp
        # Store IMU data for decision making
        self._imu_data = {
            'orientation': np.array([
                msg.orientation.x, msg.orientation.y,
                msg.orientation.z, msg.orientation.w
            ]),
            'angular_vel': np.array([
                msg.angular_velocity.x, msg.angular_velocity.y,
                msg.angular_velocity.z
            ]),
            'linear_acc': np.array([
                msg.linear_acceleration.x, msg.linear_acceleration.y,
                msg.linear_acceleration.z
            ])
        }
    
    def _joint_state_callback(self, msg: JointState):
        """Process joint states."""
        # Store joint data
        self._joint_data = {
            'positions': dict(zip(msg.name, msg.position)),
            'velocities': dict(zip(msg.name, msg.velocity)) if msg.velocity else {},
            'efforts': dict(zip(msg.name, msg.effort)) if msg.effort else {}
        }
    
    def _contact_callback(self, msg: String):
        """Process contact sensor data."""
        try:
            contacts = json.loads(msg.data)
            self._contact_data = {
                'left': contacts.get('left_foot', False),
                'right': contacts.get('right_foot', False)
            }
        except json.JSONDecodeError:
            pass
    
    def _battery_callback(self, msg: Float32):
        """Process battery level."""
        self._battery_level = msg.data
    
    def _position_callback(self, msg: Point):
        """Update robot position."""
        self.robot_position = np.array([msg.x, msg.y])
        # Environment interpreter updated during main update loop
    
    def _health_callback(self, msg: String):
        """Receive health status from health_node."""
        try:
            data = json.loads(msg.data)
            old_faults = set(self.system_health.active_faults)
            
            self.system_health.overall = FaultState[data.get('overall', 'NONE')]
            self.system_health.locomotion = FaultState[data.get('locomotion', 'NONE')]
            self.system_health.battery = FaultState[data.get('battery', 'NONE')]
            self.system_health.active_faults = data.get('active_faults', [])
            
            new_faults = set(self.system_health.active_faults)
            
            # Learn from new faults
            if hasattr(self, 'learning') and new_faults - old_faults:
                for fault in new_faults - old_faults:
                    # Use learning engine to decide response
                    response = self.learning.handle_fault(
                        fault_type=fault,
                        context={
                            'battery': self._battery_level,
                            'height': getattr(self, '_robot_height', 0.68),
                            'mission_active': self.is_mission_active
                        }
                    )
                    # Only abort on truly critical
                    if response.get('go_to_charger'):
                        self._emergency_go_to_charger(f"Critical fault: {fault}")
                    # Otherwise continue - logged to knowledge already
                    
        except (json.JSONDecodeError, KeyError):
            pass
    
    def _mission_command_callback(self, msg: String):
        """Handle mission commands."""
        command = msg.data.lower()
        
        if command == 'start':
            self._start_mission()
        elif command == 'stop':
            self._stop_mission()
        elif command == 'pause':
            self._pause_mission()
        elif command == 'resume':
            self._resume_mission()
        elif command == 'home':
            self._go_home()
        else:
            self.get_logger().warn(f"Unknown command: {command}")
    
    # === MISSION CONTROL ===
    
    def _auto_start_mission_once(self):
        """Auto-start mission after delay (one-shot)."""
        # Cancel timer to make it one-shot
        if hasattr(self, '_auto_start_timer'):
            self._auto_start_timer.cancel()
        self._auto_start_mission()
    
    def _auto_start_mission(self):
        """Auto-start mission after delay."""
        self.get_logger().info("Auto-starting mission...")
        self._start_mission()
    
    def _start_mission(self):
        """Start the delivery mission."""
        if self.is_mission_active:
            self.get_logger().warn("Mission already active")
            return
        
        self.get_logger().info("=" * 40)
        self.get_logger().info("  STARTING DELIVERY MISSION")
        self.get_logger().info("=" * 40)
        
        # Initialize mission start time for grace period
        self._mission_start_time = time.time()
        
        # Define hospital mission waypoints (positions adjacent to beds, not on top)
        # Beds have collision - waypoints are at accessible positions beside each bed
        waypoints = [
            {'name': 'supply', 'position': [2.0, 0.0], 'action': 'load', 'duration': 3.0},
            {'name': 'bed_1', 'position': [-3.0, -1.3], 'action': 'deliver', 'duration': 2.0},
            {'name': 'supply', 'position': [2.0, 0.0], 'action': 'load', 'duration': 3.0},
            {'name': 'bed_2', 'position': [-5.0, -1.3], 'action': 'deliver', 'duration': 2.0},
            {'name': 'supply', 'position': [2.0, 0.0], 'action': 'load', 'duration': 3.0},
            {'name': 'bed_3', 'position': [-5.0, 1.3], 'action': 'deliver', 'duration': 2.0},
            {'name': 'charging', 'position': [3.5, -4.5], 'action': 'charge', 'duration': 0.0},
        ]
        
        self._waypoints = waypoints
        self._waypoint_index = 0
        self.is_mission_active = True
        
        # Start with first waypoint
        self._advance_to_next_waypoint()
    
    def _stop_mission(self):
        """Stop the current mission."""
        self.get_logger().info("Stopping mission...")
        self.is_mission_active = False
        self.current_target = None
        self._waypoints = []
        self._waypoint_index = 0
        self._current_waypoint = None
    
    def _pause_mission(self):
        """Pause the mission."""
        self.get_logger().info("Pausing mission...")
        # Simple pause: just log (mission continues when resumed)
    
    def _resume_mission(self):
        """Resume the mission."""
        self.get_logger().info("Resuming mission...")
    
    def _go_home(self):
        """Navigate to charging station."""
        self.get_logger().info("Navigating to charging station...")
        self._send_nav_target(np.array([3.5, -4.5]))
    
    def _emergency_go_to_charger(self, reason: str):
        """Emergency protocol: go to charging station immediately."""
        self._emergency_mode = True
        
        self.get_logger().error("=" * 50)
        self.get_logger().error(f"  EMERGENCY: {reason}")
        self.get_logger().error("  Executing emergency protocol: GO TO CHARGER")
        self.get_logger().error("=" * 50)
        
        # Publish emergency status
        emergency_msg = String()
        emergency_msg.data = json.dumps({
            'emergency': True,
            'reason': reason,
            'action': 'GO_TO_CHARGER',
            'timestamp': time.time()
        })
        self.emergency_pub.publish(emergency_msg)
        
        # Override current mission - go directly to charger
        self._waypoints = [{
            'name': 'charging', 
            'position': [3.5, -4.5], 
            'action': 'emergency_charge', 
            'duration': 0.0
        }]
        self._waypoint_index = 0
        self.is_mission_active = True
        self._advance_to_next_waypoint()
    
    def _enter_static_equilibrium(self):
        """Enter static equilibrium mode at charging station."""
        self._static_equilibrium = True
        self._charging_mode = True
        
        self.get_logger().info("=" * 50)
        self.get_logger().info("  ENTERING STATIC EQUILIBRIUM MODE")
        self.get_logger().info("  Robot will remain stationary while charging")
        self.get_logger().info("=" * 50)
        
        # Send equilibrium command to robot
        eq_msg = Bool()
        eq_msg.data = True
        self.equilibrium_pub.publish(eq_msg)
        
        # Stop navigation
        self.current_target = None
        self.is_mission_active = False
    
    def _simulate_disconnect_callback(self, msg: Bool):
        """Handle simulated disconnect request from GUI."""
        if msg.data:
            self._simulated_disconnect = True
            self._disconnect_end_time = time.time() + 10.0  # 10 seconds simulated disconnect
            self.get_logger().error("=" * 50)
            self.get_logger().error("  SIMULATED ROS2 DISCONNECT ACTIVATED")
            self.get_logger().error("  Duration: 10 seconds")
            self.get_logger().error("=" * 50)
        else:
            self._simulated_disconnect = False
            self.get_logger().info("Simulated disconnect cancelled")
    
    def _check_ros_connection(self):
        """
        Watchdog: Check if ROS2 connection is alive.
        
        IMPORTANT: Does NOT auto-reconnect. Only:
        1. Cuts ROS2 authority (stops accepting remote commands)
        2. Makes autonomous decision based on criticality
        3. Notifies for manual reconnection
        """
        # Handle simulated disconnect
        if self._simulated_disconnect:
            if time.time() < self._disconnect_end_time:
                # Still in simulated disconnect
                if self._ros_connected:
                    self._ros_connected = False
                    self._ros_authority = False  # Cut ROS2 authority
                    self.get_logger().error("=" * 50)
                    self.get_logger().error("  WATCHDOG: ROS2 AUTHORITY REVOKED")
                    self.get_logger().error("  Robot now in AUTONOMOUS mode")
                    self.get_logger().error("  Manual reconnection required")
                    self.get_logger().error("=" * 50)
                    
                    # Autonomous decision based on criticality
                    self._make_autonomous_decision("SIMULATED DISCONNECT")
                return
            else:
                # Simulated disconnect ended - but DON'T auto-restore authority
                self._simulated_disconnect = False
                self.get_logger().warn("Simulated disconnect period ended")
                self.get_logger().warn("Manual authority restoration required via /restore_ros_authority")
        
        elapsed = time.time() - self._last_sensor_time
        
        if elapsed > self._connection_timeout:
            if self._ros_connected:
                # Connection just lost - CUT AUTHORITY
                self._ros_connected = False
                self._ros_authority = False
                self.get_logger().error("=" * 50)
                self.get_logger().error(f"  WATCHDOG: SENSOR TIMEOUT ({elapsed:.1f}s)")
                self.get_logger().error("  ROS2 AUTHORITY REVOKED")
                self.get_logger().error("  Robot now in AUTONOMOUS mode")
                self.get_logger().error("=" * 50)
                
                # Autonomous decision
                self._make_autonomous_decision(f"SENSOR TIMEOUT ({elapsed:.1f}s)")
                
                # Notify for manual reconnection
                self._notify_manual_reconnection_required()
        # NOTE: We do NOT auto-restore connection. Manual intervention required.
    
    def _make_autonomous_decision(self, reason: str):
        """
        Make autonomous decision based on current state criticality.
        
        PHILOSOPHY: Complete the mission unless truly impossible.
        - ROS2 disconnection is NOT a reason to abort
        - Continue autonomously unless battery critical or robot fallen
        """
        # Evaluate criticality using learning engine
        battery = self._battery_level
        height = getattr(self, '_robot_height', 0.68)
        
        # Ask learning engine
        abort_check = self.learning.should_abort_mission(
            battery=battery,
            height=height,
            ros_connected=self._ros_connected
        )
        
        if abort_check['should_abort']:
            # TRULY CRITICAL - must abort
            decision = "GO_TO_CHARGER"
            criticality = "CRITICAL"
            reasons_str = "; ".join(abort_check['reasons'])
            self.get_logger().error(f"CRITICAL ABORT: {reasons_str}")
            
            # Record to knowledge
            self.learning.kb.record_experience(
                event_type="DECISION",
                system="EXECUTIVE",
                sensor="WATCHDOG",
                severity="CRITICAL",
                message=f"Mission abort: {reasons_str}",
                context={'battery': battery, 'height': height, 'reason': reason},
                outcome="ABORT"
            )
            
            # Execute emergency
            self._emergency_go_to_charger(reason)
        else:
            # NOT CRITICAL - CONTINUE MISSION
            decision = "CONTINUE_AUTONOMOUS"
            criticality = "LOW"
            
            self.get_logger().warn(f"AUTONOMOUS MODE: Continuing mission (Battery: {battery:.1f}%, Height: {height:.2f}m)")
            self.get_logger().warn(f"  Trigger: {reason}")
            self.get_logger().warn(f"  Decision: Continue mission autonomously")
            
            # Record to knowledge
            self.learning.kb.record_experience(
                event_type="DECISION",
                system="EXECUTIVE",
                sensor="WATCHDOG",
                severity="INFO",
                message=f"Autonomous continuation: {reason}",
                context={'battery': battery, 'height': height, 'reason': reason},
                outcome="CONTINUE"
            )
            
            # Notify but DO NOT abort
            msg = String()
            msg.data = json.dumps({
                'autonomous_mode': True,
                'reason': reason,
                'action': 'CONTINUING_MISSION',
                'battery': battery,
                'timestamp': time.time()
            })
            self.emergency_pub.publish(msg)
    
    def _notify_manual_reconnection_required(self):
        """Publish notification that manual intervention is required."""
        msg = String()
        msg.data = json.dumps({
            'emergency': True,
            'type': 'ROS2_AUTHORITY_LOST',
            'reason': 'Watchdog triggered - manual reconnection required',
            'action_required': 'Publish to /restore_ros_authority to restore control',
            'timestamp': time.time()
        })
        self.emergency_pub.publish(msg)
    
    def _advance_to_next_waypoint(self):
        """Move to next waypoint in mission."""
        if self._waypoint_index >= len(self._waypoints):
            self.get_logger().info("=" * 40)
            self.get_logger().info("  MISSION COMPLETED!")
            self.get_logger().info("=" * 40)
            self.is_mission_active = False
            self._current_waypoint = None
            return
        
        waypoint = self._waypoints[self._waypoint_index]
        self._current_waypoint = waypoint
        self._waypoint_index += 1
        
        self.current_target = np.array(waypoint['position'])
        self.get_logger().info(f"Navigating to: {waypoint['name']} at {waypoint['position']}")
        self._send_nav_target(self.current_target)
    
    def _send_nav_target(self, target: np.ndarray):
        """Publish navigation target."""
        msg = Point()
        msg.x = float(target[0])
        msg.y = float(target[1])
        msg.z = 0.0
        self.nav_target_pub.publish(msg)
    
    # === MAIN UPDATE LOOP ===
    
    def _update_callback(self):
        """Main thinking loop - runs at update_rate Hz."""
        # Always check ROS connection
        self._check_ros_connection()
        
        # If in static equilibrium, just maintain it
        if self._static_equilibrium:
            # Periodically resend equilibrium command
            if not hasattr(self, '_last_eq_publish') or time.time() - self._last_eq_publish > 2.0:
                eq_msg = Bool()
                eq_msg.data = True
                self.equilibrium_pub.publish(eq_msg)
                self._last_eq_publish = time.time()
            
            # Check if battery full to exit equilibrium
            if self._battery_level >= 99.0 and self._charging_mode:
                self.get_logger().info("Battery fully charged - ready for new mission")
                self._charging_mode = False
                # Keep static equilibrium until new mission command
            return
        
        if not self.is_mission_active:
            return
        
        # 1. Check system health - stop only on CONFIRMED critical faults
        # Grace period: don't react to faults in first 8 seconds of mission
        if not hasattr(self, '_mission_start_time'):
            self._mission_start_time = time.time()
        
        mission_runtime = time.time() - self._mission_start_time
        grace_period_active = mission_runtime < 8.0
        
        # After grace period, check for critical faults
        if not grace_period_active:
            critical_faults = ['FALL']  # Only confirmed FALL stops mission
            has_critical_fault = (
                self.system_health.overall == FaultState.TRUE and
                any(f in critical_faults for f in self.system_health.active_faults)
            )
            
            if has_critical_fault:
                self.get_logger().error(f"CRITICAL FAULT: {self.system_health.active_faults}")
                self.get_logger().error("Stopping mission for safety!")
                self._stop_mission()
                return
            
            # Log warnings for non-critical faults but continue mission
            if self.system_health.overall == FaultState.SUSPECT:
                # Only log occasionally to avoid spam
                if hasattr(self, '_last_warning_time'):
                    if time.time() - self._last_warning_time > 5.0:
                        self.get_logger().warn(f"SUSPECT faults: {self.system_health.active_faults}")
                        self._last_warning_time = time.time()
                else:
                    self._last_warning_time = time.time()
        
        # 2. Build context from sensor data
        context = {
            'robot_position': self.robot_position,
            'battery_level': self._battery_level,
            'health': self.system_health,
            'contacts': self._contact_data,
        }
        
        # 3. Check if reached current waypoint
        if self.current_target is not None:
            distance = np.linalg.norm(self.robot_position - self.current_target)
            
            if distance < 0.3:  # Reached waypoint
                waypoint = self._current_waypoint
                if waypoint:
                    self.get_logger().info(f"Reached {waypoint['name']}")
                    
                    # Execute waypoint action
                    action = waypoint.get('action', 'none')
                    duration = waypoint.get('duration', 0.0)
                    
                    if action == 'load':
                        self.get_logger().info(f"  Loading medicine ({duration}s)...")
                    elif action == 'deliver':
                        self.get_logger().info(f"  Delivering medicine ({duration}s)...")
                    elif action == 'charge':
                        self.get_logger().info("  At charging station - entering equilibrium")
                        self._enter_static_equilibrium()
                        return  # Don't advance, stay at charger
                    elif action == 'emergency_charge':
                        self.get_logger().info("  EMERGENCY CHARGE - entering static equilibrium")
                        self._emergency_mode = False  # Clear emergency after reaching charger
                        self._enter_static_equilibrium()
                        return
                    
                    # Mark waypoint complete and advance
                    self._advance_to_next_waypoint()
        
        # 4. Publish current action
        action_msg = String()
        action_msg.data = json.dumps({
            'action': 'navigate' if self.is_mission_active else 'idle',
            'target': self.current_target.tolist() if self.current_target is not None else None,
            'phase': 'ACTIVE' if self.is_mission_active else 'IDLE'
        })
        self.action_pub.publish(action_msg)
    
    def _publish_status(self):
        """Publish system status (1 Hz)."""
        # Mission status
        mission_msg = String()
        mission_msg.data = json.dumps({
            'active': self.is_mission_active,
            'phase': 'ACTIVE' if self.is_mission_active else 'IDLE',
            'current_waypoint': self._current_waypoint,
            'progress': f"{self._waypoint_index}/{len(self._waypoints)}" if self._waypoints else "0/0",
            'position': self.robot_position.tolist()
        })
        self.mission_status_pub.publish(mission_msg)
        
        # System status
        system_msg = String()
        system_msg.data = json.dumps({
            'health': self.system_health.overall.name,
            'battery': self._battery_level,
            'locomotion': 'walking' if self._contact_data.get('left') or self._contact_data.get('right') else 'airborne',
            'active_faults': self.system_health.active_faults
        })
        self.system_status_pub.publish(system_msg)


def main(args=None):
    """Entry point for thinking_node."""
    rclpy.init(args=args)
    
    node = ThinkingNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
