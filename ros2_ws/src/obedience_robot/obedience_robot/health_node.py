#!/usr/bin/env python3
"""
Health Node - Integrated System Health Management (ISHM)

This node monitors system health using FMEA-based fault detection:
- Monitors all sensor data for anomalies
- Detects and propagates faults through FMEA chains  
- Supports fault injection for testing
- Coordinates recovery actions

Communication:
- Subscribes to sensor topics for monitoring
- Subscribes to /inject_fault and /clear_fault for testing
- Publishes /health_status for thinking_node
- Publishes /active_faults for logging
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
import numpy as np
from typing import Dict, List, Optional
import json
import time

# ROS2 message types
from std_msgs.msg import String, Float32, Bool
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Point

# Health subsystems
from .health.fault_detector import FaultDetector, FaultType, FaultState, Fault
from .health.health_monitor import HealthMonitor, HealthStatus
from .health.recovery_manager import RecoveryManager, RecoveryType


class HealthNode(Node):
    """
    System Health Monitoring ROS2 Node.
    
    Implements ISHM (Integrated System Health Management):
    1. Monitors sensor data for anomalies
    2. Detects faults using FMEA rules
    3. Propagates faults through dependency chains
    4. Suggests recovery actions
    5. Supports fault injection for testing
    """
    
    def __init__(self):
        super().__init__('health_node')
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("  HEALTH NODE - Integrated System Health Management")
        self.get_logger().info("=" * 60)
        
        # Declare parameters
        self.declare_parameter('update_rate', 20.0)  # Hz - faster than thinking
        self.declare_parameter('enable_fault_injection', True)
        
        update_rate = self.get_parameter('update_rate').value
        
        # Initialize health subsystems
        self._init_health_subsystems()
        
        # QoS profiles
        qos_reliable = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=10
        )
        
        qos_sensor = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=1
        )
        
        # === SUBSCRIBERS (Sensors) ===
        self.imu_sub = self.create_subscription(
            Imu, '/imu', self._imu_callback, qos_sensor)
        
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_state_callback, qos_sensor)
        
        self.contact_sub = self.create_subscription(
            String, '/contacts', self._contact_callback, qos_sensor)
        
        self.battery_sub = self.create_subscription(
            Float32, '/battery_level', self._battery_callback, qos_sensor)
        
        self.position_sub = self.create_subscription(
            Point, '/robot_position', self._position_callback, qos_sensor)
        
        # === SUBSCRIBERS (Fault Injection) ===
        if self.get_parameter('enable_fault_injection').value:
            self.inject_fault_sub = self.create_subscription(
                String, '/inject_fault', self._inject_fault_callback, qos_reliable)
            
            self.clear_fault_sub = self.create_subscription(
                String, '/clear_fault', self._clear_fault_callback, qos_reliable)
            
            self.get_logger().info("  ⚠ Fault injection ENABLED")
        
        # === PUBLISHERS ===
        # Health status (for thinking_node)
        self.health_status_pub = self.create_publisher(
            String, '/health_status', qos_reliable)
        
        # Active faults (detailed)
        self.active_faults_pub = self.create_publisher(
            String, '/active_faults', qos_reliable)
        
        # Recovery suggestions
        self.recovery_pub = self.create_publisher(
            String, '/recovery_action', qos_reliable)
        
        # Alerts (critical events)
        self.alert_pub = self.create_publisher(
            String, '/health_alert', qos_reliable)
        
        # === TIMERS ===
        self.monitor_timer = self.create_timer(
            1.0 / update_rate, self._monitor_callback)
        
        self.status_timer = self.create_timer(
            0.5, self._publish_health_status)  # 2 Hz status
        
        # State tracking
        self.last_imu_time = time.time()
        self.last_joint_time = time.time()
        self.last_contact_time = time.time()
        self.robot_position = np.array([0.0, 0.0])
        self.robot_height = 0.68  # Expected standing height
        
        # Sensor data buffers
        self.imu_data = {'orientation': None, 'angular_vel': None, 'linear_acc': None}
        self.joint_data = {'positions': {}, 'velocities': {}, 'efforts': {}}
        self.contact_data = {'left': False, 'right': False}
        self.battery_level = 100.0
        
        # === FMEA Intelligent Detection State ===
        # Contact loss timing (normal walking has brief air phases ~0.1s)
        self._no_contact_start = None
        self._contact_loss_threshold = 0.35  # seconds without contact = anomaly
        
        # Fall detection state
        self._fall_detected = False
        self._fall_time = None
        
        # IMU anomaly buffers for correlation
        self._imu_anomaly_count = 0
        self._imu_anomaly_threshold = 3  # Need N consecutive anomalies
        
        # Warm-up period - don't detect faults until sensors are streaming
        self._startup_time = time.time()
        self._warmup_duration = 3.0  # seconds to wait before fault detection
        self._sensors_ready = False
        
        # Injected faults (for testing)
        self.injected_faults: Dict[FaultType, bool] = {}
        
        self.get_logger().info(f"  Monitor rate: {update_rate} Hz")
        self.get_logger().info(f"  Warm-up period: {self._warmup_duration}s")
        self.get_logger().info("  FMEA-based fault detection enabled")
        self.get_logger().info("Health Node initialized")
    
    def _init_health_subsystems(self):
        """Initialize health monitoring subsystems."""
        self.get_logger().info("Initializing Health Subsystems...")
        
        # Fault detector with FMEA chains
        self.fault_detector = FaultDetector()
        self.fault_detector.set_callbacks(
            on_detected=self._on_fault_detected,
            on_cleared=self._on_fault_cleared
        )
        self.get_logger().info("  ✓ Fault Detector (FMEA)")
        
        # Health monitor (aggregates status)
        self.health_monitor = HealthMonitor()
        self.get_logger().info("  ✓ Health Monitor")
        
        # Recovery manager
        self.recovery_manager = RecoveryManager()
        self.get_logger().info("  ✓ Recovery Manager")
    
    # === SENSOR CALLBACKS ===
    
    def _imu_callback(self, msg: Imu):
        """Process IMU data for anomaly detection."""
        self.last_imu_time = time.time()
        
        self.imu_data['orientation'] = np.array([
            msg.orientation.x, msg.orientation.y,
            msg.orientation.z, msg.orientation.w
        ])
        self.imu_data['angular_vel'] = np.array([
            msg.angular_velocity.x, msg.angular_velocity.y,
            msg.angular_velocity.z
        ])
        self.imu_data['linear_acc'] = np.array([
            msg.linear_acceleration.x, msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])
    
    def _joint_state_callback(self, msg: JointState):
        """Process joint states."""
        self.last_joint_time = time.time()
        
        self.joint_data['positions'] = dict(zip(msg.name, msg.position))
        if msg.velocity:
            self.joint_data['velocities'] = dict(zip(msg.name, msg.velocity))
        if msg.effort:
            self.joint_data['efforts'] = dict(zip(msg.name, msg.effort))
    
    def _contact_callback(self, msg: String):
        """Process contact sensor data."""
        self.last_contact_time = time.time()
        
        try:
            contacts = json.loads(msg.data)
            self.contact_data['left'] = contacts.get('left_foot', False)
            self.contact_data['right'] = contacts.get('right_foot', False)
        except json.JSONDecodeError:
            pass
    
    def _battery_callback(self, msg: Float32):
        """Process battery level."""
        self.battery_level = msg.data
    
    def _position_callback(self, msg: Point):
        """Update robot position and height."""
        self.robot_position = np.array([msg.x, msg.y])
        self.robot_height = msg.z
    
    # === FAULT INJECTION CALLBACKS ===
    
    def _inject_fault_callback(self, msg: String):
        """
        Handle fault injection request.
        
        JSON format:
        {
            "fault_type": "fall|trip|contact_loss|imu_drift|...",
            "severity": 0.0-1.0 (optional),
            "duration": seconds (optional, 0=permanent)
        }
        """
        try:
            data = json.loads(msg.data)
            fault_type_str = data.get('fault_type', '')
            
            # Map string to FaultType enum
            fault_type = None
            for ft in FaultType:
                if ft.name.lower() == fault_type_str.lower():
                    fault_type = ft
                    break
            
            if fault_type is None:
                self.get_logger().warn(f"Unknown fault type: {fault_type_str}")
                return
            
            severity = data.get('severity', 1.0)
            duration = data.get('duration', 0)
            
            self.get_logger().warn(f"⚠ INJECTING FAULT: {fault_type.name} (severity={severity})")
            
            self.injected_faults[fault_type] = True
            
            # Create the fault
            self.fault_detector.inject_fault(fault_type, time.time())
            
            # Publish alert
            alert_msg = String()
            alert_msg.data = json.dumps({
                'type': 'FAULT_INJECTED',
                'fault': fault_type.name,
                'severity': severity,
                'timestamp': time.time()
            })
            self.alert_pub.publish(alert_msg)
            
            # Schedule auto-clear if duration specified
            if duration > 0:
                def make_auto_clear(ft):
                    timer_holder = [None]
                    def callback():
                        self._auto_clear_fault(ft)
                        if timer_holder[0]:
                            timer_holder[0].cancel()
                    timer_holder[0] = self.create_timer(duration, callback)
                make_auto_clear(fault_type)
                
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Invalid inject_fault JSON: {e}")
    
    def _clear_fault_callback(self, msg: String):
        """
        Handle fault clearing request.
        
        JSON format:
        {"fault_type": "fall|all"}
        """
        try:
            data = json.loads(msg.data)
            fault_type_str = data.get('fault_type', 'all')
            
            if fault_type_str.lower() == 'all':
                self.get_logger().info("Clearing ALL injected faults")
                self.injected_faults.clear()
                # Clear all active faults
                for fault in list(self.fault_detector.get_active_faults()):
                    self.fault_detector.clear_fault(fault.fault_type)
            else:
                for ft in FaultType:
                    if ft.name.lower() == fault_type_str.lower():
                        self.get_logger().info(f"Clearing fault: {ft.name}")
                        self.injected_faults.pop(ft, None)
                        self.fault_detector.clear_fault(ft)
                        break
                        
        except json.JSONDecodeError as e:
            self.get_logger().error(f"Invalid clear_fault JSON: {e}")
    
    def _auto_clear_fault(self, fault_type: FaultType):
        """Auto-clear a temporary injected fault."""
        self.get_logger().info(f"Auto-clearing fault: {fault_type.name}")
        self.injected_faults.pop(fault_type, None)
        self.fault_detector.clear_fault(fault_type)
    
    # === FAULT DETECTION CALLBACKS ===
    
    def _on_fault_detected(self, fault: Fault):
        """Called when fault detector identifies a fault."""
        self.get_logger().error(f"🚨 FAULT DETECTED: {fault.fault_type.name}")
        self.get_logger().error(f"   State: {fault.state.name}")
        self.get_logger().error(f"   Source: {fault.source}")
        
        # Publish alert
        alert_msg = String()
        alert_msg.data = json.dumps({
            'type': 'FAULT_DETECTED',
            'fault': fault.fault_type.name,
            'state': fault.state.name,
            'source': fault.source,
            'timestamp': fault.timestamp
        })
        self.alert_pub.publish(alert_msg)
        
        # Get recovery suggestion
        recovery = self.recovery_manager.get_recovery_plan(fault.fault_type)
        if recovery:
            recovery_msg = String()
            recovery_msg.data = json.dumps({
                'fault': fault.fault_type.name,
                'actions': [a.name for a in recovery.actions],
                'description': recovery.description,
                'priority': recovery.priority
            })
            self.recovery_pub.publish(recovery_msg)
    
    def _on_fault_cleared(self, fault_type: FaultType):
        """Called when a fault is cleared."""
        self.get_logger().info(f"✓ Fault cleared: {fault_type.name}")
        
        alert_msg = String()
        alert_msg.data = json.dumps({
            'type': 'FAULT_CLEARED',
            'fault': fault_type.name,
            'timestamp': time.time()
        })
        self.alert_pub.publish(alert_msg)
    
    # === MAIN MONITORING LOOP ===
    
    def _monitor_callback(self):
        """Main health monitoring loop."""
        current_time = time.time()
        
        # Warm-up period - skip fault detection until sensors are ready
        if not self._sensors_ready:
            elapsed = current_time - self._startup_time
            if elapsed < self._warmup_duration:
                return  # Still warming up
            
            # Check if we have valid sensor data
            if self.imu_data['linear_acc'] is not None:
                self._sensors_ready = True
                self.get_logger().info("Sensors ready - fault detection enabled")
                # Clear any initial faults
                for fault in list(self.fault_detector.get_active_faults()):
                    self.fault_detector.clear_fault(fault.fault_type)
            else:
                return  # Still waiting for sensors
        
        # 1. Check sensor timeouts (communication faults)
        self._check_sensor_timeouts(current_time)
        
        # 2. Check for naturally detected faults (not injected)
        if FaultType.FALL not in self.injected_faults:
            self._check_fall_condition()
        
        if FaultType.IMU_DRIFT not in self.injected_faults:
            self._check_imu_drift()
        
        if FaultType.CONTACT_LOSS not in self.injected_faults:
            self._check_contact_loss()
        
        if FaultType.ANOMALOUS_DISCHARGE not in self.injected_faults:
            self._check_battery_anomaly()
        
        # 3. Fault propagation happens automatically in detect_fault()
        
        # 4. Update health monitor - it will query the fault_detector internally
        # Note: health_monitor.update() only takes timestamp
    
    def _check_sensor_timeouts(self, current_time: float):
        """Check if sensors stopped reporting (communication fault)."""
        timeout = 1.0  # 1 second timeout
        
        if current_time - self.last_imu_time > timeout:
            self.fault_detector.detect_fault(
                fault_type=FaultType.ROS2_DISCONNECT, 
                state=FaultState.SUSPECT,
                source="health_node",
                description="IMU timeout",
                timestamp=current_time
            )
        
        if current_time - self.last_joint_time > timeout:
            self.fault_detector.detect_fault(
                fault_type=FaultType.ROS2_DISCONNECT,
                state=FaultState.SUSPECT, 
                source="health_node",
                description="Joint state timeout",
                timestamp=current_time
            )
    
    def _check_fall_condition(self):
        """
        FMEA-based fall detection using multiple sensor correlation.
        
        Fall is confirmed when:
        1. Robot height < 0.35m (TRUE fall), OR
        2. IMU shows high tilt + no contact for extended time (SUSPECT)
        """
        current_time = time.time()
        
        # Primary indicator: Robot height
        if self.robot_height < 0.35:
            if not self._fall_detected:
                self._fall_detected = True
                self._fall_time = current_time
                self.fault_detector.detect_fault(
                    fault_type=FaultType.FALL,
                    state=FaultState.TRUE,
                    source="health_node/FMEA",
                    description=f"CONFIRMED FALL: height={self.robot_height:.2f}m",
                    timestamp=current_time
                )
            return
        
        # Check IMU for high tilt/acceleration (secondary indicator)
        if self.imu_data['linear_acc'] is not None:
            acc = self.imu_data['linear_acc']
            vertical_acc = acc[2]
            lateral_acc = np.sqrt(acc[0]**2 + acc[1]**2)
            
            # High lateral acceleration + abnormal vertical = falling
            is_imu_anomaly = (lateral_acc > 8.0 or abs(vertical_acc - 9.81) > 6.0)
            
            if is_imu_anomaly:
                self._imu_anomaly_count += 1
            else:
                self._imu_anomaly_count = max(0, self._imu_anomaly_count - 1)
            
            # Multiple consecutive IMU anomalies = SUSPECT fall
            if self._imu_anomaly_count >= self._imu_anomaly_threshold:
                self.fault_detector.detect_fault(
                    fault_type=FaultType.BALANCE_LOSS,
                    state=FaultState.SUSPECT,
                    source="health_node/FMEA",
                    description=f"IMU anomaly: lat_acc={lateral_acc:.1f}, vert_acc={vertical_acc:.1f}",
                    timestamp=current_time
                )
        
        # Clear fall state if recovered
        if self._fall_detected and self.robot_height > 0.5:
            self._fall_detected = False
            self.fault_detector.clear_fault(FaultType.FALL)
            self.get_logger().info("Robot recovered from fall")
    
    def _check_imu_drift(self):
        """Detect IMU sensor drift."""
        if self.imu_data['orientation'] is not None:
            # Check for unrealistic orientation values
            orientation = self.imu_data['orientation']
            norm = np.linalg.norm(orientation)
            
            if abs(norm - 1.0) > 0.1:  # Quaternion should have unit norm
                self.fault_detector.detect_fault(
                    fault_type=FaultType.IMU_DRIFT,
                    state=FaultState.SUSPECT,
                    source="health_node",
                    description=f"Quaternion norm={norm:.3f}",
                    timestamp=time.time()
                )
    
    def _check_contact_loss(self):
        """
        FMEA-based contact loss detection.
        
        Normal walking has brief air phases (~0.1s during step transitions).
        Only report anomaly when:
        1. Both feet have no contact for > threshold (0.35s)
        2. Correlated with IMU anomaly (high acceleration)
        
        This eliminates false positives during normal walking.
        """
        current_time = time.time()
        both_feet_off = not self.contact_data['left'] and not self.contact_data['right']
        
        if both_feet_off:
            # Start timing if not already
            if self._no_contact_start is None:
                self._no_contact_start = current_time
            else:
                # Check duration
                no_contact_duration = current_time - self._no_contact_start
                
                if no_contact_duration > self._contact_loss_threshold:
                    # Correlate with IMU for confirmation
                    imu_confirms = False
                    if self.imu_data['linear_acc'] is not None:
                        acc = self.imu_data['linear_acc']
                        # During fall/jump, vertical acc deviates from gravity
                        imu_confirms = abs(acc[2] - 9.81) > 3.0
                    
                    if imu_confirms:
                        self.fault_detector.detect_fault(
                            fault_type=FaultType.CONTACT_LOSS,
                            state=FaultState.TRUE,
                            source="health_node/FMEA",
                            description=f"Airborne {no_contact_duration:.2f}s + IMU anomaly",
                            timestamp=current_time
                        )
                    elif no_contact_duration > 0.5:
                        # Extended air time without IMU anomaly = still suspicious
                        self.fault_detector.detect_fault(
                            fault_type=FaultType.CONTACT_LOSS,
                            state=FaultState.SUSPECT,
                            source="health_node/FMEA",
                            description=f"Extended airborne: {no_contact_duration:.2f}s",
                            timestamp=current_time
                        )
        else:
            # Contact restored - reset timer and clear fault
            if self._no_contact_start is not None:
                self._no_contact_start = None
                # Only clear if we had actually reported a fault
                self.fault_detector.clear_fault(FaultType.CONTACT_LOSS)
    
    def _check_battery_anomaly(self):
        """Detect battery issues."""
        # Critical low battery
        if self.battery_level < 10.0:
            self.fault_detector.detect_fault(
                fault_type=FaultType.ANOMALOUS_DISCHARGE,
                state=FaultState.TRUE,
                source="health_node",
                description=f"Critical battery: {self.battery_level:.1f}%",
                timestamp=time.time()
            )
        elif self.battery_level < 20.0:
            self.fault_detector.detect_fault(
                fault_type=FaultType.ANOMALOUS_DISCHARGE,
                state=FaultState.SUSPECT,
                source="health_node",
                description=f"Low battery: {self.battery_level:.1f}%",
                timestamp=time.time()
            )
    
    def _publish_health_status(self):
        """Publish aggregated health status."""
        status = self.health_monitor.status
        
        # During warmup, publish healthy status (no faults yet)
        if not self._sensors_ready:
            health_msg = String()
            health_msg.data = json.dumps({
                'overall': 'FALSE',
                'locomotion': 'FALSE',
                'battery': 'FALSE',
                'sensors': 'FALSE',
                'communication': 'FALSE',
                'active_faults': [],
                'warmup': True,
                'timestamp': time.time()
            })
            self.health_status_pub.publish(health_msg)
            return
        
        active_faults = self.fault_detector.get_active_faults()
        
        # Determine overall health
        overall = FaultState.FALSE  # Healthy
        
        if any(f.state == FaultState.TRUE for f in active_faults):
            overall = FaultState.TRUE
        elif any(f.state == FaultState.SUSPECT for f in active_faults):
            overall = FaultState.SUSPECT
        
        # Health status message
        health_msg = String()
        health_msg.data = json.dumps({
            'overall': overall.name,
            'locomotion': self._get_subsystem_health(
                [FaultType.FALL, FaultType.TRIP, FaultType.CONTACT_LOSS]).name,
            'battery': self._get_subsystem_health(
                [FaultType.ANOMALOUS_DISCHARGE, FaultType.CHARGE_FAILURE]).name,
            'sensors': self._get_subsystem_health(
                [FaultType.IMU_DRIFT, FaultType.LIDAR_OBSTRUCTION]).name,
            'communication': self._get_subsystem_health(
                [FaultType.ROS2_DISCONNECT]).name,
            'active_faults': [f.fault_type.name for f in active_faults],
            'timestamp': time.time()
        })
        self.health_status_pub.publish(health_msg)
        
        # Detailed faults message
        faults_msg = String()
        faults_msg.data = json.dumps([{
            'type': f.fault_type.name,
            'state': f.state.name,
            'source': f.source,
            'description': f.description,
            'timestamp': f.timestamp
        } for f in active_faults])
        self.active_faults_pub.publish(faults_msg)
    
    def _get_subsystem_health(self, fault_types: List[FaultType]) -> FaultState:
        """Get health state for a subsystem based on related fault types."""
        active_faults = self.fault_detector.get_active_faults()
        
        for fault in active_faults:
            if fault.fault_type in fault_types:
                if fault.state == FaultState.TRUE:
                    return FaultState.TRUE
                elif fault.state == FaultState.SUSPECT:
                    return FaultState.SUSPECT
        
        return FaultState.FALSE


def main(args=None):
    """Entry point for health_node."""
    rclpy.init(args=args)
    
    node = HealthNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
