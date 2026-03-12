"""
Fault Injection Interface

Provides interfaces for injecting faults for testing:
1. ROS2 Topics: /inject_fault, /clear_fault
2. Simple GUI: Tkinter-based button interface

Use for FMEA validation and fault response testing.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from typing import Optional, Dict, Callable
import json

# Import fault types from local module
import sys
import os

# Will be imported when running as ROS2 node
try:
    from ..health.fault_detector import FaultType, FaultState
except ImportError:
    # Define locally for standalone testing
    from enum import Enum, auto
    
    class FaultType(Enum):
        FALL = auto()
        TRIP = auto()
        CONTACT_LOSS = auto()
        BALANCE_LOSS = auto()
        ANOMALOUS_DISCHARGE = auto()
        CHARGE_FAILURE = auto()
        LOW_VOLTAGE = auto()
        IMU_DRIFT = auto()
        LIDAR_OBSTRUCTION = auto()
        SENSOR_TIMEOUT = auto()
        WRONG_MEDICINE = auto()
        WRONG_BED = auto()
        MISSION_TIMEOUT = auto()
        ROS2_DISCONNECT = auto()
        MESSAGE_TIMEOUT = auto()


class FaultInjectionNode(Node):
    """
    ROS2 Node for fault injection.
    
    Topics:
    - /inject_fault (String): JSON with fault_type to inject
    - /clear_fault (String): JSON with fault_type to clear
    - /fault_status (String): Published fault status
    """
    
    def __init__(self, injection_callback: Optional[Callable] = None,
                 clear_callback: Optional[Callable] = None):
        """Initialize fault injection node."""
        super().__init__('fault_injection_node')
        
        # Callbacks to actual fault injection system
        self._inject_callback = injection_callback
        self._clear_callback = clear_callback
        
        # Subscribers
        self.inject_sub = self.create_subscription(
            String, '/inject_fault', self._on_inject, 10
        )
        self.clear_sub = self.create_subscription(
            String, '/clear_fault', self._on_clear, 10
        )
        
        # Publisher for status
        self.status_pub = self.create_publisher(String, '/fault_status', 10)
        
        # Available faults for reference
        self._fault_names = [ft.name for ft in FaultType]
        
        self.get_logger().info('Fault Injection Node initialized')
        self.get_logger().info(f'Available faults: {", ".join(self._fault_names)}')
    
    def _on_inject(self, msg: String):
        """Handle fault injection request."""
        try:
            data = json.loads(msg.data)
            fault_name = data.get('fault_type', '').upper()
            
            if fault_name not in self._fault_names:
                self.get_logger().error(f'Unknown fault type: {fault_name}')
                return
            
            fault_type = FaultType[fault_name]
            
            self.get_logger().warn(f'INJECTING FAULT: {fault_name}')
            
            if self._inject_callback:
                self._inject_callback(fault_type)
            
            # Publish status
            status = String()
            status.data = json.dumps({
                'action': 'injected',
                'fault_type': fault_name,
                'timestamp': self.get_clock().now().nanoseconds / 1e9
            })
            self.status_pub.publish(status)
            
        except json.JSONDecodeError:
            self.get_logger().error(f'Invalid JSON: {msg.data}')
        except Exception as e:
            self.get_logger().error(f'Injection error: {e}')
    
    def _on_clear(self, msg: String):
        """Handle fault clear request."""
        try:
            data = json.loads(msg.data)
            fault_name = data.get('fault_type', '').upper()
            
            if fault_name == 'ALL':
                # Clear all faults
                self.get_logger().info('Clearing ALL faults')
                for ft in FaultType:
                    if self._clear_callback:
                        self._clear_callback(ft)
            elif fault_name in self._fault_names:
                fault_type = FaultType[fault_name]
                self.get_logger().info(f'Clearing fault: {fault_name}')
                
                if self._clear_callback:
                    self._clear_callback(fault_type)
            else:
                self.get_logger().error(f'Unknown fault type: {fault_name}')
                return
            
            # Publish status
            status = String()
            status.data = json.dumps({
                'action': 'cleared',
                'fault_type': fault_name,
                'timestamp': self.get_clock().now().nanoseconds / 1e9
            })
            self.status_pub.publish(status)
            
        except json.JSONDecodeError:
            self.get_logger().error(f'Invalid JSON: {msg.data}')
        except Exception as e:
            self.get_logger().error(f'Clear error: {e}')
    
    def set_callbacks(self, inject: Callable, clear: Callable):
        """Set injection/clear callbacks."""
        self._inject_callback = inject
        self._clear_callback = clear


def create_fault_injection_gui(inject_callback: Callable, clear_callback: Callable):
    """
    Create a simple Tkinter GUI for fault injection.
    
    Args:
        inject_callback: Function to call when injecting fault
        clear_callback: Function to call when clearing fault
        
    Returns:
        Tkinter window (call mainloop() to run)
    """
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        print("Tkinter not available - GUI disabled")
        return None
    
    # Create window
    root = tk.Tk()
    root.title("Fault Injection Panel")
    root.geometry("400x500")
    
    # Title
    title = tk.Label(root, text="FMEA Fault Injection", font=('Arial', 14, 'bold'))
    title.pack(pady=10)
    
    # Categories
    categories = {
        "Locomotion": [
            FaultType.FALL,
            FaultType.TRIP,
            FaultType.CONTACT_LOSS,
            FaultType.BALANCE_LOSS,
        ],
        "Battery": [
            FaultType.ANOMALOUS_DISCHARGE,
            FaultType.CHARGE_FAILURE,
            FaultType.LOW_VOLTAGE,
        ],
        "Sensors": [
            FaultType.IMU_DRIFT,
            FaultType.LIDAR_OBSTRUCTION,
            FaultType.SENSOR_TIMEOUT,
        ],
        "Mission": [
            FaultType.WRONG_MEDICINE,
            FaultType.WRONG_BED,
            FaultType.MISSION_TIMEOUT,
        ],
        "Communication": [
            FaultType.ROS2_DISCONNECT,
            FaultType.MESSAGE_TIMEOUT,
        ],
    }
    
    # Create notebook for categories
    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True, padx=10, pady=10)
    
    # Create tab for each category
    for category, faults in categories.items():
        frame = ttk.Frame(notebook)
        notebook.add(frame, text=category)
        
        for fault in faults:
            fault_frame = ttk.Frame(frame)
            fault_frame.pack(fill='x', padx=5, pady=2)
            
            label = ttk.Label(fault_frame, text=fault.name, width=20)
            label.pack(side='left')
            
            inject_btn = ttk.Button(
                fault_frame, text="Inject",
                command=lambda f=fault: inject_callback(f)
            )
            inject_btn.pack(side='left', padx=2)
            
            clear_btn = ttk.Button(
                fault_frame, text="Clear",
                command=lambda f=fault: clear_callback(f)
            )
            clear_btn.pack(side='left', padx=2)
    
    # Clear all button
    clear_all_btn = ttk.Button(
        root, text="Clear All Faults",
        command=lambda: [clear_callback(ft) for ft in FaultType]
    )
    clear_all_btn.pack(pady=10)
    
    # Status label
    status_var = tk.StringVar(value="Ready")
    status_label = tk.Label(root, textvariable=status_var, fg='gray')
    status_label.pack(pady=5)
    
    return root


def main_gui(args=None):
    """
    Launch fault injection GUI with ROS2 publisher.
    
    This creates a GUI that:
    1. Publishes to /inject_fault for health_node detection
    2. Publishes to /robot_perturbation for REAL physical effects in MuJoCo
    """
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from std_msgs.msg import String
    import json
    import threading
    
    # Initialize ROS2
    rclpy.init(args=args)
    
    # Create QoS profiles
    qos_reliable = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        depth=10
    )
    qos_best_effort = QoSProfile(depth=10)
    
    # Create node for publishing
    node = rclpy.create_node('fault_injection_gui')
    
    # Publishers
    inject_pub = node.create_publisher(String, '/inject_fault', qos_reliable)
    clear_pub = node.create_publisher(String, '/clear_fault', qos_reliable)
    perturbation_pub = node.create_publisher(String, '/robot_perturbation', qos_best_effort)
    
    # Map fault types to physical perturbations
    FAULT_TO_PERTURBATION = {
        FaultType.FALL: {'type': 'fall'},
        FaultType.TRIP: {'type': 'trip'},
        FaultType.BALANCE_LOSS: {'type': 'push', 'force_x': 60.0, 'force_y': 30.0, 'force_z': 0, 'duration': 0.2},
        FaultType.CONTACT_LOSS: {'type': 'push', 'force_x': 0, 'force_y': 0, 'force_z': 100.0, 'duration': 0.3},
        FaultType.IMU_DRIFT: {'type': 'sensor_fault', 'enabled': True},
        FaultType.SENSOR_TIMEOUT: {'type': 'sensor_fault', 'enabled': True},
        FaultType.LOW_VOLTAGE: {'type': 'motor_fault', 'enabled': True},
    }
    
    def inject_fault(fault_type):
        """Publish fault injection + physical perturbation."""
        # 1. Notify health_node
        msg = String()
        msg.data = json.dumps({'fault_type': fault_type.name.lower()})
        inject_pub.publish(msg)
        
        # 2. Apply physical perturbation to robot
        if fault_type in FAULT_TO_PERTURBATION:
            pert_msg = String()
            pert_msg.data = json.dumps(FAULT_TO_PERTURBATION[fault_type])
            perturbation_pub.publish(pert_msg)
            node.get_logger().warn(f'INJECTED: {fault_type.name} + physical perturbation')
        else:
            node.get_logger().info(f'Injected (no physics): {fault_type.name}')
    
    def clear_fault(fault_type):
        """Clear fault and perturbations."""
        # Clear health_node fault
        msg = String()
        msg.data = json.dumps({'fault_type': fault_type.name.lower()})
        clear_pub.publish(msg)
        
        # Clear physical perturbations
        pert_msg = String()
        pert_msg.data = json.dumps({'type': 'clear'})
        perturbation_pub.publish(pert_msg)
        node.get_logger().info(f'Cleared: {fault_type.name}')
    
    # Start ROS2 spinner in background
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    
    # Create and run GUI
    gui = create_fault_injection_gui(inject_fault, clear_fault)
    if gui:
        node.get_logger().info('=' * 50)
        node.get_logger().info('FAULT INJECTION GUI - REAL PHYSICS MODE')
        node.get_logger().info('=' * 50)
        node.get_logger().info('Topics:')
        node.get_logger().info('  /inject_fault -> health_node')
        node.get_logger().info('  /robot_perturbation -> MuJoCo forces')
        node.get_logger().info('')
        node.get_logger().info('Faults with physics: FALL, TRIP, BALANCE_LOSS, MOTOR_FAILURE')
        try:
            gui.mainloop()
        except KeyboardInterrupt:
            pass
    
    # Cleanup
    node.destroy_node()
    rclpy.shutdown()


# Example usage commands for ROS2 topics:
"""
# Inject a fault:
ros2 topic pub /inject_fault std_msgs/msg/String "data: '{\"fault_type\": \"fall\"}'" --once

# Clear a fault:
ros2 topic pub /clear_fault std_msgs/msg/String "data: '{\"fault_type\": \"fall\"}'" --once

# Clear all faults:
ros2 topic pub /clear_fault std_msgs/msg/String "data: '{\"fault_type\": \"all\"}'" --once

# Monitor fault status:
ros2 topic echo /fault_status
"""
