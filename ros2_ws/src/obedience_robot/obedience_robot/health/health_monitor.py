"""
Health Monitor - Central Health Aggregation

Aggregates health information from all subsystems and provides
unified health status for decision making.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Any, TYPE_CHECKING
from .fault_detector import FaultDetector, FaultType, FaultState, Fault

# Conditional import to avoid circular dependencies
if TYPE_CHECKING:
    from ..thinking.sensory import (
        IMUProcessor, JointStateProcessor, ContactProcessor,
        LiDARProcessor, BatteryMonitor
    )


@dataclass
class HealthStatus:
    """Overall system health status."""
    overall: FaultState = FaultState.NONE
    locomotion: FaultState = FaultState.NONE
    battery: FaultState = FaultState.NONE
    sensors: FaultState = FaultState.NONE
    mission: FaultState = FaultState.NONE
    communication: FaultState = FaultState.NONE
    
    active_faults: List[str] = field(default_factory=list)
    timestamp: float = 0.0
    
    def is_safe_to_operate(self) -> bool:
        """Check if system is safe for operation."""
        return self.overall != FaultState.TRUE
    
    def needs_attention(self) -> bool:
        """Check if any subsystem needs attention."""
        return self.overall in [FaultState.TRUE, FaultState.SUSPECT]


class HealthMonitor:
    """
    Central health monitoring system.
    
    Aggregates data from all sensors and the fault detector
    to provide unified health status.
    """
    
    def __init__(self):
        """Initialize health monitor."""
        # Fault detector
        self.fault_detector = FaultDetector()
        
        # Sensor processors (references, initialized externally)
        # Using Any to avoid import issues
        self.imu: Optional[Any] = None
        self.joints: Optional[Any] = None
        self.contacts: Optional[Any] = None
        self.lidar: Optional[Any] = None
        self.battery: Optional[Any] = None
        
        # Current status
        self.status = HealthStatus()
        
        # Communication monitoring
        self._last_ros_message_time: float = 0.0
        self._ros_timeout: float = 1.0  # seconds
        
        # Setup fault detector callbacks
        self.fault_detector.set_callbacks(
            on_detected=self._on_fault_detected,
            on_cleared=self._on_fault_cleared
        )
    
    def set_sensors(self, imu: Any, joints: Any, contacts: Any,
                    lidar: Any, battery: Any):
        """Set sensor processor references."""
        self.imu = imu
        self.joints = joints
        self.contacts = contacts
        self.lidar = lidar
        self.battery = battery
    
    def update(self, timestamp: float) -> HealthStatus:
        """
        Update health status from all sources.
        
        Args:
            timestamp: Current simulation time
            
        Returns:
            Updated health status
        """
        # Check sensor faults
        self._check_sensor_faults(timestamp)
        
        # Check communication
        self._check_communication(timestamp)
        
        # Update status from fault detector
        summary = self.fault_detector.get_status_summary()
        
        self.status.locomotion = FaultState(summary["locomotion"])
        self.status.battery = FaultState(summary["battery"])
        self.status.sensors = FaultState(summary["sensors"])
        self.status.mission = FaultState(summary["mission"])
        self.status.communication = FaultState(summary["communication"])
        
        # Calculate overall status
        states = [
            self.status.locomotion,
            self.status.battery,
            self.status.sensors,
            self.status.mission,
            self.status.communication
        ]
        
        if FaultState.TRUE in states:
            self.status.overall = FaultState.TRUE
        elif FaultState.SUSPECT in states:
            self.status.overall = FaultState.SUSPECT
        elif all(s == FaultState.FALSE for s in states):
            self.status.overall = FaultState.FALSE
        else:
            self.status.overall = FaultState.NONE
        
        # Update active faults list
        self.status.active_faults = [
            str(f) for f in self.fault_detector.get_active_faults()
        ]
        self.status.timestamp = timestamp
        
        return self.status
    
    def _check_sensor_faults(self, timestamp: float):
        """Check sensors for faults and report to detector."""
        # IMU faults
        if self.imu and self.imu.status.fault_state == FaultState.TRUE:
            self.fault_detector.detect_fault(
                FaultType.IMU_DRIFT,
                FaultState.TRUE,
                "imu_processor",
                "IMU anomaly detected",
                timestamp
            )
        elif self.imu and self.imu.status.fault_state == FaultState.SUSPECT:
            self.fault_detector.detect_fault(
                FaultType.IMU_DRIFT,
                FaultState.SUSPECT,
                "imu_processor",
                "IMU possible drift",
                timestamp
            )
        
        # Contact faults (fall detection)
        if self.contacts and self.contacts.status.fault_state == FaultState.TRUE:
            self.fault_detector.detect_fault(
                FaultType.CONTACT_LOSS,
                FaultState.TRUE,
                "contact_processor",
                "Extended contact loss - possible fall",
                timestamp
            )
        
        # LiDAR faults
        if self.lidar and self.lidar.status.fault_state == FaultState.TRUE:
            self.fault_detector.detect_fault(
                FaultType.LIDAR_OBSTRUCTION,
                FaultState.TRUE,
                "lidar_processor",
                "LiDAR sensor blocked",
                timestamp
            )
        
        # Battery faults
        if self.battery:
            if self.battery.status.fault_state == FaultState.TRUE:
                self.fault_detector.detect_fault(
                    FaultType.LOW_VOLTAGE,
                    FaultState.TRUE,
                    "battery_monitor",
                    f"Critical battery: {self.battery.get_soc_percent():.1f}%",
                    timestamp
                )
            elif self.battery.status.fault_state == FaultState.SUSPECT:
                # Check if it's anomalous discharge
                if self.battery._discharge_rate > self.battery.config.anomaly_threshold:
                    self.fault_detector.detect_fault(
                        FaultType.ANOMALOUS_DISCHARGE,
                        FaultState.SUSPECT,
                        "battery_monitor",
                        f"High discharge rate: {self.battery._discharge_rate*100:.2f}%/s",
                        timestamp
                    )
        
        # Joint faults
        if self.joints:
            faulty = self.joints.get_faulty_joints()
            if faulty:
                self.fault_detector.detect_fault(
                    FaultType.BALANCE_LOSS,
                    FaultState.SUSPECT,
                    "joint_processor",
                    f"Joint issues: {', '.join(faulty)}",
                    timestamp
                )
    
    def _check_communication(self, timestamp: float):
        """Check ROS2 communication health."""
        if timestamp - self._last_ros_message_time > self._ros_timeout:
            self.fault_detector.detect_fault(
                FaultType.ROS2_DISCONNECT,
                FaultState.SUSPECT,
                "health_monitor",
                "No ROS2 messages received",
                timestamp
            )
    
    def ros_message_received(self, timestamp: float):
        """Record ROS2 message reception."""
        self._last_ros_message_time = timestamp
        # Clear communication fault if active
        if self.fault_detector.is_fault_active(FaultType.ROS2_DISCONNECT):
            self.fault_detector.clear_fault(FaultType.ROS2_DISCONNECT)
    
    def _on_fault_detected(self, fault: Fault):
        """Callback when fault is detected."""
        print(f"[HEALTH] FAULT DETECTED: {fault}")
    
    def _on_fault_cleared(self, fault_type: FaultType):
        """Callback when fault is cleared."""
        print(f"[HEALTH] Fault cleared: {fault_type.name}")
    
    def inject_fault(self, fault_type: FaultType, timestamp: float):
        """Inject a fault for testing."""
        self.fault_detector.inject_fault(fault_type, timestamp)
    
    def clear_injected_fault(self, fault_type: FaultType):
        """Clear an injected fault."""
        self.fault_detector.clear_injected_fault(fault_type)
    
    def get_health_report(self) -> Dict:
        """Get comprehensive health report."""
        return {
            "overall": self.status.overall.value,
            "subsystems": {
                "locomotion": self.status.locomotion.value,
                "battery": self.status.battery.value,
                "sensors": self.status.sensors.value,
                "mission": self.status.mission.value,
                "communication": self.status.communication.value,
            },
            "active_faults": self.status.active_faults,
            "safe_to_operate": self.status.is_safe_to_operate(),
            "needs_attention": self.status.needs_attention(),
            "timestamp": self.status.timestamp,
        }
