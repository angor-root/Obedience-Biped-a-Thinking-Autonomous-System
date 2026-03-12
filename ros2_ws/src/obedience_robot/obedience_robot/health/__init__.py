"""
Health Module - Integrated System Health Management (ISHM)

Monitors overall system health and manages fault detection/recovery.

Components:
- HealthMonitor: Central health aggregation
- FaultDetector: FMEA-based fault detection and propagation
- RecoveryManager: Fault recovery actions
- FaultInjection: Testing interface for fault injection
"""

from .health_monitor import HealthMonitor
from .fault_detector import FaultDetector, FaultType
from .recovery_manager import RecoveryManager, RecoveryAction
from .fault_injection import FaultInjectionNode, create_fault_injection_gui

__all__ = [
    'HealthMonitor',
    'FaultDetector',
    'FaultType',
    'RecoveryManager',
    'RecoveryAction',
    'FaultInjectionNode',
    'create_fault_injection_gui',
]
