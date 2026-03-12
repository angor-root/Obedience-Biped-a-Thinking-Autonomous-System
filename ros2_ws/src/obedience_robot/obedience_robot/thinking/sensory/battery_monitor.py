"""
Battery Monitor - Energy State Monitoring

Monitors battery for:
- Anomalous discharge rate
- Low voltage warning
- Charge failure

Fault Modes:
- Anomalous discharge: Battery draining faster than expected
- Low voltage: Below safe operating threshold
- Charge failure: Not charging when expected
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable
from collections import deque
from ..common import FaultState, SensorStatus, SensorType


@dataclass
class BatteryConfig:
    """Battery monitor configuration."""
    # Capacity
    nominal_voltage: float = 24.0  # V
    capacity_ah: float = 5.0  # Amp-hours
    
    # Thresholds
    low_voltage: float = 20.0  # V - warning
    critical_voltage: float = 18.0  # V - critical
    
    # Discharge monitoring
    max_discharge_rate: float = 0.02  # %/s - normal walking
    anomaly_threshold: float = 0.05  # %/s - too fast
    
    # Charge monitoring
    expected_charge_rate: float = 0.01  # %/s
    charge_timeout: float = 10.0  # seconds - max time without charge progress


class BatteryMonitor:
    """
    Battery monitor with fault detection.
    
    Monitors state of charge and detects anomalies.
    """
    
    def __init__(self, config: Optional[BatteryConfig] = None):
        """Initialize battery monitor."""
        self.config = config or BatteryConfig()
        
        # Status
        self.status = SensorStatus(
            sensor_type=SensorType.BATTERY,
            name="battery",
            fault_state=FaultState.NONE
        )
        
        # Current state
        self.soc: float = 1.0  # State of charge (0.0 - 1.0)
        self.voltage: float = self.config.nominal_voltage
        self.current: float = 0.0  # A (positive = discharging)
        self.is_charging: bool = False
        
        # History for discharge rate
        self._soc_history: deque = deque(maxlen=100)
        self._discharge_rate: float = 0.0
        
        # Charge monitoring
        self._charge_start_time: Optional[float] = None
        self._charge_start_soc: float = 0.0
        
        # Injected fault
        self._injected_fault: Optional[str] = None
        
        # Callbacks
        self._on_low: Optional[Callable] = None
        self._on_critical: Optional[Callable] = None
        self._on_anomaly: Optional[Callable] = None
    
    def update(self, soc: float, voltage: float, current: float,
               is_charging: bool, timestamp: float) -> FaultState:
        """
        Update battery state and check for faults.
        
        Args:
            soc: State of charge (0.0 - 1.0)
            voltage: Measured voltage
            current: Measured current (positive = discharge)
            is_charging: True if on charger
            timestamp: Current simulation time
            
        Returns:
            Current fault state
        """
        # Apply injected fault
        if self._injected_fault == "anomalous_discharge":
            soc = max(0, self.soc - 0.1)  # Fast discharge
        elif self._injected_fault == "low_voltage":
            voltage = self.config.low_voltage - 1
        elif self._injected_fault == "charge_fail":
            is_charging = False  # Can't charge
        
        # Calculate discharge rate
        self._soc_history.append((timestamp, soc))
        if len(self._soc_history) >= 2:
            t0, s0 = self._soc_history[0]
            t1, s1 = self._soc_history[-1]
            dt = t1 - t0
            if dt > 0:
                self._discharge_rate = (s0 - s1) / dt  # positive = discharging
        
        # Update state
        prev_soc = self.soc
        self.soc = soc
        self.voltage = voltage
        self.current = current
        self.is_charging = is_charging
        
        self.status.last_update = timestamp
        self.status.last_value = soc
        
        # Check faults
        fault_state = self._check_faults(timestamp)
        self.status.fault_state = fault_state
        
        # Trigger callbacks
        if soc < 0.20 and prev_soc >= 0.20 and self._on_low:
            self._on_low(soc)
        if soc < 0.10 and prev_soc >= 0.10 and self._on_critical:
            self._on_critical(soc)
        
        return fault_state
    
    def _check_faults(self, timestamp: float) -> FaultState:
        """Check for battery faults."""
        # Critical voltage
        if self.voltage < self.config.critical_voltage:
            return FaultState.TRUE
        
        # Low voltage warning
        if self.voltage < self.config.low_voltage:
            return FaultState.SUSPECT
        
        # Anomalous discharge
        if self._discharge_rate > self.config.anomaly_threshold:
            if self._on_anomaly:
                self._on_anomaly(self._discharge_rate, self.soc)
            return FaultState.SUSPECT
        
        # Check charge progress
        if self.is_charging:
            if self._charge_start_time is None:
                self._charge_start_time = timestamp
                self._charge_start_soc = self.soc
            else:
                elapsed = timestamp - self._charge_start_time
                soc_gained = self.soc - self._charge_start_soc
                
                if elapsed > self.config.charge_timeout:
                    expected_gain = self.config.expected_charge_rate * elapsed
                    if soc_gained < expected_gain * 0.5:
                        return FaultState.SUSPECT  # Not charging properly
        else:
            self._charge_start_time = None
        
        return FaultState.FALSE
    
    def set_callbacks(self, on_low: Callable, on_critical: Callable,
                      on_anomaly: Optional[Callable] = None):
        """Set event callbacks."""
        self._on_low = on_low
        self._on_critical = on_critical
        self._on_anomaly = on_anomaly
    
    def get_soc_percent(self) -> float:
        """Get state of charge as percentage."""
        return self.soc * 100.0
    
    def get_estimated_time_remaining(self) -> Optional[float]:
        """Estimate time remaining at current discharge rate."""
        if self._discharge_rate <= 0:
            return None  # Not discharging
        
        return self.soc / self._discharge_rate
    
    def is_low(self) -> bool:
        """Check if battery is low."""
        return self.soc < 0.20
    
    def is_critical(self) -> bool:
        """Check if battery is critical."""
        return self.soc < 0.10
    
    def inject_fault(self, fault_type: str):
        """Inject fault (anomalous_discharge, low_voltage, charge_fail)."""
        self._injected_fault = fault_type
    
    def clear_fault(self):
        """Clear injected fault."""
        self._injected_fault = None
    
    def get_status_dict(self) -> dict:
        """Get battery status as dictionary."""
        return {
            "soc_percent": self.get_soc_percent(),
            "voltage": self.voltage,
            "current": self.current,
            "is_charging": self.is_charging,
            "discharge_rate": self._discharge_rate * 100,  # %/s
            "fault_state": self.status.fault_state.value,
            "time_remaining": self.get_estimated_time_remaining(),
        }
