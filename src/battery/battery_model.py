"""
Non-linear Battery Model for Bipedal Robot.

This module implements a realistic battery discharge model based on
lithium-ion battery characteristics. The model accounts for:
- State of Charge (SoC) dependent discharge rate
- Activity-based power consumption (walking vs idle)
- Non-linear discharge curve (faster at extremes)

Mathematical Model:
    dSoC/dt = -P(t) * η(SoC) / E_nom
    
    where:
    - P(t) = power consumption at time t
    - η(SoC) = non-linear efficiency factor
    - E_nom = nominal energy capacity

    η(SoC) = 1 + k1*(1-SoC)² + k2*SoC²
    
    This creates a U-shaped efficiency curve where discharge is
    slightly faster at very high and very low SoC levels.
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional


class RobotState(Enum):
    """Robot activity states affecting power consumption."""
    IDLE = "idle"
    WALKING = "walking"
    CHARGING = "charging"
    DELIVERING = "delivering"


@dataclass
class BatteryConfig:
    """Battery configuration parameters."""
    # Nominal capacity (Wh)
    capacity_wh: float = 100.0
    
    # Power consumption by state (W)
    power_idle: float = 5.0
    power_walking: float = 50.0
    power_delivering: float = 10.0
    
    # Charging power (W)
    charging_power: float = 30.0
    
    # Non-linearity coefficients
    k1: float = 0.15  # Low SoC penalty
    k2: float = 0.05  # High SoC penalty
    
    # Thresholds
    low_threshold: float = 0.20    # 20% - return to charge
    critical_threshold: float = 0.10  # 10% - emergency stop
    
    # Anomaly detection
    max_discharge_rate: float = 0.02  # %/s - anomaly threshold


class BatteryModel:
    """
    Non-linear battery model simulating realistic Li-ion behavior.
    
    Features:
    - Non-linear discharge curve
    - Activity-based consumption
    - Anomaly detection for abnormal discharge
    - Charge estimation for path planning
    """
    
    def __init__(self, config: Optional[BatteryConfig] = None, initial_soc: float = 1.0):
        """
        Initialize battery model.
        
        Args:
            config: Battery configuration parameters
            initial_soc: Initial state of charge (0.0 to 1.0)
        """
        self.config = config or BatteryConfig()
        self._soc = np.clip(initial_soc, 0.0, 1.0)
        self._state = RobotState.IDLE
        
        # History for anomaly detection
        self._soc_history = []
        self._time_history = []
        self._discharge_rates = []
        
        # Callbacks
        self._on_low_battery: Optional[Callable] = None
        self._on_critical_battery: Optional[Callable] = None
        self._on_anomaly: Optional[Callable] = None
        
        # Flags
        self._low_battery_triggered = False
        self._critical_triggered = False
    
    @property
    def soc(self) -> float:
        """Current state of charge (0.0 - 1.0)."""
        return self._soc
    
    @property
    def soc_percent(self) -> float:
        """Current state of charge in percentage."""
        return self._soc * 100.0
    
    @property
    def state(self) -> RobotState:
        """Current robot state."""
        return self._state
    
    @property
    def is_low(self) -> bool:
        """Check if battery is below low threshold."""
        return self._soc <= self.config.low_threshold
    
    @property
    def is_critical(self) -> bool:
        """Check if battery is below critical threshold."""
        return self._soc <= self.config.critical_threshold
    
    def set_state(self, state: RobotState):
        """Update robot activity state."""
        self._state = state
    
    def _get_power_consumption(self) -> float:
        """Get current power consumption based on state."""
        power_map = {
            RobotState.IDLE: self.config.power_idle,
            RobotState.WALKING: self.config.power_walking,
            RobotState.DELIVERING: self.config.power_delivering,
            RobotState.CHARGING: -self.config.charging_power,
        }
        return power_map.get(self._state, self.config.power_idle)
    
    def _compute_efficiency(self, soc: float) -> float:
        """
        Compute non-linear efficiency factor.
        
        The efficiency factor creates a U-shaped curve where discharge
        is slightly accelerated at very high and very low SoC.
        
        Args:
            soc: Current state of charge
            
        Returns:
            Efficiency multiplier (>= 1.0)
        """
        k1, k2 = self.config.k1, self.config.k2
        # η(SoC) = 1 + k1*(1-SoC)² + k2*SoC²
        eta = 1.0 + k1 * (1.0 - soc)**2 + k2 * soc**2
        return eta
    
    def update(self, dt: float, current_time: float) -> float:
        """
        Update battery state for one timestep.
        
        Args:
            dt: Time step in seconds
            current_time: Current simulation time
            
        Returns:
            Change in SoC (negative for discharge)
        """
        # Get power and efficiency
        power = self._get_power_consumption()
        eta = self._compute_efficiency(self._soc)
        
        # Compute SoC change
        # dSoC = -P * η * dt / E_nom (converted to fraction)
        energy_capacity_ws = self.config.capacity_wh * 3600  # Wh to Ws
        
        if power > 0:  # Discharging
            d_soc = -(power * eta * dt) / energy_capacity_ws
        else:  # Charging
            # Charging efficiency (simplified - constant)
            charging_eta = 0.95
            d_soc = -(power * charging_eta * dt) / energy_capacity_ws
        
        # Update SoC
        old_soc = self._soc
        self._soc = np.clip(self._soc + d_soc, 0.0, 1.0)
        
        # Record history for anomaly detection
        self._soc_history.append(self._soc)
        self._time_history.append(current_time)
        
        # Keep last 100 samples
        if len(self._soc_history) > 100:
            self._soc_history.pop(0)
            self._time_history.pop(0)
        
        # Check for anomalies
        self._check_anomaly()
        
        # Trigger callbacks
        self._check_thresholds()
        
        return d_soc
    
    def _check_anomaly(self):
        """Check for abnormal discharge rate."""
        if len(self._soc_history) < 10:
            return
        
        # Compute recent discharge rate
        recent_soc = self._soc_history[-10:]
        recent_time = self._time_history[-10:]
        
        if recent_time[-1] - recent_time[0] > 0:
            rate = -(recent_soc[-1] - recent_soc[0]) / (recent_time[-1] - recent_time[0])
            self._discharge_rates.append(rate)
            
            # Check if rate exceeds threshold
            if rate > self.config.max_discharge_rate and self._on_anomaly:
                self._on_anomaly(rate, self._soc)
    
    def _check_thresholds(self):
        """Check battery thresholds and trigger callbacks."""
        if self.is_critical and not self._critical_triggered:
            self._critical_triggered = True
            if self._on_critical_battery:
                self._on_critical_battery(self._soc)
        
        if self.is_low and not self._low_battery_triggered:
            self._low_battery_triggered = True
            if self._on_low_battery:
                self._on_low_battery(self._soc)
    
    def estimate_range(self, target_soc: float = None) -> float:
        """
        Estimate remaining operation time in seconds.
        
        Args:
            target_soc: Target SoC (default: low threshold)
            
        Returns:
            Estimated time until target SoC (seconds)
        """
        if target_soc is None:
            target_soc = self.config.low_threshold
        
        if self._soc <= target_soc:
            return 0.0
        
        # Use current consumption rate (simplified estimation)
        power = self._get_power_consumption()
        if power <= 0:
            return float('inf')
        
        # Average efficiency over remaining range
        avg_eta = self._compute_efficiency((self._soc + target_soc) / 2)
        
        # Time = (SoC_diff * E_nom) / (P * η)
        soc_diff = self._soc - target_soc
        energy_diff = soc_diff * self.config.capacity_wh * 3600
        time_estimate = energy_diff / (power * avg_eta)
        
        return time_estimate
    
    def can_complete_mission(self, estimated_duration: float, 
                            safety_margin: float = 1.2) -> bool:
        """
        Check if battery can complete a mission.
        
        Args:
            estimated_duration: Estimated mission duration (seconds)
            safety_margin: Safety factor multiplier
            
        Returns:
            True if mission is feasible
        """
        available_time = self.estimate_range()
        return available_time >= estimated_duration * safety_margin
    
    def reset(self, soc: float = 1.0):
        """Reset battery to specified SoC."""
        self._soc = np.clip(soc, 0.0, 1.0)
        self._soc_history.clear()
        self._time_history.clear()
        self._discharge_rates.clear()
        self._low_battery_triggered = False
        self._critical_triggered = False
    
    def set_callbacks(self, 
                     on_low: Optional[Callable] = None,
                     on_critical: Optional[Callable] = None,
                     on_anomaly: Optional[Callable] = None):
        """Set callback functions for battery events."""
        self._on_low_battery = on_low
        self._on_critical_battery = on_critical
        self._on_anomaly = on_anomaly
    
    def get_status(self) -> dict:
        """Get comprehensive battery status."""
        return {
            "soc": self._soc,
            "soc_percent": self.soc_percent,
            "state": self._state.value,
            "is_low": self.is_low,
            "is_critical": self.is_critical,
            "estimated_range_s": self.estimate_range(),
            "power_consumption_w": self._get_power_consumption(),
        }
    
    def __repr__(self) -> str:
        return f"Battery(SoC={self.soc_percent:.1f}%, state={self._state.value})"
