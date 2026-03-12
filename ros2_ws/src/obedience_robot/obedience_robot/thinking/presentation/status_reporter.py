"""
Status Reporter - System Status Presentation

Aggregates and formats system status for display and logging.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Callable
from datetime import datetime
from ..common import FaultState


@dataclass
class StatusReport:
    """Complete system status report."""
    # Robot state
    robot_status: str = "Idle"
    current_position: str = "(0.0, 0.0)"
    current_zone: Optional[str] = None
    
    # Energy state
    battery_percent: float = 100.0
    energy_alert: str = ""
    
    # Balance/stability
    balance_status: str = "Stable"
    tilt_angle: float = 0.0
    
    # Mission state
    mission_status: str = "No active mission"
    mission_progress: float = 0.0
    current_goal: Optional[str] = None
    
    # Health state
    health_status: str = "NOMINAL"
    active_faults: List[str] = field(default_factory=list)
    
    # Alarms
    alarm_active: bool = False
    alarm_signal: str = ""
    
    # Timestamp
    timestamp: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "robot_status": self.robot_status,
            "position": self.current_position,
            "zone": self.current_zone,
            "battery": f"{self.battery_percent:.1f}%",
            "energy_alert": self.energy_alert,
            "balance": self.balance_status,
            "tilt": f"{self.tilt_angle:.2f} rad",
            "mission": self.mission_status,
            "progress": f"{self.mission_progress*100:.0f}%",
            "goal": self.current_goal,
            "health": self.health_status,
            "faults": self.active_faults,
            "alarm": self.alarm_signal if self.alarm_active else "None",
        }
    
    def format_display(self) -> str:
        """Format for console display."""
        lines = [
            "=" * 50,
            f"ROBOT STATUS: {self.robot_status}",
            "=" * 50,
            f"Position: {self.current_position}",
            f"Zone: {self.current_zone or 'Transit'}",
            f"Battery: {self.battery_percent:.1f}% {self.energy_alert}",
            f"Balance: {self.balance_status} (tilt: {self.tilt_angle:.2f})",
            "",
            f"Mission: {self.mission_status}",
            f"Progress: {self.mission_progress*100:.0f}%",
            f"Goal: {self.current_goal or 'None'}",
            "",
            f"Health: {self.health_status}",
        ]
        
        if self.active_faults:
            lines.append(f"Faults: {', '.join(self.active_faults)}")
        
        if self.alarm_active:
            lines.append(f"!!! ALARM: {self.alarm_signal} !!!")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)


class StatusReporter:
    """
    Aggregates and presents system status.
    
    Collects status from various subsystems and formats
    for display/logging/transmission.
    """
    
    def __init__(self):
        """Initialize status reporter."""
        self.current_report = StatusReport()
        self._report_history: List[StatusReport] = []
        
        # Callbacks for notifications
        self._on_alarm: Optional[Callable[[str], None]] = None
        self._on_status_change: Optional[Callable[[StatusReport], None]] = None
        
        # Report interval tracking
        self._last_report_time: float = 0.0
        self.report_interval: float = 2.0  # seconds
    
    def update(self, robot_status: str, position: tuple, zone: Optional[str],
               battery: float, balance_status: str, tilt: float,
               mission_status: str, progress: float, goal: Optional[str],
               health_status: str, faults: List[str], timestamp: float) -> StatusReport:
        """
        Update status report with new data.
        
        Returns updated StatusReport.
        """
        # Create new report
        report = StatusReport(
            robot_status=robot_status,
            current_position=f"({position[0]:.2f}, {position[1]:.2f})",
            current_zone=zone,
            battery_percent=battery * 100,
            balance_status=balance_status,
            tilt_angle=tilt,
            mission_status=mission_status,
            mission_progress=progress,
            current_goal=goal,
            health_status=health_status,
            active_faults=faults,
            timestamp=timestamp
        )
        
        # Energy alert
        if battery < 0.10:
            report.energy_alert = "!!! CRITICAL !!!"
            report.alarm_active = True
            report.alarm_signal = "BEEP/FLASH - LOW BATTERY"
        elif battery < 0.20:
            report.energy_alert = "LOW BATTERY - ALARM ON"
            report.alarm_active = True
            report.alarm_signal = "BEEP - Low Battery"
        
        # Health alarm
        if health_status == "CRITICAL":
            report.alarm_active = True
            if report.alarm_signal:
                report.alarm_signal += " + FAULT"
            else:
                report.alarm_signal = "ALERT - System Fault"
        
        # Check for alarm trigger
        if report.alarm_active and not self.current_report.alarm_active:
            if self._on_alarm:
                self._on_alarm(report.alarm_signal)
        
        # Store report
        self.current_report = report
        
        # Periodic history storage
        if timestamp - self._last_report_time >= self.report_interval:
            self._report_history.append(report)
            self._last_report_time = timestamp
            
            if self._on_status_change:
                self._on_status_change(report)
        
        return report
    
    def get_quick_status(self) -> str:
        """Get one-line status summary."""
        r = self.current_report
        alarm = " [ALARM]" if r.alarm_active else ""
        return (f"{r.robot_status} | Bat: {r.battery_percent:.0f}% | "
                f"{r.health_status}{alarm}")
    
    def set_callbacks(self, on_alarm: Callable[[str], None],
                      on_status_change: Optional[Callable[[StatusReport], None]] = None):
        """Set notification callbacks."""
        self._on_alarm = on_alarm
        self._on_status_change = on_status_change
    
    def print_status(self):
        """Print current status to console."""
        print(self.current_report.format_display())
    
    def get_history(self, limit: int = 10) -> List[Dict]:
        """Get recent status history as dictionaries."""
        return [r.to_dict() for r in self._report_history[-limit:]]
