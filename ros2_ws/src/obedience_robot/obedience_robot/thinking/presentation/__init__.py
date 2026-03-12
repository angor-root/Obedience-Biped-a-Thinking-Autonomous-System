"""
Presentation Subsystem - Status Reporting and Alerts

From the Thought System architecture:
- Viewing
- Projection
- Experience
- Email Address (notifications)

Manages output/presentation of robot status:
- self.robot_status = "Walking to Zone B"
- self.energy_alert = "LOW BATTERY - ALARM ON"
- self.balance_status = "Stable"
- self.alarm_signal = "Beep/Flash"
"""

from .status_reporter import StatusReporter, StatusReport

__all__ = [
    'StatusReporter',
    'StatusReport',
]
