"""
Executive Subsystem - Task Execution Management

Manages execution of individual tasks within a mission.
- Monitors task progress
- Handles task transitions
- Reports execution status
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Callable, Dict
import numpy as np


class TaskState(Enum):
    """States for task execution."""
    IDLE = auto()
    NAVIGATING = auto()
    WAITING = auto()
    EXECUTING = auto()
    COMPLETED = auto()
    FAILED = auto()


@dataclass
class ExecutionContext:
    """Context for current execution."""
    task_type: str
    target_position: Optional[np.ndarray] = None
    target_name: Optional[str] = None
    wait_duration: float = 0.0
    elapsed_time: float = 0.0
    distance_to_target: float = float('inf')


class ExecutiveSubsystem:
    """
    Task execution manager.
    
    Handles low-level task execution for the mission subsystem.
    """
    
    # Navigation thresholds
    ARRIVAL_THRESHOLD = 0.4  # meters
    
    def __init__(self):
        """Initialize executive subsystem."""
        self.state = TaskState.IDLE
        self.context: Optional[ExecutionContext] = None
        
        # Callbacks
        self._on_task_complete: Optional[Callable] = None
        self._on_task_failed: Optional[Callable] = None
    
    def start_task(self, task_type: str, target_position: Optional[np.ndarray],
                   target_name: str, wait_duration: float = 0.0):
        """Start executing a task."""
        self.context = ExecutionContext(
            task_type=task_type,
            target_position=target_position,
            target_name=target_name,
            wait_duration=wait_duration
        )
        
        if target_position is not None:
            self.state = TaskState.NAVIGATING
        elif wait_duration > 0:
            self.state = TaskState.WAITING
        else:
            self.state = TaskState.EXECUTING
        
        print(f"[EXEC] Started: {task_type} -> {target_name}")
    
    def update(self, robot_position: np.ndarray, dt: float) -> TaskState:
        """
        Update execution state.
        
        Args:
            robot_position: Current robot position [x, y, z] or [x, y]
            dt: Time step
            
        Returns:
            Current task state
        """
        if self.context is None:
            return TaskState.IDLE
        
        robot_2d = robot_position[:2] if len(robot_position) > 2 else robot_position
        
        if self.state == TaskState.NAVIGATING:
            # Check if arrived
            if self.context.target_position is not None:
                distance = np.linalg.norm(
                    self.context.target_position - robot_2d
                )
                self.context.distance_to_target = distance
                
                if distance < self.ARRIVAL_THRESHOLD:
                    # Arrived - start waiting if needed
                    if self.context.wait_duration > 0:
                        self.state = TaskState.WAITING
                        self.context.elapsed_time = 0.0
                        print(f"[EXEC] Arrived at {self.context.target_name}, waiting {self.context.wait_duration}s")
                    else:
                        self._complete_task()
        
        elif self.state == TaskState.WAITING:
            # Update wait timer
            self.context.elapsed_time += dt
            
            if self.context.elapsed_time >= self.context.wait_duration:
                self._complete_task()
        
        elif self.state == TaskState.EXECUTING:
            # Generic execution (immediate completion)
            self._complete_task()
        
        return self.state
    
    def _complete_task(self):
        """Mark current task as complete."""
        if self.context:
            print(f"[EXEC] Completed: {self.context.task_type} at {self.context.target_name}")
            
            if self._on_task_complete:
                self._on_task_complete(self.context)
        
        self.state = TaskState.COMPLETED
        self.context = None
    
    def fail_task(self, reason: str):
        """Fail current task."""
        if self.context:
            print(f"[EXEC] FAILED: {self.context.task_type} - {reason}")
            
            if self._on_task_failed:
                self._on_task_failed(self.context, reason)
        
        self.state = TaskState.FAILED
        self.context = None
    
    def get_navigation_target(self) -> Optional[np.ndarray]:
        """Get current navigation target position."""
        if self.context and self.state == TaskState.NAVIGATING:
            return self.context.target_position
        return None
    
    def is_waiting(self) -> bool:
        """Check if currently waiting."""
        return self.state == TaskState.WAITING
    
    def set_callbacks(self, on_complete: Callable, on_failed: Callable):
        """Set task completion callbacks."""
        self._on_task_complete = on_complete
        self._on_task_failed = on_failed
    
    def get_status(self) -> Dict:
        """Get executive status."""
        return {
            "state": self.state.name,
            "task_type": self.context.task_type if self.context else None,
            "target": self.context.target_name if self.context else None,
            "distance": self.context.distance_to_target if self.context else None,
            "wait_progress": (
                f"{self.context.elapsed_time:.1f}/{self.context.wait_duration:.1f}s"
                if self.context and self.state == TaskState.WAITING else None
            ),
        }
