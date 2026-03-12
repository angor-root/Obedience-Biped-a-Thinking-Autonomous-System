"""
Mission Subsystem - High-Level Mission Management

Manages mission definition, tracking, and state.

Based on Thought System's Will module:
- self.mission = "Transport medicines from Zone A to Zone B"
- self.current_goal = "Deliver" / "Recharge"
- self.energy_level
- self.energy_threshold
- self.alarm_active
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np


class MissionType(Enum):
    """Types of missions."""
    DELIVERY = auto()
    PATROL = auto()
    RETURN_HOME = auto()
    CHARGE = auto()
    IDLE = auto()


class GoalStatus(Enum):
    """Status of individual goals."""
    PENDING = auto()
    ACTIVE = auto()
    COMPLETED = auto()
    FAILED = auto()
    SKIPPED = auto()


@dataclass
class MissionGoal:
    """Individual goal within a mission."""
    goal_id: str
    goal_type: str  # "navigate", "load", "deliver", "charge"
    target_position: Optional[np.ndarray] = None
    target_name: Optional[str] = None
    duration: float = 0.0  # seconds to spend at goal
    status: GoalStatus = GoalStatus.PENDING
    
    def __str__(self) -> str:
        return f"Goal({self.goal_id}: {self.goal_type} -> {self.target_name}, {self.status.name})"


@dataclass
class Mission:
    """Complete mission definition."""
    mission_id: str
    mission_type: MissionType
    description: str
    goals: List[MissionGoal] = field(default_factory=list)
    priority: int = 5  # 1 = highest
    energy_required: float = 0.3  # minimum battery needed
    
    # Runtime state
    current_goal_idx: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    
    @property
    def current_goal(self) -> Optional[MissionGoal]:
        if 0 <= self.current_goal_idx < len(self.goals):
            return self.goals[self.current_goal_idx]
        return None
    
    @property
    def progress(self) -> float:
        if not self.goals:
            return 1.0
        completed = sum(1 for g in self.goals if g.status == GoalStatus.COMPLETED)
        return completed / len(self.goals)
    
    @property
    def is_complete(self) -> bool:
        return all(g.status in [GoalStatus.COMPLETED, GoalStatus.SKIPPED] 
                   for g in self.goals)


class MissionSubsystem:
    """
    Mission management subsystem.
    
    Tracks mission state and provides goal sequencing.
    """
    
    def __init__(self):
        """Initialize mission subsystem."""
        # Active mission
        self.active_mission: Optional[Mission] = None
        
        # Mission queue
        self.mission_queue: List[Mission] = []
        
        # History
        self.completed_missions: List[Mission] = []
        
        # State from Thought System diagram
        self.energy_level: float = 1.0
        self.energy_threshold: float = 0.15
        self.alarm_active: bool = False
    
    def create_delivery_mission(self, supply_pos: np.ndarray,
                                 bed_positions: List[tuple]) -> Mission:
        """
        Create a supply-to-bed delivery mission.
        
        Args:
            supply_pos: Position of supply zone [x, y]
            bed_positions: List of (bed_name, [x, y]) tuples
            
        Returns:
            Created mission
        """
        goals = []
        goal_idx = 0
        
        for bed_name, bed_pos in bed_positions:
            # Go to supply
            goals.append(MissionGoal(
                goal_id=f"load_{goal_idx}",
                goal_type="load",
                target_position=np.array(supply_pos),
                target_name="supply_zone",
                duration=3.0
            ))
            goal_idx += 1
            
            # Go to bed
            goals.append(MissionGoal(
                goal_id=f"deliver_{goal_idx}",
                goal_type="deliver",
                target_position=np.array(bed_pos),
                target_name=bed_name,
                duration=2.0
            ))
            goal_idx += 1
        
        # Return home
        goals.append(MissionGoal(
            goal_id="return_home",
            goal_type="navigate",
            target_position=np.array([3.5, -4.5]),  # Charging station
            target_name="charging_station",
            duration=0.0
        ))
        
        return Mission(
            mission_id=f"delivery_{len(self.completed_missions) + 1}",
            mission_type=MissionType.DELIVERY,
            description="Deliver medicine to patient beds",
            goals=goals,
            priority=3
        )
    
    def start_mission(self, mission: Mission, timestamp: float):
        """Start a mission."""
        self.active_mission = mission
        mission.start_time = timestamp
        mission.current_goal_idx = 0
        
        if mission.goals:
            mission.goals[0].status = GoalStatus.ACTIVE
        
        print(f"[MISSION] Started: {mission.description}")
        print(f"[MISSION] Goals: {len(mission.goals)}")
    
    def complete_current_goal(self, timestamp: float) -> Optional[MissionGoal]:
        """Mark current goal as complete and advance."""
        if not self.active_mission:
            return None
        
        current = self.active_mission.current_goal
        if current:
            current.status = GoalStatus.COMPLETED
            print(f"[MISSION] Completed: {current}")
            
            # Advance to next goal
            self.active_mission.current_goal_idx += 1
            next_goal = self.active_mission.current_goal
            
            if next_goal:
                next_goal.status = GoalStatus.ACTIVE
                return next_goal
            else:
                # Mission complete
                self._complete_mission(timestamp)
        
        return None
    
    def fail_current_goal(self, reason: str, timestamp: float):
        """Mark current goal as failed."""
        if not self.active_mission:
            return
        
        current = self.active_mission.current_goal
        if current:
            current.status = GoalStatus.FAILED
            print(f"[MISSION] FAILED: {current} - {reason}")
    
    def abort_mission(self, reason: str, timestamp: float):
        """Abort active mission."""
        if self.active_mission:
            print(f"[MISSION] ABORTED: {reason}")
            self.active_mission.end_time = timestamp
            # Don't add to completed (it wasn't completed)
            self.active_mission = None
    
    def _complete_mission(self, timestamp: float):
        """Mark mission as complete."""
        if self.active_mission:
            self.active_mission.end_time = timestamp
            self.completed_missions.append(self.active_mission)
            print(f"[MISSION] COMPLETED: {self.active_mission.description}")
            print(f"[MISSION] Duration: {timestamp - self.active_mission.start_time:.1f}s")
            self.active_mission = None
    
    def update_energy(self, energy_level: float):
        """Update energy level and check thresholds."""
        self.energy_level = energy_level
        
        if energy_level < self.energy_threshold and not self.alarm_active:
            self.alarm_active = True
            print(f"[MISSION] LOW ENERGY ALARM: {energy_level*100:.1f}%")
        elif energy_level >= self.energy_threshold:
            self.alarm_active = False
    
    def get_status(self) -> Dict:
        """Get mission subsystem status."""
        return {
            "active_mission": self.active_mission.description if self.active_mission else None,
            "current_goal": str(self.active_mission.current_goal) if self.active_mission else None,
            "progress": self.active_mission.progress if self.active_mission else 0.0,
            "missions_completed": len(self.completed_missions),
            "energy_level": self.energy_level,
            "alarm_active": self.alarm_active,
        }
