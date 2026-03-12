#!/usr/bin/env python3
"""
Learning Engine - Experiential Learning for Autonomous Bipedal Robot

This module implements a learning system that:
1. Records experiences (events, faults, recoveries)
2. Builds patterns from repeated events
3. Adapts behavior based on learned knowledge
4. Persists knowledge to disk for long-term memory

Knowledge Types:
- Fault patterns: What faults occur and how to recover
- Perturbation responses: How to handle disturbances
- Mission patterns: Optimal paths and timing
- Recovery strategies: What works and what doesn't
"""

import json
import time
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from collections import defaultdict
from datetime import datetime
import hashlib


@dataclass
class Experience:
    """Single experience/event record."""
    id: str
    timestamp: float
    event_type: str  # FAULT, PERTURBATION, RECOVERY, MISSION, DECISION
    system: str      # LOCOMOTION, BALANCE, BATTERY, COMMUNICATION, MISSION
    sensor: str      # IMU, CONTACTS, BATTERY, WATCHDOG, etc.
    severity: str    # INFO, WARN, ERROR, CRITICAL
    message: str
    context: Dict[str, Any] = field(default_factory=dict)
    outcome: str = ""  # SUCCESS, FAILURE, PARTIAL, ONGOING
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)
    
    def __str__(self):
        return f"[{self.id}] {self.timestamp:.2f} | {self.system}/{self.sensor} | {self.severity} | {self.message}"


@dataclass
class Pattern:
    """Learned pattern from repeated experiences."""
    pattern_type: str
    key: str
    occurrences: int = 0
    success_rate: float = 0.0
    avg_recovery_time: float = 0.0
    best_response: str = ""
    context_factors: Dict[str, Any] = field(default_factory=dict)
    
    def update(self, success: bool, recovery_time: float = 0.0):
        """Update pattern statistics."""
        self.occurrences += 1
        # Running average for success rate
        self.success_rate = (self.success_rate * (self.occurrences - 1) + (1.0 if success else 0.0)) / self.occurrences
        # Running average for recovery time
        if recovery_time > 0:
            self.avg_recovery_time = (self.avg_recovery_time * (self.occurrences - 1) + recovery_time) / self.occurrences


class KnowledgeBase:
    """
    Persistent knowledge storage with learning capabilities.
    """
    
    def __init__(self, storage_path: str = None):
        self.storage_path = storage_path or "/tmp/obedience_knowledge"
        os.makedirs(self.storage_path, exist_ok=True)
        
        # Experience log (recent experiences)
        self.experiences: List[Experience] = []
        self.max_experiences = 1000
        
        # Learned patterns
        self.patterns: Dict[str, Pattern] = {}
        
        # Statistics
        self.stats = {
            'total_faults': 0,
            'total_recoveries': 0,
            'total_perturbations': 0,
            'missions_completed': 0,
            'missions_aborted': 0,
            'avg_mission_time': 0.0,
            'uptime_seconds': 0.0,
        }
        
        # Current session
        self.session_start = time.time()
        self.experience_counter = 0
        
        # Load persisted knowledge
        self._load()
    
    def _generate_id(self) -> str:
        """Generate unique experience ID."""
        self.experience_counter += 1
        ts = int(time.time() * 1000)
        return f"EXP-{ts}-{self.experience_counter:04d}"
    
    def record_experience(self, 
                         event_type: str,
                         system: str,
                         sensor: str,
                         severity: str,
                         message: str,
                         context: Dict = None,
                         outcome: str = "ONGOING") -> Experience:
        """Record a new experience."""
        exp = Experience(
            id=self._generate_id(),
            timestamp=time.time(),
            event_type=event_type,
            system=system,
            sensor=sensor,
            severity=severity,
            message=message,
            context=context or {},
            outcome=outcome
        )
        
        self.experiences.append(exp)
        
        # Trim if too many
        if len(self.experiences) > self.max_experiences:
            self.experiences = self.experiences[-self.max_experiences:]
        
        # Update stats
        if event_type == "FAULT":
            self.stats['total_faults'] += 1
        elif event_type == "RECOVERY":
            self.stats['total_recoveries'] += 1
        elif event_type == "PERTURBATION":
            self.stats['total_perturbations'] += 1
        
        # Learn from experience
        self._learn_from_experience(exp)
        
        return exp
    
    def _learn_from_experience(self, exp: Experience):
        """Extract patterns and update knowledge."""
        # Create pattern key
        pattern_key = f"{exp.event_type}:{exp.system}:{exp.sensor}"
        
        if pattern_key not in self.patterns:
            self.patterns[pattern_key] = Pattern(
                pattern_type=exp.event_type,
                key=pattern_key
            )
        
        # Update pattern
        success = exp.outcome in ["SUCCESS", "RECOVERED"]
        recovery_time = exp.context.get('recovery_time', 0.0)
        self.patterns[pattern_key].update(success, recovery_time)
        
        # Learn context factors
        for key, value in exp.context.items():
            if key not in self.patterns[pattern_key].context_factors:
                self.patterns[pattern_key].context_factors[key] = []
            factors = self.patterns[pattern_key].context_factors[key]
            factors.append(value)
            # Keep last 100 values
            self.patterns[pattern_key].context_factors[key] = factors[-100:]
    
    def get_pattern(self, event_type: str, system: str, sensor: str) -> Optional[Pattern]:
        """Get learned pattern for given event type."""
        key = f"{event_type}:{system}:{sensor}"
        return self.patterns.get(key)
    
    def get_recommendation(self, event_type: str, system: str, context: Dict = None) -> Dict:
        """Get recommended action based on learned knowledge."""
        # Find all matching patterns
        matching = []
        for key, pattern in self.patterns.items():
            if pattern.pattern_type == event_type and system in key:
                matching.append(pattern)
        
        if not matching:
            return {
                'action': 'CONTINUE',  # Default: continue mission
                'confidence': 0.0,
                'reason': 'No prior experience'
            }
        
        # Find best pattern
        best = max(matching, key=lambda p: p.success_rate * p.occurrences)
        
        # Recommend based on success rate
        if best.success_rate > 0.8:
            action = 'CONTINUE'
            reason = f"High success rate ({best.success_rate:.1%}) over {best.occurrences} occurrences"
        elif best.success_rate > 0.5:
            action = 'CAUTION'
            reason = f"Moderate success rate ({best.success_rate:.1%})"
        else:
            action = 'ABORT'
            reason = f"Low success rate ({best.success_rate:.1%})"
        
        return {
            'action': action,
            'confidence': best.success_rate,
            'reason': reason,
            'pattern': best.key,
            'occurrences': best.occurrences
        }
    
    def mission_completed(self, duration: float):
        """Record completed mission."""
        self.stats['missions_completed'] += 1
        n = self.stats['missions_completed']
        self.stats['avg_mission_time'] = (self.stats['avg_mission_time'] * (n-1) + duration) / n
        self._save()
    
    def mission_aborted(self, reason: str):
        """Record aborted mission."""
        self.stats['missions_aborted'] += 1
        self.record_experience(
            event_type="MISSION",
            system="MISSION",
            sensor="EXECUTIVE",
            severity="WARN",
            message=f"Mission aborted: {reason}",
            outcome="FAILURE"
        )
        self._save()
    
    def get_summary(self) -> Dict:
        """Get knowledge summary."""
        return {
            'total_experiences': len(self.experiences),
            'patterns_learned': len(self.patterns),
            'stats': self.stats,
            'session_uptime': time.time() - self.session_start,
            'top_patterns': [
                {'key': p.key, 'occurrences': p.occurrences, 'success_rate': p.success_rate}
                for p in sorted(self.patterns.values(), key=lambda x: x.occurrences, reverse=True)[:5]
            ]
        }
    
    def _save(self):
        """Persist knowledge to disk."""
        try:
            # Save experiences
            exp_path = os.path.join(self.storage_path, "experiences.json")
            with open(exp_path, 'w') as f:
                json.dump([e.to_dict() for e in self.experiences[-500:]], f, indent=2)
            
            # Save patterns
            pat_path = os.path.join(self.storage_path, "patterns.json")
            patterns_data = {k: asdict(v) for k, v in self.patterns.items()}
            with open(pat_path, 'w') as f:
                json.dump(patterns_data, f, indent=2)
            
            # Save stats
            stats_path = os.path.join(self.storage_path, "stats.json")
            with open(stats_path, 'w') as f:
                json.dump(self.stats, f, indent=2)
                
        except Exception as e:
            print(f"[Knowledge] Save error: {e}")
    
    def _load(self):
        """Load persisted knowledge."""
        try:
            # Load experiences
            exp_path = os.path.join(self.storage_path, "experiences.json")
            if os.path.exists(exp_path):
                with open(exp_path, 'r') as f:
                    data = json.load(f)
                    self.experiences = [Experience.from_dict(d) for d in data]
            
            # Load patterns
            pat_path = os.path.join(self.storage_path, "patterns.json")
            if os.path.exists(pat_path):
                with open(pat_path, 'r') as f:
                    data = json.load(f)
                    self.patterns = {k: Pattern(**v) for k, v in data.items()}
            
            # Load stats
            stats_path = os.path.join(self.storage_path, "stats.json")
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    loaded_stats = json.load(f)
                    self.stats.update(loaded_stats)
                    
        except Exception as e:
            print(f"[Knowledge] Load error: {e}")


class LearningEngine:
    """
    Learning engine that uses knowledge to improve behavior.
    
    Philosophy: "Complete the mission, learn from everything"
    - Never abort mission unless truly critical
    - Learn from every perturbation and recovery
    - Adapt behavior based on past success/failure
    """
    
    def __init__(self, knowledge_base: KnowledgeBase = None):
        self.kb = knowledge_base or KnowledgeBase()
        
        # Current mission tracking
        self.mission_start_time = None
        self.mission_waypoints_completed = 0
        self.current_perturbation_count = 0
        
        # Learning thresholds (can be adapted)
        self.critical_battery_threshold = 15.0  # Only this is truly critical
        self.low_battery_threshold = 25.0
        self.fall_height_threshold = 0.3  # Below this = fallen
        
    def start_mission(self):
        """Record mission start."""
        self.mission_start_time = time.time()
        self.mission_waypoints_completed = 0
        self.current_perturbation_count = 0
        
        self.kb.record_experience(
            event_type="MISSION",
            system="MISSION",
            sensor="EXECUTIVE",
            severity="INFO",
            message="Mission started",
            outcome="ONGOING"
        )
    
    def waypoint_reached(self, waypoint_name: str):
        """Record waypoint completion."""
        self.mission_waypoints_completed += 1
        self.kb.record_experience(
            event_type="MISSION",
            system="NAVIGATION",
            sensor="WAYPOINT",
            severity="INFO",
            message=f"Waypoint reached: {waypoint_name}",
            context={'waypoint': waypoint_name, 'count': self.mission_waypoints_completed},
            outcome="SUCCESS"
        )
    
    def complete_mission(self):
        """Record mission completion."""
        if self.mission_start_time:
            duration = time.time() - self.mission_start_time
            self.kb.mission_completed(duration)
            self.kb.record_experience(
                event_type="MISSION",
                system="MISSION",
                sensor="EXECUTIVE",
                severity="INFO",
                message=f"Mission completed in {duration:.1f}s",
                context={
                    'duration': duration,
                    'waypoints': self.mission_waypoints_completed,
                    'perturbations': self.current_perturbation_count
                },
                outcome="SUCCESS"
            )
        self.mission_start_time = None
    
    def handle_perturbation(self, perturbation_type: str, force_magnitude: float = 0.0) -> Dict:
        """
        Handle perturbation and decide response.
        
        Philosophy: Keep going unless it's impossible.
        """
        self.current_perturbation_count += 1
        
        # Record experience
        exp = self.kb.record_experience(
            event_type="PERTURBATION",
            system="LOCOMOTION",
            sensor="IMU",
            severity="WARN",
            message=f"Perturbation detected: {perturbation_type}",
            context={
                'type': perturbation_type,
                'force': force_magnitude,
                'count_this_mission': self.current_perturbation_count
            },
            outcome="ONGOING"
        )
        
        # Get recommendation from knowledge
        rec = self.kb.get_recommendation("PERTURBATION", "LOCOMOTION")
        
        # Decision: Almost always continue
        if rec['confidence'] > 0.7 and rec['action'] == 'CONTINUE':
            return {
                'action': 'CONTINUE',
                'message': f"Perturbation handled (learned: {rec['reason']})",
                'experience_id': exp.id
            }
        else:
            return {
                'action': 'CONTINUE',  # Still continue!
                'message': f"Perturbation encountered, continuing mission",
                'experience_id': exp.id
            }
    
    def handle_fault(self, fault_type: str, context: Dict = None) -> Dict:
        """
        Handle fault and decide response.
        
        Philosophy: Continue unless truly impossible.
        """
        severity = self._assess_fault_severity(fault_type, context or {})
        
        exp = self.kb.record_experience(
            event_type="FAULT",
            system=self._fault_to_system(fault_type),
            sensor=self._fault_to_sensor(fault_type),
            severity=severity,
            message=f"Fault detected: {fault_type}",
            context=context or {},
            outcome="ONGOING"
        )
        
        # Only CRITICAL faults should stop mission
        if severity == "CRITICAL":
            return {
                'action': 'ABORT',
                'message': f"Critical fault, mission abort required",
                'experience_id': exp.id,
                'go_to_charger': True
            }
        else:
            return {
                'action': 'CONTINUE',
                'message': f"Fault detected ({severity}), continuing mission",
                'experience_id': exp.id,
                'go_to_charger': False
            }
    
    def should_abort_mission(self, battery: float, height: float, ros_connected: bool) -> Dict:
        """
        Decide if mission should be aborted.
        
        Only truly critical conditions warrant abort:
        - Battery < 15% (robot might not make it back)
        - Height < 0.3m (robot has fallen)
        
        ROS disconnection is NOT a reason to abort - robot continues autonomously.
        """
        reasons = []
        should_abort = False
        
        # Check battery - only critical level
        if battery < self.critical_battery_threshold:
            should_abort = True
            reasons.append(f"Battery critical ({battery:.1f}%)")
        
        # Check if fallen
        if height < self.fall_height_threshold:
            should_abort = True
            reasons.append(f"Robot has fallen (height: {height:.2f}m)")
        
        # ROS disconnection - NOT a reason to abort!
        if not ros_connected:
            self.kb.record_experience(
                event_type="COMMUNICATION",
                system="COMMUNICATION",
                sensor="WATCHDOG",
                severity="WARN",
                message="ROS2 disconnected - continuing autonomous operation",
                context={'battery': battery, 'height': height},
                outcome="ONGOING"
            )
            # Continue mission, robot is autonomous
        
        return {
            'should_abort': should_abort,
            'reasons': reasons,
            'continue_autonomous': not ros_connected  # Flag that we're in autonomous mode
        }
    
    def handle_recovery(self, fault_type: str, recovery_time: float, success: bool):
        """Record recovery from fault."""
        self.kb.record_experience(
            event_type="RECOVERY",
            system=self._fault_to_system(fault_type),
            sensor=self._fault_to_sensor(fault_type),
            severity="INFO" if success else "WARN",
            message=f"Recovery from {fault_type}: {'SUCCESS' if success else 'PARTIAL'}",
            context={'fault_type': fault_type, 'recovery_time': recovery_time},
            outcome="SUCCESS" if success else "PARTIAL"
        )
    
    def get_knowledge_summary(self) -> Dict:
        """Get learning summary."""
        return self.kb.get_summary()
    
    def _assess_fault_severity(self, fault_type: str, context: Dict) -> str:
        """Assess fault severity."""
        # Only these are CRITICAL
        if fault_type in ['FALL'] and context.get('height', 1.0) < 0.3:
            return "CRITICAL"
        if fault_type == 'LOW_VOLTAGE' and context.get('battery', 100) < self.critical_battery_threshold:
            return "CRITICAL"
        
        # Most faults are just warnings - robot can handle them
        if fault_type in ['BALANCE_LOSS', 'CONTACT_LOSS', 'TRIP']:
            return "WARN"
        
        return "INFO"
    
    def _fault_to_system(self, fault_type: str) -> str:
        """Map fault type to system."""
        mapping = {
            'FALL': 'LOCOMOTION',
            'TRIP': 'LOCOMOTION',
            'BALANCE_LOSS': 'BALANCE',
            'CONTACT_LOSS': 'LOCOMOTION',
            'LOW_VOLTAGE': 'BATTERY',
            'MOTOR_FAULT': 'ACTUATORS',
            'SENSOR_FAULT': 'SENSORS',
        }
        return mapping.get(fault_type, 'GENERAL')
    
    def _fault_to_sensor(self, fault_type: str) -> str:
        """Map fault type to primary sensor."""
        mapping = {
            'FALL': 'IMU',
            'TRIP': 'IMU',
            'BALANCE_LOSS': 'IMU',
            'CONTACT_LOSS': 'CONTACTS',
            'LOW_VOLTAGE': 'BATTERY',
            'MOTOR_FAULT': 'JOINTS',
            'SENSOR_FAULT': 'GENERAL',
        }
        return mapping.get(fault_type, 'GENERAL')
