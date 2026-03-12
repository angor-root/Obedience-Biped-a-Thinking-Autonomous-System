"""
Knowledge Subsystem - Learning and Memory for Thinking Systems

Implements persistent knowledge that allows the robot to:
1. Learn from experiences (perturbations, faults, recoveries)
2. Remember mission patterns
3. Adapt behavior based on past events
"""

from .learning_engine import LearningEngine, Experience, KnowledgeBase

__all__ = ['LearningEngine', 'Experience', 'KnowledgeBase']
