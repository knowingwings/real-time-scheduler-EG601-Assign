#!/usr/bin/env python3
"""
Real-Time Task Scheduling System for Raspberry Pi 3
Author: Student Name
Module: EG6801 - Real-Time Embedded System
"""

import numpy as np
import threading
import queue # Not explicitly used by plan changes, but was in original
import time
import psutil
import logging
import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any, Callable
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns # Not explicitly used by plan changes, but was in original
from collections import deque
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import pandas as pd # Not explicitly used by plan changes, but was in original
from datetime import datetime
import multiprocessing as mp # Not explicitly used by plan changes, but was in original
import os
import sys
from scipy import stats  # For statistical tests (optional - handle ImportError)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
HIGH_PRIORITY = 1
MEDIUM_PRIORITY = 2
LOW_PRIORITY = 3

# Task specifications as per assignment
TASK_SPECS = {
    HIGH_PRIORITY: {
        'count': 20,
        'lambda_param': 3.0,  # Poisson distribution parameter
        'service_range': (2, 5),
        'name': 'High'
    },
    MEDIUM_PRIORITY: {
        'count': 20,
        'lambda_param': 5.0,
        'service_range': (3, 7),
        'name': 'Medium'
    },
    LOW_PRIORITY: {
        'count': 10,
        'lambda_param': 7.0,
        'service_range': (5, 10),
        'name': 'Low'
    }
}

class TaskState(Enum):
    """Task execution states"""
    WAITING = "WAITING"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    PREEMPTED = "PREEMPTED"

@dataclass
class Task:
    """Task representation with all required attributes"""
    task_id: str
    arrival_time: float
    service_time: float
    priority: int
    deadline: Optional[float] = None
    state: TaskState = TaskState.WAITING
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    preemption_count: int = 0
    remaining_service_time: float = field(init=False)
    predicted_wait_time: Optional[float] = None # Added for MLScheduler if used elsewhere, plan doesn't use directly on Task
    
    def __post_init__(self):
        self.remaining_service_time = self.service_time
        if self.deadline is None:
            # Set deadline based on priority (tighter for high priority)
            deadline_factor = {1: 1.5, 2: 2.0, 3: 2.5}
            self.deadline = self.arrival_time + self.service_time * deadline_factor[self.priority]
    
    @property
    def waiting_time(self) -> float:
        """Calculate waiting time"""
        if self.start_time is None or self.arrival_time is None: # Added arrival_time check for robustness
            return 0
        return self.start_time - self.arrival_time
    
    @property
    def response_time(self) -> float: # In original file, response_time was turnaround_time. Plan doesn't change this.
        """Calculate response time"""
        if self.completion_time is None or self.arrival_time is None: # Added arrival_time check
            return 0
        return self.completion_time - self.arrival_time
    
    @property
    def turnaround_time(self) -> float:
        """Calculate turnaround time"""
        return self.response_time # turnaround_time is same as response_time in this context
    
    def __lt__(self, other):
        """For priority queue comparison"""
        return self.priority < other.priority
    
    def to_dict(self) -> dict:
        """Convert task to dictionary for serialization"""
        return {
            'task_id': self.task_id,
            'arrival_time': self.arrival_time,
            'service_time': self.service_time,
            'priority': self.priority,
            'deadline': self.deadline
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Task':
        """Create task from dictionary"""
        return cls(
            task_id=data['task_id'],
            arrival_time=data['arrival_time'],
            service_time=data['service_time'],
            priority=data['priority'],
            deadline=data.get('deadline')
        )

class TaskGenerator:
    """Generate tasks according to Poisson distribution"""
    
    @staticmethod
    def generate_tasks(seed: Optional[int] = None) -> List[Task]:
        """Generate 50 tasks with specified distributions"""
        if seed is not None:
            np.random.seed(seed)
        
        tasks = []
        task_counter = 1
        
        for priority, spec in TASK_SPECS.items():
            current_time = 0
            
            for i in range(spec['count']):
                inter_arrival = np.random.exponential(spec['lambda_param'])
                current_time += inter_arrival
                
                service_time = np.random.uniform(*spec['service_range'])
                
                task = Task(
                    task_id=f"Task_{task_counter}",
                    arrival_time=current_time,
                    service_time=service_time,
                    priority=priority
                )
                
                tasks.append(task)
                task_counter += 1
                
                logger.debug(f"Generated {task.task_id}: arrival={task.arrival_time:.2f}, "
                           f"service={task.service_time:.2f}, priority={spec['name']}")
        
        tasks.sort(key=lambda t: t.arrival_time)
        return tasks
    
    @staticmethod
    def export_tasks(tasks: List[Task], filename: str):
        task_data = {
            'generator_info': {
                'total_tasks': len(tasks),
                'timestamp': datetime.now().isoformat(),
                'task_specs': TASK_SPECS
            },
            'tasks': [task.to_dict() for task in tasks]
        }
        
        with open(filename, 'w') as f:
            json.dump(task_data, f, indent=2)
        
        logger.info(f"Exported {len(tasks)} tasks to {filename}")
    
    @staticmethod
    def import_tasks(filename: str) -> List[Task]:
        with open(filename, 'r') as f:
            task_data = json.load(f)
        
        tasks = [Task.from_dict(t) for t in task_data['tasks']]
        logger.info(f"Imported {len(tasks)} tasks from {filename}")
        
        return tasks

class BaseScheduler:
    """Base class for all scheduling algorithms"""
    
    def __init__(self, name: str):
        self.name = name
        self.ready_queue = []
        self.completed_tasks = []
        self.current_time = 0
        self.metrics = {
            'queue_lengths': [],
            'memory_usage': [],
            'cpu_usage': [],
            'timestamps': []
        }
    
    def add_task(self, task: Task):
        raise NotImplementedError
    
    def get_next_task(self) -> Optional[Task]:
        raise NotImplementedError
    
    def record_metrics(self):
        self.metrics['queue_lengths'].append(len(self.ready_queue))
        try: # Add try-except for psutil calls if they can fail
            self.metrics['memory_usage'].append(psutil.Process().memory_info().rss / 1024 / 1024)  # MB
            self.metrics['cpu_usage'].append(psutil.cpu_percent(interval=None)) # interval=0.1 in original, plan doesn't specify. Using None for non-blocking.
        except psutil.Error as e:
            logger.warning(f"Could not record psutil metrics: {e}")
            self.metrics['memory_usage'].append(0)
            self.metrics['cpu_usage'].append(0)
        self.metrics['timestamps'].append(self.current_time)

class FCFSScheduler(BaseScheduler):
    """First-Come-First-Served Scheduler"""
    
    def __init__(self):
        super().__init__("FCFS")
        self.ready_queue = deque() # Original uses deque
    
    def add_task(self, task: Task):
        self.ready_queue.append(task)
    
    def get_next_task(self) -> Optional[Task]:
        if self.ready_queue:
            return self.ready_queue.popleft()
        return None

class EDFScheduler(BaseScheduler):
    """Earliest Deadline First Scheduler"""
    
    def __init__(self):
        super().__init__("EDF")
    
    def add_task(self, task: Task):
        self.ready_queue.append(task)
        self.ready_queue.sort(key=lambda t: t.deadline if t.deadline is not None else float('inf')) # Handle None deadline
    
    def get_next_task(self) -> Optional[Task]:
        if self.ready_queue:
            return self.ready_queue.pop(0)
        return None

class PriorityScheduler(BaseScheduler):
    """Priority-Based Scheduler with Priority Inheritance"""
    
    def __init__(self):
        super().__init__("Priority")
        self.priority_inheritance_active = {}
        # Added attributes from plan
        self.priority_inversion_events = []
        self.priority_inversion_stats = {
            'count': 0,
            'total_duration': 0,
            'affected_tasks': set(),
            'resolution_times': []
        }
    
    def add_task(self, task: Task):
        self.ready_queue.append(task)
        self.ready_queue.sort(key=lambda t: t.priority)
    
    def get_next_task(self) -> Optional[Task]:
        if self.ready_queue:
            return self.ready_queue.pop(0)
        return None
    
    def handle_priority_inversion(self, high_priority_task: Task, blocking_task: Task):
        """Implement priority inheritance with tracking"""
        event_time = self.current_time
        event = {
            'time': event_time,
            'high_priority_task': high_priority_task.task_id,
            'blocking_task': blocking_task.task_id,
            'original_priority': blocking_task.priority,
            'start_time': event_time  # Add start time for resolution tracking
        }
        
        self.priority_inversion_events.append(event)
        self.priority_inversion_stats['count'] += 1
        self.priority_inversion_stats['affected_tasks'].add(high_priority_task.task_id)
        self.priority_inversion_stats['affected_tasks'].add(blocking_task.task_id)
        
        logger.info(f"Priority inversion detected: {high_priority_task.task_id} blocked by {blocking_task.task_id}")
        
        original_priority = blocking_task.priority
        blocking_task.priority = high_priority_task.priority
        self.priority_inheritance_active[blocking_task.task_id] = {
            'original_priority': original_priority,
            'event_index': len(self.priority_inversion_events) - 1  # Track which event this relates to
        }
        
        self.ready_queue.sort(key=lambda t: t.priority)

    def restore_priority(self, task: Task):
        if task.task_id in self.priority_inheritance_active:
            info = self.priority_inheritance_active[task.task_id]
            task.priority = info['original_priority']
            
            # Calculate resolution time
            event_index = info['event_index']
            if event_index < len(self.priority_inversion_events):
                event = self.priority_inversion_events[event_index]
                resolution_time = self.current_time - event['start_time']
                event['resolution_time'] = self.current_time
                event['duration'] = resolution_time
                
                self.priority_inversion_stats['resolution_times'].append(resolution_time)
                self.priority_inversion_stats['total_duration'] += resolution_time
            
            del self.priority_inheritance_active[task.task_id]
            logger.debug(f"Restored priority for {task.task_id} to {task.priority}")

    # Added get_priority_inversion_report method from plan
    def get_priority_inversion_report(self) -> str:
        """Generate detailed priority inversion report"""
        if self.priority_inversion_stats['count'] == 0:
            return "No priority inversion events detected."
        
        avg_res_time_str = "N/A"
        max_res_time_str = "N/A"
        
        # Check if resolution_times list is not empty before calculating mean/max
        if self.priority_inversion_stats['resolution_times']:
            avg_res_time_val = np.mean(self.priority_inversion_stats['resolution_times'])
            avg_res_time_str = f"{avg_res_time_val:.2f}s" if not np.isnan(avg_res_time_val) else "N/A"
            
            max_res_time_val = max(self.priority_inversion_stats['resolution_times'])
            max_res_time_str = f"{max_res_time_val:.2f}s"
        else: # Handle case where resolution_times is empty
            avg_res_time_str = "0.00s (no data)" # Or "N/A"
            max_res_time_str = "0.00s (no data)" # Or "N/A"


        report = f"""
Priority Inversion Analysis:
- Total Events: {self.priority_inversion_stats['count']}
- Affected Tasks: {len(self.priority_inversion_stats['affected_tasks'])}
- Average Resolution Time: {avg_res_time_str}
- Maximum Resolution Time: {max_res_time_str}

Detailed Events (Last 10):
"""
        for event in self.priority_inversion_events[-10:]:  # Last 10 events
            report += f"  - Time: {event['time']:.2f}, HighP: {event['high_priority_task']}, Blocking: {event['blocking_task']}, OrigPrioBlock: {event['original_priority']}\n"
        
        return report

class MLScheduler(BaseScheduler):
    """Machine Learning based scheduler using Linear Regression"""
    
    def __init__(self):
        super().__init__("ML-LinearRegression") # Name updated as per plan's scheduler dicts
        self.model = LinearRegression()
        self.training_data = [] # Not directly used in plan's methods
        self.is_trained = False
    
    def add_task(self, task: Task):
        self.ready_queue.append(task)
        
        if len(self.completed_tasks) > 5 and not self.is_trained:
            self.train_model()
        
        if self.is_trained:
            # Add controlled randomness to predictions
            for t_in_queue in self.ready_queue:
                base_prediction = self.predict_wait_time(t_in_queue)
                # Add 5-15% random variation
                variation = np.random.uniform(0.95, 1.15)
                t_in_queue.predicted_wait_time = base_prediction * variation
            
            # Sort with some randomness for ties
            self.ready_queue.sort(key=lambda t: (
                t.predicted_wait_time if t.predicted_wait_time is not None else float('inf'),
                np.random.random()  # Random tiebreaker
            ))
    
    def get_next_task(self) -> Optional[Task]:
        if self.ready_queue:
            return self.ready_queue.pop(0)
        return None
    
    def train_model(self):
        if len(self.completed_tasks) < 5: # Min 5 samples to train
            return
        
        X = []
        y = []
        
        for task_item in self.completed_tasks: # Renamed task to task_item
            features = [
                task_item.priority,
                task_item.service_time,
                len(self.ready_queue), 
                self.current_time - task_item.arrival_time 
            ]
            X.append(features)
            y.append(task_item.waiting_time)
        
        if not X or not y: # Ensure X and y are not empty
            logger.warning("ML model training skipped: no valid training data.")
            return

        self.model.fit(X, y)
        self.is_trained = True
        logger.info("ML model (Linear Regression) trained on completed tasks")
    
    def predict_wait_time(self, task_to_predict: Task) -> float:
        if not self.is_trained:
            return task_to_predict.service_time
        
        # Add some randomness to prevent overfitting to historical data
        queue_size = len(self.ready_queue)
        
        features = [[
            task_to_predict.priority,
            task_to_predict.service_time,
            queue_size,
            max(0, self.current_time - task_to_predict.arrival_time)
        ]]
        
        try:
            prediction = self.model.predict(features)[0]
            # Add some noise to prevent overfitting
            noise = np.random.normal(0, prediction * 0.1)  # 10% noise
            return max(0, prediction + noise)
        except Exception as e:
            logger.error(f"Error in ML prediction: {e}")
            return task_to_predict.service_time

# Added DecisionTreeScheduler class from plan
class DecisionTreeScheduler(BaseScheduler):
    """Decision Tree based scheduler for task categorization"""
    
    def __init__(self):
        super().__init__("ML-DecisionTree")
        self.tree_model = DecisionTreeRegressor(max_depth=5, random_state=42)
        self.is_trained = False
        # Use performance-based queues instead of priority-based
        self.performance_queues = {
            'urgent': deque(),      # Tasks predicted to need immediate attention
            'normal': deque(),      # Tasks with normal requirements
            'deferred': deque()     # Tasks that can wait
        }
        self.ready_queue = deque()  # Use deque for main queue
    
    def add_task(self, task: Task):
        # Always add to main ready queue first
        self.ready_queue.append(task)
        
        # Train model when enough data
        if len(self.completed_tasks) > 10 and not self.is_trained:
            self._train_tree_model()
    
    def get_next_task(self) -> Optional[Task]:
        if not self.ready_queue:
            return None
        
        if self.is_trained and len(self.ready_queue) > 1:
            # Score all tasks in ready queue
            task_scores = []
            for i, task in enumerate(self.ready_queue):
                score = self._calculate_task_urgency_score(task)
                task_scores.append((i, task, score))
            
            # Sort by score (highest first)
            task_scores.sort(key=lambda x: x[2], reverse=True)
            
            # Get the most urgent task
            index, selected_task, _ = task_scores[0]
            
            # Remove from queue efficiently
            temp_list = list(self.ready_queue)
            temp_list.pop(index)
            self.ready_queue = deque(temp_list)
            
            return selected_task
        else:
            # Default FCFS behavior
            return self.ready_queue.popleft()
    
    def _calculate_task_urgency_score(self, task: Task) -> float:
        """Calculate urgency score using decision tree prediction"""
        if not self.is_trained:
            # Default scoring based on basic heuristics
            wait_time = self.current_time - task.arrival_time
            deadline_urgency = 0
            if task.deadline:
                slack = task.deadline - self.current_time - task.remaining_service_time
                deadline_urgency = 10 / (1 + max(0, slack))
            
            # Use inverse priority for better load distribution
            priority_factor = (4 - task.priority) * 5
            return priority_factor + wait_time * 0.5 + deadline_urgency
        
        # Use trained model to predict performance ratio
        slack_time = 0
        if task.deadline is not None and task.arrival_time is not None:
            slack_time = task.deadline - self.current_time - task.remaining_service_time
        
        # Add queue pressure feature
        queue_pressure = len(self.ready_queue) / 10.0  # Normalize
        
        features = [[
            task.service_time,
            task.priority,
            slack_time,
            self.current_time - task.arrival_time,  # Current waiting time
            queue_pressure,
            task.remaining_service_time / task.service_time  # Progress ratio
        ]]
        
        try:
            # Predict the performance ratio (waiting_time / service_time)
            predicted_ratio = self.tree_model.predict(features)[0]
            
            # Convert to urgency score
            urgency = 50 / (1 + max(0, predicted_ratio))
            
            # Strong starvation prevention
            wait_time = self.current_time - task.arrival_time
            if wait_time > 100:  # If waiting more than 100s
                urgency += wait_time * 0.5
            else:
                urgency += wait_time * 0.2
            
            # Priority boost for high priority tasks
            if task.priority == 1:
                urgency *= 1.5
            
            return urgency
        except Exception as e:
            logger.error(f"Error in decision tree prediction: {e}")
            # Fallback to enhanced heuristic
            wait_factor = (self.current_time - task.arrival_time) * 0.5
            priority_factor = (4 - task.priority) * 10
            return priority_factor + wait_factor
    
    def _train_tree_model(self):
        """Train decision tree on completed tasks"""
        if len(self.completed_tasks) < 10:
            return
        
        X = []
        y = []
        
        # Calculate average queue length from historical data
        avg_queue_length = np.mean(self.metrics['queue_lengths']) if self.metrics['queue_lengths'] else 5
        
        for task in self.completed_tasks:
            slack_time = 0
            if task.deadline is not None and task.start_time is not None:
                slack_time = task.deadline - task.start_time - task.service_time
            
            # Use actual waiting time at the time task started
            actual_wait = task.waiting_time if task.waiting_time is not None else 0
            
            features = [
                task.service_time,
                task.priority,
                slack_time,
                actual_wait,  # Actual wait time
                avg_queue_length / 10.0,  # Normalized queue pressure
                1.0  # Progress ratio (completed tasks have ratio 1.0)
            ]
            X.append(features)
            
            # Target: actual performance ratio
            performance_ratio = actual_wait / task.service_time if task.service_time > 0 else 0
            y.append(performance_ratio)
        
        if not X or not y:
            logger.warning("Decision Tree model training skipped: no valid training data.")
            return
        
        # Use a deeper tree for more complex patterns
        self.tree_model = DecisionTreeRegressor(max_depth=7, min_samples_split=5, random_state=42)
        self.tree_model.fit(X, y)
        self.is_trained = True
        logger.info("Decision Tree model trained on completed tasks")

# Added RLScheduler class from plan
class RLScheduler(BaseScheduler):
    """Simple Q-learning based scheduler"""
    
    def __init__(self):
        super().__init__("ML-QLearning")
        self.q_table = {}
        self.epsilon = 0.3  # Increased from 0.1 for more exploration
        self.alpha = 0.1
        self.gamma = 0.9
        self.last_state = None
        self.last_action = None
        self.ready_queue = deque()
        self.epsilon_decay = 0.995  # Add epsilon decay
        self.min_epsilon = 0.1
    
    def add_task(self, task: Task):
        self.ready_queue.append(task)
        # Sort based on Q-values if available (plan does not specify this sorting in add_task, only implies it for get_next_task)
        # The plan's _sort_by_q_values is defined but not called in add_task. It's called in RLScheduler itself in the plan snippet but not in the actual methods.
        # Re-evaluating plan: `_sort_by_q_values` is mentioned in `add_task` in the plan.
        if self.q_table: # Check if q_table has entries
            self._sort_by_q_values() 
    
    def get_next_task(self) -> Optional[Task]:
        if not self.ready_queue:
            return None
        
        state = self._get_state()
        
        # Adaptive epsilon based on learning progress
        if len(self.completed_tasks) > 50:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        
        num_possible_actions = min(3, len(self.ready_queue))
        
        if num_possible_actions == 0:
            return None
        
        action_idx = 0
        if np.random.random() < self.epsilon:
            # Exploration: Create dynamic weights that favor middle positions
            if num_possible_actions == 1:
                action_idx = 0
            else:
                # Create bell-curve-like weights
                weights = []
                for i in range(num_possible_actions):
                    # Higher weight for middle positions
                    distance_from_middle = abs(i - (num_possible_actions - 1) / 2)
                    weight = 1.0 / (1.0 + distance_from_middle * 0.5)
                    weights.append(weight)
                
                # Normalize to sum to 1
                total = sum(weights)
                weights = [w / total for w in weights]
                
                action_idx = np.random.choice(num_possible_actions, p=weights)
        else:
            # Exploit with occasional random selection
            if np.random.random() < 0.1:  # 10% random even in exploitation
                action_idx = np.random.randint(0, num_possible_actions)
            else:
                action_idx = self._get_best_action(state, num_possible_actions)
        
        # Update Q-values based on previous action
        if self.last_state is not None and self.last_action is not None:
            reward = self._calculate_reward()
            self._update_q_value(self.last_state, self.last_action, reward, state)
        
        self.last_state = state
        self.last_action = action_idx
        
        # Extract selected task more efficiently
        if isinstance(self.ready_queue, deque):
            # For deque, convert to list once
            queue_list = list(self.ready_queue)
            selected_task = queue_list.pop(action_idx)
            self.ready_queue = deque(queue_list)
        else:
            # For list
            selected_task = self.ready_queue.pop(action_idx)
        
        return selected_task
    
    def _get_state(self) -> tuple:
        """Encode current state"""
        queue_length = len(self.ready_queue)
        if queue_length == 0:
            return (0, 0, 0)
        
        queue_cat = min(queue_length // 5, 4)
        
        # Consider first 5 tasks for avg priority and service time
        sample_tasks = list(self.ready_queue)[:5]
        if not sample_tasks:
            return (queue_cat, 0, 0)
        
        # Calculate averages with safety checks
        priorities = [t.priority for t in sample_tasks]
        service_times = [t.service_time for t in sample_tasks]
        
        avg_priority = int(round(np.mean(priorities))) if priorities else 2
        avg_priority = max(1, min(3, avg_priority))
        
        avg_service = np.mean(service_times) if service_times else 5.0
        service_cat = min(int(avg_service // 2), 4)
        
        return (queue_cat, avg_priority, service_cat)
    
    def _get_best_action(self, state: tuple, num_actions: int) -> int:
        """Get action (index) with highest Q-value"""
        best_action_idx = 0
        best_value = float('-inf')
        
        if num_actions == 0: return 0 # Should not happen

        for action in range(num_actions): # Actions are 0, 1, ..., num_actions-1
            q_key = (state, action)
            value = self.q_table.get(q_key, 0) # Default to 0 if state-action not seen
            if value > best_value:
                best_value = value
                best_action_idx = action
        
        return best_action_idx
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on recent performance"""
        if not self.completed_tasks:
            return 0
        
        recent_task = self.completed_tasks[-1]
        wait_ratio = 0
        if recent_task.service_time > 0:
            wait_ratio = recent_task.waiting_time / recent_task.service_time
        else: # Avoid division by zero
            wait_ratio = 0 if recent_task.waiting_time == 0 else float('inf')

        if wait_ratio < 0.5: return 1.0
        elif wait_ratio < 1.0: return 0.5
        elif wait_ratio < 2.0: return -0.5
        else: return -1.0
    
    def _update_q_value(self, state: tuple, action: int, reward: float, next_state: tuple):
        """Update Q-value using Q-learning formula"""
        q_key = (state, action)
        old_value = self.q_table.get(q_key, 0)
        
        # Get max Q-value for next state (considering up to 3 actions)
        next_max_q = float('-inf') # Initialize with very small number
        # Number of actions possible in next_state depends on queue length at that time.
        # For simplicity, let's assume up to 3 actions can be taken from any state if queue allows.
        # This might need refinement if next_state implies fewer than 3 tasks.
        # Max over actions available from next_state. The plan uses range(3).
        possible_next_actions = 3 # As per plan's loop for next_q_key
        
        for a_next in range(possible_next_actions):
            next_q_key = (next_state, a_next)
            next_max_q = max(next_max_q, self.q_table.get(next_q_key, 0)) # Default to 0 if not seen
        
        if next_max_q == float('-inf'): # If no actions were possible or no Q-values found
            next_max_q = 0

        new_value = old_value + self.alpha * (reward + self.gamma * next_max_q - old_value)
        self.q_table[q_key] = new_value
    
    def _sort_by_q_values(self):
        """Sort ready queue based on Q-values (first few elements)"""
        if not self.ready_queue or not self.q_table:
            return

        state = self._get_state()
        
        num_sortable_tasks = min(len(self.ready_queue), 3)
        if num_sortable_tasks <= 1: # No need to sort if 0 or 1 task to consider
            return

        # Get scores for the first 'num_sortable_tasks' tasks if they were chosen
        task_scores = []
        for i in range(num_sortable_tasks):
            task = self.ready_queue[i] # The task itself
            q_key = (state, i) # (state, action_index_if_chosen)
            score = self.q_table.get(q_key, 0) # Default to 0
            task_scores.append({'task': task, 'score': score, 'original_index': i})
        
        # Sort these tasks by score
        task_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Reorder the beginning of the ready_queue
        # This is tricky with deque. Convert to list, modify, convert back.
        current_tasks_list = list(self.ready_queue)
        
        # Extract the elements that were scored and the rest of the queue
        sorted_first_part = [item['task'] for item in task_scores]
        remaining_part = current_tasks_list[num_sortable_tasks:]
        
        # Check if the sorted part actually changed order relative to original indices of considered tasks
        original_first_part_tasks = current_tasks_list[:num_sortable_tasks]

        # Only reorder if the actual task objects in sorted_first_part are different in order
        # from original_first_part_tasks
        needs_reordering = False
        if len(sorted_first_part) == len(original_first_part_tasks):
            for i in range(len(sorted_first_part)):
                if sorted_first_part[i] is not original_first_part_tasks[i]: # Compare object identity
                    needs_reordering = True
                    break
        else: # Should not happen if logic is correct
            needs_reordering = True 

        if needs_reordering:
            self.ready_queue = deque(sorted_first_part + remaining_part)


class Simulator:
    """Main simulation engine"""
    
    def __init__(self, scheduler: BaseScheduler, num_processors: int = 1):
        self.scheduler = scheduler
        self.num_processors = num_processors
        self.processors: List[Optional[Task]] = [None] * num_processors
        self.processor_locks = [threading.Lock() for _ in range(num_processors)]
        self.simulation_time = 0.0 # Ensure float
        self.max_simulation_time = 2000.0 # Ensure float
        self.time_step = 0.1 # Ensure float
        self.running = False
        # Added from plan for Part 2.4
        self.processor_metrics: Dict[str, Any] = {
            'utilization': [[] for _ in range(num_processors)], 
            'task_count': [0] * num_processors,
            'total_service_time': [0.0] * num_processors
        }
        
    def find_idle_processor(self) -> Optional[int]:
        """Find the best processor for load balancing"""
        if self.num_processors == 1:
            return 0 if self.processors[0] is None else None
        
        idle_processors = [i for i in range(self.num_processors) if self.processors[i] is None]
        
        if not idle_processors:
            return None
        
        # If scheduler has a ready queue, consider the next task's characteristics
        if hasattr(self.scheduler, 'ready_queue') and self.scheduler.ready_queue:
            # Peek at next task
            if isinstance(self.scheduler.ready_queue, deque):
                next_task = self.scheduler.ready_queue[0] if self.scheduler.ready_queue else None
            else:
                next_task = self.scheduler.ready_queue[0] if self.scheduler.ready_queue else None
            
            if next_task:
                # For high priority tasks, use least loaded processor
                if next_task.priority == 1:
                    return min(idle_processors, 
                            key=lambda i: self.processor_metrics['total_service_time'][i])
                # For low priority tasks, use round-robin approach
                elif next_task.priority == 3:
                    # Simple round-robin based on task count
                    return min(idle_processors,
                            key=lambda i: self.processor_metrics['task_count'][i])
        
        # Default: least loaded processor
        return min(idle_processors, 
                key=lambda i: self.processor_metrics['total_service_time'][i])

    def execute_task_on_processor(self, task: Task, processor_id: int):
        self.processor_metrics['task_count'][processor_id] += 1

        with self.processor_locks[processor_id]:
            if self.processors[processor_id] is not None:
                logger.warning(f"Processor {processor_id} was not idle when trying to assign {task.task_id}")
                return

            self.processors[processor_id] = task
            task.state = TaskState.EXECUTING
            
            if task.start_time is None:
                task.start_time = self.simulation_time
            
            # Check for priority inversion before execution
            if isinstance(self.scheduler, PriorityScheduler) and self.scheduler.ready_queue:
                # Check if any higher priority task is waiting
                for waiting_task in self.scheduler.ready_queue:
                    if waiting_task.priority < task.priority:  # Higher priority is waiting
                        # This is a priority inversion scenario
                        self.scheduler.handle_priority_inversion(waiting_task, task)
                        break
            
            # Simulate task execution
            while task.remaining_service_time > 0 and self.running:
                time.sleep(self.time_step * 0.01)
                task.remaining_service_time -= self.time_step
                task.remaining_service_time = max(0, task.remaining_service_time)

                if isinstance(self.scheduler, PriorityScheduler):
                    if self.check_preemption_needed(task, processor_id):
                        task.state = TaskState.PREEMPTED
                        task.preemption_count += 1
                        self.processors[processor_id] = None
                        self.scheduler.add_task(task)
                        logger.debug(f"Task {task.task_id} preempted on P{processor_id} at {self.simulation_time:.2f}")
                        return
            
            if not self.running and task.remaining_service_time > 0:
                logger.info(f"Task {task.task_id} stopped mid-execution as simulation ended.")
                self.processors[processor_id] = None
                return

            if task.remaining_service_time <= 0:
                task.completion_time = self.simulation_time
                self.processor_metrics['total_service_time'][processor_id] += task.service_time
                
                task.state = TaskState.COMPLETED
                self.scheduler.completed_tasks.append(task)
                logger.debug(f"Task {task.task_id} completed on P{processor_id} at {self.simulation_time:.2f}")

                if isinstance(self.scheduler, PriorityScheduler):
                    self.scheduler.restore_priority(task)
            
            self.processors[processor_id] = None
    
    # Modified check_preemption_needed, added processor_id for context (though not used by original logic)
    def check_preemption_needed(self, current_task: Task, processor_id: int) -> bool:
        if not self.scheduler.ready_queue: # Check the central ready queue
            return False
        
        # This logic is simplistic: it just looks at the head of the ready queue.
        # A more robust preemption check might involve iterating or specific scheduler logic.
        # Original code: highest_priority_waiting = min(self.scheduler.ready_queue, key=lambda t: t.priority)
        # This assumes ready_queue is sortable or min works as expected.
        # If ready_queue is deque, min() will work but might not be sorted if scheduler is not always sorting.
        # For PriorityScheduler, add_task keeps it sorted.
        
        # Peek at the next task from the scheduler IF the scheduler supports peeking or if we get it
        # This is tricky as get_next_task() usually removes the task.
        # For PriorityScheduler, the ready_queue[0] after sorting is the highest priority.
        if isinstance(self.scheduler.ready_queue, list) and self.scheduler.ready_queue:
            if not self.scheduler.ready_queue: return False
            # Need to ensure it's sorted for this to be highest priority
            # PriorityScheduler sorts on add.
            # Let's assume it's sorted if it's PriorityScheduler
            highest_priority_waiting_task = self.scheduler.ready_queue[0]
            return highest_priority_waiting_task.priority < current_task.priority
        elif isinstance(self.scheduler.ready_queue, deque) and self.scheduler.ready_queue:
            # For deque, [0] is peek. Again, assumes sorted.
            highest_priority_waiting_task = self.scheduler.ready_queue[0]
            return highest_priority_waiting_task.priority < current_task.priority
            
        return False # Default no preemption
    
    def check_for_priority_inversions(self):
        """Check for priority inversions when high priority tasks arrive"""
        if not isinstance(self.scheduler, PriorityScheduler):
            return
        
        # Get all executing tasks
        executing_tasks = []
        for i, task in enumerate(self.processors):
            if task is not None:
                executing_tasks.append((i, task))
        
        if not executing_tasks or not self.scheduler.ready_queue:
            return
        
        # Check each waiting high-priority task
        for waiting_task in self.scheduler.ready_queue:
            if waiting_task.priority == HIGH_PRIORITY:  # High priority waiting
                # Check if any lower priority task is executing
                for proc_id, exec_task in executing_tasks:
                    if exec_task.priority > waiting_task.priority:  # Lower priority executing
                        # Priority inversion detected
                        self.scheduler.handle_priority_inversion(waiting_task, exec_task)
                        logger.info(f"Priority inversion: {waiting_task.task_id} (P{waiting_task.priority}) blocked by {exec_task.task_id} (P{exec_task.priority})")

    def run(self, tasks: List[Task]):
        self.running = True
        pending_tasks = deque(sorted(tasks, key=lambda t: t.arrival_time))
        
        active_threads = []
        
        while self.running:
            if not pending_tasks and not self.scheduler.ready_queue and all(p is None for p in self.processors):
                logger.info("All tasks processed and processors idle.")
                break
            
            # Add arrived tasks to scheduler
            tasks_added_this_cycle = []
            while pending_tasks and pending_tasks[0].arrival_time <= self.simulation_time:
                task_to_add = pending_tasks.popleft()
                self.scheduler.add_task(task_to_add)
                tasks_added_this_cycle.append(task_to_add)
                logger.debug(f"{task_to_add.task_id} arrived at {self.simulation_time:.2f}")
            
            # Check for priority inversions after new tasks arrive
            if tasks_added_this_cycle:
                self.check_for_priority_inversions()
            
            # Schedule tasks on idle processors
            processor_id = self.find_idle_processor()
            while processor_id is not None:
                next_task_to_run = self.scheduler.get_next_task()
                if next_task_to_run is None:
                    break
                
                logger.debug(f"Assigning {next_task_to_run.task_id} to P{processor_id}")
                thread = threading.Thread(
                    target=self.execute_task_on_processor,
                    args=(next_task_to_run, processor_id)
                )
                active_threads.append(thread)
                thread.start()
                
                processor_id = self.find_idle_processor()
            
            # Record metrics
            self.scheduler.current_time = self.simulation_time
            self.scheduler.record_metrics()
            
            # Advance simulation time
            self.simulation_time += self.time_step
            self.simulation_time = round(self.simulation_time, 2)
            
            # Clean up finished threads
            active_threads = [t for t in active_threads if t.is_alive()]
            
            if self.simulation_time > self.max_simulation_time:
                logger.warning(f"Simulation time limit ({self.max_simulation_time}s) reached.")
                self.running = False
                break
            
            time.sleep(self.time_step * 0.001)
        
        self.running = False
        
        logger.info("Waiting for active threads to complete...")
        for thread in active_threads:
            thread.join(timeout=1.0)
            if thread.is_alive():
                logger.warning(f"Thread did not complete in time.")
        
        logger.info(f"Simulation completed. Total tasks: {len(self.scheduler.completed_tasks)}")

    # Added from plan for Part 2.4
    def get_processor_balance_metrics(self) -> Dict:
        """Calculate load balancing metrics across processors"""
        if self.num_processors == 1:
            return {'balanced': True, 'variance': 0.0, 'task_count_variance': 0.0, 'utilization_variance': 0.0} # Added more keys
        
        task_counts = self.processor_metrics['task_count']
        # avg_tasks = np.mean(task_counts) # Not used in return dict by plan
        variance = np.var(task_counts) if task_counts else 0.0
        
        utilizations = []
        for proc_id in range(self.num_processors):
            # Simulation time might be 0 if sim didn't run or very short.
            sim_time_for_util = self.simulation_time
            if sim_time_for_util <= 0: # Avoid division by zero if sim time is 0 or negative
                # This can happen if no tasks were run or sim ended prematurely.
                # Fallback to a small positive number or handle as 0% utilization.
                # If max_simulation_time was used to calculate utilization instead:
                # sim_time_for_util = self.max_simulation_time # Or total time tasks were available
                # For now, if actual simulation_time is 0, util is 0.
                 utilizations.append(0.0)
                 continue

            util = (self.processor_metrics['total_service_time'][proc_id] / 
                   sim_time_for_util) * 100
            utilizations.append(util)
    
        util_variance = np.var(utilizations) if utilizations else 0.0
        balance_score = 0.0
        if (1.0 + variance) > 0 : # Avoid division by zero
            balance_score = 1.0 / (1.0 + variance)

        return {
            'task_count_per_processor': task_counts,
            'task_count_variance': variance,
            'utilization_per_processor': utilizations,
            'utilization_variance': util_variance,
            'balance_score': balance_score 
        }

class Visualizer: # Original Visualizer, plan doesn't modify this directly
    @staticmethod
    def plot_gantt_chart(tasks: List[Task], title: str):
        fig, ax = plt.subplots(figsize=(12, 8))
        tasks_sorted = sorted([t for t in tasks if t.start_time is not None and t.completion_time is not None], 
                            key=lambda t: t.start_time)
        
        colors = {1: 'red', 2: 'gold', 3: 'green'} # yellow -> gold for better visibility
        priority_names = {p: TASK_SPECS[p]['name'] for p in TASK_SPECS}
        
        # Create legend handles manually if not all priorities appear
        legend_handles = [plt.Rectangle((0,0),1,1, color=colors[p], label=priority_names[p]) for p in sorted(colors.keys())]

        for i, task in enumerate(tasks_sorted):
            # Ensure start_time and completion_time are valid
            if task.start_time is not None and task.completion_time is not None and task.completion_time > task.start_time:
                 ax.barh(i, task.completion_time - task.start_time, 
                       left=task.start_time, height=0.8,
                       color=colors.get(task.priority, 'gray')) # Default color for unknown prio
                
                 ax.text(task.start_time + (task.completion_time - task.start_time) / 2, 
                       i, task.task_id, ha='center', va='center', fontsize=8, color='black')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Tasks (ordered by start time)') # More descriptive ylabel
        ax.set_yticks([]) # Hide y-axis task numbers if too many tasks
        ax.set_title(f'Gantt Chart - {title}')
        ax.legend(handles=legend_handles, title="Priority")
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_metrics(scheduler: BaseScheduler):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        if scheduler.metrics['timestamps'] and scheduler.metrics['queue_lengths']:
            axes[0, 0].plot(scheduler.metrics['timestamps'], scheduler.metrics['queue_lengths'])
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Queue Length')
        axes[0, 0].set_title('Queue Length Over Time')
        axes[0, 0].grid(True)
        
        if scheduler.metrics['timestamps'] and scheduler.metrics['memory_usage']:
            axes[0, 1].plot(scheduler.metrics['timestamps'], scheduler.metrics['memory_usage'])
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Memory Usage (MB)')
        axes[0, 1].set_title('Memory Usage Over Time')
        axes[0, 1].grid(True)
        
        if scheduler.metrics['timestamps'] and scheduler.metrics['cpu_usage']:
            axes[1, 0].plot(scheduler.metrics['timestamps'], scheduler.metrics['cpu_usage'])
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('CPU Usage (%)')
        axes[1, 0].set_title('CPU Usage Over Time')
        axes[1, 0].grid(True)
        
        completion_times = {1: [], 2: [], 3: []}
        for task in scheduler.completed_tasks:
            if task.completion_time and task.priority in completion_times: # Ensure priority is valid
                completion_times[task.priority].append(task.turnaround_time)
        
        priority_names = [TASK_SPECS[p]['name'] for p in sorted(completion_times.keys())]
        avg_times = [np.mean(completion_times[p]) if completion_times[p] else 0 
                    for p in sorted(completion_times.keys())]
        
        bar_colors = ['red', 'gold', 'green'] # Match Gantt
        axes[1, 1].bar(priority_names, avg_times, color=bar_colors[:len(priority_names)])
        axes[1, 1].set_xlabel('Priority')
        axes[1, 1].set_ylabel('Average Turnaround Time (s)')
        axes[1, 1].set_title('Average Turnaround Time by Priority')
        axes[1, 1].grid(True, axis='y')
        
        plt.suptitle(f'Performance Metrics - {scheduler.name}')
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
        return fig
    
    @staticmethod
    def generate_report(schedulers_results: Dict[str, Dict]):
        report = "# Real-Time Task Scheduling Analysis Report\n\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        report += "## Summary Statistics\n\n"
        
        for name, results_data in schedulers_results.items(): # Renamed results to results_data
            scheduler = results_data['scheduler']
            report += f"### {name}\n"
            num_completed = len(scheduler.completed_tasks)
            report += f"- Total tasks completed: {num_completed}\n"
            
            if num_completed > 0:
                avg_waiting = np.mean([t.waiting_time for t in scheduler.completed_tasks if t.waiting_time is not None])
                avg_turnaround = np.mean([t.turnaround_time for t in scheduler.completed_tasks if t.turnaround_time is not None])
                max_queue_length = max(scheduler.metrics['queue_lengths']) if scheduler.metrics['queue_lengths'] else 0
                avg_memory = np.mean(scheduler.metrics['memory_usage']) if scheduler.metrics['memory_usage'] else 0
                
                report += f"- Average waiting time: {avg_waiting:.2f}s\n"
                report += f"- Average turnaround time: {avg_turnaround:.2f}s\n"
                report += f"- Maximum queue length: {max_queue_length}\n"
                report += f"- Average memory usage: {avg_memory:.2f}MB\n"
            else:
                report += "- No tasks completed to calculate statistics.\n"
            report += "\n"
        
        report += "## Priority-Based Analysis\n\n"
        
        for name, results_data in schedulers_results.items():
            scheduler = results_data['scheduler']
            report += f"### {name}\n"
            
            for priority_val in sorted(TASK_SPECS.keys()): # Renamed priority to priority_val
                priority_tasks = [t for t in scheduler.completed_tasks if t.priority == priority_val]
                if priority_tasks:
                    avg_wait = np.mean([t.waiting_time for t in priority_tasks if t.waiting_time is not None])
                    avg_turn = np.mean([t.turnaround_time for t in priority_tasks if t.turnaround_time is not None])
                    report += f"- Priority {TASK_SPECS[priority_val]['name']}:\n"
                    report += f"  - Tasks: {len(priority_tasks)}\n"
                    report += f"  - Avg waiting time: {avg_wait:.2f}s\n"
                    report += f"  - Avg turnaround time: {avg_turn:.2f}s\n"
            report += "\n"
            
            # Include Priority Inversion Report if applicable
            if isinstance(scheduler, PriorityScheduler):
                report += scheduler.get_priority_inversion_report() + "\n"

        return report

# Added ComprehensiveTestSuite class from plan (Part 2.1)
# Make sure it's placed before GUI or where it's instantiated.
class ComprehensiveTestSuite:
    """Automated comprehensive testing suite for all schedulers"""
    
    def __init__(self):
        self.test_scenarios = {
            'baseline': {
                'name': 'Baseline Test', 'seed': 42, 'processors': 1,
                'description': 'Standard configuration baseline'
            },
            'high_load': {
                'name': 'High Load Test', 'seed': 123, 'processors': 1, 
                'lambda_multiplier': 0.5, # Halves lambda_param, so higher arrival rate
                'description': 'Testing under high task arrival rate'
            },
            'multi_processor_2': {
                'name': 'Dual-Processor Test', 'seed': 42, 'processors': 2,
                'description': 'Testing dual-processor efficiency'
            },
            'multi_processor_4': {
                'name': 'Quad-Processor Test', 'seed': 42, 'processors': 4,
                'description': 'Testing quad-processor efficiency'
            },
            'priority_skew_high': {
                'name': 'High Priority Dominant', 'seed': 456, 'processors': 2,
                'priority_distribution': [0.7, 0.2, 0.1],
                'description': 'Testing with majority high-priority tasks'
            },
            'priority_skew_low': {
                'name': 'Low Priority Dominant', 'seed': 789, 'processors': 2,
                'priority_distribution': [0.1, 0.2, 0.7],
                'description': 'Testing with majority low-priority tasks'
            },
            'extreme_service_times': {
                'name': 'Extreme Service Times', 'seed': 111, 'processors': 1,
                'service_multiplier': 2.0,
                'description': 'Testing with extended service times'
            },
            'stress_test': { # Plan has 'task_count': 100, this overrides TASK_SPECS counts
                'name': 'Stress Test', 'seed': 999, 'processors': 1,
                'task_count': 100, # This means total tasks, not per priority.
                'description': 'Stress testing with 100 tasks'
            }
        }
        
        self.results: Dict[str, Dict[str, Any]] = {}
        self.statistical_results: Dict[str, Any] = {} # Type hint could be more specific
    
    def run_all_tests(self, progress_callback: Optional[Callable[[int, int, str], None]] = None) -> Dict:
        """Run all test scenarios for all schedulers"""
        scheduler_classes = {
            'FCFS': FCFSScheduler,
            'EDF': EDFScheduler,
            'Priority': PriorityScheduler,
            'ML-Linear': MLScheduler,
            'ML-DecisionTree': DecisionTreeScheduler,
            'ML-QLearning': RLScheduler
        }
        
        total_tests = len(self.test_scenarios) * len(scheduler_classes)
        current_test = 0
        
        for scenario_name, scenario_config in self.test_scenarios.items():
            self.results[scenario_name] = {}
            logger.info(f"Starting Test Scenario: {scenario_config['name']}")
            
            for sched_name, sched_class in scheduler_classes.items():
                current_test += 1
                if progress_callback:
                    progress_callback(current_test, total_tests, 
                                    f"Running {scenario_config['name']} with {sched_name}")
                logger.info(f"  Scheduler: {sched_name}")
                
                tasks_for_scenario = self._generate_scenario_tasks(scenario_config)
                
                if not tasks_for_scenario:
                    logger.warning(f"    No tasks generated for {scenario_config['name']}, skipping {sched_name}")
                    self.results[scenario_name][sched_name] = {
                        'scheduler': None, 'execution_time': 0, 
                        'completed_tasks': 0, 'metrics': {}, 'balance_metrics': {}
                    }
                    continue

                scheduler_instance = sched_class()
                num_processors = scenario_config.get('processors', 1)
                simulator = Simulator(scheduler_instance, num_processors)
                
                task_copies = [Task(
                    task_id=t.task_id, arrival_time=t.arrival_time,
                    service_time=t.service_time, priority=t.priority,
                    deadline=t.deadline 
                ) for t in tasks_for_scenario]
                
                start_time = time.time()
                simulator.run(task_copies)
                execution_time = time.time() - start_time
                
                # Pass num_processors to _calculate_detailed_metrics
                detailed_metrics = self._calculate_detailed_metrics(scheduler_instance, num_processors)
                
                self.results[scenario_name][sched_name] = {
                    'scheduler': scheduler_instance,
                    'execution_time': execution_time,
                    'completed_tasks': len(scheduler_instance.completed_tasks),
                    'metrics': detailed_metrics
                }

                if num_processors > 1:
                    balance_metrics = simulator.get_processor_balance_metrics()
                    self.results[scenario_name][sched_name]['balance_metrics'] = balance_metrics
                else:
                    self.results[scenario_name][sched_name]['balance_metrics'] = {}
        
        self._calculate_statistical_results()
        logger.info("Comprehensive test suite finished.")
        return self.results
    
    def _generate_scenario_tasks(self, scenario_config: Dict) -> List[Task]:
        seed = scenario_config.get('seed', int(time.time()))
        np.random.seed(seed)
        
        tasks = []
        task_counter = 1
        
        lambda_mult = scenario_config.get('lambda_multiplier', 1.0)
        service_mult = scenario_config.get('service_multiplier', 1.0)
        
        default_dist = [TASK_SPECS[HIGH_PRIORITY]['count'], TASK_SPECS[MEDIUM_PRIORITY]['count'], TASK_SPECS[LOW_PRIORITY]['count']]
        total_default_tasks = sum(default_dist)
        default_dist_ratios = [c/total_default_tasks for c in default_dist]

        priority_dist_ratios = scenario_config.get('priority_distribution', default_dist_ratios)
        
        scenario_total_tasks = scenario_config.get('task_count', sum(spec['count'] for spec in TASK_SPECS.values()))

        # Calculate tasks per priority based on ratios and scenario_total_tasks
        task_counts_by_priority: Dict[int, int] = {}
        
        if len(priority_dist_ratios) == 3:
            task_counts_by_priority[HIGH_PRIORITY] = int(scenario_total_tasks * priority_dist_ratios[0])
            task_counts_by_priority[MEDIUM_PRIORITY] = int(scenario_total_tasks * priority_dist_ratios[1])
            task_counts_by_priority[LOW_PRIORITY] = scenario_total_tasks - (task_counts_by_priority[HIGH_PRIORITY] + task_counts_by_priority[MEDIUM_PRIORITY])
        else:
            logger.warning("Priority distribution in scenario has incorrect length. Using default counts.")
            for p, spec in TASK_SPECS.items():
                task_counts_by_priority[p] = spec['count']
        
        # Ensure all priority levels have at least one task
        for priority_val in [HIGH_PRIORITY, MEDIUM_PRIORITY, LOW_PRIORITY]:
            if priority_val not in task_counts_by_priority or task_counts_by_priority[priority_val] <= 0:
                task_counts_by_priority[priority_val] = max(1, int(scenario_total_tasks * 0.1))
        
        # Adjust to ensure total matches
        total_assigned = sum(task_counts_by_priority.values())
        if total_assigned != scenario_total_tasks:
            # Adjust the largest group
            largest_priority = max(task_counts_by_priority, key=task_counts_by_priority.get)
            task_counts_by_priority[largest_priority] += (scenario_total_tasks - total_assigned)

        for priority_val, count in task_counts_by_priority.items():
            if count <= 0: continue
                
            spec = TASK_SPECS[priority_val]
            current_arrival_time_offset = 0.0
            
            for _ in range(count):
                inter_arrival = np.random.exponential(spec['lambda_param'] * lambda_mult) 
                current_arrival_time_offset += inter_arrival
                
                service_range = spec['service_range']
                service_time = np.random.uniform(
                    service_range[0] * service_mult,
                    service_range[1] * service_mult
                )
                service_time = max(0.01, service_time)

                task = Task(
                    task_id=f"TaskGen_{task_counter}",
                    arrival_time=current_arrival_time_offset,
                    service_time=service_time,
                    priority=priority_val
                )
                tasks.append(task)
                task_counter += 1
        
        tasks.sort(key=lambda t: t.arrival_time)
        return tasks

    def _calculate_detailed_metrics(self, scheduler: BaseScheduler, num_processors: int = 1) -> Dict:
        if not scheduler.completed_tasks:
            return {'error': 'No completed tasks to calculate metrics.'}
        
        # Basic metrics extraction
        waiting_times = [t.waiting_time for t in scheduler.completed_tasks if t.waiting_time is not None]
        turnaround_times = [t.turnaround_time for t in scheduler.completed_tasks if t.turnaround_time is not None]
        response_times = [t.response_time for t in scheduler.completed_tasks if t.response_time is not None]

        # Handle cases where lists might be empty after filtering Nones
        if not waiting_times: waiting_times = [0.0]
        if not turnaround_times: turnaround_times = [0.0]
        if not response_times: response_times = [0.0]

        priority_metrics = {}
        for priority_val in sorted(TASK_SPECS.keys()):
            priority_tasks = [t for t in scheduler.completed_tasks if t.priority == priority_val]
            if priority_tasks:
                p_waiting = [t.waiting_time for t in priority_tasks if t.waiting_time is not None]
                p_turnaround = [t.turnaround_time for t in priority_tasks if t.turnaround_time is not None]
                
                if not p_waiting: p_waiting = [0.0]
                if not p_turnaround: p_turnaround = [0.0]

                priority_metrics[priority_val] = {
                    'count': len(priority_tasks),
                    'avg_waiting': np.mean(p_waiting), 'std_waiting': np.std(p_waiting),
                    'p50_waiting': np.percentile(p_waiting, 50),
                    'p90_waiting': np.percentile(p_waiting, 90),
                    'p99_waiting': np.percentile(p_waiting, 99),
                    'avg_turnaround': np.mean(p_turnaround),
                }
        
        total_service_time = sum(t.service_time for t in scheduler.completed_tasks if t.service_time is not None)
        
        effective_total_time = 1.0
        if scheduler.completed_tasks:
            max_comp_time = max(t.completion_time for t in scheduler.completed_tasks if t.completion_time is not None)
            if max_comp_time is not None:
                effective_total_time = max(1.0, max_comp_time)
        
        if hasattr(scheduler, 'current_time') and scheduler.current_time > effective_total_time:
            effective_total_time = scheduler.current_time

        # Fixed CPU utilization calculation
        cpu_util = 0
        if effective_total_time > 0:
            # For multi-processor systems, max possible service time is time * processors
            max_possible_service_time = effective_total_time * num_processors
            cpu_util = (total_service_time / max_possible_service_time) * 100
            cpu_util = min(cpu_util, 100.0)

        metrics = {
            'avg_waiting': np.mean(waiting_times), 'std_waiting': np.std(waiting_times),
            'p50_waiting': np.percentile(waiting_times, 50),
            'p90_waiting': np.percentile(waiting_times, 90),
            'p99_waiting': np.percentile(waiting_times, 99),
            'max_waiting': max(waiting_times) if waiting_times else 0,
            'min_waiting': min(waiting_times) if waiting_times else 0,

            'avg_turnaround': np.mean(turnaround_times), 'std_turnaround': np.std(turnaround_times),
            'p50_turnaround': np.percentile(turnaround_times, 50),
            'p90_turnaround': np.percentile(turnaround_times, 90),
            'p99_turnaround': np.percentile(turnaround_times, 99),
            
            'avg_response': np.mean(response_times),
            
            'cpu_utilization': cpu_util,
            'throughput': len(scheduler.completed_tasks) / effective_total_time if effective_total_time > 0 else 0.0,
            
            'max_queue_length': max(scheduler.metrics['queue_lengths']) if scheduler.metrics['queue_lengths'] else 0,
            'avg_queue_length': np.mean(scheduler.metrics['queue_lengths']) if scheduler.metrics['queue_lengths'] else 0,
            'priority_metrics': priority_metrics
        }
        
        n = len(waiting_times)
        if n > 1:
            std_err_waiting = np.std(waiting_times) / np.sqrt(n)
            metrics['ci_95_waiting'] = 1.96 * std_err_waiting
            
            if len(turnaround_times) > 1:
                std_err_turnaround = np.std(turnaround_times) / np.sqrt(len(turnaround_times))
                metrics['ci_95_turnaround'] = 1.96 * std_err_turnaround
            else:
                metrics['ci_95_turnaround'] = 0.0
        else:
            metrics['ci_95_waiting'] = 0.0
            metrics['ci_95_turnaround'] = 0.0
            
        return metrics

    def _calculate_statistical_results(self):
        try:
            from scipy import stats
        except ImportError:
            logger.warning("scipy not available, skipping statistical t-tests.")
            self.statistical_results = {"error": "scipy not found"}
            return
        
        self.statistical_results = {}
        
        for scenario_name, scenario_data in self.results.items():
            scenario_stats = {}
            scheduler_names = list(scenario_data.keys())
            
            if len(scheduler_names) < 2: continue
            
            for i, sched1_name in enumerate(scheduler_names):
                for sched2_name in scheduler_names[i+1:]:
                    res1 = scenario_data.get(sched1_name)
                    res2 = scenario_data.get(sched2_name)

                    if not res1 or not res2 or \
                    not res1.get('scheduler') or not res2.get('scheduler'):
                        logger.warning(f"Skipping t-test between {sched1_name} and {sched2_name} due to missing data in {scenario_name}")
                        continue

                    tasks1_completed = res1['scheduler'].completed_tasks
                    tasks2_completed = res2['scheduler'].completed_tasks
                    
                    if tasks1_completed and tasks2_completed:
                        waiting1 = [t.waiting_time for t in tasks1_completed if t.waiting_time is not None]
                        waiting2 = [t.waiting_time for t in tasks2_completed if t.waiting_time is not None]
                        
                        if len(waiting1) > 1 and len(waiting2) > 1:
                            # Check for variance - if all values are the same, skip t-test
                            if np.var(waiting1) == 0 and np.var(waiting2) == 0:
                                if np.mean(waiting1) == np.mean(waiting2):
                                    scenario_stats[f"{sched1_name}_vs_{sched2_name}"] = {
                                        't_statistic': 0, 'p_value': 1.0,
                                        'significant': False,
                                        'mean_diff': 0
                                    }
                                else:
                                    # Different means but no variance
                                    scenario_stats[f"{sched1_name}_vs_{sched2_name}"] = {
                                        't_statistic': float('inf'), 'p_value': 0,
                                        'significant': True,
                                        'mean_diff': np.mean(waiting1) - np.mean(waiting2)
                                    }
                                continue
                            
                            # Perform t-test
                            t_stat, p_value = stats.ttest_ind(waiting1, waiting2, equal_var=False, nan_policy='omit')
                            
                            comparison_key = f"{sched1_name}_vs_{sched2_name}"
                            scenario_stats[comparison_key] = {
                                't_statistic': t_stat, 'p_value': p_value,
                                'significant': p_value < 0.05,
                                'mean_diff': np.mean(waiting1) - np.mean(waiting2)
                            }
                        else:
                            logger.info(f"Not enough data for t-test between {sched1_name} and {sched2_name} in {scenario_name}")
            
            self.statistical_results[scenario_name] = scenario_stats

    def generate_comprehensive_report(self) -> str:
        report = f"# Comprehensive Testing Suite Results\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        report += "## Executive Summary\n"
        best_performers = self._find_best_performers()
        if best_performers:
            report += "### Best Performers by Metric (Baseline Scenario):\n" # Clarify scenario
            for metric, performer_info in best_performers.items(): # Renamed performer to performer_info
                if performer_info and performer_info.get('scheduler'):
                     report += f"- **{metric}**: {performer_info['scheduler']} ({performer_info['value']:.3f})\n"
                else:
                     report += f"- **{metric}**: Not available\n"
        else:
            report += "No baseline results to determine best performers.\n"
        report += "\n"

        report += "## Detailed Scenario Results\n"
        for scenario_name_key, scenario_results_data in self.results.items(): # Renamed scenario_name, scenario_results
            scenario_info_dict = self.test_scenarios.get(scenario_name_key, {}) # Renamed scenario_info
            report += f"\n### Scenario: {scenario_info_dict.get('name', scenario_name_key)}\n"
            report += f"*{scenario_info_dict.get('description', 'N/A')}*\n"
            report += f"*Processors: {scenario_info_dict.get('processors', 'N/A')}*\n\n"
            
            # Comparison table header
            report += "| Scheduler         | Avg Wait (s) | P90 Wait (s) | P99 Wait (s) | CPU Util (%) | Throughput (t/s) |\n"
            report += "|-------------------|--------------|--------------|--------------|--------------|------------------|\n"
            
            for sched_name_key, result_data in scenario_results_data.items(): # Renamed sched_name, result
                metrics = result_data.get('metrics', {})
                if not metrics or metrics.get('error'): # Handle case where metrics might be empty or error
                    report += f"| {sched_name_key:<17} | N/A          | N/A          | N/A          | N/A          | N/A              |\n"
                    continue

                avg_w = metrics.get('avg_waiting', 0)
                p90_w = metrics.get('p90_waiting', 0)
                p99_w = metrics.get('p99_waiting', 0)
                cpu_u = metrics.get('cpu_utilization', 0)
                thru  = metrics.get('throughput', 0)
                report += (f"| {sched_name_key:<17} | {avg_w:12.2f} | {p90_w:12.2f} | {p99_w:12.2f} | "
                           f"{cpu_u:12.1f} | {thru:16.2f} |\n")

            # Statistical significance (if available for this scenario)
            if scenario_name_key in self.statistical_results and self.statistical_results[scenario_name_key]:
                report += "\n**Statistical Significance (p < 0.05 for waiting time difference):**\n"
                for comparison, stats_data in self.statistical_results[scenario_name_key].items(): # Renamed stats to stats_data
                    if stats_data.get('significant'):
                        report += f"- {comparison}: p={stats_data['p_value']:.4f}, "
                        report += f"mean diff={stats_data['mean_diff']:.2f}s\n"
            report += "\n"
        
        report += "## Priority-Specific Performance\n"
        # Scenarios to analyze for priority performance
        priority_analysis_scenarios = ['baseline', 'priority_skew_high', 'priority_skew_low']
        for scenario_name_key in priority_analysis_scenarios:
            if scenario_name_key not in self.results: continue
                
            scenario_info_dict = self.test_scenarios.get(scenario_name_key, {})
            report += f"\n### Scenario: {scenario_info_dict.get('name', scenario_name_key)}\n"
            
            for prio_val in sorted(TASK_SPECS.keys()): # Renamed priority to prio_val
                priority_name_str = TASK_SPECS[prio_val].get('name', f'Priority {prio_val}') # Renamed priority_name
                report += f"\n**{priority_name_str} Priority Tasks:**\n"
                
                scenario_data = self.results.get(scenario_name_key, {})
                for sched_name_key, result_data in scenario_data.items():
                    priority_metrics_dict = result_data.get('metrics', {}).get('priority_metrics', {}).get(prio_val)
                    if priority_metrics_dict:
                        avg_w = priority_metrics_dict.get('avg_waiting', 0)
                        p90_w = priority_metrics_dict.get('p90_waiting', 0)
                        report += f"- {sched_name_key}: avg_wait={avg_w:.2f}s, p90_wait={p90_w:.2f}s\n"
            report += "\n"
        
        # Add processor comparison analysis from its own method
        report += self.generate_processor_comparison_report()
        
        # Include overall Priority Inversion Summary if any PriorityScheduler was run
        # This is a general summary, not per-scenario for this report section
        # Find if any PriorityScheduler was run and has events
        priority_inversion_summary = ""
        for scenario_data_val in self.results.values():
            for result_data_val in scenario_data_val.values():
                scheduler_inst = result_data_val.get('scheduler')
                if isinstance(scheduler_inst, PriorityScheduler) and scheduler_inst.priority_inversion_stats['count'] > 0:
                    if not priority_inversion_summary: # Add header once
                        priority_inversion_summary += "\n## Overall Priority Inversion Notes (from all scenarios):\n"
                    priority_inversion_summary += f"\n### Priority Inversion for {result_data_val.get('scheduler_name', scheduler_inst.name)} in a scenario:\n" # Need scheduler name better
                    priority_inversion_summary += scheduler_inst.get_priority_inversion_report() + "\n"
        if priority_inversion_summary:
             report += priority_inversion_summary

        return report
    
    def _find_best_performers(self) -> Dict[str, Dict[str, Any]]:
        """Find best performing scheduler for key metrics in the baseline scenario."""
        best_performers: Dict[str, Dict[str, Any]] = {
            'Average Waiting Time': {'scheduler': None, 'value': float('inf')},
            'P99 Waiting Time':     {'scheduler': None, 'value': float('inf')},
            'CPU Utilization':      {'scheduler': None, 'value': float('-inf')}, # Higher is better
            'Throughput':           {'scheduler': None, 'value': float('-inf')}  # Higher is better
        }
        
        baseline_results = self.results.get('baseline')
        if not baseline_results:
            logger.warning("Baseline scenario results not found for determining best performers.")
            return best_performers # Return empty/initial if no baseline

        for sched_name, result_data in baseline_results.items(): # Renamed result to result_data
            metrics = result_data.get('metrics')
            if not metrics or metrics.get('error'): continue

            # Avg Waiting Time (lower is better)
            avg_wait = metrics.get('avg_waiting', float('inf'))
            if avg_wait < best_performers['Average Waiting Time']['value']:
                best_performers['Average Waiting Time'] = {'scheduler': sched_name, 'value': avg_wait}

            # P99 Waiting Time (lower is better)
            p99_wait = metrics.get('p99_waiting', float('inf'))
            if p99_wait < best_performers['P99 Waiting Time']['value']:
                best_performers['P99 Waiting Time'] = {'scheduler': sched_name, 'value': p99_wait}

            # CPU Utilization (higher is better)
            cpu_util = metrics.get('cpu_utilization', float('-inf'))
            if cpu_util > best_performers['CPU Utilization']['value']:
                best_performers['CPU Utilization'] = {'scheduler': sched_name, 'value': cpu_util}

            # Throughput (higher is better)
            throughput = metrics.get('throughput', float('-inf'))
            if throughput > best_performers['Throughput']['value']:
                best_performers['Throughput'] = {'scheduler': sched_name, 'value': throughput}
        
        return best_performers

    def generate_processor_comparison_report(self) -> str:
        """Generate detailed comparison between single and multi-processor performance."""
        report = "\n## Processor Scaling Analysis\n"
        
        processor_scenarios_map = {
            'baseline': 1,
            'multi_processor_2': 2,
            'multi_processor_4': 4
        }
        
        report += "### Performance Scaling by Number of Processors\n"
        
        schedulers_to_compare = ['FCFS', 'EDF', 'Priority', 'ML-Linear', 'ML-DecisionTree', 'ML-QLearning']

        for sched_name_key in schedulers_to_compare: 
            report += f"\n**{sched_name_key} Scheduler:**\n"
            report += "| Processors | Avg Wait (s) | Throughput (t/s) | CPU Util (%) | Speedup (vs 1P) | Efficiency (%) |\n"
            report += "|------------|--------------|------------------|--------------|-----------------|----------------|\n"
            
            baseline_throughput_val: Optional[float] = None 

            for scenario_key, num_procs_val in processor_scenarios_map.items(): 
                if scenario_key in self.results and sched_name_key in self.results[scenario_key]:
                    metrics_dict = self.results[scenario_key][sched_name_key].get('metrics', {}) 
                    if not metrics_dict or metrics_dict.get('error'):
                        report += f"| {num_procs_val:<10} | N/A          | N/A              | N/A          | N/A             | N/A            |\n"
                        continue

                    avg_w = metrics_dict.get('avg_waiting', 0)
                    thru = metrics_dict.get('throughput', 0)
                    cpu_u = metrics_dict.get('cpu_utilization', 0)
                    
                    balance_metrics_data = self.results[scenario_key][sched_name_key].get('balance_metrics', {})
                    if num_procs_val > 1 and balance_metrics_data and 'utilization_per_processor' in balance_metrics_data:
                        per_proc_utils = balance_metrics_data['utilization_per_processor']
                        if per_proc_utils: 
                             cpu_u = np.mean(per_proc_utils) 

                    speedup = 1.0
                    efficiency = 100.0 if num_procs_val == 1 else 0.0

                    if num_procs_val == 1: 
                        baseline_throughput_val = thru
                    
                    if baseline_throughput_val is not None and baseline_throughput_val > 0:
                        speedup = thru / baseline_throughput_val
                        if num_procs_val > 0:
                             efficiency = (speedup / num_procs_val) * 100
                    elif num_procs_val > 1 : 
                        speedup = 0.0
                        efficiency = 0.0
                    
                    report += (f"| {num_procs_val:<10} | {avg_w:12.2f} | {thru:16.2f} | "
                               f"{cpu_u:12.1f} | {speedup:15.2f}x | {efficiency:14.1f} |\n")
                else: 
                    report += f"| {num_procs_val:<10} | Data N/A     | Data N/A         | Data N/A     | Data N/A        | Data N/A       |\n"
            report += "\n"
        
        report += "### Load Balancing Analysis (for Multi-Processor Scenarios)\n"
        multi_proc_scenarios_keys = ['multi_processor_2', 'multi_processor_4'] 
        for scenario_key in multi_proc_scenarios_keys:
            if scenario_key not in self.results: continue
            
            num_procs_val = processor_scenarios_map.get(scenario_key)
            if num_procs_val is None: continue 
            
            report += f"\n**Scenario: {self.test_scenarios[scenario_key]['name']} ({num_procs_val} Processors):**\n"
            
            scenario_data = self.results[scenario_key]
            for sched_name_key, result_data in scenario_data.items(): 
                balance_metrics_dict = result_data.get('balance_metrics') 
                if balance_metrics_dict:
                    task_var = balance_metrics_dict.get('task_count_variance', 'N/A')
                    util_var = balance_metrics_dict.get('utilization_variance', 'N/A')
                    balance_s = balance_metrics_dict.get('balance_score', 'N/A')
                    tasks_per_p = balance_metrics_dict.get('task_count_per_processor', [])
                    utils_per_p = balance_metrics_dict.get('utilization_per_processor', [])

                    report += f"- **{sched_name_key}**:\n"
                    
                    # Corrected f-string formatting below
                    task_var_str = f"{task_var:.2f}" if isinstance(task_var, float) else str(task_var)
                    report += f"  - Task Count Variance: {task_var_str}\n"
                    
                    report += f"  - Tasks per Processor: {tasks_per_p}\n"
                    
                    util_var_str = f"{util_var:.2f}" if isinstance(util_var, float) else str(util_var)
                    report += f"  - Utilization Variance: {util_var_str}\n"
                    
                    report += f"  - Utilizations per Processor: [{', '.join([f'{u:.1f}%' for u in utils_per_p])}]\n"
                    
                    balance_s_str = f"{balance_s:.3f}" if isinstance(balance_s, float) else str(balance_s)
                    report += f"  - Balance Score (task count based): {balance_s_str}\n"
                else: 
                     if num_procs_val > 1:
                         report += f"- {sched_name_key}: Load balancing metrics not available.\n"
            report += "\n"
        
        return report

    def generate_processor_comparison_visualization(self, save_dir: str):
        """Generate visualization comparing single vs multi-processor performance."""
        try:
            import matplotlib.pyplot as plt # Local import as per plan
            # Apply a style for better looking plots
            plt.style.use('seaborn-v0_8-whitegrid')
        except ImportError:
            logger.warning("matplotlib not available, skipping processor comparison visualization.")
            return

        # Define scenarios and processor counts for these plots
        # These must match keys in self.test_scenarios and have 'processors' field.
        processor_plot_scenarios = ['baseline', 'multi_processor_2', 'multi_processor_4']
        processor_counts = [self.test_scenarios[s].get('processors') for s in processor_plot_scenarios]
        
        # Filter out None counts if any scenario was misconfigured
        valid_indices = [i for i, count in enumerate(processor_counts) if count is not None]
        processor_plot_scenarios = [processor_plot_scenarios[i] for i in valid_indices]
        processor_counts = [processor_counts[i] for i in valid_indices]

        if not processor_counts or len(processor_counts) < 2 : # Need at least 2 points to plot trend
            logger.warning("Not enough valid processor scenarios to generate comparison plots.")
            return

        # Schedulers to plot - typically the non-ML ones for clear scaling, or a subset
        # Plan: 'FCFS', 'EDF', 'Priority', 'ML-Linear'
        schedulers_for_plot = ['FCFS', 'EDF', 'Priority', 'ML-Linear'] 
        
        num_metrics_to_plot = 4 # Throughput, Avg Wait, Speedup, Efficiency
        fig, axes = plt.subplots(2, 2, figsize=(14, 12)) # Slightly larger figure
        axes = axes.flatten() # Flatten to 1D array for easier indexing

        plot_titles = ['Throughput Scaling', 'Avg Waiting Time vs Processors', 
                       'Speedup vs Processors', 'Parallel Efficiency']
        plot_ylabels = ['Throughput (tasks/s)', 'Average Waiting Time (s)', 
                        'Speedup (Factor)', 'Efficiency (%)']

        # Plot 1: Throughput scaling
        ax = axes[0]
        for sched_name in schedulers_for_plot:
            throughputs = []
            for scenario_key in processor_plot_scenarios:
                val = self.results.get(scenario_key, {}).get(sched_name, {}).get('metrics', {}).get('throughput')
                if val is not None: throughputs.append(val)
                else: throughputs.append(np.nan) # Use NaN for missing data
            if not all(np.isnan(throughputs)): # Only plot if some data exists
                 ax.plot(processor_counts, throughputs, marker='o', linestyle='-', label=sched_name)
        
        # Plot 2: Average waiting time
        ax = axes[1]
        for sched_name in schedulers_for_plot:
            wait_times = []
            for scenario_key in processor_plot_scenarios:
                val = self.results.get(scenario_key, {}).get(sched_name, {}).get('metrics', {}).get('avg_waiting')
                if val is not None: wait_times.append(val)
                else: wait_times.append(np.nan)
            if not all(np.isnan(wait_times)):
                 ax.plot(processor_counts, wait_times, marker='o', linestyle='-', label=sched_name)

        # Plot 3: Speedup (relative to 1-processor baseline of the same scheduler)
        ax = axes[2]
        for sched_name in schedulers_for_plot:
            speedups = []
            # Baseline throughput for this scheduler (from 1-processor scenario)
            baseline_scenario_key = processor_plot_scenarios[processor_counts.index(1)] if 1 in processor_counts else None
            
            baseline_throughput = np.nan
            if baseline_scenario_key:
                 baseline_throughput = self.results.get(baseline_scenario_key, {}).get(sched_name, {}).get('metrics', {}).get('throughput')
            
            if np.isnan(baseline_throughput) or baseline_throughput == 0: # Handle missing or zero baseline
                speedups = [np.nan] * len(processor_plot_scenarios)
            else:
                for scenario_key in processor_plot_scenarios:
                    current_throughput = self.results.get(scenario_key, {}).get(sched_name, {}).get('metrics', {}).get('throughput')
                    if current_throughput is not None:
                        speedups.append(current_throughput / baseline_throughput)
                    else:
                        speedups.append(np.nan)
            if not all(np.isnan(speedups)):
                ax.plot(processor_counts, speedups, marker='o', linestyle='-', label=sched_name)
        
        # Ideal speedup line for reference
        ax.plot(processor_counts, processor_counts, 'k--', label='Ideal Speedup', alpha=0.7)
        
        # Plot 4: Efficiency
        ax = axes[3]
        for sched_name in schedulers_for_plot:
            efficiencies = []
            baseline_scenario_key = processor_plot_scenarios[processor_counts.index(1)] if 1 in processor_counts else None
            baseline_throughput = np.nan
            if baseline_scenario_key:
                baseline_throughput = self.results.get(baseline_scenario_key, {}).get(sched_name, {}).get('metrics', {}).get('throughput')

            if np.isnan(baseline_throughput) or baseline_throughput == 0:
                efficiencies = [np.nan] * len(processor_plot_scenarios)
            else:
                for i, scenario_key in enumerate(processor_plot_scenarios):
                    num_procs = processor_counts[i]
                    current_throughput = self.results.get(scenario_key, {}).get(sched_name, {}).get('metrics', {}).get('throughput')
                    if current_throughput is not None and num_procs > 0:
                        speedup = current_throughput / baseline_throughput
                        efficiencies.append((speedup / num_procs) * 100)
                    else:
                        efficiencies.append(np.nan)
            if not all(np.isnan(efficiencies)):
                 ax.plot(processor_counts, efficiencies, marker='o', linestyle='-', label=sched_name)
        ax.set_ylim(0, 110) # Efficiency typically 0-100%, allow slight overshoot
        ax.axhline(100, color='k', linestyle=':', alpha=0.7, label='Ideal Efficiency (100%)')


        # Common formatting for all subplots
        for i, ax_curr in enumerate(axes):
            ax_curr.set_xlabel('Number of Processors')
            ax_curr.set_ylabel(plot_ylabels[i])
            ax_curr.set_title(plot_titles[i])
            ax_curr.legend(loc='best', fontsize='small')
            ax_curr.grid(True, linestyle='--', alpha=0.7)
            ax_curr.set_xticks(processor_counts) # Ensure ticks are at 1, 2, 4

        fig.suptitle('Processor Scaling Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout for suptitle
        
        # Ensure save_dir exists
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'processor_comparison.png')
        try:
            plt.savefig(save_path)
            logger.info(f"Processor comparison visualization saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save processor comparison plot: {e}")
        plt.close(fig) # Close the figure to free memory

    # 1. Add the visualization methods to the ComprehensiveTestSuite class (after generate_processor_comparison_visualization method):

    def generate_comprehensive_visualizations(self, save_dir: str):
        """Generate all visualizations for comprehensive test results"""
        logger.info("Generating comprehensive visualizations...")
        
        # Create subdirectories for organization
        gantt_dir = os.path.join(save_dir, 'gantt_charts')
        metrics_dir = os.path.join(save_dir, 'metrics_plots')
        os.makedirs(gantt_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Generate visualizations for each scenario
        for scenario_name, scenario_results in self.results.items():
            scenario_info = self.test_scenarios.get(scenario_name, {})
            scenario_display_name = scenario_info.get('name', scenario_name)
            
            # Create scenario-specific subdirectories
            scenario_gantt_dir = os.path.join(gantt_dir, scenario_name)
            scenario_metrics_dir = os.path.join(metrics_dir, scenario_name)
            os.makedirs(scenario_gantt_dir, exist_ok=True)
            os.makedirs(scenario_metrics_dir, exist_ok=True)
            
            logger.info(f"Generating visualizations for scenario: {scenario_display_name}")
            
            for sched_name, result_data in scenario_results.items():
                scheduler = result_data.get('scheduler')
                if not scheduler or not scheduler.completed_tasks:
                    logger.warning(f"  Skipping {sched_name} - no completed tasks")
                    continue
                
                try:
                    # Generate Gantt chart
                    gantt_fig = Visualizer.plot_gantt_chart(
                        scheduler.completed_tasks, 
                        f"{sched_name} - {scenario_display_name}"
                    )
                    gantt_path = os.path.join(scenario_gantt_dir, f"{sched_name.lower()}_gantt.png")
                    gantt_fig.savefig(gantt_path, dpi=300, bbox_inches='tight')
                    plt.close(gantt_fig)
                    logger.debug(f"  Saved Gantt chart: {gantt_path}")
                    
                    # Generate metrics plot
                    metrics_fig = Visualizer.plot_metrics(scheduler)
                    metrics_path = os.path.join(scenario_metrics_dir, f"{sched_name.lower()}_metrics.png")
                    metrics_fig.savefig(metrics_path, dpi=300, bbox_inches='tight')
                    plt.close(metrics_fig)
                    logger.debug(f"  Saved metrics plot: {metrics_path}")
                    
                except Exception as e:
                    logger.error(f"  Error generating visualizations for {sched_name}: {e}")
        
        # Generate comparison visualizations
        comparison_dir = os.path.join(save_dir, 'comparisons')
        os.makedirs(comparison_dir, exist_ok=True)
        
        # Generate scenario comparison plots
        self._generate_scenario_comparison_plots(comparison_dir)
        
        # Generate priority performance comparison
        self._generate_priority_comparison_plots(comparison_dir)
        
        logger.info(f"All visualizations saved to: {save_dir}")

    def _generate_scenario_comparison_plots(self, save_dir: str):
        """Generate plots comparing schedulers across scenarios"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.style.use('seaborn-v0_8-whitegrid')
        except ImportError:
            logger.warning("matplotlib/seaborn not available for comparison plots")
            return
        
        # Prepare data for comparison
        schedulers = ['FCFS', 'EDF', 'Priority', 'ML-Linear', 'ML-DecisionTree', 'ML-QLearning']
        scenarios = ['baseline', 'high_load', 'multi_processor_2', 'stress_test']
        
        # Create comparison matrices
        avg_wait_matrix = []
        throughput_matrix = []
        
        for scenario in scenarios:
            wait_row = []
            throughput_row = []
            for scheduler in schedulers:
                metrics = self.results.get(scenario, {}).get(scheduler, {}).get('metrics', {})
                wait_row.append(metrics.get('avg_waiting', 0))
                throughput_row.append(metrics.get('throughput', 0))
            avg_wait_matrix.append(wait_row)
            throughput_matrix.append(throughput_row)
        
        # Plot heatmap for average waiting time
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Average waiting time heatmap
        sns.heatmap(avg_wait_matrix, 
                    xticklabels=schedulers,
                    yticklabels=[self.test_scenarios[s]['name'] for s in scenarios],
                    annot=True, fmt='.2f', cmap='YlOrRd',
                    ax=ax1, cbar_kws={'label': 'Avg Waiting Time (s)'})
        ax1.set_title('Average Waiting Time Comparison')
        ax1.set_xlabel('Scheduler')
        ax1.set_ylabel('Scenario')
        
        # Throughput heatmap
        sns.heatmap(throughput_matrix,
                    xticklabels=schedulers,
                    yticklabels=[self.test_scenarios[s]['name'] for s in scenarios],
                    annot=True, fmt='.2f', cmap='YlGnBu',
                    ax=ax2, cbar_kws={'label': 'Throughput (tasks/s)'})
        ax2.set_title('Throughput Comparison')
        ax2.set_xlabel('Scheduler')
        ax2.set_ylabel('Scenario')
        
        plt.suptitle('Scheduler Performance Across Scenarios', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, 'scenario_comparison_heatmaps.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved scenario comparison heatmaps: {save_path}")

    def _generate_priority_comparison_plots(self, save_dir: str):
        """Generate plots comparing priority-based performance"""
        try:
            import matplotlib.pyplot as plt
            plt.style.use('seaborn-v0_8-whitegrid')
        except ImportError:
            logger.warning("matplotlib not available for priority comparison plots")
            return
        
        # Compare priority performance in baseline scenario
        baseline_results = self.results.get('baseline', {})
        if not baseline_results:
            logger.warning("No baseline results for priority comparison")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        schedulers = ['FCFS', 'EDF', 'Priority', 'ML-Linear', 'ML-DecisionTree', 'ML-QLearning']
        priorities = [HIGH_PRIORITY, MEDIUM_PRIORITY, LOW_PRIORITY]
        priority_names = ['High', 'Medium', 'Low']
        colors = ['red', 'gold', 'green']
        
        # Plot 1: Average waiting time by priority
        ax = axes[0]
        x = np.arange(len(schedulers))
        width = 0.25
        
        for i, (prio, prio_name, color) in enumerate(zip(priorities, priority_names, colors)):
            wait_times = []
            for sched in schedulers:
                prio_metrics = baseline_results.get(sched, {}).get('metrics', {}).get('priority_metrics', {}).get(prio, {})
                wait_times.append(prio_metrics.get('avg_waiting', 0))
            
            ax.bar(x + i*width, wait_times, width, label=prio_name, color=color, alpha=0.8)
        
        ax.set_xlabel('Scheduler')
        ax.set_ylabel('Average Waiting Time (s)')
        ax.set_title('Average Waiting Time by Priority')
        ax.set_xticks(x + width)
        ax.set_xticklabels(schedulers, rotation=45)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        # Plot 2: P90 waiting time by priority
        ax = axes[1]
        for i, (prio, prio_name, color) in enumerate(zip(priorities, priority_names, colors)):
            p90_times = []
            for sched in schedulers:
                prio_metrics = baseline_results.get(sched, {}).get('metrics', {}).get('priority_metrics', {}).get(prio, {})
                p90_times.append(prio_metrics.get('p90_waiting', 0))
            
            ax.bar(x + i*width, p90_times, width, label=prio_name, color=color, alpha=0.8)
        
        ax.set_xlabel('Scheduler')
        ax.set_ylabel('P90 Waiting Time (s)')
        ax.set_title('90th Percentile Waiting Time by Priority')
        ax.set_xticks(x + width)
        ax.set_xticklabels(schedulers, rotation=45)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        # Plot 3: Task completion count by priority
        ax = axes[2]
        for i, (prio, prio_name, color) in enumerate(zip(priorities, priority_names, colors)):
            counts = []
            for sched in schedulers:
                prio_metrics = baseline_results.get(sched, {}).get('metrics', {}).get('priority_metrics', {}).get(prio, {})
                counts.append(prio_metrics.get('count', 0))
            
            ax.bar(x + i*width, counts, width, label=prio_name, color=color, alpha=0.8)
        
        ax.set_xlabel('Scheduler')
        ax.set_ylabel('Completed Tasks')
        ax.set_title('Task Completion Count by Priority')
        ax.set_xticks(x + width)
        ax.set_xticklabels(schedulers, rotation=45)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        
        # Plot 4: Priority fairness index (ratio of wait times)
        ax = axes[3]
        fairness_indices = []
        
        for sched in schedulers:
            metrics = baseline_results.get(sched, {}).get('metrics', {}).get('priority_metrics', {})
            high_wait = metrics.get(HIGH_PRIORITY, {}).get('avg_waiting', 1)
            low_wait = metrics.get(LOW_PRIORITY, {}).get('avg_waiting', 1)
            
            # Fairness index: ratio of high priority to low priority wait time
            # Lower is better (high priority should wait less)
            if low_wait > 0:
                fairness = high_wait / low_wait
            else:
                fairness = 0
            fairness_indices.append(fairness)
        
        bars = ax.bar(schedulers, fairness_indices, color='purple', alpha=0.7)
        ax.set_xlabel('Scheduler')
        ax.set_ylabel('Fairness Index (High/Low Wait Ratio)')
        ax.set_title('Priority Fairness Index (Lower is Better)')
        ax.set_xticklabels(schedulers, rotation=45)
        ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Equal treatment')
        ax.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, fairness_indices):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.2f}', ha='center', va='bottom')
        
        plt.suptitle('Priority-Based Performance Analysis (Baseline Scenario)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, 'priority_performance_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved priority comparison plots: {save_path}")

class GUI:
    """Graphical User Interface for the scheduler"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Real-Time Task Scheduler") # Simpler title
        self.root.geometry("1200x800")
        
        self.tasks: List[Task] = []
        self.results: Dict[str, Dict[str, Any]] = {} # Store results from regular sim
        
        self.setup_ui()
    
    def setup_ui(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.config_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.config_frame, text="Configuration")
        self.setup_config_tab()
        
        self.sim_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.sim_frame, text="Simulation")
        self.setup_simulation_tab()
        
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Results")
        self.setup_results_tab()
    
    def setup_config_tab(self):
        gen_frame = ttk.LabelFrame(self.config_frame, text="Task Generation", padding=10)
        gen_frame.grid(row=0, column=0, padx=10, pady=10, sticky='ew', columnspan=2)
        
        ttk.Label(gen_frame, text="Random Seed:").grid(row=0, column=0, sticky='w', padx=5, pady=5)
        self.seed_var = tk.IntVar(value=42)
        ttk.Entry(gen_frame, textvariable=self.seed_var, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Button(gen_frame, text="Generate Tasks", command=self.generate_tasks).grid(row=0, column=2, padx=10, pady=5)
        ttk.Button(gen_frame, text="Export Tasks", command=self.export_tasks).grid(row=0, column=3, padx=5, pady=5)
        ttk.Button(gen_frame, text="Import Tasks", command=self.import_tasks).grid(row=0, column=4, padx=5, pady=5)
        
        self.task_text = scrolledtext.ScrolledText(self.config_frame, width=80, height=15) # Reduced height
        self.task_text.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky='ew')
        
        sched_frame = ttk.LabelFrame(self.config_frame, text="Scheduler Configuration (for basic simulation)", padding=10)
        sched_frame.grid(row=2, column=0, padx=10, pady=10, sticky='ew')
        
        ttk.Label(sched_frame, text="Number of Processors:").grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.num_processors_var = tk.IntVar(value=1)
        ttk.Spinbox(sched_frame, from_=1, to=mp.cpu_count(), textvariable=self.num_processors_var, width=5).grid(row=0, column=1, padx=5, pady=2) # Max to CPU count
        
        ttk.Label(sched_frame, text="Schedulers to Compare:").grid(row=1, column=0, columnspan=2, sticky='w', padx=5, pady=2)
        
        # Updated scheduler_vars as per plan (Part 3)
        self.scheduler_vars = {
            'FCFS': tk.BooleanVar(value=True),
            'EDF': tk.BooleanVar(value=True),
            'Priority': tk.BooleanVar(value=True),
            'ML-Linear': tk.BooleanVar(value=True),
            'ML-DecisionTree': tk.BooleanVar(value=True),
            'ML-QLearning': tk.BooleanVar(value=True)
        }
        
        # Layout checkboxes in a grid
        row, col = 2, 0
        for name, var in self.scheduler_vars.items():
            ttk.Checkbutton(sched_frame, text=name, variable=var).grid(row=row, column=col, sticky='w', padx=5)
            col += 1
            if col >= 3: # Max 3 per row
                col = 0
                row += 1
        
        # Placeholder for future Pi-specific config
        pi_config_frame = ttk.LabelFrame(self.config_frame, text="Raspberry Pi Specific (Placeholder)", padding=10)
        pi_config_frame.grid(row=2, column=1, padx=10, pady=10, sticky='nsew')
        ttk.Label(pi_config_frame, text="Pi-specific options would go here.").pack()

        self.config_frame.columnconfigure(0, weight=1)
        self.config_frame.columnconfigure(1, weight=1)


    def setup_simulation_tab(self):
        control_frame = ttk.Frame(self.sim_frame, padding=10)
        control_frame.pack(fill='x')
        
        ttk.Button(control_frame, text="Run Simulation", command=self.run_simulation).pack(side='left', padx=5, pady=5)
        # ttk.Button(control_frame, text="Stop Simulation", command=self.stop_simulation).pack(side='left', padx=5, pady=5) # Stop not fully implemented
        
        # Added button for comprehensive tests (Part 2.2)
        ttk.Button(control_frame, text="Run Comprehensive Tests", 
                  command=self.run_comprehensive_tests).pack(side='left', padx=5, pady=5)

        self.progress_var = tk.StringVar(value="Ready to simulate.")
        ttk.Label(self.sim_frame, textvariable=self.progress_var).pack(pady=5, fill='x', padx=10)
        
        log_frame = ttk.LabelFrame(self.sim_frame, text="Simulation Log", padding=10)
        log_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, width=100, height=25, wrap=tk.WORD)
        self.log_text.pack(fill='both', expand=True)
    
    def setup_results_tab(self):
        control_frame = ttk.Frame(self.results_frame, padding=10)
        control_frame.pack(fill='x')
        
        ttk.Button(control_frame, text="Generate Report (from last basic sim)", 
                  command=self.generate_basic_report).pack(side='left', padx=5, pady=5) # Clarified button
        ttk.Button(control_frame, text="Save Results (last basic sim)", 
                  command=self.save_basic_results).pack(side='left', padx=5, pady=5) # Clarified button
        
        self.results_text = scrolledtext.ScrolledText(self.results_frame, width=100, height=30, wrap=tk.WORD)
        self.results_text.pack(fill='both', expand=True, padx=10, pady=10)
    
    def _log_to_gui(self, message: str, end="\n"):
        """Helper to log messages to GUI log_text and console."""
        self.log_text.insert(tk.END, message + end)
        self.log_text.see(tk.END)
        logger.info(message) # Also log to console logger
        self.root.update_idletasks() # Keep GUI responsive

    def generate_tasks(self):
        seed = self.seed_var.get()
        self.tasks = TaskGenerator.generate_tasks(seed)
        
        self.task_text.delete(1.0, tk.END)
        self.task_text.insert(tk.END, f"Generated {len(self.tasks)} tasks (Seed: {seed}):\n\n")
        for task in self.tasks[:10]: # Display first 10 tasks as sample
            deadline_str = f"{task.deadline:.2f}" if task.deadline is not None else "N/A"
            self.task_text.insert(tk.END, 
                f"{task.task_id: <10} Prio:{task.priority} Arrival:{task.arrival_time:5.2f}s "
                f"Service:{task.service_time:5.2f}s Deadline:{deadline_str}\n")
        if len(self.tasks) > 10: self.task_text.insert(tk.END, f"... and {len(self.tasks)-10} more tasks.\n")
        
        self.progress_var.set(f"Generated {len(self.tasks)} tasks.")
        self._log_to_gui(f"Generated {len(self.tasks)} tasks with seed {seed}.")

    def export_tasks(self):
        if not self.tasks:
            messagebox.showwarning("No Tasks", "Please generate tasks first!")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Export Tasks As"
        )
        if filename:
            TaskGenerator.export_tasks(self.tasks, filename)
            messagebox.showinfo("Export Successful", f"Tasks exported to {filename}")
            self._log_to_gui(f"Tasks exported to {filename}")

    def import_tasks(self):
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Import Tasks From"
        )
        if filename:
            try:
                self.tasks = TaskGenerator.import_tasks(filename)
                self.task_text.delete(1.0, tk.END)
                self.task_text.insert(tk.END, f"Imported {len(self.tasks)} tasks from {os.path.basename(filename)}:\n\n")
                for task in self.tasks[:10]: # Display sample
                    deadline_str = f"{task.deadline:.2f}" if task.deadline is not None else "N/A"
                    self.task_text.insert(tk.END, 
                        f"{task.task_id: <10} Prio:{task.priority} Arrival:{task.arrival_time:5.2f}s "
                        f"Service:{task.service_time:5.2f}s Deadline:{deadline_str}\n")
                if len(self.tasks) > 10: self.task_text.insert(tk.END, f"... and {len(self.tasks)-10} more tasks.\n")
                
                self.progress_var.set(f"Imported {len(self.tasks)} tasks.")
                self._log_to_gui(f"Imported {len(self.tasks)} tasks from {filename}")
                messagebox.showinfo("Import Successful", f"Tasks imported from {filename}")
            except Exception as e:
                messagebox.showerror("Import Error", f"Failed to import tasks: {str(e)}")
                self._log_to_gui(f"Error importing tasks: {e}")

    def run_simulation_thread(self):
        """Target for running basic simulation in a thread."""
        if not self.tasks:
            messagebox.showwarning("No Tasks", "Please generate tasks first!")
            self.progress_var.set("Task generation needed.")
            return

        self.results = {} # Clear previous basic simulation results
        self.log_text.delete(1.0, tk.END)
        self._log_to_gui("Starting basic simulation...")

        # Updated schedulers dictionary as per plan (Part 3)
        schedulers_to_run = {
            'FCFS': FCFSScheduler, 'EDF': EDFScheduler, 'Priority': PriorityScheduler,
            'ML-Linear': MLScheduler, 'ML-DecisionTree': DecisionTreeScheduler,
            'ML-QLearning': RLScheduler
        }
        
        num_processors = self.num_processors_var.get()
        if num_processors <=0: num_processors = 1 # Ensure positive
        
        selected_schedulers_count = 0
        for name, scheduler_class_ref in schedulers_to_run.items(): # Renamed scheduler_class to scheduler_class_ref
            if self.scheduler_vars.get(name) and self.scheduler_vars[name].get():
                selected_schedulers_count +=1
                self._log_to_gui(f"\n--- Running {name} Scheduler with {num_processors} Processor(s) ---")
                
                task_copies = [Task( # Fresh copies for each scheduler
                    task_id=t.task_id, arrival_time=t.arrival_time, service_time=t.service_time,
                    priority=t.priority, deadline=t.deadline
                ) for t in self.tasks]
                
                scheduler_inst = scheduler_class_ref() # Renamed scheduler to scheduler_inst
                simulator = Simulator(scheduler_inst, num_processors)
                
                sim_start_time = time.time()
                simulator.run(task_copies) # This is blocking, but it's in a thread
                sim_duration = time.time() - sim_start_time
                
                self.results[name] = {
                    'scheduler': scheduler_inst, # Store the scheduler instance with its completed_tasks and metrics
                    'num_processors': num_processors,
                    'simulation_wall_time': sim_duration
                }
                self._log_to_gui(f"{name} completed: {len(scheduler_inst.completed_tasks)} tasks. Wall time: {sim_duration:.2f}s.")
        
        if selected_schedulers_count == 0:
            messagebox.showwarning("No Schedulers", "Please select at least one scheduler to run!")
            self.progress_var.set("No schedulers selected for basic simulation.")
            return

        self.progress_var.set("Basic simulation(s) completed!")
        self._log_to_gui("\nAll selected basic simulations finished.")
        
        # Automatically generate and display report in results tab
        self.generate_basic_report() 
        # Switch to results tab
        self.notebook.select(self.results_frame)


    def run_simulation(self):
        """Runs the basic simulation in a new thread to keep GUI responsive."""
        sim_thread = threading.Thread(target=self.run_simulation_thread, daemon=True)
        sim_thread.start()

    # def stop_simulation(self): # Plan doesn't implement this fully
    #     self.progress_var.set("Stop command issued (implementation pending).")
    #     # This would require signaling all active simulator instances and threads to stop.
    #     # For example, by setting a global `self.simulation_should_stop = True`
    #     # and checking this flag in Simulator's run loop and execute_task_on_processor loop.
    #     logger.warning("Stop simulation functionality is not fully implemented.")


    def generate_basic_report(self):
        """Generates and displays report for the basic simulation runs."""
        if not self.results:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "No basic simulation results available to generate report.")
            if not self.tasks: messagebox.showinfo("Info", "Generate tasks and run a simulation first.")
            return
        
        # Use the existing Visualizer.generate_report for basic results
        report_content = Visualizer.generate_report(self.results)
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, report_content)
        self._log_to_gui("Basic simulation report generated and displayed.")
        
        # Optionally save plots for basic simulation too
        # Create a timestamped directory for these results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join("results", f"basic_sim_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        
        self._log_to_gui(f"Saving basic simulation plots to: {results_dir}")
        for name, result_data in self.results.items():
            scheduler_inst = result_data['scheduler']
            if scheduler_inst and scheduler_inst.completed_tasks:
                try:
                    gantt_fig = Visualizer.plot_gantt_chart(scheduler_inst.completed_tasks, f"{name} Scheduler (Basic Sim)")
                    gantt_fig.savefig(os.path.join(results_dir, f"gantt_basic_{name.lower()}.png"))
                    plt.close(gantt_fig)

                    metrics_fig = Visualizer.plot_metrics(scheduler_inst) # scheduler_inst has its metrics
                    metrics_fig.savefig(os.path.join(results_dir, f"metrics_basic_{name.lower()}.png"))
                    plt.close(metrics_fig)
                except Exception as e:
                    self._log_to_gui(f"Error generating plots for {name}: {e}")
        
        self.results_text.insert(tk.END, f"\n\nPlots and basic report components saved to {results_dir}")


    def save_basic_results(self):
        """Saves results of the basic simulation."""
        if not self.results:
            messagebox.showwarning("No Results", "No basic simulation results to save!")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join("results", f"basic_sim_save_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save full report text
        report_path = os.path.join(results_dir, "basic_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(self.results_text.get(1.0, tk.END)) # Get report from text widget
        
        # Save raw results data (scheduler objects might be complex, consider pickling relevant parts)
        # For simplicity, let's try to pickle the self.results dict.
        # Caution: pickling Tkinter vars or complex objects can fail.
        # The plan's comprehensive test pickles 'results'.
        data_to_pickle = {'tasks': [t.to_dict() for t in self.tasks], 'results': {}}
        for name, res_data in self.results.items():
            # Store simplified scheduler data, not the whole object for basic save
            data_to_pickle['results'][name] = {
                'completed_tasks': [t.to_dict() for t in res_data['scheduler'].completed_tasks],
                'metrics': res_data['scheduler'].metrics, # This should be pickleable
                'num_processors': res_data['num_processors'],
                'simulation_wall_time': res_data.get('simulation_wall_time')
            }

        raw_data_path = os.path.join(results_dir, "basic_raw_data.pkl")
        try:
            with open(raw_data_path, 'wb') as f:
                pickle.dump(data_to_pickle, f)
        except Exception as e:
            self._log_to_gui(f"Error pickling basic results: {e}")
            messagebox.showerror("Pickle Error", f"Could not pickle basic results: {e}")

        # Save task configuration if tasks were involved
        if self.tasks:
            task_config_path = os.path.join(results_dir, "task_configuration.json")
            TaskGenerator.export_tasks(self.tasks, task_config_path)
            
        self.progress_var.set(f"Basic simulation results saved to {results_dir}")
        messagebox.showinfo("Save Successful", f"Basic results (report, data, plots) saved to {results_dir}")
        self._log_to_gui(f"Basic results bundle saved to {results_dir}")

    # Added run_comprehensive_tests method to GUI class (Part 2.2)
    def run_comprehensive_tests_thread(self):
        """Target for running comprehensive tests in a thread."""
        self._log_to_gui("Starting Comprehensive Test Suite...")
        self.progress_var.set("Comprehensive tests running...")

        # Create progress dialog elements (managed by the main thread later)
        # This part needs to be done carefully with Tkinter threading.
        # For now, progress_callback will log to GUI.
        
        def progress_callback_gui(current, total, message):
            # This callback is from a worker thread. GUI updates must be scheduled.
            # A simple way: update stringvar and log.
            # A more robust way: use root.after or a queue.
            # For now, direct update (might cause issues on some platforms).
            try:
                progress_percent = (current / total) * 100
                self.progress_var.set(f"Test {current}/{total} ({progress_percent:.1f}%): {message}")
                self._log_to_gui(f"Progress: [{current}/{total}] {message}", end='\r') # Overwrite line
                if current == total: self._log_to_gui("", end="\n") # Newline at end
            except tk.TclError: # If root window is destroyed
                pass

        test_suite = ComprehensiveTestSuite()
        # Run all tests; results is a dict
        comp_results = test_suite.run_all_tests(progress_callback=progress_callback_gui) 
        
        self._log_to_gui("\nComprehensive tests data collection finished. Generating report...", end="\n")
        report_content = test_suite.generate_comprehensive_report()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join("results", f"comprehensive_test_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        
        report_file_path = os.path.join(results_dir, "comprehensive_report.txt")
        with open(report_file_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        self._log_to_gui(f"Comprehensive report saved to: {report_file_path}")
        
        detailed_results_path = os.path.join(results_dir, "detailed_results.pkl")
        try:
            with open(detailed_results_path, 'wb') as f:
                # comp_results contains scheduler objects. Consider serializing parts.
                # For now, trying to pickle the whole thing as per plan.
                pickle.dump(comp_results, f)
            self._log_to_gui(f"Detailed results pickled to: {detailed_results_path}")
        except Exception as e:
            self._log_to_gui(f"Error pickling comprehensive results: {e}")

        # Generate processor comparison visualization (Part 2.6 update)
        try:
            test_suite.generate_processor_comparison_visualization(results_dir)
            self._log_to_gui(f"Processor comparison visualization saved in {results_dir}")
        except Exception as e:
             self._log_to_gui(f"Error generating processor comparison plot: {e}")

        # Generate comprehensive visualizations (add this after the processor comparison visualization)
        try:
            test_suite.generate_comprehensive_visualizations(results_dir)
            self._log_to_gui(f"Comprehensive visualizations saved in {results_dir}")
        except Exception as e:
            self._log_to_gui(f"Error generating comprehensive visualizations: {e}")


        # Update GUI results tab with the comprehensive report
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, report_content[:5000] + "\n...(report truncated in GUI, full report saved to file)...") # Show part
        self.results_text.see(tk.END)
        
        self.progress_var.set(f"Comprehensive tests complete! Results in {results_dir}")
        self._log_to_gui(f"\nComprehensive tests completed! Results saved to: {results_dir}", end="\n")
        messagebox.showinfo("Comprehensive Tests Complete", 
                            f"Results saved to: {results_dir}\nReport also displayed in Results tab.")
        self.notebook.select(self.results_frame) # Switch to results tab

    def run_comprehensive_tests(self):
        """Runs the comprehensive test suite in a new thread."""
        # No need to check self.tasks here as ComprehensiveTestSuite generates its own.
        # However, some initial setup might be good (e.g. clear log).
        # self.log_text.delete(1.0, tk.END) # Cleared by the thread method.

        # Progress dialog setup (optional, as callback updates main GUI directly)
        # The plan includes a Toplevel progress window. This is complex with threading.
        # I'll simplify: the progress_callback_gui updates the main window's progress_var and log.
        
        test_thread = threading.Thread(target=self.run_comprehensive_tests_thread, daemon=True)
        test_thread.start()

    def run(self):
        self.root.mainloop()

def run_headless(args): # Modified for Comprehensive Tests (Part 2.3)
    logger.info("Running in headless mode...")

    if args.run_comprehensive_tests:
        logger.info("Starting Comprehensive Test Suite (headless)...")
        test_suite = ComprehensiveTestSuite()
        
        def progress_callback_headless(current, total, message):
            logger.info(f"Progress: [{current}/{total}] {message}")
        
        # ComprehensiveTestSuite.run_all_tests returns the results dictionary
        comp_results_headless = test_suite.run_all_tests(progress_callback=progress_callback_headless)
        report_headless = test_suite.generate_comprehensive_report()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir_headless = os.path.join("results", f"comprehensive_headless_{timestamp}")
        os.makedirs(results_dir_headless, exist_ok=True)
        
        report_path_headless = os.path.join(results_dir_headless, "comprehensive_report.txt")
        with open(report_path_headless, 'w', encoding='utf-8') as f:
            f.write(report_headless)
        
        # Save detailed pickled results
        detailed_results_path_headless = os.path.join(results_dir_headless, "detailed_results.pkl")
        try:
            with open(detailed_results_path_headless, 'wb') as f:
                pickle.dump(comp_results_headless, f)
        except Exception as e:
             logger.error(f"Error pickling comprehensive headless results: {e}")

        # Generate processor comparison plot if matplotlib is available
        try:
            test_suite.generate_processor_comparison_visualization(results_dir_headless)
        except Exception as e:
            logger.error(f"Error generating processor comparison plot in headless mode: {e}")

        try:
            test_suite.generate_comprehensive_visualizations(results_dir_headless)
            logger.info(f"Comprehensive visualizations saved in {results_dir_headless}")
        except Exception as e:
            logger.error(f"Error generating comprehensive visualizations: {e}")


        print(f"\nComprehensive tests completed! Results saved to: {results_dir_headless}")
        print("\n--- Comprehensive Report Summary (Headless) ---")
        # Print first ~20 lines of report as summary to console
        print('\n'.join(report_headless.splitlines()[:20]))
        print("...\n(Full report saved to file)")
        return # Exit after comprehensive tests

    # Original headless simulation logic (if not running comprehensive tests)
    logger.info("Running basic headless simulation...")
    tasks_headless: List[Task]
    if args.import_tasks:
        tasks_headless = TaskGenerator.import_tasks(args.import_tasks)
    else:
        tasks_headless = TaskGenerator.generate_tasks(seed=args.seed)
    
    logger.info(f"Using {len(tasks_headless)} tasks for basic headless simulation.")
    
    results_headless: Dict[str, Dict[str, Any]] = {} # Type hint
    
    # Schedulers for basic headless (matches GUI's updated list for consistency)
    # Part 3: Update Scheduler Dictionary (implied for headless too)
    schedulers_map_headless = {
        'FCFS': FCFSScheduler, 'EDF': EDFScheduler, 'Priority': PriorityScheduler,
        'ML-Linear': MLScheduler, 'ML-DecisionTree': DecisionTreeScheduler,
        'ML-QLearning': RLScheduler
    }
    
    for name, scheduler_class_ref in schedulers_map_headless.items():
        # Argparse flags like --enable-ml-linear would be needed for granular control.
        # The plan doesn't add these. Defaulting to True for getattr if flag missing.
        # Example: args.enable_fcfs, args.enable_ml_linear (if defined in parser)
        # Current plan's parser only has original flags.
        # For simplicity, run if flag exists and is True, or if flag doesn't exist (getattr default True).
        # This means new schedulers will run unless explicitly disabled via new flags.
        # The existing enable_ml would control ML-Linear if key was 'ML'.
        # Given the plan, I'll assume for basic headless, it runs if its old equivalent flag was true or no flag.
        # This is a bit ambiguous from plan.
        
        # Simplification: rely on original flags for original schedulers, new ones always run in basic headless.
        # Or, use a generic check (e.g., if args.scheduler_filter is not None and name not in it, skip)
        # For now, let's use the getattr approach; it will default to True if specific flag is not found.
        # This means ML-DecisionTree will try to use args.enable_ml-decisiontree
        enable_flag_name = f'enable_{name.lower().replace("-", "_")}' # e.g. enable_ml_linear

        if getattr(args, enable_flag_name, True): # Default to True if flag not present
            logger.info(f"\nRunning {name} scheduler (headless)...")
            
            task_copies_headless = [Task(
                task_id=t.task_id, arrival_time=t.arrival_time, service_time=t.service_time,
                priority=t.priority, deadline=t.deadline
            ) for t in tasks_headless]
            
            scheduler_inst_headless = scheduler_class_ref()
            simulator_headless = Simulator(scheduler_inst_headless, args.num_processors)
            
            sim_start_headless = time.time()
            simulator_headless.run(task_copies_headless)
            sim_dur_headless = time.time() - sim_start_headless
            
            results_headless[name] = {
                'scheduler': scheduler_inst_headless,
                'num_processors': args.num_processors,
                'simulation_wall_time': sim_dur_headless
            }
            logger.info(f"{name} completed: {len(scheduler_inst_headless.completed_tasks)} tasks. Wall time: {sim_dur_headless:.2f}s.")
    
    report_content_headless = Visualizer.generate_report(results_headless)
    print("\n--- Basic Headless Simulation Report ---")
    print(report_content_headless)
    
    timestamp_headless_basic = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir_headless_basic = os.path.join("results", f"basic_headless_{timestamp_headless_basic}")
    os.makedirs(results_dir_headless_basic, exist_ok=True)
    
    report_path_headless_basic = os.path.join(results_dir_headless_basic, "basic_report.txt")
    with open(report_path_headless_basic, 'w', encoding='utf-8') as f:
        f.write(report_content_headless)
    
    if args.generate_plots:
        logger.info(f"Generating plots for basic headless sim in {results_dir_headless_basic}...")
        for name, result_data in results_headless.items():
            scheduler_inst = result_data['scheduler']
            if scheduler_inst and scheduler_inst.completed_tasks:
                try:
                    g_fig = Visualizer.plot_gantt_chart(scheduler_inst.completed_tasks, f"{name} Scheduler (Headless)")
                    g_fig.savefig(os.path.join(results_dir_headless_basic, f"gantt_{name.lower()}.png"))
                    plt.close(g_fig)
                    
                    m_fig = Visualizer.plot_metrics(scheduler_inst)
                    m_fig.savefig(os.path.join(results_dir_headless_basic, f"metrics_{name.lower()}.png"))
                    plt.close(m_fig)
                except Exception as e:
                    logger.error(f"Error generating plots for {name} in headless: {e}")
    
    logger.info(f"Basic headless simulation results saved to {results_dir_headless_basic}")


def main():
    import argparse # Ensure argparse is imported here
    
    parser = argparse.ArgumentParser(description="Real-Time Task Scheduler Simulation")
    parser.add_argument('--mode', choices=['gui', 'headless'], default='gui', help='Run mode (default: gui)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for task generation (None for time-based)') # Default None
    parser.add_argument('--import-tasks', type=str, help='Import tasks from JSON file')
    parser.add_argument('--num-processors', type=int, default=1, help='Number of processors (default: 1)')
    
    # Flags for basic headless schedulers (original ones)
    parser.add_argument('--enable-fcfs', action='store_true', help='Enable FCFS scheduler in basic headless')
    parser.add_argument('--enable-edf', action='store_true', help='Enable EDF scheduler in basic headless')
    parser.add_argument('--enable-priority', action='store_true', help='Enable Priority scheduler in basic headless')
    parser.add_argument('--enable-ml-linear', action='store_true', help='Enable ML-Linear scheduler in basic headless')
    # Need to add flags for new schedulers if granular control is desired for basic headless mode
    parser.add_argument('--enable-ml-decisiontree', action='store_true', help='Enable ML-DecisionTree scheduler in basic headless')
    parser.add_argument('--enable-ml-qlearning', action='store_true', help='Enable ML-QLearning scheduler in basic headless')
    # If specific enable flags are not used, getattr will default to True, running them.
    # For explicit control, one might pass a list of schedulers --run-schedulers FCFS,EDF ...

    parser.add_argument('--generate-plots', action='store_true', default=False, help='Generate plots in basic headless mode (default: False)')
    
    # Added from plan for Part 2.3 (Comprehensive Tests)
    parser.add_argument('--run-comprehensive-tests', action='store_true',
                       help='Run comprehensive test suite automatically (overrides basic simulation in headless)')
    
    # Default all enable flags to False, so only explicitly enabled ones run in basic headless.
    # Or, if no enable flags are given, run all (current getattr behavior with default True).
    # Setting defaults for enable_ flags to False provides more control if user wants specific basic set.
    parser.set_defaults(enable_fcfs=False, enable_edf=False, enable_priority=False, 
                        enable_ml_linear=False, enable_ml_decisiontree=False, enable_ml_qlearning=False)


    args = parser.parse_args()

    # If no specific schedulers are enabled for basic headless, and not comprehensive, enable a default set.
    if args.mode == 'headless' and not args.run_comprehensive_tests:
        # Check if any scheduler was explicitly enabled
        any_scheduler_enabled = (args.enable_fcfs or args.enable_edf or args.enable_priority or
                                 args.enable_ml_linear or args.enable_ml_decisiontree or args.enable_ml_qlearning)
        if not any_scheduler_enabled:
            logger.info("No specific schedulers enabled for basic headless. Running a default set (FCFS, Priority).")
            args.enable_fcfs = True
            args.enable_priority = True
            # By default, generate_plots can be True for headless if not specified
            if not hasattr(args, 'generate_plots') or args.generate_plots is None:
                 args.generate_plots = True


    # Create results directory if it doesn't exist
    if not os.path.exists("results"):
        os.makedirs("results")

    if args.mode == 'gui':
        gui = GUI()
        gui.run()
    else: # headless mode
        run_headless(args)

if __name__ == "__main__":
    # This check is important for multiprocessing safety on Windows
    # if sys.platform.startswith('win'):
    #    mp.freeze_support() # Not strictly needed as plan doesn't use mp directly for tasks
    main()
