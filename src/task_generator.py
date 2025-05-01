"""
Task Generator Module

This module handles the generation of tasks with different priorities
using Poisson distribution for arrival times.
"""

import numpy as np
from enum import Enum, auto
import time
import uuid
import logging

logger = logging.getLogger(__name__)

# Constants for validation
MAX_WAITING_TIME = 60.0   # Maximum reasonable waiting time (seconds)
MAX_SERVICE_TIME = 20.0   # Maximum reasonable service time (seconds)
MAX_TURNAROUND_TIME = 120.0  # Maximum reasonable turnaround time (seconds)

class Priority(Enum):
    """Task priority levels"""
    HIGH = 1
    MEDIUM = 2
    LOW = 3

class Task:
    """Task class representing a real-time task with attributes"""
    
    def __init__(self, task_id, arrival_time, service_time, priority, deadline=None):
        """
        Initialize task with validation
        
        Args:
            task_id: Unique identifier for the task
            arrival_time: Time when the task arrives
            service_time: Execution time required by the task
            priority: Priority level (HIGH, MEDIUM, LOW)
            deadline: Optional deadline for the task
        """
        self.id = task_id
        
        # Validate arrival time (must be non-negative)
        if arrival_time < 0:
            logger.warning(f"Negative arrival time for Task {task_id}: {arrival_time}. Setting to 0.")
            self.arrival_time = 0.0
        else:
            self.arrival_time = arrival_time
        
        # Validate service time (must be positive and reasonable)
        if service_time <= 0:
            logger.warning(f"Invalid service time for Task {task_id}: {service_time}. Setting to 0.1s.")
            self.service_time = 0.1
        elif service_time > MAX_SERVICE_TIME:
            logger.warning(
                f"Excessive service time for Task {task_id}: {service_time}s. "
                f"Capping to {MAX_SERVICE_TIME}s."
            )
            self.service_time = MAX_SERVICE_TIME
        else:
            self.service_time = service_time
        
        # Store priority
        self.priority = priority
        
        # Set deadline if provided
        self.deadline = deadline
        
        # Initialize execution metrics
        self.start_time = None
        self.completion_time = None
        self.waiting_time = None
        self.response_time = None
        self.turnaround_time = None
        
        # For ML prediction
        self.predicted_time = None
        self.prediction_error = None
    
    def __str__(self):
        """String representation of the task"""
        return f"Task {self.id}: Priority={self.priority.name}, Arrival={self.arrival_time:.2f}s, Service={self.service_time:.2f}s"
    
    
    def calculate_metrics(self):
        """
        Calculate waiting time and other metrics once task is processed
        
        Returns:
            Dictionary containing timing metrics or None if task processing is incomplete
        """
        if self.start_time is None or self.completion_time is None:
            return None
            
        # Calculate waiting time (time between arrival and start)
        self.waiting_time = max(0, self.start_time - self.arrival_time)
        
        # Validate waiting time
        if self.waiting_time > MAX_WAITING_TIME:
            logger.warning(
                f"Excessive waiting time for Task {self.id}: {self.waiting_time}s. "
                f"Capping to {MAX_WAITING_TIME}s."
            )
            self.waiting_time = MAX_WAITING_TIME
            
        # Calculate turnaround time (time between arrival and completion)
        self.turnaround_time = max(0, self.completion_time - self.arrival_time)
        
        # Validate turnaround time
        if self.turnaround_time > MAX_TURNAROUND_TIME:
            logger.warning(
                f"Excessive turnaround time for Task {self.id}: {self.turnaround_time}s. "
                f"Capping to {MAX_TURNAROUND_TIME}s."
            )
            self.turnaround_time = MAX_TURNAROUND_TIME
            
        # Calculate response time (time between arrival and start of execution)
        self.response_time = max(0, self.start_time - self.arrival_time)
        
        # Validate and round all metrics for consistency
        self.waiting_time = round(self.waiting_time, 3)
        self.turnaround_time = round(self.turnaround_time, 3)
        self.response_time = round(self.response_time, 3)
        
        return {
            'waiting_time': self.waiting_time,
            'turnaround_time': self.turnaround_time,
            'response_time': self.response_time
        }
        
    def is_deadline_met(self):
        """
        Check if the task met its deadline
        
        Returns:
            True if deadline was met, False if missed, None if not applicable or not completed
        """
        if self.deadline is None or self.completion_time is None:
            return None
            
        return self.completion_time <= self.deadline

class TaskGenerator:
    """Generates tasks with Poisson distribution for arrival times"""
    
    def __init__(self, config=None):
        """
        Initialise task generator with configuration validation
        
        Args:
            config: Dictionary containing configuration parameters
        """
        # Default configuration
        default_config = {
            Priority.HIGH: {
                'count': 20,
                'lambda': 3,  # Average arrival rate (λ) in seconds
                'service_min': 2,
                'service_max': 5
            },
            Priority.MEDIUM: {
                'count': 20,
                'lambda': 5,
                'service_min': 3,
                'service_max': 7
            },
            Priority.LOW: {
                'count': 10,
                'lambda': 7,
                'service_min': 5,
                'service_max': 10
            }
        }
        
        # Use provided config or default
        self.config = config or default_config
        
        # Validate configuration
        self._validate_config()
        
    def _validate_config(self):
        """Validate task generation configuration"""
        for priority, params in self.config.items():
            # Ensure count is positive
            if 'count' not in params or params['count'] <= 0:
                logger.warning(f"Invalid count for {priority.name} priority. Setting to 10.")
                params['count'] = 10
                
            # Ensure lambda (arrival rate) is positive
            if 'lambda' not in params or params['lambda'] <= 0:
                logger.warning(f"Invalid lambda for {priority.name} priority. Setting to 5.0.")
                params['lambda'] = 5.0
                
            # Ensure service time range is valid
            if ('service_min' not in params or 
                'service_max' not in params or 
                params['service_min'] <= 0 or 
                params['service_max'] <= params['service_min']):
                
                logger.warning(f"Invalid service time range for {priority.name} priority. Using defaults.")
                
                if priority == Priority.HIGH:
                    params['service_min'] = 2
                    params['service_max'] = 5
                elif priority == Priority.MEDIUM:
                    params['service_min'] = 3
                    params['service_max'] = 7
                else:  # LOW
                    params['service_min'] = 5
                    params['service_max'] = 10
        
    def generate_tasks(self, start_time=0):
        """
        Generate tasks according to configuration with validation
        
        Args:
            start_time: Starting time for the simulation
            
        Returns:
            List of Task objects sorted by arrival time
        """
        tasks = []
        
        for priority, params in self.config.items():
            count = params['count']
            arrival_lambda = params['lambda']
            service_min = params['service_min']
            service_max = params['service_max']
            
            # Handle invalid arrival lambda
            if arrival_lambda <= 0:
                logger.warning(f"Invalid arrival lambda for {priority.name}: {arrival_lambda}. Using 5.0.")
                arrival_lambda = 5.0
            
            # Generate interarrival times using Poisson distribution
            # Note: numpy's exponential function generates samples from exp distribution with scale=1/lambda
            try:
                interarrival_times = np.random.exponential(scale=1/arrival_lambda, size=count)
            except Exception as e:
                logger.error(f"Error generating interarrival times: {e}. Using uniform distribution.")
                interarrival_times = np.random.uniform(1, 10, count)
            
            # Calculate absolute arrival times
            arrival_times = np.cumsum(interarrival_times) + start_time
            
            # Generate service times (uniform distribution between min and max)
            try:
                service_times = np.random.uniform(service_min, service_max, count)
            except Exception as e:
                logger.error(f"Error generating service times: {e}. Using defaults.")
                service_times = np.array([service_min] * count)
            
            # Create tasks
            for i in range(count):
                task_id = f"{priority.name[0]}{i+1}"  # H1, H2, M1, M2, L1, L2, etc.
                
                # Calculate deadline based on priority
                if priority == Priority.HIGH:
                    # High priority tasks have tighter deadlines
                    deadline_factor = 1.5
                elif priority == Priority.MEDIUM:
                    # Medium priority tasks have moderate deadlines
                    deadline_factor = 2.0
                else:  # LOW
                    # Low priority tasks have relaxed deadlines
                    deadline_factor = 3.0
                
                # Set deadline based on priority (only for HIGH priority by default)
                deadline = None
                if priority == Priority.HIGH:
                    deadline = arrival_times[i] + service_times[i] * deadline_factor
                
                task = Task(
                    task_id=task_id,
                    arrival_time=arrival_times[i],
                    service_time=service_times[i],
                    priority=priority,
                    deadline=deadline
                )
                tasks.append(task)
        
        # Sort tasks by arrival time
        tasks.sort(key=lambda x: x.arrival_time)
        
        return tasks

if __name__ == "__main__":
    # Simple test
    generator = TaskGenerator()
    tasks = generator.generate_tasks()
    
    print(f"Generated {len(tasks)} tasks:")
    for i, task in enumerate(tasks[:10]):
        print(f"{i+1}. {task}")
    print("...")