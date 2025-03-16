"""
Task Generator Module

This module handles the generation of tasks with different priorities
using Poisson distribution for arrival times.
"""

import numpy as np
from enum import Enum, auto
import time
import uuid

class Priority(Enum):
    """Task priority levels"""
    HIGH = 1
    MEDIUM = 2
    LOW = 3

class Task:
    """Task class representing a real-time task with attributes"""
    
    def __init__(self, task_id, arrival_time, service_time, priority, deadline=None):
        self.id = task_id
        self.arrival_time = arrival_time
        self.service_time = service_time
        self.priority = priority
        self.deadline = deadline
        self.start_time = None
        self.completion_time = None
        self.waiting_time = None
    
    def __str__(self):
        return f"Task {self.id}: Priority={self.priority.name}, Arrival={self.arrival_time:.2f}s, Service={self.service_time:.2f}s"
    
    def calculate_metrics(self):
        """Calculate waiting time and other metrics once task is processed"""
        if self.start_time and self.completion_time:
            self.waiting_time = self.start_time - self.arrival_time
            return {
                'waiting_time': self.waiting_time,
                'turnaround_time': self.completion_time - self.arrival_time,
                'response_time': self.start_time - self.arrival_time
            }
        return None

class TaskGenerator:
    """Generates tasks with Poisson distribution for arrival times"""
    
    def __init__(self, config=None):
        """
        Initialize task generator with configuration
        
        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config or {
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
        
    def generate_tasks(self, start_time=0):
        """
        Generate tasks according to configuration
        
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
            
            # Generate interarrival times using Poisson distribution
            interarrival_times = np.random.exponential(scale=1/arrival_lambda, size=count)
            
            # Calculate absolute arrival times
            arrival_times = np.cumsum(interarrival_times) + start_time
            
            # Generate service times (uniform distribution between min and max)
            service_times = np.random.uniform(service_min, service_max, count)
            
            # Create tasks
            for i in range(count):
                task_id = f"{priority.name[0]}{i+1}"  # H1, H2, M1, M2, L1, L2, etc.
                task = Task(
                    task_id=task_id,
                    arrival_time=arrival_times[i],
                    service_time=service_times[i],
                    priority=priority,
                    # Set a deadline for high priority tasks
                    deadline=arrival_times[i] + service_times[i] * 1.5 if priority == Priority.HIGH else None
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