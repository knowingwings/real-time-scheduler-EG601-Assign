# First-Come-First-Served scheduler 
"""
First-Come-First-Served (FCFS) Scheduler

Implements a non-preemptive FCFS scheduling algorithm.
"""

import time
from queue import Queue
import threading
import psutil
import logging

class FCFSScheduler:
    """
    First-Come-First-Served Scheduler
    
    Tasks are executed in the order they arrive, without preemption.
    """
    
    def __init__(self):
        self.task_queue = Queue()
        self.current_task = None
        self.completed_tasks = []
        self.lock = threading.Lock()
        self.running = False
        self.current_time = 0
        self.metrics = {
            'queue_length': [],
            'memory_usage': [],
            'timestamp': []
        }
        self.logger = logging.getLogger(__name__)
    
    def add_task(self, task):
        """Add a task to the scheduler queue"""
        self.task_queue.put(task)
        self.logger.info(f"Task {task.id} added to FCFS queue")
        
    def run(self, simulation=False, speed_factor=1.0):
        """
        Run the scheduler
        
        Args:
            simulation: If True, time is simulated rather than real
            speed_factor: Speed up or slow down simulation (only used if simulation=True)
        """
        self.running = True
        self.current_time = 0
        start_time = time.time()  # Store real-time start
        
        # Start metrics collection in a separate thread
        metrics_thread = threading.Thread(target=self._collect_metrics)
        metrics_thread.daemon = True
        metrics_thread.start()
        
        while self.running:
            # Record current state
            with self.lock:
                queue_size = self.task_queue.qsize()
                self.metrics['queue_length'].append(queue_size)
                
            if not self.task_queue.empty():
                # Get the next task
                task = self.task_queue.get()
                self.current_task = task
                
                if simulation:
                    # In simulation mode, advance time to task arrival if needed
                    if self.current_time < task.arrival_time:
                        self.current_time = task.arrival_time
                
                # Record start time (using relative time in simulation mode)
                if simulation:
                    task.start_time = self.current_time
                else:
                    task.start_time = time.time() - start_time  # Store relative time
                    
                self.logger.info(f"Starting execution of {task.id} at time {task.start_time:.2f}")
                
                # Execute the task
                if simulation:
                    # Simulate execution by advancing time
                    self.current_time += task.service_time
                    time.sleep(task.service_time / speed_factor)  # Still sleep a bit for visualisation
                else:
                    # Actually sleep for the service time in real execution
                    time.sleep(task.service_time)
                
                # Record completion time
                if simulation:
                    task.completion_time = self.current_time
                else:
                    task.completion_time = time.time() - start_time  # Store relative time
                    
                self.logger.info(f"Completed execution of {task.id} at time {task.completion_time:.2f}")
                
                # Calculate metrics
                task.calculate_metrics()
                
                # Add to completed tasks
                with self.lock:
                    self.completed_tasks.append(task)
                    self.current_task = None
            else:
                # No tasks to process
                if not simulation:
                    time.sleep(0.1)  # Prevent CPU hogging
                else:
                    # In simulation, advance time to next expected arrival if known
                    time.sleep(0.01 / speed_factor)
    
    def stop(self):
        """Stop the scheduler"""
        self.running = False


    def _initialize_metrics(self):
        """Initialize metrics with default values to prevent NaN issues"""
        with self.lock:
            # Basic counters
            self.completed_tasks = []
            
            # Queue metrics
            self.queue_length_history = []
            
            # Timing metrics with proper initialization
            self.waiting_times = []
            self.response_times = []
            self.turnaround_times = []
            
            # Priority metrics with proper initialization
            self.waiting_times_by_priority = {
                'HIGH': [],
                'MEDIUM': [],
                'LOW': []
            }
            
            # Deadline metrics
            self.deadline_misses = 0
            self.deadline_met = 0
            self.deadline_tasks = 0
            
            # System metrics
            self.memory_usage_history = []
            self.timestamp_history = []# Add this method to each scheduler class to standardize metrics collection
    
    def _collect_metrics(self):
        """Collect system metrics during execution"""
        start_time = time.time()  # Record the absolute start time
        
        while self.running:
            current_time = time.time()
            relative_time = round(current_time - start_time, 3)  # Calculate relative time in seconds
            memory_percent = psutil.virtual_memory().percent
            
            with self.lock:
                self.metrics['memory_usage'].append(memory_percent)
                self.metrics['timestamp'].append(relative_time)  # Store relative time
                self.metrics['queue_length'].append(self.task_queue.qsize())
            
            time.sleep(0.5)  # Collect metrics every 0.5 seconds
    
    def get_metrics(self):
        """Get execution metrics"""
        with self.lock:
            # Calculate average waiting time
            waiting_times = [task.waiting_time for task in self.completed_tasks 
                            if task.waiting_time is not None]
            avg_waiting_time = sum(waiting_times) / len(waiting_times) if waiting_times else 0
            
            # Calculate waiting times by priority
            waiting_by_priority = {'HIGH': [], 'MEDIUM': [], 'LOW': []}
            for task in self.completed_tasks:
                if task.waiting_time is not None and hasattr(task, 'priority'):
                    priority_name = task.priority.name if hasattr(task.priority, 'name') else str(task.priority)
                    if priority_name in waiting_by_priority:
                        waiting_by_priority[priority_name].append(task.waiting_time)
            
            avg_waiting_by_priority = {}
            for priority, times in waiting_by_priority.items():
                avg_waiting_by_priority[priority] = round(sum(times) / len(times), 3) if times else 0
            
            # Count tasks by priority
            tasks_by_priority = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
            for task in self.completed_tasks:
                if hasattr(task, 'priority'):
                    priority_name = task.priority.name if hasattr(task.priority, 'name') else str(task.priority)
                    if priority_name in tasks_by_priority:
                        tasks_by_priority[priority_name] += 1
            
            # Round timing values for better readability
            return {
                'completed_tasks': len(self.completed_tasks),
                'avg_waiting_time': round(avg_waiting_time, 3),  # Rounded to 3 decimal places
                'avg_waiting_by_priority': avg_waiting_by_priority,  # Already rounded
                'tasks_by_priority': tasks_by_priority,
                'queue_length_history': self.metrics['queue_length'],
                'memory_usage_history': self.metrics['memory_usage'],
                'timestamp_history': self.metrics['timestamp']
            }