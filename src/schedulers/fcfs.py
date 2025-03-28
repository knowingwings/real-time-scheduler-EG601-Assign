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
                
                # Record start time
                task.start_time = time.time() if not simulation else self.current_time
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
                task.completion_time = time.time() if not simulation else self.current_time
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
    
    def _collect_metrics(self):
        """Collect system metrics during execution"""
        while self.running:
            memory_percent = psutil.virtual_memory().percent
            timestamp = time.time()
            
            with self.lock:
                self.metrics['memory_usage'].append(memory_percent)
                self.metrics['timestamp'].append(timestamp)
            
            time.sleep(0.5)  # Collect metrics every 0.5 seconds
    
    def get_metrics(self):
        """Get execution metrics"""
        with self.lock:
            # Calculate average waiting time
            waiting_times = [task.waiting_time for task in self.completed_tasks 
                            if task.waiting_time is not None]
            avg_waiting_time = sum(waiting_times) / len(waiting_times) if waiting_times else 0
            
            return {
                'completed_tasks': len(self.completed_tasks),
                'average_waiting_time': avg_waiting_time,
                'queue_length_history': self.metrics['queue_length'],
                'memory_usage_history': self.metrics['memory_usage'],
                'timestamp_history': self.metrics['timestamp']
            }