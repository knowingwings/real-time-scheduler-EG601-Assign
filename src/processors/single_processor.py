"""
Single Processor Implementation

This module handles task execution on a single processor system.
"""

import threading
import time
import logging
import psutil

class SingleProcessor:
    """
    Single Processor Implementation
    
    Handles task scheduling and execution on a single processor.
    """
    
    def __init__(self, scheduler, name="CPU-1"):
        """
        Initialize single processor with a scheduler
        
        Args:
            scheduler: The scheduler instance to use
            name: Name identifier for this processor
        """
        self.name = name
        self.scheduler = scheduler
        self.is_running = False
        self.simulation_speed = 1.0  # For simulation mode
        self.tasks = []
        self.simulation_mode = False
        self.logger = logging.getLogger(__name__)
        
        # Performance metrics
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'timestamp': [],
            'throughput': [],  # Tasks per second
            'utilization': []  # Percentage of time the processor is busy
        }
    
    def add_tasks(self, tasks):
        """
        Add tasks to the processor
        
        Args:
            tasks: List of Task objects to add
        """
        self.tasks.extend(tasks)
        self.logger.info(f"Added {len(tasks)} tasks to {self.name}")
    
    def run(self, simulation=False, speed_factor=1.0):
        """
        Run the processor
        
        Args:
            simulation: If True, run in simulation mode
            speed_factor: Speed factor for simulation (higher = faster)
        """
        self.is_running = True
        self.simulation_mode = simulation
        self.simulation_speed = speed_factor
        
        # Start metrics collection
        metrics_thread = threading.Thread(target=self._collect_metrics)
        metrics_thread.daemon = True
        metrics_thread.start()
        
        # Start scheduler in a separate thread
        scheduler_thread = threading.Thread(target=self._run_scheduler)
        scheduler_thread.daemon = True
        scheduler_thread.start()
        
        # Feed tasks to the scheduler based on arrival times
        self._feed_tasks()
        
        # Wait for scheduler to complete if running in simulation mode
        if simulation:
            scheduler_thread.join()
            self.is_running = False
            self.logger.info(f"{self.name} simulation completed")
    
    def _run_scheduler(self):
        """Run the scheduler"""
        self.logger.info(f"Starting scheduler on {self.name}")
        self.scheduler.run(simulation=self.simulation_mode, speed_factor=self.simulation_speed)
    
    def _feed_tasks(self):
        """Feed tasks to the scheduler based on their arrival times"""
        if not self.tasks:
            self.logger.warning(f"No tasks to feed to {self.name}")
            return
        
        # Sort tasks by arrival time
        tasks = sorted(self.tasks, key=lambda x: x.arrival_time)
        
        start_time = time.time()
        
        for task in tasks:
            if not self.is_running:
                break
                
            # Calculate time to wait before adding this task
            if self.simulation_mode:
                # In simulation mode, we don't actually wait
                self.scheduler.add_task(task)
                self.logger.info(f"Added task {task.id} to {self.name} at simulated time {task.arrival_time:.2f}s")
            else:
                # In real mode, wait until the task's arrival time
                current_time = time.time() - start_time
                wait_time = max(0, task.arrival_time - current_time)
                
                if wait_time > 0:
                    time.sleep(wait_time)
                
                self.scheduler.add_task(task)
                self.logger.info(f"Added task {task.id} to {self.name} at time {time.time() - start_time:.2f}s")
        
        # If in simulation mode, signal that all tasks have been added
        if self.simulation_mode:
            self.logger.info(f"All tasks added to {self.name} in simulation mode")
            
            # Wait for all tasks to complete
            all_completed = False
            while not all_completed and self.is_running:
                if self.scheduler.task_queue.empty() and not self.scheduler.current_task:
                    all_completed = True
                    self.scheduler.stop()
                else:
                    time.sleep(0.1)
    
    def stop(self):
        """Stop the processor"""
        self.is_running = False
        self.scheduler.stop()
        self.logger.info(f"Stopping {self.name}")
    
    def _collect_metrics(self):
        """Collect performance metrics"""
        last_completed = 0
        last_time = time.time()
        
        while self.is_running:
            current_time = time.time()
            cpu_percent = psutil.cpu_percent(interval=None)
            memory_percent = psutil.virtual_memory().percent
            
            # Get number of completed tasks
            completed = len(self.scheduler.completed_tasks)
            
            # Calculate throughput (tasks per second)
            elapsed = current_time - last_time
            if elapsed > 0:
                throughput = (completed - last_completed) / elapsed
            else:
                throughput = 0
                
            # Calculate processor utilization
            if hasattr(self.scheduler, 'current_task') and self.scheduler.current_task:
                utilization = 100  # Processor is busy
            else:
                utilization = 0    # Processor is idle
            
            # Update metrics
            with self.scheduler.lock:
                self.metrics['cpu_usage'].append(cpu_percent)
                self.metrics['memory_usage'].append(memory_percent)
                self.metrics['timestamp'].append(current_time)
                self.metrics['throughput'].append(throughput)
                self.metrics['utilization'].append(utilization)
            
            # Update last values
            last_completed = completed
            last_time = current_time
            
            time.sleep(0.5)  # Collect metrics every 0.5 seconds
    
    def get_metrics(self):
        """Get processor metrics combined with scheduler metrics"""
        scheduler_metrics = self.scheduler.get_metrics()
        
        # Calculate average CPU usage and memory usage
        avg_cpu = sum(self.metrics['cpu_usage']) / len(self.metrics['cpu_usage']) if self.metrics['cpu_usage'] else 0
        avg_memory = sum(self.metrics['memory_usage']) / len(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0
        avg_throughput = sum(self.metrics['throughput']) / len(self.metrics['throughput']) if self.metrics['throughput'] else 0
        avg_utilization = sum(self.metrics['utilization']) / len(self.metrics['utilization']) if self.metrics['utilization'] else 0
        
        processor_metrics = {
            'name': self.name,
            'avg_cpu_usage': avg_cpu,
            'avg_memory_usage': avg_memory,
            'avg_throughput': avg_throughput,
            'avg_utilization': avg_utilization,
            'cpu_usage_history': self.metrics['cpu_usage'],
            'memory_usage_history': self.metrics['memory_usage'],
            'timestamp_history': self.metrics['timestamp'],
            'throughput_history': self.metrics['throughput']
        }
        
        # Combine processor and scheduler metrics
        combined_metrics = {**processor_metrics, **scheduler_metrics}
        
        return combined_metrics