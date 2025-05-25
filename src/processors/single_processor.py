"""
Single Processor Implementation

This module handles task execution on a single processor system.
"""

import threading
import time
import logging
import psutil
from config.params import SIMULATION

class SingleProcessor:
    """
    Single Processor Implementation
    
    Handles task scheduling and execution on a single processor.
    """
    
    def __init__(self, scheduler, name="CPU-1"):
        """
        Initialise single processor with a scheduler
        
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
            'utilisation': []  # Percentage of time the processor is busy
        }
        self.metrics_lock = threading.Lock()
        self.last_metrics_time = time.time()
        self.last_completed_count = 0
    
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
        
        # Start resource monitoring
        resource_monitor_thread = threading.Thread(target=self._monitor_resources)
        resource_monitor_thread.daemon = True
        resource_monitor_thread.start()
        
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

    def _initialize_metrics(self):
        """Initialize metrics with proper defaults to prevent NaN issues"""
        self.metrics = {
            'cpu_usage': [],
            'memory_usage': [],
            'timestamp': [],
            'throughput': [],
            'utilisation': []
        }
    
    def _collect_metrics(self):
        """Collect system metrics during execution with improved error handling and initialization"""
        # Initialize metrics if not already done
        if not all(key in self.metrics for key in ['cpu_usage', 'memory_usage', 'timestamp', 'throughput', 'utilisation']):
            self.metrics = {
                'cpu_usage': [],
                'memory_usage': [],
                'timestamp': [],
                'throughput': [],  # Tasks per second
                'utilisation': []  # Percentage of time the processor is busy
            }
        
        # Ensure all lists are initialized even if empty
        for key in ['cpu_usage', 'memory_usage', 'timestamp', 'throughput', 'utilisation']:
            if key not in self.metrics or self.metrics[key] is None:
                self.metrics[key] = []
        
        last_completed = 0
        last_time = time.time()
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Get CPU and memory metrics safely
                try:
                    cpu_percent = psutil.cpu_percent(interval=None)
                except:
                    cpu_percent = 0
                    
                try:
                    memory_percent = psutil.virtual_memory().percent
                except:
                    memory_percent = 0
                
                # Get number of completed tasks safely
                completed = 0
                if hasattr(self.scheduler, 'completed_tasks'):
                    completed = len(self.scheduler.completed_tasks)
                
                # Calculate throughput (tasks per second) safely
                elapsed = current_time - last_time
                throughput = 0
                if elapsed > 0:
                    throughput = (completed - last_completed) / elapsed
                
                # Calculate processor utilisation safely
                utilisation = 0
                if hasattr(self.scheduler, 'current_task') and self.scheduler.current_task:
                    utilisation = SIMULATION['processor_busy_utilization']
                
                # Update metrics safely
                with self.scheduler.lock:
                    self.metrics['cpu_usage'].append(cpu_percent)
                    self.metrics['memory_usage'].append(memory_percent)
                    self.metrics['timestamp'].append(current_time)
                    self.metrics['throughput'].append(throughput)
                    self.metrics['utilisation'].append(utilisation)
                
                # Update last values
                last_completed = completed
                last_time = current_time
                
                time.sleep(SIMULATION['metrics_collection_interval'])
            except Exception as e:
                # Log error but continue collecting
                print(f"Error collecting metrics: {str(e)}")
                time.sleep(SIMULATION['metrics_collection_interval'])  # Continue collecting

    def _monitor_resources(self):
        """Monitor CPU and memory utilization"""
        while self.is_running:
            try:
                # Get process-specific CPU usage
                process = psutil.Process()
                cpu_percent = process.cpu_percent(interval=0.1)
                memory_percent = process.memory_percent()
                
                with self.metrics_lock:
                    # Append valid measurements only
                    if 0 <= cpu_percent <= 100:
                        self.metrics['cpu_usage'].append(cpu_percent)
                    if 0 <= memory_percent <= 100:
                        self.metrics['memory_usage'].append(memory_percent)
                    
                    # Calculate throughput for current interval
                    current_time = time.time()
                    elapsed = current_time - self.last_metrics_time
                    completed_since_last = len(self.scheduler.completed_tasks) - self.last_completed_count
                    
                    if elapsed >= 0.1:  # Minimum 100ms interval
                        throughput = completed_since_last / elapsed
                        if throughput >= 0:  # Only record valid throughput
                            self.metrics['throughput'].append(throughput)
                            
                        # Update counters
                        self.last_metrics_time = current_time
                        self.last_completed_count = len(self.scheduler.completed_tasks)
                
                # Sleep briefly to prevent excessive CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error monitoring resources: {e}")
                time.sleep(0.1)

    def get_metrics(self):
        """Get processor metrics with validation"""
        with self.metrics_lock:
            # Calculate CPU usage average
            cpu_usage = self.metrics.get('cpu_usage', [])
            avg_cpu_usage = (sum(cpu_usage) / len(cpu_usage) 
                           if cpu_usage else 0)
            
            # Calculate memory usage average
            memory_usage = self.metrics.get('memory_usage', [])
            avg_memory_usage = (sum(memory_usage) / len(memory_usage) 
                              if memory_usage else 0)
            
            # Calculate throughput metrics
            throughput = self.metrics.get('throughput', [])
            avg_throughput = (sum(throughput) / len(throughput) 
                            if throughput else 0)
            
            # Get scheduler metrics
            scheduler_metrics = self.scheduler.get_metrics()
            
            # Combine and validate metrics
            metrics = {
                'name': self.name,
                'avg_cpu_usage': round(max(0, min(100, avg_cpu_usage)), 1),
                'avg_memory_usage': round(max(0, min(100, avg_memory_usage)), 1),
                'avg_throughput': round(max(0, avg_throughput), 3),
                'avg_utilisation': round(max(0, min(100, avg_cpu_usage)), 1),
                
                # Include history data
                'cpu_usage_history': [round(max(0, min(100, x)), 1) for x in cpu_usage],
                'memory_usage_history': [round(max(0, min(100, x)), 1) for x in memory_usage],
                'timestamp_history': self.metrics.get('timestamp', []),
                'throughput_history': [round(max(0, x), 3) for x in throughput],
                
                # Include scheduler metrics with validation
                'completed_tasks': max(0, scheduler_metrics.get('completed_tasks', 0)),
                'avg_waiting_time': max(0, scheduler_metrics.get('avg_waiting_time', 0)),
                'avg_completion_time': max(0, scheduler_metrics.get('avg_completion_time', 0)),
                'avg_waiting_by_priority': {
                    k: max(0, v) for k, v in scheduler_metrics.get('avg_waiting_by_priority', {}).items()
                },
                'tasks_by_priority': {
                    k: max(0, v) for k, v in scheduler_metrics.get('tasks_by_priority', {}).items()
                }
            }
            
            # Add additional scheduler-specific metrics if available
            for key in ['priority_inversions', 'deadline_miss_rate', 'prediction_errors']:
                if key in scheduler_metrics:
                    metrics[key] = max(0, scheduler_metrics[key])
            
            return metrics