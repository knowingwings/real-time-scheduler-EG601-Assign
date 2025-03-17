"""
Multi-Processor Implementation

This module handles task execution on a multi-processor system,
implementing load balancing and task distribution strategies.
"""

import threading
import time
import logging
import psutil
import numpy as np
from typing import List, Dict, Any, Type, Optional
from src.processors.single_processor import SingleProcessor
from src.task_generator import Task, Priority

class MultiProcessor:
    """
    Multi-Processor Implementation
    
    Handles task scheduling and distribution across multiple processors.
    Implements load balancing and system-wide metrics collection.
    """
    
    def __init__(self, schedulers, processor_count=4, strategy="round_robin"):
        """
        Initialize multi-processor system
        
        Args:
            schedulers: List of scheduler instances to use (one per processor)
            processor_count: Number of processors to simulate
            strategy: Load balancing strategy ('round_robin', 'least_loaded', 'priority_based')
        """
        if len(schedulers) != processor_count:
            raise ValueError(f"Expected {processor_count} schedulers, got {len(schedulers)}")
            
        self.processors = []
        for i in range(processor_count):
            processor = SingleProcessor(schedulers[i], name=f"CPU-{i+1}")
            self.processors.append(processor)
            
        self.processor_count = processor_count
        self.strategy = strategy
        self.is_running = False
        self.simulation_speed = 1.0
        self.tasks = []
        self.simulation_mode = False
        self.logger = logging.getLogger(__name__)
        
        # For round-robin strategy
        self.next_processor_index = 0
        
        # For system-wide metrics
        self.metrics = {
            'total_completed_tasks': 0,
            'avg_waiting_time': 0,
            'processor_load_balance': 0,  # Standard deviation of processor loads
            'system_throughput': 0        # Tasks per second across all processors
        }
    
    def add_tasks(self, tasks):
        """
        Add tasks to the system
        
        Tasks will be distributed across processors according to
        the chosen load balancing strategy.
        
        Args:
            tasks: List of Task objects to add
        """
        self.tasks.extend(tasks)
        self.logger.info(f"Added {len(tasks)} tasks to multi-processor system")
    
    def run(self, simulation=False, speed_factor=1.0):
        """
        Run the multi-processor system
        
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
        
        # Start all processors
        processor_threads = []
        for processor in self.processors:
            thread = threading.Thread(target=self._run_processor, args=(processor,))
            thread.daemon = True
            thread.start()
            processor_threads.append(thread)
            
        # Feed tasks to processors based on arrival times and strategy
        self._feed_tasks()
        
        # Wait for all processors to complete if running in simulation mode
        if simulation:
            for thread in processor_threads:
                thread.join()
            self.is_running = False
            self.logger.info("Multi-processor simulation completed")
    
    def _run_processor(self, processor):
        """Run a processor"""
        processor.run(simulation=self.simulation_mode, speed_factor=self.simulation_speed)
    
    def _feed_tasks(self):
        """Feed tasks to processors based on arrival times and distribution strategy"""
        if not self.tasks:
            self.logger.warning("No tasks to feed to processors")
            return
        
        # Sort tasks by arrival time
        tasks = sorted(self.tasks, key=lambda x: x.arrival_time)
        
        start_time = time.time()
        
        for task in tasks:
            if not self.is_running:
                break
                
            # Calculate time to wait before distributing this task
            if not self.simulation_mode:
                # In real mode, wait until the task's arrival time
                current_time = time.time() - start_time
                wait_time = max(0, task.arrival_time - current_time)
                
                if wait_time > 0:
                    time.sleep(wait_time)
            
            # Select processor based on strategy
            processor = self._select_processor(task)
            
            # Add task to selected processor
            processor.scheduler.add_task(task)
            self.logger.info(f"Added task {task.id} to {processor.name} using {self.strategy} strategy")
        
        # If in simulation mode, signal that all tasks have been added
        if self.simulation_mode:
            self.logger.info("All tasks distributed in simulation mode")
            
            # Wait for all tasks to complete
            all_completed = False
            while not all_completed and self.is_running:
                all_empty = True
                for processor in self.processors:
                    if (not processor.scheduler.task_queue.empty() or 
                        processor.scheduler.current_task):
                        all_empty = False
                        break
                
                if all_empty:
                    all_completed = True
                    for processor in self.processors:
                        processor.stop()
                else:
                    time.sleep(0.1)
    
    def _select_processor(self, task):
        """
        Select a processor for the task based on load balancing strategy
        
        Args:
            task: Task to assign to a processor
            
        Returns:
            Selected processor
        """
        if self.strategy == "round_robin":
            # Simple round-robin
            processor = self.processors[self.next_processor_index]
            self.next_processor_index = (self.next_processor_index + 1) % self.processor_count
            return processor
            
        elif self.strategy == "least_loaded":
            # Assign to processor with fewest queued tasks
            min_queue = float('inf')
            selected_processor = None
            
            for processor in self.processors:
                queue_size = processor.scheduler.task_queue.qsize()
                if queue_size < min_queue:
                    min_queue = queue_size
                    selected_processor = processor
            
            return selected_processor
            
        elif self.strategy == "priority_based":
            # Distribute by priority
            # High priority tasks go to processors 0-1
            # Medium priority tasks go to processor 2
            # Low priority tasks go to processor 3
            if task.priority == Priority.HIGH:
                # Distribute high priority tasks between first two processors
                if self.next_processor_index < 2:
                    processor = self.processors[self.next_processor_index]
                    self.next_processor_index = (self.next_processor_index + 1) % 2
                else:
                    processor = self.processors[0]
                    self.next_processor_index = 1
            elif task.priority == Priority.MEDIUM:
                processor = self.processors[2]
            else:  # LOW
                processor = self.processors[3]
            
            return processor
            
        else:
            # Default to first processor
            return self.processors[0]
    
    def stop(self):
        """Stop all processors"""
        self.is_running = False
        for processor in self.processors:
            processor.stop()
        self.logger.info("Stopping multi-processor system")
    
    def _collect_metrics(self):
        """Collect system-wide metrics"""
        while self.is_running:
            cpu_percent = psutil.cpu_percent(interval=None)
            memory_percent = psutil.virtual_memory().percent
            # Calculate total completed tasks
            total_completed = sum(len(p.scheduler.completed_tasks) for p in self.processors)
            
            # Calculate average waiting time across all processors
            all_waiting_times = []
            for processor in self.processors:
                waiting_times = [task.waiting_time for task in processor.scheduler.completed_tasks 
                              if task.waiting_time is not None]
                all_waiting_times.extend(waiting_times)
            
            avg_waiting_time = sum(all_waiting_times) / len(all_waiting_times) if all_waiting_times else 0
            
            # Calculate load balance (standard deviation of queue sizes)
            queue_sizes = [p.scheduler.task_queue.qsize() for p in self.processors]
            load_balance = np.std(queue_sizes) if queue_sizes else 0
            
            # Calculate system throughput (completed tasks per second)
            time_elapsed = max(p.metrics['timestamp'][-1] - p.metrics['timestamp'][0] 
                             for p in self.processors if p.metrics['timestamp']) if any(p.metrics['timestamp'] for p in self.processors) else 1
            
            system_throughput = total_completed / time_elapsed if time_elapsed > 0 else 0
            
            # Update metrics
            self.metrics['cpu_usage'] = cpu_percent
            self.metrics['memory_usage'] = memory_percent
            self.metrics['total_completed_tasks'] = total_completed
            self.metrics['avg_waiting_time'] = avg_waiting_time
            self.metrics['processor_load_balance'] = load_balance
            self.metrics['system_throughput'] = system_throughput
            
            time.sleep(1.0)  # Collect metrics every second
    
    def get_metrics(self):
        """
        Get comprehensive system metrics
        
        Returns:
            Dictionary containing system-wide and per-processor metrics
        """
        # Get metrics from each processor
        processor_metrics = [p.get_metrics() for p in self.processors]
        
        # Calculate final values from the most recent data
        total_completed = sum(metrics['completed_tasks'] for metrics in processor_metrics)
        
        # Calculate average waiting time across all processors
        all_waiting_times = []
        for processor in self.processors:
            waiting_times = [task.waiting_time for task in processor.scheduler.completed_tasks 
                          if task.waiting_time is not None]
            all_waiting_times.extend(waiting_times)
        
        avg_waiting_time = sum(all_waiting_times) / len(all_waiting_times) if all_waiting_times else 0
        
        # Calculate load balance as coefficient of variation
        completed_per_processor = [metrics['completed_tasks'] for metrics in processor_metrics]
        mean_completed = np.mean(completed_per_processor) if completed_per_processor else 0
        std_completed = np.std(completed_per_processor) if completed_per_processor else 0
        cv_completed = (std_completed / mean_completed) * 100 if mean_completed > 0 else 0
        
        # Calculate average CPU and memory usage across processors
        avg_cpu_usage = np.mean([metrics['avg_cpu_usage'] for metrics in processor_metrics])
        avg_memory_usage = np.mean([metrics['avg_memory_usage'] for metrics in processor_metrics])
        
        # Calculate system throughput
        start_times = [min(p.metrics['timestamp']) for p in self.processors if p.metrics['timestamp']]
        end_times = [max(p.metrics['timestamp']) for p in self.processors if p.metrics['timestamp']]
        
        if start_times and end_times:
            overall_start = min(start_times)
            overall_end = max(end_times)
            time_elapsed = overall_end - overall_start
            system_throughput = total_completed / time_elapsed if time_elapsed > 0 else 0
        else:
            system_throughput = 0
        
        # Compile system-wide metrics
        system_metrics = {
            'processor_count': self.processor_count,
            'strategy': self.strategy,
            'total_completed_tasks': total_completed,
            'avg_waiting_time': avg_waiting_time,
            'load_balance_cv': cv_completed,  # Lower is better (more even distribution)
            'system_throughput': system_throughput,
            'avg_cpu_usage': avg_cpu_usage,
            'avg_memory_usage': avg_memory_usage,
            'per_processor_metrics': processor_metrics
        }
        
        return system_metrics