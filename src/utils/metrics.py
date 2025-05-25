"""
Performance Metrics Calculation

This module calculates and analyses performance metrics for task scheduling.
"""

import numpy as np
import statistics
from collections import defaultdict
from src.task_generator import Priority

class MetricsCalculator:
    """Calculator for analysing scheduler performance metrics"""
    
    def __init__(self):
        """Initialise metrics calculator"""
        pass
        
    def calculate_task_metrics(self, completed_tasks):
        """
        Calculate metrics for completed tasks
        
        Args:
            completed_tasks: List of completed Task objects
            
        Returns:
            Dictionary containing calculated metrics
        """
        if not completed_tasks:
            return {
                'count': 0,
                'avg_waiting_time': 0,
                'avg_response_time': 0,
                'avg_turnaround_time': 0,
                'waiting_times_by_priority': {p.name: 0 for p in Priority}
            }
            
        # Basic metrics
        count = len(completed_tasks)
        
        # Calculate waiting time metrics
        waiting_times = [task.waiting_time for task in completed_tasks if task.waiting_time is not None]
        avg_waiting_time = sum(waiting_times) / len(waiting_times) if waiting_times else 0
        
        # Group by priority
        tasks_by_priority = defaultdict(list)
        for task in completed_tasks:
            tasks_by_priority[task.priority].append(task)
            
        waiting_times_by_priority = {}
        for priority, tasks in tasks_by_priority.items():
            priority_waiting_times = [task.waiting_time for task in tasks if task.waiting_time is not None]
            waiting_times_by_priority[priority.name] = (
                sum(priority_waiting_times) / len(priority_waiting_times) 
                if priority_waiting_times else 0
            )
        
        # Calculate response time (time between arrival and start of execution)
        response_times = [
            task.start_time - task.arrival_time 
            for task in completed_tasks 
            if task.start_time is not None
        ]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Calculate turnaround time (time between arrival and completion)
        turnaround_times = [
            task.completion_time - task.arrival_time 
            for task in completed_tasks 
            if task.completion_time is not None
        ]
        avg_turnaround_time = sum(turnaround_times) / len(turnaround_times) if turnaround_times else 0
        
        return {
            'count': count,
            'avg_waiting_time': avg_waiting_time,
            'avg_response_time': avg_response_time,
            'avg_turnaround_time': avg_turnaround_time,
            'waiting_times_by_priority': waiting_times_by_priority
        }
    
    def calculate_queue_metrics(self, queue_length_history):
        """
        Calculate metrics for queue length history
        
        Args:
            queue_length_history: List of queue lengths over time
            
        Returns:
            Dictionary containing calculated metrics
        """
        if not queue_length_history:
            return {
                'avg_queue_length': 0,
                'max_queue_length': 0,
                'queue_variance': 0
            }
            
        avg_queue_length = sum(queue_length_history) / len(queue_length_history)
        max_queue_length = max(queue_length_history)
        queue_variance = statistics.variance(queue_length_history) if len(queue_length_history) > 1 else 0
        
        return {
            'avg_queue_length': avg_queue_length,
            'max_queue_length': max_queue_length,
            'queue_variance': queue_variance
        }
    
    def calculate_deadline_metrics(self, completed_tasks):
        """
        Calculate metrics related to deadline handling
        
        Args:
            completed_tasks: List of completed Task objects
            
        Returns:
            Dictionary containing calculated metrics
        """
        if not completed_tasks:
            return {
                'deadline_tasks': 0,
                'deadline_met': 0,
                'deadline_miss_rate': 0,
                'avg_deadline_margin': 0
            }
            
        # Filter tasks with deadlines
        deadline_tasks = [task for task in completed_tasks if task.deadline is not None]
        
        if not deadline_tasks:
            return {
                'deadline_tasks': 0,
                'deadline_met': 0,
                'deadline_miss_rate': 0,
                'avg_deadline_margin': 0
            }
            
        # Count tasks that met their deadline
        deadline_met = sum(
            1 for task in deadline_tasks 
            if task.completion_time <= task.deadline
        )
        
        # Calculate deadline miss rate
        deadline_miss_rate = (len(deadline_tasks) - deadline_met) / len(deadline_tasks)
        
        # Calculate average margin (positive = met deadline, negative = missed deadline)
        deadline_margins = [
            task.deadline - task.completion_time 
            for task in deadline_tasks 
            if task.completion_time is not None
        ]
        avg_deadline_margin = sum(deadline_margins) / len(deadline_margins) if deadline_margins else 0
        
        return {
            'deadline_tasks': len(deadline_tasks),
            'deadline_met': deadline_met,
            'deadline_miss_rate': deadline_miss_rate,
            'avg_deadline_margin': avg_deadline_margin
        }
    
    def calculate_priority_metrics(self, completed_tasks, priority_inversions=0):
        """
        Calculate metrics related to priority handling
        
        Args:
            completed_tasks: List of completed Task objects
            priority_inversions: Number of priority inversions detected
            
        Returns:
            Dictionary containing calculated metrics
        """
        if not completed_tasks:
            return {
                'priority_inversions': priority_inversions,
                'priority_ratio': {}
            }
            
        # Group tasks by priority
        tasks_by_priority = defaultdict(list)
        for task in completed_tasks:
            tasks_by_priority[task.priority].append(task)
            
        # Calculate average waiting time ratios between priorities
        waiting_times_by_priority = {}
        for priority, tasks in tasks_by_priority.items():
            waiting_times = [task.waiting_time for task in tasks if task.waiting_time is not None]
            waiting_times_by_priority[priority] = (
                sum(waiting_times) / len(waiting_times) 
                if waiting_times else 0
            )
        
        # Calculate ratios between priority levels (how much longer low priority tasks wait)
        priority_ratio = {}
        if Priority.HIGH in waiting_times_by_priority and waiting_times_by_priority[Priority.HIGH] > 0:
            # Medium to High ratio
            if Priority.MEDIUM in waiting_times_by_priority:
                ratio = waiting_times_by_priority[Priority.MEDIUM] / waiting_times_by_priority[Priority.HIGH]
                priority_ratio['MEDIUM_TO_HIGH'] = ratio
                
            # Low to High ratio
            if Priority.LOW in waiting_times_by_priority:
                ratio = waiting_times_by_priority[Priority.LOW] / waiting_times_by_priority[Priority.HIGH]
                priority_ratio['LOW_TO_HIGH'] = ratio
        
        return {
            'priority_inversions': priority_inversions,
            'priority_ratio': priority_ratio
        }
    
    def calculate_processor_metrics(self, processor_metrics, throughput_history=None):
        """
        Calculate metrics for processor utilisation
        
        Args:
            processor_metrics: Dictionary containing processor metrics
            throughput_history: List of throughput values over time
            
        Returns:
            Dictionary containing calculated metrics
        """
        avg_cpu_usage = processor_metrics.get('avg_cpu_usage', 0)
        avg_memory_usage = processor_metrics.get('avg_memory_usage', 0)
        
        # Calculate throughput metrics if available
        throughput_metrics = {}
        if throughput_history:
            throughput_metrics['avg_throughput'] = sum(throughput_history) / len(throughput_history)
            throughput_metrics['max_throughput'] = max(throughput_history)
            throughput_metrics['throughput_variance'] = statistics.variance(throughput_history) if len(throughput_history) > 1 else 0
        
        return {
            'avg_cpu_usage': avg_cpu_usage,
            'avg_memory_usage': avg_memory_usage,
            **throughput_metrics
        }
    
    def compare_algorithms(self, metrics_by_algorithm):
        """
        Compare performance across different algorithms
        
        Args:
            metrics_by_algorithm: Dictionary mapping algorithm names to their metrics
            
        Returns:
            Dictionary containing comparison results
        """
        if not metrics_by_algorithm:
            return {}
            
        comparison = {}
        
        # Compare waiting times
        waiting_times = {
            algo: metrics.get('avg_waiting_time', float('inf'))
            for algo, metrics in metrics_by_algorithm.items()
        }
        
        best_waiting = min(waiting_times.items(), key=lambda x: x[1])
        comparison['best_waiting_time'] = {
            'algorithm': best_waiting[0],
            'value': best_waiting[1]
        }
        
        # Compare deadline handling (if applicable)
        deadline_metrics = {
            algo: metrics.get('deadline_miss_rate', 1.0)
            for algo, metrics in metrics_by_algorithm.items()
            if 'deadline_miss_rate' in metrics
        }
        
        if deadline_metrics:
            best_deadline = min(deadline_metrics.items(), key=lambda x: x[1])
            comparison['best_deadline_handling'] = {
                'algorithm': best_deadline[0],
                'miss_rate': best_deadline[1]
            }
        
        # Compare priority handling (if applicable)
        priority_metrics = {
            algo: metrics.get('priority_inversions', float('inf'))
            for algo, metrics in metrics_by_algorithm.items()
            if 'priority_inversions' in metrics
        }
        
        if priority_metrics:
            best_priority = min(priority_metrics.items(), key=lambda x: x[1])
            comparison['best_priority_handling'] = {
                'algorithm': best_priority[0],
                'inversions': best_priority[1]
            }
        
        # Compare throughput (if available)
        throughput_metrics = {
            algo: metrics.get('avg_throughput', 0)
            for algo, metrics in metrics_by_algorithm.items()
            if 'avg_throughput' in metrics
        }
        
        if throughput_metrics:
            best_throughput = max(throughput_metrics.items(), key=lambda x: x[1])
            comparison['best_throughput'] = {
                'algorithm': best_throughput[0],
                'value': best_throughput[1]
            }
        
        return comparison
    
    def compare_processors(self, single_processor_metrics, multi_processor_metrics):
        """
        Compare performance between single and multi-processor systems
        
        Args:
            single_processor_metrics: Metrics from single processor
            multi_processor_metrics: Metrics from multi-processor
            
        Returns:
            Dictionary containing comparison results
        """
        if not single_processor_metrics or not multi_processor_metrics:
            return {}
        
        comparison = {}
        
        # Calculate speedup
        single_throughput = single_processor_metrics.get('avg_throughput', 0)
        multi_throughput = multi_processor_metrics.get('system_throughput', 0)
        
        if single_throughput > 0:
            speedup = multi_throughput / single_throughput
        else:
            speedup = 0
        
        # Calculate efficiency (speedup / processor count)
        processor_count = multi_processor_metrics.get('processor_count', 0)
        if processor_count > 0:
            efficiency = speedup / processor_count
        else:
            efficiency = 0
        
        # Compare waiting times
        single_waiting = single_processor_metrics.get('avg_waiting_time', 0)
        multi_waiting = multi_processor_metrics.get('avg_waiting_time', 0)
        waiting_improvement = ((single_waiting - multi_waiting) / single_waiting) * 100 if single_waiting > 0 else 0
        
        comparison['speedup'] = speedup
        comparison['efficiency'] = efficiency
        comparison['waiting_improvement'] = waiting_improvement
        
        # Determine if multi-processor was worthwhile
        if speedup > 1.5 and efficiency > 0.5:
            comparison['recommendation'] = "Multi-processor configuration provides significant benefits."
        elif speedup > 1.0:
            comparison['recommendation'] = "Multi-processor provides some benefits, but efficiency could be improved."
        else:
            comparison['recommendation'] = "Single processor may be more cost-effective for this workload."
        
        return comparison
    
    def analyse_ml_metrics(self, ml_metrics):
        """
        Analyse machine learning scheduler performance
        
        Args:
            ml_metrics: Metrics from ML-based scheduler
            
        Returns:
            Dictionary containing analysis results
        """
        if not ml_metrics:
            return {}
        
        analysis = {}
        
        # Analyse prediction error
        avg_error = ml_metrics.get('average_prediction_error', 0)
        prediction_errors = ml_metrics.get('prediction_errors', [])
        
        if prediction_errors:
            min_error = min(prediction_errors)
            max_error = max(prediction_errors)
            error_std = np.std(prediction_errors) if len(prediction_errors) > 1 else 0
            
            analysis['prediction_performance'] = {
                'avg_error': avg_error,
                'min_error': min_error,
                'max_error': max_error,
                'error_std': error_std
            }
            
            # Evaluate ML effectiveness
            if avg_error < 0.5:  # Less than 0.5 second error on average
                analysis['effectiveness'] = "Excellent prediction accuracy"
            elif avg_error < 1.0:
                analysis['effectiveness'] = "Good prediction accuracy"
            elif avg_error < 2.0:
                analysis['effectiveness'] = "Moderate prediction accuracy"
            else:
                analysis['effectiveness'] = "Poor prediction accuracy"
        
        # Compare with waiting time performance
        avg_waiting = ml_metrics.get('average_waiting_time', 0)
        analysis['avg_waiting_time'] = avg_waiting
        
        return analysis
    
"""
Standardized Metrics Collector

This module provides a unified interface for collecting metrics across 
different schedulers to ensure consistent data collection.
"""

import time
import threading
import psutil
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Union
from collections import defaultdict
from enum import Enum

from src.task_generator import Priority

logger = logging.getLogger(__name__)

class MetricsCollector:
    """
    Unified metrics collector that can be used with any scheduler.
    
    Ensures consistent metric collection across different scheduler implementations.
    """
    
    def __init__(self, scheduler_name: str, processor_name: str = "Unknown"):
        """
        Initialize metrics collector
        
        Args:
            scheduler_name: Name of the scheduler (FCFS, EDF, Priority, ML)
            processor_name: Name of the processor (e.g., CPU-1)
        """
        self.scheduler_name = scheduler_name
        self.processor_name = processor_name
        self.start_time = time.time()
        self.lock = threading.RLock()
        self.running = False
        self.collection_thread = None
        
        # Scheduler and processor metrics
        self.queue_length_history = []
        self.memory_usage_history = []
        self.timestamp_history = []
        self.cpu_usage_history = []
        self.throughput_history = []
        self.completed_task_count_history = []
        
        # Priority-specific metrics with defaults
        self.waiting_times_by_priority = {'HIGH': [], 'MEDIUM': [], 'LOW': []}
        self.response_times_by_priority = {'HIGH': [], 'MEDIUM': [], 'LOW': []}
        self.turnaround_times_by_priority = {'HIGH': [], 'MEDIUM': [], 'LOW': []}
        
        # Deadline tracking
        self.deadline_tasks = 0
        self.deadline_met = 0
        self.deadline_misses = 0
        self.deadline_margins = []
        
        # System performance
        self.last_collection_time = time.time()
        self.last_completed_count = 0
        
        # Priority inversion tracking
        self.priority_inversions = 0
        self.priority_inheritance_events = 0
        self.priority_inversion_durations = []
        
        # ML-specific metrics
        self.prediction_errors = []
        self.training_events = []
        self.feature_importances = {}
        
        # Collection settings
        self.collection_interval = SIMULATION['metrics_collection_interval']
        self.min_collection_interval = 0.1  # Minimum 100ms between collections
    
    def start_collection(self):
        """Start the metrics collection process"""
        with self.lock:
            self.running = True
            self.start_time = time.time()
            self.collection_thread = threading.Thread(target=self._collect_metrics)
            self.collection_thread.daemon = True
            self.collection_thread.start()
            logger.info(f"Started metrics collection for {self.scheduler_name} on {self.processor_name}")
    
    def stop_collection(self):
        """Stop the metrics collection process"""
        with self.lock:
            self.running = False
            logger.info(f"Stopped metrics collection for {self.scheduler_name} on {self.processor_name}")
            # Don't join thread here to avoid blocking
    
    def _collect_metrics(self):
        """Background thread for collecting system metrics"""
        logger.debug(f"Metrics collection thread started for {self.scheduler_name}")
        last_sample_time = time.time()
        
        while self.running:
            try:
                current_time = time.time()
                time_since_last = current_time - last_sample_time
                
                # Only collect if minimum interval has passed
                if time_since_last >= self.min_collection_interval:
                    relative_time = round(current_time - self.start_time, 3)
                    last_sample_time = current_time
                    
                    with self.lock:
                        # Only collect real metrics, no synthetic data
                        try:
                            memory_percent = psutil.virtual_memory().percent
                            cpu_percent = psutil.cpu_percent(interval=None)
                            queue_size = self._get_queue_size()
                            completed_count = len(self.completed_tasks)
                            
                            # Only calculate throughput from real task completions
                            if completed_count > self.last_completed_count:
                                interval_completed = completed_count - self.last_completed_count
                                throughput = interval_completed / time_since_last
                                self.throughput_history.append(throughput)
                                
                            # Only append real collected values
                            self.memory_usage_history.append(memory_percent)
                            self.cpu_usage_history.append(cpu_percent)
                            self.timestamp_history.append(relative_time)
                            self.queue_length_history.append(queue_size)
                            self.completed_task_count_history.append(completed_count)
                            
                            self.last_completed_count = completed_count
                            
                        except Exception as e:
                            # Log error but don't generate synthetic data
                            logger.error(f"Error collecting metrics: {str(e)}")
                
                # Sleep for a short time to prevent busy waiting
                time.sleep(min(self.collection_interval, self.min_collection_interval))
                
            except Exception as e:
                logger.error(f"Critical error in metrics collection: {str(e)}")
                time.sleep(self.collection_interval)
    
    def _get_queue_size(self) -> int:
        """
        Get current queue size from associated scheduler
        
        Returns:
            Current queue size or 0 if not available
        """
        try:
            # This is a placeholder - in actual implementation we would
            # get queue size from the scheduler instance
            # Since we don't have direct access to the scheduler here,
            # we'll rely on task_queue_updated method to be called
            return self.queue_length_history[-1] if self.queue_length_history else 0
        except Exception as e:
            logger.warning(f"Error getting queue size: {str(e)}")
            return 0
    
    def _update_throughput(self):
        """Calculate and update throughput metrics properly"""
        current_time = time.time()
        execution_time = current_time - self.start_time
        
        if execution_time > 0:
            # Calculate instantaneous throughput
            completed_since_last = len(self.completed_tasks) - self.last_completed_count
            time_since_last = current_time - self.last_collection_time
            
            if time_since_last > 0:
                instant_throughput = completed_since_last / time_since_last
                
                # Use exponential moving average for stability
                alpha = 0.3  # Smoothing factor
                if not self.throughput_history:
                    smoothed_throughput = instant_throughput
                else:
                    smoothed_throughput = (alpha * instant_throughput + 
                                         (1 - alpha) * self.throughput_history[-1])
                
                # Ensure throughput is non-negative and reasonable
                smoothed_throughput = max(0, min(smoothed_throughput, 1000))
                self.throughput_history.append(smoothed_throughput)
                
            # Update counters
            self.last_completed_count = len(self.completed_tasks)
            self.last_collection_time = current_time

    def _track_priority_inversions(self, task):
        """Improved priority inversion detection and tracking"""
        if not hasattr(task, 'priority') or not self.current_task:
            return
            
        current_priority = getattr(self.current_task, 'priority', None)
        new_task_priority = task.priority
        
        if current_priority and new_task_priority:
            # Check if higher priority task is waiting for lower priority task
            if new_task_priority.value > current_priority.value:
                self.priority_inversions += 1
                
                # Track duration of inversion
                self.priority_inversion_durations.append({
                    'start_time': time.time(),
                    'high_priority_task': task.id,
                    'low_priority_task': self.current_task.id,
                    'duration': 0  # Will be updated when inversion ends
                })
                
                # Signal priority inheritance if supported
                if hasattr(self, 'apply_priority_inheritance'):
                    self.priority_inheritance_events += 1
                    self.apply_priority_inheritance(self.current_task, new_task_priority)

    def task_queue_updated(self, queue_size: int):
        """
        Update the task queue size metric
        
        Args:
            queue_size: Current queue size
        """
        with self.lock:
            # Record current time
            current_time = time.time()
            relative_time = round(current_time - self.start_time, 3)
            
            self.queue_length_history.append(queue_size)
            self.timestamp_history.append(relative_time)
    
    def task_started(self, task):
        """
        Record that a task has started execution
        
        Args:
            task: The task that started
        """
        with self.lock:
            self.current_task = task
            
            # Record start time if not set
            if task.start_time is None:
                task.start_time = round(time.time() - self.start_time, 3)
                logger.debug(f"Task {task.id} started at {task.start_time:.2f}s")
    
    def task_completed(self, task):
        """
        Record that a task has completed
        
        Args:
            task: The completed task
        """
        with self.lock:
            completion_time = time.time()
            task.completion_time = completion_time
            
            # Calculate and validate waiting time
            if task.start_time and task.arrival_time:
                task.waiting_time = task.start_time - task.arrival_time
                if task.waiting_time < 0:
                    logger.warning(f"Negative waiting time for task {task.id}, setting to 0")
                    task.waiting_time = 0
            
            # Update priority-specific metrics
            if hasattr(task, 'priority'):
                priority_name = task.priority.name if hasattr(task.priority, 'name') else str(task.priority)
                if priority_name in self.waiting_times_by_priority:
                    if task.waiting_time is not None:
                        self.waiting_times_by_priority[priority_name].append(task.waiting_time)
            
            # Check if this completion resolves any priority inversions
            current_time = time.time()
            for inversion in self.priority_inversion_durations:
                if (inversion['duration'] == 0 and 
                    (inversion['low_priority_task'] == task.id or 
                     inversion['high_priority_task'] == task.id)):
                    inversion['duration'] = current_time - inversion['start_time']
            
            self.completed_tasks.append(task)
            self._update_throughput()
    
    def priority_inversion_detected(self):
        """Record that a priority inversion was detected"""
        with self.lock:
            self.priority_inversions += 1
    
    def priority_inheritance_applied(self):
        """Record that priority inheritance was applied"""
        with self.lock:
            self.priority_inheritance_events += 1
    
    def ml_training_occurred(self, feature_importances=None):
        """
        Record that ML model training occurred
        
        Args:
            feature_importances: Optional dictionary of feature importances
        """
        with self.lock:
            self.training_events.append(round(time.time() - self.start_time, 3))
            
            if feature_importances:
                self.feature_importances = feature_importances
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get all collected metrics
        
        Returns:
            Dictionary containing all metrics
        """
        with self.lock:
            # Calculate waiting time metrics safely
            waiting_times = [t.waiting_time for t in self.completed_tasks if t.waiting_time is not None]
            avg_waiting_time = sum(waiting_times) / len(waiting_times) if waiting_times else 0
            
            # Calculate waiting times by priority safely
            avg_wait_by_priority = {}
            for priority, times in self.waiting_times_by_priority.items():
                avg_wait_by_priority[priority] = round(sum(times) / len(times), 3) if times else 0
            
            # Count tasks by priority safely
            tasks_by_priority = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
            for task in self.completed_tasks:
                if hasattr(task, 'priority'):
                    priority_name = task.priority.name if hasattr(task.priority, 'name') else str(task.priority)
                    if priority_name in tasks_by_priority:
                        tasks_by_priority[priority_name] += 1
            
            # Calculate throughput metrics
            current_time = time.time()
            execution_time = current_time - self.start_time
            throughput = len(self.completed_tasks) / execution_time if execution_time > 0 else 0
            
            # Calculate deadline metrics
            if self.deadline_tasks > 0:
                deadline_miss_rate = (self.deadline_tasks - self.deadline_met) / self.deadline_tasks
            else:
                deadline_miss_rate = 0
            
            # Calculate average prediction error for ML metrics
            avg_prediction_error = (sum(self.prediction_errors) / len(self.prediction_errors) 
                                 if self.prediction_errors else 0)
            
            # Calculate priority inversion metrics
            avg_inversion_duration = 0
            if self.priority_inversion_durations:
                completed_inversions = [inv['duration'] for inv in self.priority_inversion_durations 
                                     if inv['duration'] > 0]
                if completed_inversions:
                    avg_inversion_duration = sum(completed_inversions) / len(completed_inversions)
            
            return {
                'completed_tasks': len(self.completed_tasks),
                'avg_waiting_time': round(avg_waiting_time, 3),
                'avg_waiting_by_priority': avg_wait_by_priority,
                'tasks_by_priority': tasks_by_priority,
                'queue_length_history': self.queue_length_history,
                
                # Performance metrics
                'avg_cpu_usage': round(np.mean(self.cpu_usage_history) if self.cpu_usage_history else 0, 3),
                'avg_memory_usage': round(np.mean(self.memory_usage_history) if self.memory_usage_history else 0, 3),
                'avg_throughput': round(throughput, 3),
                'throughput_history': self.throughput_history,
                
                # Deadline metrics
                'deadline_tasks': self.deadline_tasks,
                'deadline_met': self.deadline_met,
                'deadline_misses': self.deadline_misses,
                'deadline_miss_rate': round(deadline_miss_rate, 3),
                
                # Priority inversion metrics
                'priority_inversions': self.priority_inversions,
                'priority_inheritance_events': self.priority_inheritance_events,
                'avg_inversion_duration': round(avg_inversion_duration, 3),
                
                # Time series data
                'cpu_usage_history': self.cpu_usage_history,
                'memory_usage_history': self.memory_usage_history,
                'timestamp_history': self.timestamp_history
            }