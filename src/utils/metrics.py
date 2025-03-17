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