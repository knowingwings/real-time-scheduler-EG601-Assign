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
from queue import Empty
from typing import List, Dict, Any, Type, Optional
from src.processors.single_processor import SingleProcessor
from src.task_generator import Task, Priority
from config.params import SIMULATION

class MultiProcessor:
    """
    Multi-Processor Implementation
    
    Handles task scheduling and distribution across multiple processors.
    Implements load balancing and system-wide metrics collection.
    """
    
    def __init__(self, schedulers, processor_count=4, strategy="round_robin"):
        """Initialize multi-processor system"""
        self.processor_count = processor_count
        self.strategy = strategy
        self.processors = []
        self.running = False
        self.tasks = []
        self.current_processor = 0
        self.logger = logging.getLogger(__name__)
        self.metrics_lock = threading.Lock()
        
        # Initialize metrics tracking
        self._initialize_metrics()
        
        # Create processors
        for i in range(processor_count):
            scheduler = schedulers[i] if i < len(schedulers) else schedulers[0].__class__()
            processor = SingleProcessor(scheduler, f"CPU-{i+1}")
            self.processors.append(processor)
    
    def _initialize_metrics(self):
        """Initialize system-wide metrics"""
        self.metrics = {
            'processor_metrics': {},
            'load_balance': {
                'cv_history': [],           # Coefficient of variation history
                'imbalance_events': 0,      # Times CV exceeded threshold
                'rebalancing_actions': 0,   # Number of load balancing actions
                'processor_loads': {},      # Load history by processor
            },
            'system_metrics': {
                'total_completed': 0,
                'avg_waiting_time': 0,
                'avg_turnaround_time': 0,
                'throughput': [],
                'cpu_usage': [],
                'memory_usage': [],
                'timestamp': []
            }
        }
        
        # Initialize per-processor tracking
        for proc in self.processors:
            self.metrics['processor_metrics'][proc.name] = {
                'queue_length': [],
                'completed_tasks': 0,
                'waiting_times': [],
                'utilization': []
            }
    
    def add_tasks(self, tasks):
        """Add tasks to the system"""
        self.tasks.extend(tasks)
        self.logger.info(f"Added {len(tasks)} tasks to multi-processor system")
    
    def run(self, simulation=False, speed_factor=1.0):
        """Run the multi-processor system"""
        self.running = True
        
        # Start metrics collection
        metrics_thread = threading.Thread(target=self._collect_metrics)
        metrics_thread.daemon = True
        metrics_thread.start()
        
        # Start load balance monitoring
        load_monitor_thread = threading.Thread(target=self._monitor_load_balance)
        load_monitor_thread.daemon = True
        load_monitor_thread.start()
        
        # Start all processors
        processor_threads = []
        for processor in self.processors:
            thread = threading.Thread(
                target=self._run_processor,
                args=(processor, simulation, speed_factor)
            )
            thread.daemon = True
            thread.start()
            processor_threads.append(thread)
        
        # Feed tasks to processors
        self._feed_tasks(simulation)
        
        # Wait for completion in simulation mode
        if simulation:
            for thread in processor_threads:
                thread.join()
            self.running = False
            self.logger.info("Multi-processor simulation completed")
    
    def _run_processor(self, processor, simulation, speed_factor):
        """Run a single processor"""
        processor.run(simulation=simulation, speed_factor=speed_factor)
    
    def _feed_tasks(self, simulation=False):
        """Feed tasks to processors based on arrival times"""
        if not self.tasks:
            self.logger.warning("No tasks to distribute")
            return
            
        # Sort tasks by arrival time
        tasks = sorted(self.tasks, key=lambda x: x.arrival_time)
        start_time = time.time()
        
        for task in tasks:
            if not self.running:
                break
                
            if simulation:
                # In simulation, add tasks immediately
                processor = self._select_processor(task)
                processor.scheduler.add_task(task)
                self.logger.debug(
                    f"Added task {task.id} to {processor.name} at "
                    f"simulated time {task.arrival_time:.2f}s"
                )
            else:
                # In real mode, respect arrival times
                current_time = time.time() - start_time
                wait_time = max(0, task.arrival_time - current_time)
                
                if wait_time > 0:
                    time.sleep(wait_time)
                    
                processor = self._select_processor(task)
                processor.scheduler.add_task(task)
                self.logger.debug(
                    f"Added task {task.id} to {processor.name} at "
                    f"time {time.time() - start_time:.2f}s"
                )
        
        # In simulation, wait for completion
        if simulation:
            all_completed = False
            while not all_completed and self.running:
                all_completed = True
                for processor in self.processors:
                    if (not processor.scheduler.task_queue.empty() or 
                        processor.scheduler.current_task):
                        all_completed = False
                        break
                if not all_completed:
                    time.sleep(0.1)
    
    def _select_processor(self, task):
        """Select processor based on strategy"""
        if self.strategy == "round_robin":
            processor = self.processors[self.current_processor]
            self.current_processor = (self.current_processor + 1) % len(self.processors)
            return processor
        elif self.strategy == "least_loaded":
            return self._select_processor_least_loaded()
        else:  # priority_based or default
            return self._select_processor_balanced()
    
    def _select_processor_balanced(self):
        """Select processor using improved load balancing"""
        best_processor = None
        best_score = float('inf')
        
        # Calculate average metrics across processors
        total_queue_size = sum(p.scheduler.task_queue.qsize() + len(getattr(p.scheduler, 'preempted_tasks', [])) 
                             for p in self.processors)
        avg_queue_size = total_queue_size / len(self.processors) if self.processors else 0
        
        for processor in self.processors:
            # Calculate normalized load score components
            queue_size = processor.scheduler.task_queue.qsize()
            preempted_count = len(getattr(processor.scheduler, 'preempted_tasks', []))
            current_task = 1 if processor.scheduler.current_task else 0
            
            # Calculate queue imbalance score
            total_load = queue_size + preempted_count + current_task
            imbalance_score = abs(total_load - avg_queue_size) / (avg_queue_size + 1)
            
            # Get recent throughput
            recent_throughput = 1.0
            try:
                with processor.metrics_lock:
                    throughput_history = processor.metrics.get('throughput', [])
                    if throughput_history:
                        # Use exponential moving average of recent throughput
                        alpha = 0.3  # Smoothing factor
                        recent_throughput = 0
                        for i, t in enumerate(reversed(throughput_history[-5:])):
                            recent_throughput = alpha * t + (1 - alpha) * recent_throughput
                        recent_throughput = max(0.1, recent_throughput)
            except Exception as e:
                self.logger.warning(f"Error calculating throughput: {e}")
            
            # Calculate ML capability score if applicable
            ml_score = 1.0
            if hasattr(processor.scheduler, 'model'):
                try:
                    with processor.scheduler.metrics_lock:
                        if processor.scheduler.trained:
                            errors = processor.scheduler.metrics.get('prediction_errors', [])
                            if errors:
                                recent_errors = errors[-5:]
                                ml_score = sum(recent_errors) / len(recent_errors)
                        else:
                            ml_score = 2.0
                except Exception as e:
                    self.logger.warning(f"Error calculating ML score: {e}")
                    ml_score = 2.0
            
            # Combined score with weighted components
            load_weight = 0.5
            throughput_weight = 0.3
            ml_weight = 0.2
            
            score = (imbalance_score * load_weight + 
                    (1/recent_throughput) * throughput_weight + 
                    ml_score * ml_weight)
            
            if score < best_score:
                best_score = score
                best_processor = processor
        
        return best_processor or self.processors[0]
    
    def _select_processor_least_loaded(self):
        """Select least loaded processor"""
        return min(self.processors, 
                  key=lambda p: (p.scheduler.task_queue.qsize() + 
                               len(getattr(p.scheduler, 'preempted_tasks', [])) +
                               (1 if p.scheduler.current_task else 0)))
    
    def _monitor_load_balance(self):
        """Monitor and correct load imbalances between processors"""
        while self.running:
            try:
                # Calculate load metrics
                loads = []
                avg_load = 0
                total_tasks = 0
                
                for processor in self.processors:
                    queue_size = processor.scheduler.task_queue.qsize()
                    preempted = len(getattr(processor.scheduler, 'preempted_tasks', []))
                    current = 1 if processor.scheduler.current_task else 0
                    load = queue_size + preempted + current
                    loads.append(load)
                    total_tasks += load
                
                if self.processors:
                    avg_load = total_tasks / len(self.processors)
                    
                    # Calculate coefficient of variation
                    if avg_load > 0:
                        variance = sum((load - avg_load) ** 2 for load in loads) / len(loads)
                        cv = (variance ** 0.5) / avg_load
                        
                        with self.metrics_lock:
                            self.metrics['load_balance']['cv_history'].append(cv)
                            
                            # Track processor loads
                            for i, load in enumerate(loads):
                                proc_name = self.processors[i].name
                                if proc_name not in self.metrics['load_balance']['processor_loads']:
                                    self.metrics['load_balance']['processor_loads'][proc_name] = []
                                self.metrics['load_balance']['processor_loads'][proc_name].append(load)
                        
                        # If CV is too high, trigger load balancing
                        if cv > 0.5:  # CV threshold of 50%
                            with self.metrics_lock:
                                self.metrics['load_balance']['imbalance_events'] += 1
                            self._rebalance_loads(loads, avg_load)
                
                time.sleep(SIMULATION['metrics_collection_interval'])
                
            except Exception as e:
                self.logger.error(f"Error in load balance monitoring: {e}")
                time.sleep(SIMULATION['metrics_collection_interval'])
    
    def _rebalance_loads(self, loads, avg_load):
        """Rebalance loads between processors"""
        try:
            # Sort processors by load difference from average
            processor_loads = list(zip(self.processors, loads))
            processor_loads.sort(key=lambda x: x[1] - avg_load)
            
            # Find overloaded and underloaded processors
            overloaded = [(p, l) for p, l in processor_loads if l > avg_load * 1.2]
            underloaded = [(p, l) for p, l in processor_loads if l < avg_load * 0.8]
            
            rebalancing_occurred = False
            
            # Balance between overloaded and underloaded processors
            for over_proc, over_load in overloaded:
                for under_proc, under_load in underloaded:
                    # Calculate how many tasks to move
                    tasks_to_move = min(
                        int((over_load - avg_load) / 2),  # Don't move all excess
                        int((avg_load - under_load))      # Don't overload receiver
                    )
                    
                    if tasks_to_move > 0:
                        # Move tasks from overloaded to underloaded processor
                        moved = 0
                        while moved < tasks_to_move:
                            try:
                                # Try to get a task from the overloaded processor
                                task = over_proc.scheduler.task_queue.get_nowait()
                                
                                # Add to underloaded processor
                                under_proc.scheduler.add_task(task)
                                moved += 1
                                rebalancing_occurred = True
                                
                            except Empty:
                                break  # No more tasks to move
                            
                        if moved > 0:
                            self.logger.info(
                                f"Moved {moved} tasks from {over_proc.name} "
                                f"to {under_proc.name} for load balancing"
                            )
            
            if rebalancing_occurred:
                with self.metrics_lock:
                    self.metrics['load_balance']['rebalancing_actions'] += 1
                    
        except Exception as e:
            self.logger.error(f"Error during load rebalancing: {e}")
    
    def _collect_metrics(self):
        """Collect system-wide metrics"""
        while self.running:
            try:
                current_time = time.time()
                
                # Collect processor metrics
                cpu_usage = []
                memory_usage = []
                completed_tasks = 0
                waiting_times = []
                
                for processor in self.processors:
                    metrics = processor.get_metrics()
                    
                    # Update processor-specific metrics
                    with self.metrics_lock:
                        proc_metrics = self.metrics['processor_metrics'][processor.name]
                        proc_metrics['queue_length'].append(
                            processor.scheduler.task_queue.qsize()
                        )
                        proc_metrics['completed_tasks'] = metrics['completed_tasks']
                        proc_metrics['waiting_times'].extend(
                            t.waiting_time for t in processor.scheduler.completed_tasks
                            if t.waiting_time is not None
                        )
                        proc_metrics['utilization'].append(metrics['avg_cpu_usage'])
                    
                    # Collect system-wide metrics
                    cpu_usage.append(metrics['avg_cpu_usage'])
                    memory_usage.append(metrics['avg_memory_usage'])
                    completed_tasks += metrics['completed_tasks']
                    waiting_times.extend(proc_metrics['waiting_times'])
                
                # Update system metrics
                with self.metrics_lock:
                    self.metrics['system_metrics']['total_completed'] = completed_tasks
                    
                    if waiting_times:
                        self.metrics['system_metrics']['avg_waiting_time'] = (
                            sum(waiting_times) / len(waiting_times)
                        )
                    
                    # Calculate system-wide averages
                    if cpu_usage:
                        self.metrics['system_metrics']['cpu_usage'].append(
                            sum(cpu_usage) / len(cpu_usage)
                        )
                    if memory_usage:
                        self.metrics['system_metrics']['memory_usage'].append(
                            sum(memory_usage) / len(memory_usage)
                        )
                    
                    # Calculate current throughput
                    elapsed = current_time - self.start_time if hasattr(self, 'start_time') else 0
                    if elapsed > 0:
                        throughput = completed_tasks / elapsed
                        self.metrics['system_metrics']['throughput'].append(throughput)
                    
                    self.metrics['system_metrics']['timestamp'].append(current_time)
                
                time.sleep(SIMULATION['metrics_collection_interval'])
                
            except Exception as e:
                self.logger.error(f"Error collecting system metrics: {e}")
                time.sleep(SIMULATION['metrics_collection_interval'])
    
    def stop(self):
        """Stop the multi-processor system"""
        self.running = False
        for processor in self.processors:
            processor.stop()
        self.logger.info("Stopped multi-processor system")
    
    def get_metrics(self):
        """Get comprehensive system metrics"""
        with self.metrics_lock:
            # Calculate system-wide averages
            system_metrics = self.metrics['system_metrics']
            processor_metrics = self.metrics['processor_metrics']
            load_balance = self.metrics['load_balance']
            
            # Calculate load balance metrics
            cv_history = load_balance['cv_history']
            avg_cv = sum(cv_history) / len(cv_history) if cv_history else 0
            
            metrics = {
                'processor_count': len(self.processors),
                'strategy': self.strategy,
                'total_completed_tasks': system_metrics['total_completed'],
                
                # System-wide averages
                'avg_waiting_time': system_metrics['avg_waiting_time'],
                'avg_turnaround_time': system_metrics.get('avg_turnaround_time', 0),
                'system_throughput': (system_metrics['throughput'][-1] 
                                    if system_metrics['throughput'] else 0),
                'avg_cpu_usage': (sum(system_metrics['cpu_usage']) / 
                                len(system_metrics['cpu_usage'])
                                if system_metrics['cpu_usage'] else 0),
                'avg_memory_usage': (sum(system_metrics['memory_usage']) / 
                                   len(system_metrics['memory_usage'])
                                   if system_metrics['memory_usage'] else 0),
                
                # Load balancing metrics
                'load_balance_cv': avg_cv,
                'imbalance_events': load_balance['imbalance_events'],
                'rebalancing_actions': load_balance['rebalancing_actions'],
                
                # Per-processor metrics
                'per_processor_metrics': []
            }
            
            # Add task priority information
            all_tasks_by_priority = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
            all_waiting_by_priority = {'HIGH': [], 'MEDIUM': [], 'LOW': []}
            
            # Collect per-processor metrics
            for processor in self.processors:
                proc_metrics = processor.get_metrics()
                
                # Update priority counts and waiting times
                for priority, count in proc_metrics['tasks_by_priority'].items():
                    all_tasks_by_priority[priority] += count
                
                for priority, wait_time in proc_metrics['avg_waiting_by_priority'].items():
                    if wait_time > 0:
                        all_waiting_by_priority[priority].append(wait_time)
                
                metrics['per_processor_metrics'].append(proc_metrics)
            
            # Calculate overall priority metrics
            metrics['tasks_by_priority'] = all_tasks_by_priority
            metrics['avg_waiting_by_priority'] = {
                priority: (sum(times) / len(times) if times else 0)
                for priority, times in all_waiting_by_priority.items()
            }
            
            return metrics