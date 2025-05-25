"""
Earliest Deadline First (EDF) Scheduler

Implements a preemptive EDF scheduling algorithm where tasks with
earlier deadlines are given higher priority.
"""

import time
import heapq
import threading
import psutil
import logging
import math
from queue import PriorityQueue

class EDFScheduler:
    """
    Earliest Deadline First Scheduler
    
    Tasks are executed based on their deadlines, with earlier deadlines
    having higher priority. EDF is an optimal dynamic priority algorithm
    for real-time systems when the system is not overloaded.
    """
    
    def __init__(self, deadline_factor=5.0):
        # Using PriorityQueue for deadline-based ordering
        self.task_queue = PriorityQueue()
        self.current_task = None
        self.completed_tasks = []
        self.lock = threading.Lock()
        self.running = False
        self.current_time = 0
        self.preempted_tasks = []
        self.deadline_factor = deadline_factor  # Multiplier for deadline calculation
        self.system_load = 0.0  # Estimated system load
        
        # Track preemption overhead
        self.preemption_count = 0
        self.total_preemption_overhead = 0
        self.avg_preemption_overhead = 0.001  # Initial estimate (1ms)
        
        self.metrics = {
            'queue_length': [],
            'memory_usage': [],
            'timestamp': [],
            'deadline_misses': 0,
            'deadline_met': 0,
            'deadline_tasks': 0,
            'preemptions': 0,
            'avg_preemption_overhead': 0.0
        }
        self.logger = logging.getLogger(__name__)
    
    def add_task(self, task):
        """
        Add a task to the scheduler queue with improved deadline calculation
        """
        if task.deadline is None:
            # More reasonable deadline calculation based on load and priority
            priority_factor = 1.0
            if hasattr(task, 'priority'):
                if task.priority.name == 'HIGH':
                    priority_factor = 1.2  # Give high priority tasks more time
                elif task.priority.name == 'MEDIUM':
                    priority_factor = 1.0
                else:  # LOW
                    priority_factor = 0.8  # Less time for low priority
            
            # Calculate load factor based on current queue size
            queue_size = self.task_queue.qsize() + len(self.preempted_tasks)
            load_factor = 1.0 + (0.1 * min(queue_size, 10))  # Cap load impact
            
            # Set deadline relative to arrival time and expected execution time
            base_deadline = task.service_time * self.deadline_factor
            task.deadline = task.arrival_time + (base_deadline * priority_factor * load_factor)
        
        # Add to queue with deadline as priority
        self.task_queue.put((task.deadline, id(task), task))
        self.logger.info(f"Task {task.id} added to EDF queue with deadline {task.deadline:.2f}")
    
    def run(self, simulation=False, speed_factor=1.0):
        """
        Run the scheduler
        
        Args:
            simulation: If True, time is simulated rather than real
            speed_factor: Speed up or slow down simulation (only used if simulation=True)
        """
        self.running = True
        self.current_time = 0
        start_time = time.time()
        
        # Start metrics collection
        metrics_thread = threading.Thread(target=self._collect_metrics)
        metrics_thread.daemon = True
        metrics_thread.start()
        
        while self.running:
            # Record current state
            with self.lock:
                queue_size = self.task_queue.qsize() + len(self.preempted_tasks)
                self.metrics['queue_length'].append(queue_size)
                
                # Update system load estimate with smoothing
                target_load = queue_size / 10.0 if queue_size > 0 else 0.0
                self.system_load = (self.system_load * 0.8) + (target_load * 0.2)
            
            # Handle task selection with preemption
            task = None
            preempt_current = False
            
            if self.preempted_tasks:
                # Sort preempted tasks by deadline
                self.preempted_tasks.sort(key=lambda x: x.deadline)
                next_preempted = self.preempted_tasks[0]
                
                if not self.task_queue.empty():
                    # Check if any new task has an earlier deadline
                    deadline, _, queue_task = self.task_queue.queue[0]
                    
                    if deadline < next_preempted.deadline:
                        # New task has an earlier deadline, use it instead
                        _, _, task = self.task_queue.get()
                    else:
                        # Resume the preempted task with earliest deadline
                        task = self.preempted_tasks.pop(0)
                else:
                    # No new tasks, resume the preempted task
                    task = self.preempted_tasks.pop(0)
            elif not self.task_queue.empty():
                # Get the task with earliest deadline
                _, _, task = self.task_queue.get()
                
                # Check if this new task should preempt the current task
                if self.current_task and task.deadline < self.current_task.deadline:
                    preempt_current = True
            
            # Handle preemption
            preemption_start = None
            if preempt_current and self.current_task:
                preemption_start = time.time() if not simulation else self.current_time
                self.logger.info(f"Preempting task {self.current_task.id} for task {task.id} with earlier deadline")
                
                # Save remaining execution time
                if simulation:
                    elapsed = self.current_time - self.current_task.start_time
                else:
                    elapsed = time.time() - start_time - self.current_task.start_time
                
                self.current_task.service_time -= min(elapsed, self.current_task.service_time)
                self.preempted_tasks.append(self.current_task)
                self.current_task = None
                
                # Track preemption
                self.preemption_count += 1
                self.metrics['preemptions'] += 1
            
            if task:
                # Account for preemption overhead
                if preemption_start:
                    preemption_end = time.time() if not simulation else self.current_time
                    overhead = preemption_end - preemption_start
                    self.total_preemption_overhead += overhead
                    self.avg_preemption_overhead = self.total_preemption_overhead / max(1, self.preemption_count)
                    self.metrics['avg_preemption_overhead'] = self.avg_preemption_overhead
                
                # If simulation, advance time if needed
                if simulation and self.current_time < task.arrival_time:
                    self.current_time = task.arrival_time
                
                # Calculate waiting time
                if task.waiting_time is None:
                    if simulation:
                        current_t = self.current_time
                    else:
                        current_t = time.time() - start_time
                    
                    task.waiting_time = round(max(0, current_t - task.arrival_time), 3)
                
                # Set as current task and record start
                self.current_task = task
                if simulation:
                    task.start_time = self.current_time
                else:
                    task.start_time = time.time() - start_time
                
                self.logger.info(f"Starting execution of {task.id} at time {task.start_time:.2f}")
                
                # Execute the task
                if simulation:
                    # Account for preemption overhead in simulation
                    if preemption_start:
                        self.current_time += self.avg_preemption_overhead
                    
                    self.current_time += task.service_time
                    time.sleep(task.service_time / speed_factor)
                else:
                    # Add small delay for preemption overhead in real execution
                    if preemption_start:
                        time.sleep(self.avg_preemption_overhead)
                    time.sleep(task.service_time)
                
                # Record completion and check deadline
                if simulation:
                    task.completion_time = self.current_time
                else:
                    task.completion_time = time.time() - start_time
                
                self.logger.info(f"Completed execution of {task.id} at time {task.completion_time:.2f}")
                
                # Track deadline metrics
                if task.deadline is not None:
                    with self.lock:
                        self.metrics['deadline_tasks'] += 1
                        if task.completion_time > task.deadline:
                            self.logger.warning(f"Task {task.id} missed deadline by {task.completion_time - task.deadline:.2f}s")
                            self.metrics['deadline_misses'] += 1
                        else:
                            self.metrics['deadline_met'] += 1
                
                # Calculate metrics and add to completed
                task.calculate_metrics()
                with self.lock:
                    self.completed_tasks.append(task)
                    self.current_task = None
            else:
                # No tasks to process
                if not simulation:
                    time.sleep(0.1)
                else:
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
        """Collect only real system metrics during execution"""
        logger.debug("Starting metrics collection for EDF scheduler")
        
        while self.running:
            try:
                current_time = time.time()
                relative_time = round(current_time - self.start_time, 3)
                
                with self.lock:
                    # Only collect real measurements
                    memory_percent = psutil.virtual_memory().percent
                    queue_size = self.task_queue.qsize()
                    
                    # Only append if we got valid measurements
                    self.metrics['timestamp'].append(relative_time)
                    self.metrics['memory_usage'].append(memory_percent)
                    self.metrics['queue_length'].append(queue_size)
            
            except Exception as e:
                logger.error(f"Error collecting metrics: {str(e)}")
            
            time.sleep(0.5)
    
    def get_metrics(self):
        """Get execution metrics"""
        with self.lock:
            # Calculate average waiting time with error handling
            waiting_times = [task.waiting_time for task in self.completed_tasks 
                            if task.waiting_time is not None]
            avg_waiting_time = sum(waiting_times) / len(waiting_times) if waiting_times else 0
            
            # Calculate average completion time
            completion_times = [(task.completion_time - task.arrival_time) 
                            for task in self.completed_tasks 
                            if task.completion_time is not None]
            avg_completion_time = sum(completion_times) / len(completion_times) if completion_times else 0
            
            # Calculate waiting times by priority
            waiting_by_priority = {'HIGH': [], 'MEDIUM': [], 'LOW': []}
            
            for task in self.completed_tasks:
                if task.waiting_time is not None and hasattr(task, 'priority'):
                    priority_name = task.priority.name if hasattr(task.priority, 'name') else str(task.priority)
                    if priority_name in waiting_by_priority:
                        waiting_by_priority[priority_name].append(task.waiting_time)
            
            avg_wait_by_priority = {}
            for priority, times in waiting_by_priority.items():
                avg_wait_by_priority[priority] = round(sum(times) / len(times), 3) if times else 0
            
            # Count tasks by priority
            tasks_by_priority = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
            for task in self.completed_tasks:
                if hasattr(task, 'priority'):
                    priority_name = task.priority.name if hasattr(task.priority, 'name') else str(task.priority)
                    if priority_name in tasks_by_priority:
                        tasks_by_priority[priority_name] += 1
            
            # Count deadline metrics properly
            deadline_tasks = 0
            deadline_met = 0
            for task in self.completed_tasks:
                if hasattr(task, 'deadline') and task.deadline is not None:
                    deadline_tasks += 1
                    if task.completion_time <= task.deadline:
                        deadline_met += 1

            # Calculate deadline miss rate safely
            if deadline_tasks > 0:
                deadline_miss_rate = (deadline_tasks - deadline_met) / deadline_tasks 
            else:
                deadline_miss_rate = 0.0

            # Ensure deadline miss rate is between 0 and 1
            deadline_miss_rate = max(0.0, min(1.0, deadline_miss_rate))

            return {
                'completed_tasks': len(self.completed_tasks),
                'avg_waiting_time': round(avg_waiting_time, 3),
                'avg_completion_time': round(avg_completion_time, 3),
                'avg_waiting_by_priority': avg_wait_by_priority,
                'tasks_by_priority': tasks_by_priority,
                'queue_length_history': self.metrics['queue_length'],
                'memory_usage_history': self.metrics['memory_usage'],
                'timestamp_history': self.metrics['timestamp'],
                'deadline_misses': self.metrics['deadline_misses'],
                'deadline_tasks': deadline_tasks,
                'deadline_met': deadline_met,
                'deadline_miss_rate': deadline_miss_rate
            }