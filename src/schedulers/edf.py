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
        self.metrics = {
            'queue_length': [],
            'memory_usage': [],
            'timestamp': [],
            'deadline_misses': 0
        }
        self.logger = logging.getLogger(__name__)
    
    def add_task(self, task):
        """
        Add a task to the scheduler queue
        
        For tasks without explicit deadlines, we create a soft deadline
        based on arrival time, service time, and system load.
        """
        if task.deadline is None:
            # Create a more realistic deadline based on priority and estimated system load
            priority_factor = 1.0
            if hasattr(task, 'priority'):
                # Adjust deadline based on priority (higher priority = lower factor)
                if task.priority.name == 'HIGH':
                    priority_factor = 0.7
                elif task.priority.name == 'MEDIUM':
                    priority_factor = 1.0
                else:  # LOW
                    priority_factor = 1.5
            
            # Estimate current queue size for load estimation
            queue_size = self.task_queue.qsize() + len(self.preempted_tasks)
            
            # Adjust deadline factor based on system load
            adjusted_factor = self.deadline_factor * (1 + (queue_size * 0.1))
            
            # Calculate deadline with priority consideration
            task.deadline = task.arrival_time + (task.service_time * adjusted_factor * priority_factor)
        
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
        start_time = time.time()  # Store real-time start
        
        # Start metrics collection in a separate thread
        metrics_thread = threading.Thread(target=self._collect_metrics)
        metrics_thread.daemon = True
        metrics_thread.start()
        
        while self.running:
            # Record current state
            with self.lock:
                queue_size = self.task_queue.qsize() + len(self.preempted_tasks)
                self.metrics['queue_length'].append(queue_size)
                
                # Update system load estimate
                self.system_load = queue_size / 10.0 if queue_size > 0 else 0.0
            
            # Check if we need to resume a preempted task first
            task = None
            preempt_current = False
            
            if self.preempted_tasks:
                # Sort preempted tasks by deadline
                self.preempted_tasks.sort(key=lambda x: x.deadline)
                next_preempted = self.preempted_tasks[0]
                
                if not self.task_queue.empty():
                    # Check if any new task has an earlier deadline
                    deadline, _, queue_task = self.task_queue.queue[0]  # Peek
                    
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
            
            # Handle preemption if needed
            if preempt_current and self.current_task:
                self.logger.info(f"Preempting task {self.current_task.id} for task {task.id} with earlier deadline")
                # Save the remaining execution time of the current task
                if simulation:
                    elapsed = self.current_time - self.current_task.start_time
                else:
                    elapsed = time.time() - start_time - self.current_task.start_time
                
                self.current_task.service_time -= min(elapsed, self.current_task.service_time)  # Avoid negative service time
                # Add to preempted tasks list
                self.preempted_tasks.append(self.current_task)
                self.current_task = None
            
            if task:
                # If simulation, we might need to advance time
                if simulation and self.current_time < task.arrival_time:
                    self.current_time = task.arrival_time
                
                # Set waiting time if not already set
                if task.waiting_time is None:
                    if simulation:
                        current_t = self.current_time
                    else:
                        current_t = time.time() - start_time
                        
                    task.waiting_time = round(max(0, current_t - task.arrival_time), 3)
                
                # Set as current task and start execution
                self.current_task = task
                
                # Record start time
                if simulation:
                    task.start_time = self.current_time
                else:
                    task.start_time = time.time() - start_time
                    
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
                    task.completion_time = time.time() - start_time
                    
                self.logger.info(f"Completed execution of {task.id} at time {task.completion_time:.2f}")
                
                # Check if deadline was missed
                if task.completion_time > task.deadline:
                    self.logger.warning(f"Task {task.id} missed deadline by {task.completion_time - task.deadline:.2f}s")
                    with self.lock:
                        self.metrics['deadline_misses'] += 1
                
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
                    # In simulation, advance time slightly
                    time.sleep(0.01 / speed_factor)
    
    def stop(self):
        """Stop the scheduler"""
        self.running = False
    
    def _collect_metrics(self):
        """Collect system metrics during execution"""
        start_time = time.time()  # Record the absolute start time
        
        while self.running:
            current_time = time.time()
            relative_time = round(current_time - start_time, 3)  # Calculate relative time in seconds
            memory_percent = psutil.virtual_memory().percent
            queue_size = self.task_queue.qsize() + len(self.preempted_tasks)
            
            with self.lock:
                self.metrics['memory_usage'].append(memory_percent)
                self.metrics['timestamp'].append(relative_time)
                self.metrics['queue_length'].append(queue_size)
            
            time.sleep(0.5)  # Collect metrics every 0.5 seconds
    
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
            
            return {
                'completed_tasks': len(self.completed_tasks),
                'avg_waiting_time': round(avg_waiting_time, 3),
                'avg_completion_time': round(avg_completion_time, 3),
                'avg_waiting_by_priority': avg_wait_by_priority,
                'tasks_by_priority': tasks_by_priority,
                'queue_length_history': self.metrics['queue_length'],
                'memory_usage_history': self.metrics['memory_usage'],
                'timestamp_history': self.metrics['timestamp'],
                'deadline_misses': self.metrics['deadline_misses']
            }