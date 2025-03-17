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
    
    def __init__(self):
        # Using PriorityQueue for deadline-based ordering
        self.task_queue = PriorityQueue()
        self.current_task = None
        self.completed_tasks = []
        self.lock = threading.Lock()
        self.running = False
        self.current_time = 0
        self.preempted_tasks = []
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
        based on arrival time plus service time.
        """
        if task.deadline is None:
            # Create a soft deadline for tasks that don't have one
            task.deadline = task.arrival_time + task.service_time * 2
        
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
        
        # Start metrics collection in a separate thread
        metrics_thread = threading.Thread(target=self._collect_metrics)
        metrics_thread.daemon = True
        metrics_thread.start()
        
        while self.running:
            # Record current state
            with self.lock:
                queue_size = self.task_queue.qsize() + len(self.preempted_tasks)
                self.metrics['queue_length'].append(queue_size)
            
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
                elapsed = time.time() - self.current_task.start_time if not simulation else self.current_time - self.current_task.start_time
                self.current_task.service_time -= elapsed
                # Add to preempted tasks list
                self.preempted_tasks.append(self.current_task)
                self.current_task = None
            
            if task:
                # If simulation, we might need to advance time
                if simulation and self.current_time < task.arrival_time:
                    self.current_time = task.arrival_time
                
                # Set as current task and start execution
                self.current_task = task
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
                'timestamp_history': self.metrics['timestamp'],
                'deadline_misses': self.metrics['deadline_misses']
            }