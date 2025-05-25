"""
Priority-Based Scheduler with Priority Inversion Handling

Implements a preemptive priority-based scheduling algorithm with
priority inheritance to handle priority inversion issues.
"""

import time
import threading
import psutil
import logging
from queue import PriorityQueue
from src.task_generator import Priority

class PriorityScheduler:
    """
    Priority-Based Scheduler with Priority Inversion Handling
    
    Tasks are executed based on their priority, with higher priority tasks
    preempting lower priority ones. Implements Priority Inheritance Protocol (PIP)
    to handle priority inversion.
    """
    
    def __init__(self):
        # Using PriorityQueue for priority-based ordering
        self.task_queue = PriorityQueue()
        self.current_task = None
        self.completed_tasks = []
        self.lock = threading.RLock()  # Changed to RLock for nested acquisitions
        self.running = False
        self.current_time = 0
        self.preempted_tasks = []
        # Resources for priority inversion handling
        self.resources = {}  # Map of resource_id -> {owner, waiters, allocation_time}
        self.metrics = {
            'queue_length': [],
            'memory_usage': [],
            'timestamp': [],
            'priority_inversions': 0,
            'priority_inheritance_events': 0
        }
        self.logger = logging.getLogger(__name__)
    
    def add_task(self, task):
        """Add a task to the scheduler queue"""
        # Lower numerical value means higher priority (1 = HIGH, 3 = LOW)
        priority_value = task.priority.value
        
        # Add to queue with priority value as key
        self.task_queue.put((priority_value, id(task), task))
        self.logger.info(f"Task {task.id} added to Priority queue with priority {task.priority.name}")
    
    def allocate_resource(self, task, resource_id):
        """
        Allocate a resource to a task
        
        Used for simulating resource conflicts and priority inversion
        """
        with self.lock:
            if resource_id in self.resources and self.resources[resource_id]['owner']:
                # Resource is already allocated
                owner = self.resources[resource_id]['owner']
                
                # Check for potential priority inversion
                if task.priority.value < owner.priority.value:  # Lower value = higher priority
                    self.logger.warning(f"Priority inversion detected: {task.id} (P:{task.priority.name}) "
                                       f"waiting for {owner.id} (P:{owner.priority.name})")
                    self.metrics['priority_inversions'] += 1
                    
                    # Apply priority inheritance
                    original_priority = owner.priority
                    owner.priority = task.priority  # Temporarily boost owner's priority
                    self.logger.info(f"Priority inheritance: {owner.id} priority boosted to {task.priority.name}")
                    self.metrics['priority_inheritance_events'] += 1
                    
                    # Add original priority to restore it later
                    if 'original_priority' not in owner.__dict__:
                        owner.original_priority = original_priority
                
                # Add to waiters list
                self.resources[resource_id]['waiters'].append(task)
                return False
            else:
                # Allocate resource
                self.resources[resource_id] = {
                    'owner': task,
                    'waiters': [],
                    'allocation_time': time.time()  # Track allocation time
                }
                return True
    
    def release_resource(self, task, resource_id):
        """
        Release a resource allocated to a task
        
        Handles priority restoration and waiter notification
        """
        with self.lock:
            if (resource_id in self.resources and 
                self.resources[resource_id]['owner'] == task):
                
                # Calculate blocking duration for metrics
                allocation_time = self.resources[resource_id].get('allocation_time', 0)
                if allocation_time > 0:
                    blocking_duration = time.time() - allocation_time
                    self.logger.info(f"Resource {resource_id} was held for {blocking_duration:.2f}s")
                
                # Restore original priority if it was boosted
                if hasattr(task, 'original_priority'):
                    task.priority = task.original_priority
                    delattr(task, 'original_priority')
                    self.logger.info(f"Priority restored for {task.id} to {task.priority.name}")
                
                # Handle waiters
                waiters = self.resources[resource_id]['waiters']
                if waiters:
                    # Allocate to next waiter
                    next_waiter = waiters.pop(0)
                    self.resources[resource_id]['owner'] = next_waiter
                    self.resources[resource_id]['allocation_time'] = time.time()
                    self.logger.info(f"Resource {resource_id} allocated to waiting task {next_waiter.id}")
                    
                    # Re-add this task to ready queue
                    self.add_task(next_waiter)
                else:
                    # No waiters, resource is free
                    self.resources[resource_id] = {
                        'owner': None,
                        'waiters': [],
                        'allocation_time': 0
                    }
                    
                return True
            return False
    
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
            
            # Check if we need to resume a preempted task first
            task = None
            preempt_current = False
            
            if self.preempted_tasks:
                # Sort preempted tasks by priority
                self.preempted_tasks.sort(key=lambda x: x.priority.value)
                next_preempted = self.preempted_tasks[0]
                
                if not self.task_queue.empty():
                    # Check if any new task has a higher priority
                    priority, _, queue_task = self.task_queue.queue[0]  # Peek
                    
                    if priority < next_preempted.priority.value:
                        # New task has higher priority, use it instead
                        _, _, task = self.task_queue.get()
                    else:
                        # Resume the preempted task with highest priority
                        task = self.preempted_tasks.pop(0)
                else:
                    # No new tasks, resume the preempted task
                    task = self.preempted_tasks.pop(0)
            elif not self.task_queue.empty():
                # Get the task with highest priority
                _, _, task = self.task_queue.get()
                
                # Check if this new task should preempt the current task
                if (self.current_task and 
                    task.priority.value < self.current_task.priority.value):
                    preempt_current = True
            
            # Handle preemption if needed
            if preempt_current and self.current_task:
                self.logger.info(f"Preempting task {self.current_task.id} for task {task.id} with higher priority")
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
                        
                    task.waiting_time = max(0, round(current_t - task.arrival_time, 3))
                    
                    if task.waiting_time > 0:
                        self.logger.info(f"Task {task.id} waited for {task.waiting_time}s")
                
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
                    # Scale sleep time by speed factor
                    sleep_time = task.service_time / max(0.1, speed_factor)
                    time.sleep(sleep_time)
                else:
                    # Actually sleep for the service time in real execution
                    time.sleep(task.service_time)
                
                # Record completion time
                if simulation:
                    task.completion_time = self.current_time
                else:
                    task.completion_time = time.time() - start_time
                    
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
                    # In simulation, advance time slightly
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
        logger.debug("Starting metrics collection for Priority scheduler")
        
        while self.running:
            try:
                current_time = time.time()
                relative_time = round(current_time - self.start_time, 3)
                
                with self.lock:
                    # Only collect real measurements
                    memory_percent = psutil.virtual_memory().percent
                    queue_size = self.task_queue.qsize() + len(self.preempted_tasks)
                    
                    # Only append actual measurements
                    self.metrics['timestamp'].append(relative_time)
                    self.metrics['memory_usage'].append(memory_percent)
                    self.metrics['queue_length'].append(queue_size)
            
            except Exception as e:
                logger.error(f"Error collecting metrics: {str(e)}")
            
            time.sleep(0.5)
    
    def get_metrics(self):
        """Get execution metrics"""
        with self.lock:
            # Calculate waiting times by priority
            waiting_times = {}
            for priority in Priority:
                priority_tasks = [task for task in self.completed_tasks 
                                if task.priority == priority and task.waiting_time is not None]
                if priority_tasks:
                    waiting_times[priority.name] = round(
                        sum(t.waiting_time for t in priority_tasks) / len(priority_tasks), 
                        3
                    )
                else:
                    waiting_times[priority.name] = 0
            
            # Calculate overall average waiting time
            all_waiting_times = [task.waiting_time for task in self.completed_tasks 
                            if task.waiting_time is not None]
            avg_waiting_time = sum(all_waiting_times) / len(all_waiting_times) if all_waiting_times else 0
            
            # Count tasks by priority
            tasks_by_priority = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
            for task in self.completed_tasks:
                if hasattr(task, 'priority'):
                    priority_name = task.priority.name if hasattr(task.priority, 'name') else str(task.priority)
                    if priority_name in tasks_by_priority:
                        tasks_by_priority[priority_name] += 1
            
            # Count deadline metrics
            deadline_tasks = 0
            deadline_met = 0
            for task in self.completed_tasks:
                if task.deadline is not None:
                    deadline_tasks += 1
                    if task.completion_time <= task.deadline:
                        deadline_met += 1
            
            deadline_miss_rate = (deadline_tasks - deadline_met) / deadline_tasks if deadline_tasks > 0 else 0
            
            return {
                'completed_tasks': len(self.completed_tasks),
                'avg_waiting_time': round(avg_waiting_time, 3),  # Add standardized overall average
                'avg_waiting_by_priority': waiting_times,  # Standardized key name
                'tasks_by_priority': tasks_by_priority,  # Add priority analysis
                'queue_length_history': self.metrics['queue_length'],
                'memory_usage_history': self.metrics['memory_usage'],
                'timestamp_history': self.metrics['timestamp'],
                'priority_inversions': self.metrics['priority_inversions'],
                'priority_inheritance_events': self.metrics['priority_inheritance_events'],
                'deadline_tasks': deadline_tasks,
                'deadline_met': deadline_met,
                'deadline_miss_rate': deadline_miss_rate
            }