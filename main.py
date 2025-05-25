#!/usr/bin/env python3
"""
Real-Time Task Scheduling System for Raspberry Pi 3
Author: Student Name
Module: EG6801 - Real-Time Embedded System
"""

import numpy as np
import threading
import queue
import time
import psutil
import logging
import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
import tkinter as tk
from tkinter import ttk, scrolledtext
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from datetime import datetime
import multiprocessing as mp
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
HIGH_PRIORITY = 1
MEDIUM_PRIORITY = 2
LOW_PRIORITY = 3

# Task specifications as per assignment
TASK_SPECS = {
    HIGH_PRIORITY: {
        'count': 20,
        'lambda_param': 3.0,  # Poisson distribution parameter
        'service_range': (2, 5),
        'name': 'High'
    },
    MEDIUM_PRIORITY: {
        'count': 20,
        'lambda_param': 5.0,
        'service_range': (3, 7),
        'name': 'Medium'
    },
    LOW_PRIORITY: {
        'count': 10,
        'lambda_param': 7.0,
        'service_range': (5, 10),
        'name': 'Low'
    }
}

class TaskState(Enum):
    """Task execution states"""
    WAITING = "WAITING"
    EXECUTING = "EXECUTING"
    COMPLETED = "COMPLETED"
    PREEMPTED = "PREEMPTED"

@dataclass
class Task:
    """Task representation with all required attributes"""
    task_id: str
    arrival_time: float
    service_time: float
    priority: int
    deadline: Optional[float] = None
    state: TaskState = TaskState.WAITING
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    preemption_count: int = 0
    remaining_service_time: float = field(init=False)
    
    def __post_init__(self):
        self.remaining_service_time = self.service_time
        if self.deadline is None:
            # Set deadline based on priority (tighter for high priority)
            deadline_factor = {1: 1.5, 2: 2.0, 3: 2.5}
            self.deadline = self.arrival_time + self.service_time * deadline_factor[self.priority]
    
    @property
    def waiting_time(self) -> float:
        """Calculate waiting time"""
        if self.start_time is None:
            return 0
        return self.start_time - self.arrival_time
    
    @property
    def response_time(self) -> float:
        """Calculate response time"""
        if self.completion_time is None:
            return 0
        return self.completion_time - self.arrival_time
    
    @property
    def turnaround_time(self) -> float:
        """Calculate turnaround time"""
        return self.response_time
    
    def __lt__(self, other):
        """For priority queue comparison"""
        return self.priority < other.priority

class TaskGenerator:
    """Generate tasks according to Poisson distribution"""
    
    @staticmethod
    def generate_tasks(seed: Optional[int] = None) -> List[Task]:
        """Generate 50 tasks with specified distributions"""
        if seed is not None:
            np.random.seed(seed)
        
        tasks = []
        task_counter = 1
        
        for priority, spec in TASK_SPECS.items():
            current_time = 0
            
            for i in range(spec['count']):
                # Generate inter-arrival time using exponential distribution
                # (Poisson process has exponential inter-arrival times)
                inter_arrival = np.random.exponential(spec['lambda_param'])
                current_time += inter_arrival
                
                # Generate service time uniformly within range
                service_time = np.random.uniform(*spec['service_range'])
                
                task = Task(
                    task_id=f"Task_{task_counter}",
                    arrival_time=current_time,
                    service_time=service_time,
                    priority=priority
                )
                
                tasks.append(task)
                task_counter += 1
                
                logger.debug(f"Generated {task.task_id}: arrival={task.arrival_time:.2f}, "
                           f"service={task.service_time:.2f}, priority={spec['name']}")
        
        # Sort tasks by arrival time
        tasks.sort(key=lambda t: t.arrival_time)
        return tasks

class BaseScheduler:
    """Base class for all scheduling algorithms"""
    
    def __init__(self, name: str):
        self.name = name
        self.ready_queue = []
        self.completed_tasks = []
        self.current_time = 0
        self.metrics = {
            'queue_lengths': [],
            'memory_usage': [],
            'cpu_usage': [],
            'timestamps': []
        }
    
    def add_task(self, task: Task):
        """Add task to ready queue"""
        raise NotImplementedError
    
    def get_next_task(self) -> Optional[Task]:
        """Get next task to execute"""
        raise NotImplementedError
    
    def record_metrics(self):
        """Record system metrics"""
        self.metrics['queue_lengths'].append(len(self.ready_queue))
        self.metrics['memory_usage'].append(psutil.Process().memory_info().rss / 1024 / 1024)  # MB
        self.metrics['cpu_usage'].append(psutil.cpu_percent(interval=0.1))
        self.metrics['timestamps'].append(self.current_time)

class FCFSScheduler(BaseScheduler):
    """First-Come-First-Served Scheduler"""
    
    def __init__(self):
        super().__init__("FCFS")
        self.ready_queue = deque()
    
    def add_task(self, task: Task):
        self.ready_queue.append(task)
    
    def get_next_task(self) -> Optional[Task]:
        if self.ready_queue:
            return self.ready_queue.popleft()
        return None

class EDFScheduler(BaseScheduler):
    """Earliest Deadline First Scheduler"""
    
    def __init__(self):
        super().__init__("EDF")
    
    def add_task(self, task: Task):
        self.ready_queue.append(task)
        # Sort by deadline
        self.ready_queue.sort(key=lambda t: t.deadline)
    
    def get_next_task(self) -> Optional[Task]:
        if self.ready_queue:
            return self.ready_queue.pop(0)
        return None

class PriorityScheduler(BaseScheduler):
    """Priority-Based Scheduler with Priority Inheritance"""
    
    def __init__(self):
        super().__init__("Priority")
        self.priority_inheritance_active = {}
    
    def add_task(self, task: Task):
        self.ready_queue.append(task)
        # Sort by priority (lower number = higher priority)
        self.ready_queue.sort(key=lambda t: t.priority)
    
    def get_next_task(self) -> Optional[Task]:
        if self.ready_queue:
            return self.ready_queue.pop(0)
        return None
    
    def handle_priority_inversion(self, high_priority_task: Task, blocking_task: Task):
        """Implement priority inheritance"""
        logger.info(f"Priority inversion detected: {high_priority_task.task_id} blocked by {blocking_task.task_id}")
        
        # Temporarily boost priority of blocking task
        original_priority = blocking_task.priority
        blocking_task.priority = high_priority_task.priority
        self.priority_inheritance_active[blocking_task.task_id] = original_priority
        
        # Re-sort ready queue
        self.ready_queue.sort(key=lambda t: t.priority)
    
    def restore_priority(self, task: Task):
        """Restore original priority after task completion"""
        if task.task_id in self.priority_inheritance_active:
            task.priority = self.priority_inheritance_active[task.task_id]
            del self.priority_inheritance_active[task.task_id]

class MLScheduler(BaseScheduler):
    """Machine Learning based scheduler using Linear Regression"""
    
    def __init__(self):
        super().__init__("ML-LinearRegression")
        self.model = LinearRegression()
        self.training_data = []
        self.is_trained = False
    
    def add_task(self, task: Task):
        self.ready_queue.append(task)
        
        if len(self.completed_tasks) > 5 and not self.is_trained:
            self.train_model()
        
        if self.is_trained:
            # Predict wait time for each task and sort accordingly
            for t in self.ready_queue:
                t.predicted_wait_time = self.predict_wait_time(t)
            
            # Sort by predicted wait time (minimize total wait)
            self.ready_queue.sort(key=lambda t: t.predicted_wait_time)
    
    def get_next_task(self) -> Optional[Task]:
        if self.ready_queue:
            return self.ready_queue.pop(0)
        return None
    
    def train_model(self):
        """Train the ML model on completed tasks"""
        if len(self.completed_tasks) < 5:
            return
        
        # Prepare training data
        X = []
        y = []
        
        for task in self.completed_tasks:
            features = [
                task.priority,
                task.service_time,
                len(self.ready_queue),  # Queue length at arrival
                self.current_time - task.arrival_time  # Time since arrival
            ]
            X.append(features)
            y.append(task.waiting_time)
        
        # Train model
        self.model.fit(X, y)
        self.is_trained = True
        logger.info("ML model trained on completed tasks")
    
    def predict_wait_time(self, task: Task) -> float:
        """Predict wait time for a task"""
        if not self.is_trained:
            return task.service_time  # Default prediction
        
        features = [[
            task.priority,
            task.service_time,
            len(self.ready_queue),
            self.current_time - task.arrival_time
        ]]
        
        prediction = self.model.predict(features)[0]
        return max(0, prediction)  # Ensure non-negative

class Simulator:
    """Main simulation engine"""
    
    def __init__(self, scheduler: BaseScheduler, num_processors: int = 1):
        self.scheduler = scheduler
        self.num_processors = num_processors
        self.processors = [None] * num_processors  # Track which task is on each processor
        self.processor_locks = [threading.Lock() for _ in range(num_processors)]
        self.simulation_time = 0
        self.max_simulation_time = 200  # Reasonable upper bound
        self.time_step = 0.1
        self.running = False
        
    def find_idle_processor(self) -> Optional[int]:
        """Find an idle processor"""
        for i in range(self.num_processors):
            if self.processors[i] is None:
                return i
        return None
    
    def execute_task_on_processor(self, task: Task, processor_id: int):
        """Execute task on specified processor"""
        with self.processor_locks[processor_id]:
            self.processors[processor_id] = task
            task.state = TaskState.EXECUTING
            
            if task.start_time is None:
                task.start_time = self.simulation_time
            
            # Simulate task execution
            while task.remaining_service_time > 0 and self.running:
                time.sleep(self.time_step * 0.1)  # Scale down for simulation
                task.remaining_service_time -= self.time_step
                
                # Check for preemption (for priority scheduler)
                if isinstance(self.scheduler, PriorityScheduler):
                    if self.check_preemption_needed(task):
                        task.state = TaskState.PREEMPTED
                        task.preemption_count += 1
                        self.processors[processor_id] = None
                        self.scheduler.add_task(task)
                        return
            
            # Task completed
            task.completion_time = self.simulation_time
            task.state = TaskState.COMPLETED
            self.scheduler.completed_tasks.append(task)
            self.processors[processor_id] = None
            
            # Restore priority if using priority inheritance
            if isinstance(self.scheduler, PriorityScheduler):
                self.scheduler.restore_priority(task)
    
    def check_preemption_needed(self, current_task: Task) -> bool:
        """Check if current task should be preempted"""
        if not self.scheduler.ready_queue:
            return False
        
        highest_priority_waiting = min(self.scheduler.ready_queue, key=lambda t: t.priority)
        return highest_priority_waiting.priority < current_task.priority
    
    def run(self, tasks: List[Task]):
        """Run the simulation"""
        self.running = True
        pending_tasks = tasks.copy()
        
        # Start processor threads
        processor_threads = []
        
        while (pending_tasks or self.scheduler.ready_queue or 
               any(p is not None for p in self.processors)) and self.running:
            
            # Add arrived tasks to ready queue
            arrived = [t for t in pending_tasks if t.arrival_time <= self.simulation_time]
            for task in arrived:
                self.scheduler.add_task(task)
                pending_tasks.remove(task)
                logger.debug(f"{task.task_id} arrived at {self.simulation_time:.2f}")
            
            # Schedule tasks on idle processors
            while True:
                processor_id = self.find_idle_processor()
                if processor_id is None:
                    break
                
                next_task = self.scheduler.get_next_task()
                if next_task is None:
                    break
                
                # Start task execution in a thread
                thread = threading.Thread(
                    target=self.execute_task_on_processor,
                    args=(next_task, processor_id)
                )
                thread.start()
                processor_threads.append(thread)
            
            # Record metrics
            self.scheduler.current_time = self.simulation_time
            self.scheduler.record_metrics()
            
            # Advance simulation time
            self.simulation_time += self.time_step
            
            # Safety check
            if self.simulation_time > self.max_simulation_time:
                logger.warning("Simulation time limit reached")
                break
        
        self.running = False
        
        # Wait for all threads to complete
        for thread in processor_threads:
            thread.join()
        
        logger.info(f"Simulation completed. Total tasks: {len(self.scheduler.completed_tasks)}")

class Visualizer:
    """Visualization and analysis tools"""
    
    @staticmethod
    def plot_gantt_chart(tasks: List[Task], title: str):
        """Create Gantt chart of task execution"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Sort tasks by start time
        tasks_sorted = sorted([t for t in tasks if t.start_time is not None], 
                            key=lambda t: t.start_time)
        
        # Color map for priorities
        colors = {1: 'red', 2: 'yellow', 3: 'green'}
        priority_names = {1: 'High', 2: 'Medium', 3: 'Low'}
        
        for i, task in enumerate(tasks_sorted):
            if task.completion_time and task.start_time:
                ax.barh(i, task.completion_time - task.start_time, 
                       left=task.start_time, height=0.8,
                       color=colors[task.priority], 
                       label=priority_names[task.priority] if i < 3 else "")
                
                # Add task ID
                ax.text(task.start_time + (task.completion_time - task.start_time) / 2, 
                       i, task.task_id, ha='center', va='center', fontsize=8)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Tasks')
        ax.set_title(f'Gantt Chart - {title}')
        ax.legend()
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_metrics(scheduler: BaseScheduler):
        """Plot performance metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Queue length over time
        axes[0, 0].plot(scheduler.metrics['timestamps'], 
                       scheduler.metrics['queue_lengths'])
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Queue Length')
        axes[0, 0].set_title('Queue Length Over Time')
        axes[0, 0].grid(True)
        
        # Memory usage
        axes[0, 1].plot(scheduler.metrics['timestamps'], 
                       scheduler.metrics['memory_usage'])
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Memory Usage (MB)')
        axes[0, 1].set_title('Memory Usage Over Time')
        axes[0, 1].grid(True)
        
        # CPU usage
        axes[1, 0].plot(scheduler.metrics['timestamps'], 
                       scheduler.metrics['cpu_usage'])
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('CPU Usage (%)')
        axes[1, 0].set_title('CPU Usage Over Time')
        axes[1, 0].grid(True)
        
        # Task completion times by priority
        completion_times = {1: [], 2: [], 3: []}
        for task in scheduler.completed_tasks:
            if task.completion_time:
                completion_times[task.priority].append(task.turnaround_time)
        
        priority_names = ['High', 'Medium', 'Low']
        avg_times = [np.mean(completion_times[i+1]) if completion_times[i+1] else 0 
                    for i in range(3)]
        
        axes[1, 1].bar(priority_names, avg_times, color=['red', 'yellow', 'green'])
        axes[1, 1].set_xlabel('Priority')
        axes[1, 1].set_ylabel('Average Turnaround Time (s)')
        axes[1, 1].set_title('Average Turnaround Time by Priority')
        axes[1, 1].grid(True, axis='y')
        
        plt.suptitle(f'Performance Metrics - {scheduler.name}')
        plt.tight_layout()
        return fig
    
    @staticmethod
    def generate_report(schedulers_results: Dict[str, Dict]):
        """Generate comparison report"""
        report = "# Real-Time Task Scheduling Analysis Report\n\n"
        report += f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Summary statistics
        report += "## Summary Statistics\n\n"
        
        for name, results in schedulers_results.items():
            scheduler = results['scheduler']
            report += f"### {name}\n"
            report += f"- Total tasks completed: {len(scheduler.completed_tasks)}\n"
            
            # Calculate average metrics
            avg_waiting = np.mean([t.waiting_time for t in scheduler.completed_tasks])
            avg_turnaround = np.mean([t.turnaround_time for t in scheduler.completed_tasks])
            max_queue_length = max(scheduler.metrics['queue_lengths'])
            avg_memory = np.mean(scheduler.metrics['memory_usage'])
            
            report += f"- Average waiting time: {avg_waiting:.2f}s\n"
            report += f"- Average turnaround time: {avg_turnaround:.2f}s\n"
            report += f"- Maximum queue length: {max_queue_length}\n"
            report += f"- Average memory usage: {avg_memory:.2f}MB\n\n"
        
        # Priority-based analysis
        report += "## Priority-Based Analysis\n\n"
        
        for name, results in schedulers_results.items():
            scheduler = results['scheduler']
            report += f"### {name}\n"
            
            for priority in [1, 2, 3]:
                priority_tasks = [t for t in scheduler.completed_tasks if t.priority == priority]
                if priority_tasks:
                    avg_wait = np.mean([t.waiting_time for t in priority_tasks])
                    avg_turn = np.mean([t.turnaround_time for t in priority_tasks])
                    report += f"- Priority {TASK_SPECS[priority]['name']}:\n"
                    report += f"  - Tasks: {len(priority_tasks)}\n"
                    report += f"  - Avg waiting time: {avg_wait:.2f}s\n"
                    report += f"  - Avg turnaround time: {avg_turn:.2f}s\n"
            report += "\n"
        
        return report

class GUI:
    """Graphical User Interface for the scheduler"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Real-Time Task Scheduler - Raspberry Pi 3")
        self.root.geometry("1200x800")
        
        self.tasks = []
        self.simulators = {}
        self.results = {}
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the GUI components"""
        # Main notebook
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Configuration tab
        self.config_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.config_frame, text="Configuration")
        self.setup_config_tab()
        
        # Simulation tab
        self.sim_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.sim_frame, text="Simulation")
        self.setup_simulation_tab()
        
        # Results tab
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Results")
        self.setup_results_tab()
    
    def setup_config_tab(self):
        """Setup configuration tab"""
        # Task generation section
        gen_frame = ttk.LabelFrame(self.config_frame, text="Task Generation", padding=10)
        gen_frame.grid(row=0, column=0, padx=10, pady=10, sticky='ew')
        
        ttk.Label(gen_frame, text="Random Seed:").grid(row=0, column=0, sticky='w')
        self.seed_var = tk.IntVar(value=42)
        ttk.Entry(gen_frame, textvariable=self.seed_var, width=10).grid(row=0, column=1)
        
        ttk.Button(gen_frame, text="Generate Tasks", 
                  command=self.generate_tasks).grid(row=0, column=2, padx=10)
        
        # Task display
        self.task_text = scrolledtext.ScrolledText(gen_frame, width=80, height=20)
        self.task_text.grid(row=1, column=0, columnspan=3, pady=10)
        
        # Scheduler configuration
        sched_frame = ttk.LabelFrame(self.config_frame, text="Scheduler Configuration", padding=10)
        sched_frame.grid(row=1, column=0, padx=10, pady=10, sticky='ew')
        
        ttk.Label(sched_frame, text="Number of Processors:").grid(row=0, column=0, sticky='w')
        self.num_processors_var = tk.IntVar(value=1)
        ttk.Spinbox(sched_frame, from_=1, to=4, textvariable=self.num_processors_var, 
                   width=10).grid(row=0, column=1)
        
        ttk.Label(sched_frame, text="Schedulers to Compare:").grid(row=1, column=0, sticky='w')
        
        self.scheduler_vars = {
            'FCFS': tk.BooleanVar(value=True),
            'EDF': tk.BooleanVar(value=True),
            'Priority': tk.BooleanVar(value=True),
            'ML': tk.BooleanVar(value=True)
        }
        
        col = 1
        for name, var in self.scheduler_vars.items():
            ttk.Checkbutton(sched_frame, text=name, variable=var).grid(row=1, column=col)
            col += 1
    
    def setup_simulation_tab(self):
        """Setup simulation tab"""
        control_frame = ttk.Frame(self.sim_frame)
        control_frame.pack(pady=10)
        
        ttk.Button(control_frame, text="Run Simulation", 
                  command=self.run_simulation).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Stop Simulation", 
                  command=self.stop_simulation).pack(side='left', padx=5)
        
        # Progress display
        self.progress_var = tk.StringVar(value="Ready to simulate")
        ttk.Label(self.sim_frame, textvariable=self.progress_var).pack(pady=5)
        
        # Log display
        log_frame = ttk.LabelFrame(self.sim_frame, text="Simulation Log", padding=10)
        log_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, width=100, height=30)
        self.log_text.pack(fill='both', expand=True)
    
    def setup_results_tab(self):
        """Setup results tab"""
        # Control buttons
        control_frame = ttk.Frame(self.results_frame)
        control_frame.pack(pady=10)
        
        ttk.Button(control_frame, text="Generate Report", 
                  command=self.generate_report).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Save Results", 
                  command=self.save_results).pack(side='left', padx=5)
        
        # Results display
        self.results_text = scrolledtext.ScrolledText(self.results_frame, width=100, height=35)
        self.results_text.pack(fill='both', expand=True, padx=10, pady=10)
    
    def generate_tasks(self):
        """Generate tasks based on configuration"""
        seed = self.seed_var.get()
        self.tasks = TaskGenerator.generate_tasks(seed)
        
        # Display tasks
        self.task_text.delete(1.0, tk.END)
        self.task_text.insert(tk.END, "Generated Tasks:\n\n")
        
        for task in self.tasks:
            self.task_text.insert(tk.END, 
                f"{task.task_id}: Priority={TASK_SPECS[task.priority]['name']}, "
                f"Arrival={task.arrival_time:.2f}s, Service={task.service_time:.2f}s, "
                f"Deadline={task.deadline:.2f}s\n")
        
        self.progress_var.set(f"Generated {len(self.tasks)} tasks")
    
    def run_simulation(self):
        """Run simulation for selected schedulers"""
        if not self.tasks:
            self.progress_var.set("Please generate tasks first!")
            return
        
        self.log_text.delete(1.0, tk.END)
        self.results = {}
        
        # Create and run simulators
        schedulers = {
            'FCFS': FCFSScheduler,
            'EDF': EDFScheduler,
            'Priority': PriorityScheduler,
            'ML': MLScheduler
        }
        
        num_processors = self.num_processors_var.get()
        
        for name, scheduler_class in schedulers.items():
            if self.scheduler_vars[name].get():
                self.log_text.insert(tk.END, f"\n--- Running {name} Scheduler ---\n")
                self.log_text.see(tk.END)
                self.root.update()
                
                # Create fresh task copies
                task_copies = [Task(
                    task_id=t.task_id,
                    arrival_time=t.arrival_time,
                    service_time=t.service_time,
                    priority=t.priority,
                    deadline=t.deadline
                ) for t in self.tasks]
                
                # Run simulation
                scheduler = scheduler_class()
                simulator = Simulator(scheduler, num_processors)
                
                # Run in thread to keep GUI responsive
                sim_thread = threading.Thread(target=simulator.run, args=(task_copies,))
                sim_thread.start()
                sim_thread.join()
                
                self.results[name] = {
                    'scheduler': scheduler,
                    'num_processors': num_processors
                }
                
                self.log_text.insert(tk.END, 
                    f"Completed: {len(scheduler.completed_tasks)} tasks\n")
                self.log_text.see(tk.END)
        
        self.progress_var.set("Simulation completed!")
        self.generate_report()
    
    def stop_simulation(self):
        """Stop running simulations"""
        # Implementation would set running flags to False
        self.progress_var.set("Simulation stopped")
    
    def generate_report(self):
        """Generate and display report"""
        if not self.results:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "No simulation results available")
            return
        
        report = Visualizer.generate_report(self.results)
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, report)
        
        # Generate plots
        for name, result in self.results.items():
            scheduler = result['scheduler']
            
            # Gantt chart
            gantt_fig = Visualizer.plot_gantt_chart(scheduler.completed_tasks, 
                                                   f"{name} Scheduler")
            gantt_fig.savefig(f"gantt_{name.lower()}.png")
            plt.close(gantt_fig)
            
            # Metrics plots
            metrics_fig = Visualizer.plot_metrics(scheduler)
            metrics_fig.savefig(f"metrics_{name.lower()}.png")
            plt.close(metrics_fig)
        
        self.results_text.insert(tk.END, "\n\nPlots saved to current directory.")
    
    def save_results(self):
        """Save results to file"""
        if not self.results:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save report
        with open(f"report_{timestamp}.txt", 'w') as f:
            f.write(Visualizer.generate_report(self.results))
        
        # Save raw data
        data = {
            'tasks': self.tasks,
            'results': self.results
        }
        
        with open(f"results_{timestamp}.pkl", 'wb') as f:
            pickle.dump(data, f)
        
        self.progress_var.set(f"Results saved with timestamp {timestamp}")
    
    def run(self):
        """Run the GUI"""
        self.root.mainloop()

def run_headless(args):
    """Run in headless mode"""
    logger.info("Running in headless mode")
    
    # Generate tasks
    tasks = TaskGenerator.generate_tasks(seed=args.seed)
    logger.info(f"Generated {len(tasks)} tasks")
    
    # Run simulations
    results = {}
    
    schedulers = {
        'FCFS': FCFSScheduler,
        'EDF': EDFScheduler,
        'Priority': PriorityScheduler,
        'ML': MLScheduler
    }
    
    for name, scheduler_class in schedulers.items():
        if getattr(args, f'enable_{name.lower()}', True):
            logger.info(f"\nRunning {name} scheduler...")
            
            # Create fresh task copies
            task_copies = [Task(
                task_id=t.task_id,
                arrival_time=t.arrival_time,
                service_time=t.service_time,
                priority=t.priority,
                deadline=t.deadline
            ) for t in tasks]
            
            scheduler = scheduler_class()
            simulator = Simulator(scheduler, args.num_processors)
            simulator.run(task_copies)
            
            results[name] = {
                'scheduler': scheduler,
                'num_processors': args.num_processors
            }
            
            logger.info(f"Completed: {len(scheduler.completed_tasks)} tasks")
    
    # Generate report
    report = Visualizer.generate_report(results)
    print("\n" + report)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(f"report_{timestamp}.txt", 'w') as f:
        f.write(report)
    
    # Generate plots if requested
    if args.generate_plots:
        for name, result in results.items():
            scheduler = result['scheduler']
            
            gantt_fig = Visualizer.plot_gantt_chart(scheduler.completed_tasks, 
                                                   f"{name} Scheduler")
            gantt_fig.savefig(f"gantt_{name.lower()}_{timestamp}.png")
            plt.close(gantt_fig)
            
            metrics_fig = Visualizer.plot_metrics(scheduler)
            metrics_fig.savefig(f"metrics_{name.lower()}_{timestamp}.png")
            plt.close(metrics_fig)
        
        logger.info(f"Plots saved with timestamp {timestamp}")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-Time Task Scheduler for Raspberry Pi 3")
    parser.add_argument('--mode', choices=['gui', 'headless'], default='gui',
                       help='Run mode (default: gui)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for task generation')
    parser.add_argument('--num-processors', type=int, default=1,
                       help='Number of processors (default: 1)')
    parser.add_argument('--enable-fcfs', action='store_true', default=True,
                       help='Enable FCFS scheduler')
    parser.add_argument('--enable-edf', action='store_true', default=True,
                       help='Enable EDF scheduler')
    parser.add_argument('--enable-priority', action='store_true', default=True,
                       help='Enable Priority scheduler')
    parser.add_argument('--enable-ml', action='store_true', default=True,
                       help='Enable ML scheduler')
    parser.add_argument('--generate-plots', action='store_true', default=True,
                       help='Generate plots in headless mode')
    
    args = parser.parse_args()
    
    if args.mode == 'gui':
        gui = GUI()
        gui.run()
    else:
        run_headless(args)

if __name__ == "__main__":
    main()