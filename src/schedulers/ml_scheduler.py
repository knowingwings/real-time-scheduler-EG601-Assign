"""
Machine Learning-Based Scheduler

Implements a scheduler that uses simple machine learning techniques 
to improve task scheduling decisions.
"""

import time
import threading
import psutil
import logging
import numpy as np
from queue import PriorityQueue
from sklearn.linear_model import LinearRegression
from collections import deque

class MLScheduler:
    """
    Machine Learning-Based Scheduler
    
    Uses a simple linear regression model to predict task execution time
    and optimise scheduling decisions.
    """
    
    def __init__(self, history_size=50, min_training_samples=20):
        # Using PriorityQueue for priority-based ordering
        self.task_queue = PriorityQueue()
        self.current_task = None
        self.completed_tasks = []
        self.lock = threading.Lock()
        self.running = False
        self.current_time = 0
        self.preempted_tasks = []
        self.min_training_samples = min_training_samples
        
        # ML components
        self.model = LinearRegression()
        self.trained = False
        self.history = deque(maxlen=history_size)  # Store recent task execution data
        self.feature_names = ['priority', 'arrival_pattern', 'queue_length', 'cpu_load', 'memory_usage']
        
        # Task history for arrival pattern detection
        self.arrival_history = deque(maxlen=20)
        self.arrival_gap_history = deque(maxlen=10)
        self.last_arrival_time = 0
        
        # Metrics
        self.metrics = {
            'queue_length': [],
            'memory_usage': [],
            'timestamp': [],
            'prediction_errors': [],
            'training_events': []
        }
        self.logger = logging.getLogger(__name__)
    
    def add_task(self, task):
        """Add a task to the scheduler queue"""
        # Track arrival pattern
        current_time = time.time()
        if self.last_arrival_time > 0:
            arrival_gap = current_time - self.last_arrival_time
            self.arrival_gap_history.append(arrival_gap)
        
        self.last_arrival_time = current_time
        self.arrival_history.append(task)
        
        # Extract features for prediction (without using service_time)
        features = self._extract_features(task)
        
        # Make prediction if model is trained
        predicted_time = None
        if self.trained:
            predicted_time = self._predict_execution_time(features)
            task.predicted_time = predicted_time
            self.logger.info(f"ML prediction for {task.id}: {predicted_time:.2f}s (actual: {task.service_time:.2f}s)")
        
        # Default to priority-based scheduling when model isn't trained
        priority_value = task.priority.value
        
        # Use a combination of priority and predicted time when model is trained
        scheduling_value = priority_value if not predicted_time else priority_value + (predicted_time / 10.0)
        
        # Add to queue with scheduling value as key (lower value = higher priority)
        self.task_queue.put((scheduling_value, id(task), task))
        self.logger.info(f"Task {task.id} added to ML queue with scheduling value {scheduling_value:.2f}")
    
    def _extract_features(self, task):
        """
        Extract features for ML model
        
        Avoids using the service_time as a feature since that's what we're trying to predict
        """
        # Calculate arrival pattern feature based on recent history
        arrival_pattern = 0
        if len(self.arrival_gap_history) > 0:
            # Lower value indicates bursty arrivals, higher value indicates periodic
            arrival_pattern = np.std(list(self.arrival_gap_history)) if len(self.arrival_gap_history) > 1 else 1.0
        
        # Calculate queue complexity based on mix of priorities
        queue_length = self.task_queue.qsize()
        
        # Get current system metrics
        cpu_load = psutil.cpu_percent()
        memory_usage = psutil.virtual_memory().percent
        
        # Return feature array without using task.service_time
        return np.array([[
            task.priority.value,               # Priority level
            arrival_pattern,                   # Arrival pattern feature
            queue_length,                      # Current queue length
            cpu_load,                          # Current CPU load
            memory_usage                       # Current memory usage
        ]])
    
    def _predict_execution_time(self, features):
        """Predict task execution time using trained model"""
        try:
            prediction = self.model.predict(features)[0]
            # Ensure prediction is positive and reasonable
            return max(0.1, min(10.0, prediction))
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return None
    
    def _train_model(self):
        """Train the ML model using execution history"""
        if len(self.history) < self.min_training_samples:
            # Need enough samples to train
            return False
        
        try:
            # Extract features and targets from history
            X = np.array([entry['features'] for entry in self.history])
            y = np.array([entry['actual_time'] for entry in self.history])
            
            # Ensure no NaN values
            if np.isnan(X).any() or np.isnan(y).any():
                self.logger.warning("Skipping training due to NaN values in data")
                return False
            
            # Train the model
            self.model.fit(X, y)
            self.trained = True
            
            # Record training event timestamp
            self.metrics['training_events'].append(time.time())
            
            self.logger.info(f"ML model trained with {len(self.history)} samples")
            
            # Calculate and log model coefficients
            coef_values = self.model.coef_
            self.logger.info("Model coefficients:")
            for name, value in zip(self.feature_names, coef_values):
                self.logger.info(f"  {name}: {value:.4f}")
            
            return True
        except Exception as e:
            self.logger.error(f"Training error: {e}")
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
        
        # Start metrics collection in a separate thread
        metrics_thread = threading.Thread(target=self._collect_metrics)
        metrics_thread.daemon = True
        metrics_thread.start()
        
        # Training thread for the ML model
        training_thread = threading.Thread(target=self._periodic_training)
        training_thread.daemon = True
        training_thread.start()
        
        while self.running:
            # Record current state
            with self.lock:
                queue_size = self.task_queue.qsize() + len(self.preempted_tasks)
                self.metrics['queue_length'].append(queue_size)
            
            task = None
            
            # Handle preempted tasks first
            if self.preempted_tasks:
                task = self.preempted_tasks.pop(0)
            elif not self.task_queue.empty():
                # Get the next task based on ML-predicted value
                _, _, task = self.task_queue.get()
            
            if task:
                # If simulation, we might need to advance time
                if simulation and self.current_time < task.arrival_time:
                    self.current_time = task.arrival_time
                
                # Set waiting time if not already set
                if task.waiting_time is None:
                    current_t = time.time() if not simulation else self.current_time
                    task.waiting_time = max(0, current_t - task.arrival_time)
                
                # Set as current task and start execution
                self.current_task = task
                task.start_time = time.time() if not simulation else self.current_time
                self.logger.info(f"Starting execution of {task.id} at time {task.start_time:.2f}")
                
                # Extract features before execution for later training
                task.features = self._extract_features(task)[0]
                
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
                
                # Calculate metrics
                task.calculate_metrics()
                
                # Add to completed tasks
                with self.lock:
                    self.completed_tasks.append(task)
                    
                    # Add to training history only if it has valid data
                    if hasattr(task, 'features') and task.service_time > 0:
                        actual_time = task.completion_time - task.start_time
                        if actual_time > 0:
                            self.history.append({
                                'features': task.features,
                                'actual_time': actual_time
                            })
                    
                    # Calculate prediction error if available
                    if hasattr(task, 'predicted_time') and task.predicted_time is not None:
                        actual_time = task.completion_time - task.start_time
                        error = abs(task.predicted_time - actual_time)
                        self.metrics['prediction_errors'].append(error)
                        self.logger.info(f"Prediction error for {task.id}: {error:.2f}s")
                    
                    self.current_task = None
            else:
                # No tasks to process
                if not simulation:
                    time.sleep(0.1)  # Prevent CPU hogging
                else:
                    # In simulation, advance time slightly
                    time.sleep(0.01 / speed_factor)
    
    def _periodic_training(self):
        """Periodically retrain the ML model"""
        while self.running:
            # Wait for enough samples before first training
            if len(self.history) >= self.min_training_samples:
                self._train_model()
            time.sleep(5)  # Retrain every 5 seconds
    
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
            # Calculate average waiting time with error handling
            waiting_times = [task.waiting_time for task in self.completed_tasks 
                           if task.waiting_time is not None]
            avg_waiting_time = sum(waiting_times) / len(waiting_times) if waiting_times else 0
            
            # Calculate average prediction error with error handling
            avg_prediction_error = sum(self.metrics['prediction_errors']) / len(self.metrics['prediction_errors']) if self.metrics['prediction_errors'] else 0
            
            # Calculate waiting times by priority
            waiting_by_priority = {'HIGH': [], 'MEDIUM': [], 'LOW': []}
            for task in self.completed_tasks:
                if task.waiting_time is not None and hasattr(task, 'priority'):
                    priority_name = task.priority.name if hasattr(task.priority, 'name') else str(task.priority)
                    if priority_name in waiting_by_priority:
                        waiting_by_priority[priority_name].append(task.waiting_time)
            
            avg_wait_by_priority = {}
            for priority, times in waiting_by_priority.items():
                avg_wait_by_priority[priority] = sum(times) / len(times) if times else 0
            
            # Count tasks by priority
            tasks_by_priority = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
            for task in self.completed_tasks:
                if hasattr(task, 'priority'):
                    priority_name = task.priority.name if hasattr(task.priority, 'name') else str(task.priority)
                    if priority_name in tasks_by_priority:
                        tasks_by_priority[priority_name] += 1
            
            return {
                'completed_tasks': len(self.completed_tasks),
                'avg_waiting_time': avg_waiting_time,
                'avg_waiting_by_priority': avg_wait_by_priority,
                'tasks_by_priority': tasks_by_priority,
                'queue_length_history': self.metrics['queue_length'],
                'memory_usage_history': self.metrics['memory_usage'],
                'timestamp_history': self.metrics['timestamp'],
                'average_prediction_error': avg_prediction_error,
                'prediction_errors': self.metrics['prediction_errors'],
                'model_trained': self.trained,
                'training_events': self.metrics['training_events'],
                'training_samples': len(self.history)
            }