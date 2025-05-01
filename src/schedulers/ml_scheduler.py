"""
Machine Learning-Based Scheduler with Decision Tree

Implements a scheduler that uses Decision Trees as a machine learning technique 
to improve task scheduling decisions.
"""

import time
import threading
import psutil
import logging
import numpy as np
from queue import PriorityQueue
from sklearn.tree import DecisionTreeRegressor
from collections import deque
from src.task_generator import Priority

# Constants for validation
MAX_WAITING_TIME = 60.0   # Maximum reasonable waiting time (seconds)
MAX_SERVICE_TIME = 20.0   # Maximum reasonable service time (seconds)
MIN_PREDICTION = 0.1      # Minimum reasonable predicted time
MAX_PREDICTION = 10.0     # Maximum reasonable predicted time

class MLScheduler:
    """
    Machine Learning-Based Scheduler
    
    Uses a decision tree model to predict task execution time
    and optimise scheduling decisions.
    """
    
    def __init__(self, history_size=50, min_training_samples=20, max_depth=5):
        # Using PriorityQueue for priority-based ordering
        self.task_queue = PriorityQueue()
        self.current_task = None
        self.completed_tasks = []
        self.lock = threading.RLock()  # Changed to RLock for nested acquisitions
        self.running = False
        self.current_time = 0
        self.preempted_tasks = []
        self.min_training_samples = min_training_samples
        
        # ML components - Decision Tree instead of Linear Regression
        self.model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
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
            'training_events': [],
            'feature_importances': {},  # Track feature importances from decision tree
            'deadline_misses': 0
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
        # For decision trees, we can give more weight to the prediction since trees can capture 
        # non-linear relationships better
        scheduling_value = priority_value if not predicted_time else priority_value + (predicted_time / 5.0)
        
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
        """Predict task execution time using trained model with validation"""
        try:
            # Make prediction
            prediction = self.model.predict(features)[0]
            
            # Validate prediction - force to reasonable range
            if prediction < MIN_PREDICTION or prediction > MAX_PREDICTION:
                self.logger.warning(
                    f"ML model produced extreme prediction: {prediction:.2f}s. "
                    f"Clamping to range [{MIN_PREDICTION}, {MAX_PREDICTION}]."
                )
                
            # Return validated prediction
            return max(MIN_PREDICTION, min(MAX_PREDICTION, prediction))
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return None
    
    def _train_model(self):
        """Train the Decision Tree model using execution history"""
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
            with self.lock:
                self.metrics['training_events'].append(time.time())
            
            self.logger.info(f"Decision Tree model trained with {len(self.history)} samples")
            
            # Calculate and log feature importances (a benefit of using trees)
            importances = self.model.feature_importances_
            with self.lock:
                for i, importance in enumerate(importances):
                    feature_name = self.feature_names[i]
                    self.metrics['feature_importances'][feature_name] = float(importance)
                    self.logger.info(f"  {feature_name}: {importance:.4f}")
            
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
        start_time = time.time()  # Store real-time start
        
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
                    if simulation:
                        current_t = self.current_time
                    else:
                        current_t = time.time() - start_time
                    
                    # Calculate with validation
                    waiting_time = max(0, current_t - task.arrival_time)
                    if waiting_time > MAX_WAITING_TIME:
                        self.logger.warning(f"Excessive waiting time: {waiting_time}s. Capping to {MAX_WAITING_TIME}s")
                        waiting_time = MAX_WAITING_TIME
                    
                    task.waiting_time = round(waiting_time, 3)
                
                # Set as current task and start execution
                self.current_task = task
                
                # Record start time
                if simulation:
                    task.start_time = self.current_time
                else:
                    task.start_time = time.time() - start_time
                    
                self.logger.info(f"Starting execution of {task.id} at time {task.start_time:.2f}")
                
                # Extract features before execution for later training
                task.features = self._extract_features(task)[0]
                
                # Execute the task
                if simulation:
                    # Use safe execution time
                    service_time = min(task.service_time, MAX_SERVICE_TIME)
                    
                    # Simulate execution by advancing time
                    self.current_time += service_time
                    # Scale sleep time by speed factor
                    sleep_time = service_time / max(0.1, speed_factor)
                    time.sleep(sleep_time)
                else:
                    # Actually sleep for the service time in real execution
                    service_time = min(task.service_time, MAX_SERVICE_TIME)
                    time.sleep(service_time)
                
                # Record completion time
                if simulation:
                    task.completion_time = self.current_time
                else:
                    task.completion_time = time.time() - start_time
                    
                self.logger.info(f"Completed execution of {task.id} at time {task.completion_time:.2f}")
                
                # Check if deadline was missed
                if task.deadline is not None and task.completion_time > task.deadline:
                    self.logger.warning(f"Task {task.id} missed deadline by {task.completion_time - task.deadline:.2f}s")
                    with self.lock:
                        self.metrics['deadline_misses'] += 1
                
                # Calculate metrics
                task.calculate_metrics()
                
                # Add to completed tasks
                with self.lock:
                    self.completed_tasks.append(task)
                    
                    # Add to training history only if it has valid data
                    if hasattr(task, 'features') and task.service_time > 0:
                        actual_time = round(task.completion_time - task.start_time, 3)
                        if actual_time > 0:
                            self.history.append({
                                'features': task.features,
                                'actual_time': min(actual_time, MAX_SERVICE_TIME)  # Cap to prevent extreme values
                            })
                    
                    # Calculate prediction error if available
                    if hasattr(task, 'predicted_time') and task.predicted_time is not None:
                        actual_time = round(task.completion_time - task.start_time, 3)
                        # Validate actual time
                        actual_time = min(actual_time, MAX_SERVICE_TIME)
                        
                        error = round(abs(task.predicted_time - actual_time), 3)
                        # Cap error to avoid extreme values
                        error = min(error, MAX_SERVICE_TIME)
                        
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
            
            # Calculate average prediction error with error handling
            avg_prediction_error = 0
            if self.metrics['prediction_errors']:
                avg_prediction_error = round(
                    sum(self.metrics['prediction_errors']) / len(self.metrics['prediction_errors']), 
                    3
                )
            
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
                'avg_waiting_time': round(avg_waiting_time, 3),  # Already using standardized key
                'avg_waiting_by_priority': avg_wait_by_priority,  # Ensure consistent naming
                'tasks_by_priority': tasks_by_priority,  # Ensure consistent format
                'queue_length_history': self.metrics['queue_length'],
                'memory_usage_history': self.metrics['memory_usage'],
                'timestamp_history': self.metrics['timestamp'],
                'average_prediction_error': avg_prediction_error,
                'prediction_errors': self.metrics['prediction_errors'],
                'model_trained': self.trained,
                'training_events': self.metrics['training_events'],
                'training_samples': len(self.history),
                'feature_importances': self.metrics['feature_importances'],
                'algorithm': 'decision_tree',
                'deadline_tasks': deadline_tasks,
                'deadline_met': deadline_met,
                'deadline_miss_rate': deadline_miss_rate
            }