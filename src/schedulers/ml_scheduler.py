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
    and optimize scheduling decisions.
    """
    
    def __init__(self, history_size=50):
        # Using PriorityQueue for priority-based ordering
        self.task_queue = PriorityQueue()
        self.current_task = None
        self.completed_tasks = []
        self.lock = threading.Lock()
        self.running = False
        self.current_time = 0
        self.preempted_tasks = []
        
        # ML components
        self.model = LinearRegression()
        self.trained = False
        self.history = deque(maxlen=history_size)  # Store recent task execution data
        
        # Metrics
        self.metrics = {
            'queue_length': [],
            'memory_usage': [],
            'timestamp': [],
            'prediction_errors': []
        }
        self.logger = logging.getLogger(__name__)
    
    def add_task(self, task):
        """Add a task to the scheduler queue"""
        # Extract features for prediction
        features = self._extract_features(task)
        
        # Make prediction if model is trained
        predicted_time = None
        if self.trained:
            predicted_time = self._predict_execution_time(features)
            task.predicted_time = predicted_time
            self.logger.info(f"ML prediction for {task.id}: {predicted_time:.2f}s (actual: {task.service_time:.2f}s)")
        
        # Use predicted time for scheduling if available, otherwise use priority
        priority_value = task.priority.value
        scheduling_value = predicted_time if predicted_time else priority_value
        
        # Add to queue with scheduling value as key
        self.task_queue.put((scheduling_value, id(task), task))
        self.logger.info(f"Task {task.id} added to ML queue with scheduling value {scheduling_value:.2f}")
    
    def _extract_features(self, task):
        """Extract features for ML model"""
        # Simple feature extraction based on task attributes
        # More sophisticated features could be used in a real implementation
        return np.array([[
            task.priority.value,                # Priority level
            task.service_time,                  # Estimated service time
            len(self.task_queue.queue),         # Current queue length
            psutil.cpu_percent(),               # Current CPU load
            psutil.virtual_memory().percent     # Current memory usage
        ]])
    
    def _predict_execution_time(self, features):
        """Predict task execution time using trained model"""
        try:
            prediction = self.model.predict(features)[0]
            # Ensure prediction is positive
            return max(0.1, prediction)
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return None
    
    def _train_model(self):
        """Train the ML model using execution history"""
        if len(self.history) < 10:  # Need enough samples to train
            return False
        
        try:
            # Extract features and targets from history
            X = np.array([entry['features'] for entry in self.history])
            y = np.array([entry['actual_time'] for entry in self.history])
            
            # Train the model
            self.model.fit(X, y)
            self.trained = True
            self.logger.info(f"ML model trained with {len(self.history)} samples")
            
            # Calculate and log model coefficients
            coef_names = ['priority', 'estimated_time', 'queue_length', 'cpu_load', 'memory_usage']
            coef_values = self.model.coef_
            self.logger.info("Model coefficients:")
            for name, value in zip(coef_names, coef_values):
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
            
            if not self.task_queue.empty():
                # Get the next task based on ML-predicted value
                _, _, task = self.task_queue.get()
            
            if task:
                # If simulation, we might need to advance time
                if simulation and self.current_time < task.arrival_time:
                    self.current_time = task.arrival_time
                
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
                    time.sleep(task.service_time / speed_factor)  # Still sleep a bit for visualization
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
                    
                    # Add to training history
                    actual_time = task.completion_time - task.start_time
                    self.history.append({
                        'features': task.features,
                        'actual_time': actual_time
                    })
                    
                    # Calculate prediction error if available
                    if hasattr(task, 'predicted_time'):
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
            # Calculate average waiting time
            waiting_times = [task.waiting_time for task in self.completed_tasks 
                           if task.waiting_time is not None]
            avg_waiting_time = sum(waiting_times) / len(waiting_times) if waiting_times else 0
            
            # Calculate average prediction error
            avg_prediction_error = sum(self.metrics['prediction_errors']) / len(self.metrics['prediction_errors']) if self.metrics['prediction_errors'] else 0
            
            return {
                'completed_tasks': len(self.completed_tasks),
                'average_waiting_time': avg_waiting_time,
                'queue_length_history': self.metrics['queue_length'],
                'memory_usage_history': self.metrics['memory_usage'],
                'timestamp_history': self.metrics['timestamp'],
                'average_prediction_error': avg_prediction_error,
                'prediction_errors': self.metrics['prediction_errors'],
                'model_trained': self.trained
            }