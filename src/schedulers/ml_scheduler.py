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

logger = logging.getLogger(__name__)

class MLScheduler:
    """
    Machine Learning-Based Scheduler
    
    Uses a decision tree model to predict task execution time
    and optimise scheduling decisions.
    """
    
    # Shared training data across all instances
    _shared_history = []
    
    def __init__(self, history_size=50, min_training_samples=20, max_depth=5):
        # Using PriorityQueue for priority-based ordering
        self.task_queue = PriorityQueue()
        self.current_task = None
        self.completed_tasks = []
        self.lock = threading.RLock()  # For general scheduler operations
        self.metrics_lock = threading.RLock()  # Separate lock for metrics
        self.training_lock = threading.Lock()  # Instance-level training lock
        self.history_lock = threading.Lock()  # Instance-level history lock
        self.running = False
        self.current_time = 0
        self.preempted_tasks = []
        self.min_training_samples = min_training_samples
        
        # ML components
        self.model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        self.trained = False
        self.local_history = deque(maxlen=history_size)
        self.feature_names = ['priority', 'arrival_pattern', 'queue_length', 'cpu_load', 'memory_usage']
        
        # Task history for arrival pattern detection
        self.arrival_history = deque(maxlen=20)
        self.arrival_gap_history = deque(maxlen=10)
        self.last_arrival_time = 0
        
        # Metrics with thread-safe access
        self._initialize_metrics()
    
    def add_task(self, task):
        """Add a task to the scheduler queue"""
        # Track arrival pattern
        current_time = time.time()
        if self.last_arrival_time > 0:
            arrival_gap = current_time - self.last_arrival_time
            self.arrival_gap_history.append(arrival_gap)
        
        self.last_arrival_time = current_time
        self.arrival_history.append(task)
        
        # Extract features for prediction
        features = self._extract_features(task)
        
        # Make prediction if model is trained
        predicted_time = None
        if self.trained:
            predicted_time = self._predict_execution_time(features)
            task.predicted_time = predicted_time
            logger.info(f"ML prediction for {task.id}: {predicted_time:.2f}s (actual: {task.service_time:.2f}s)")
        
        # Use a balanced scheduling value calculation
        priority_value = task.priority.value
        scheduling_value = priority_value
        
        if predicted_time is not None:
            # Cap predicted time influence and use a smaller weight
            capped_prediction = min(predicted_time, 10.0)  # Cap at 10 seconds
            scheduling_value = (priority_value * 0.7) + (capped_prediction * 0.3)
        
        # Add to queue with scheduling value as key (lower value = higher priority)
        self.task_queue.put((scheduling_value, id(task), task))
        logger.info(f"Task {task.id} added to ML queue with scheduling value {scheduling_value:.2f}")
    
    def _extract_features(self, task):
        """
        Extract features for ML model with improved normalization and feature engineering
        """
        # Calculate arrival pattern feature
        arrival_pattern = 0
        if len(self.arrival_gap_history) > 1:
            gaps = list(self.arrival_gap_history)
            mean_gap = sum(gaps) / len(gaps)
            # Normalize by mean gap to make it scale-independent
            arrival_pattern = np.std(gaps) / (mean_gap if mean_gap > 0 else 1.0)
        
        # Normalize queue length relative to max observed
        queue_length = self.task_queue.qsize()
        normalized_queue = queue_length / max(20, queue_length)  # Cap at 20 for normalization
        
        # Get system metrics with normalization
        cpu_load = psutil.cpu_percent() / 100.0  # Normalize to 0-1
        memory_usage = psutil.virtual_memory().percent / 100.0  # Normalize to 0-1
        
        # Normalize priority and calculate secondary features
        priority_value = (task.priority.value - 2.0) / 1.0  # Center around 0 (-1 to 1 range)
        
        return np.array([[
            priority_value,          # Normalized priority
            arrival_pattern,         # Normalized arrival pattern
            normalized_queue,        # Normalized queue length
            cpu_load,               # Normalized CPU load
            memory_usage            # Normalized memory usage
        ]])
    
    def _predict_execution_time(self, features):
        """Predict task execution time using trained model with validation"""
        try:
            # Make prediction with thread safety
            with self.training_lock:
                prediction = self.model.predict(features)[0]
            
            # Only ensure non-negative prediction
            if prediction < 0:
                logger.warning(f"ML model produced negative prediction: {prediction:.2f}s. Setting to 0.1s")
                prediction = 0.1
                
            return prediction
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None

    def _train_model(self):
        """Train the Decision Tree model using shared execution history with improved feature importance calculation"""
        # Quick check without lock first
        total_samples = len(self.local_history)
        with self.history_lock:
            total_samples += len(MLScheduler._shared_history)
            
        if total_samples < self.min_training_samples:
            return False
            
        # Prevent too frequent training attempts
        current_time = time.time()
        with self.metrics_lock:
            if current_time - self.metrics['last_training_time'] < 1.0:
                return False
            self.metrics['last_training_time'] = current_time
        
        # Try to acquire training lock without blocking
        if not self.training_lock.acquire(blocking=False):
            return False  # Another instance is training
            
        try:
            # Get training data with minimal lock holding
            with self.history_lock:
                training_data = MLScheduler._shared_history.copy()
            training_data.extend(list(self.local_history))
            
            if not training_data:
                return False
            
            # Extract features and targets
            X = np.array([entry['features'] for entry in training_data])
            y = np.array([entry['actual_time'] for entry in training_data])
            
            # Validate data
            if np.isnan(X).any() or np.isnan(y).any():
                logger.warning("Skipping training due to NaN values in data")
                return False
            
            # Train the model
            self.model.fit(X, y)
            self.trained = True
            
            # Update metrics with normalized feature importances
            with self.metrics_lock:
                self.metrics['training_events'].append(time.time())
                importances = self.model.feature_importances_
                
                # Normalize importances to reduce priority dominance
                total_importance = sum(importances)
                if total_importance > 0:
                    normalized_importances = importances / total_importance
                    # Apply soft cap to prevent any feature from dominating
                    capped_importances = np.minimum(normalized_importances, 0.5)
                    # Re-normalize after capping
                    total_capped = sum(capped_importances)
                    if total_capped > 0:
                        final_importances = capped_importances / total_capped
                        
                        for i, importance in enumerate(final_importances):
                            feature_name = self.feature_names[i]
                            self.metrics['feature_importances'][feature_name] = float(importance)
            
            logger.info(f"Decision Tree model trained with {len(training_data)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            return False
        finally:
            self.training_lock.release()
    
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

        # Start metrics collection in background
        metrics_thread = threading.Thread(target=self._collect_metrics)
        metrics_thread.daemon = True
        metrics_thread.start()
        
        # Start training thread with lower frequency in multi-processor mode
        training_thread = threading.Thread(target=self._periodic_training)
        training_thread.daemon = True
        training_thread.start()

        while self.running:
            task = None
            
            # Critical section for task selection
            with self.lock:
                # Handle preempted tasks first
                if self.preempted_tasks:
                    task = self.preempted_tasks.pop(0)
                elif not self.task_queue.empty():
                    _, _, task = self.task_queue.get()
                
                if task:
                    # Update task state while holding lock
                    if simulation and self.current_time < task.arrival_time:
                        self.current_time = task.arrival_time
                        
                    # Calculate waiting time
                    current_t = self.current_time if simulation else (time.time() - start_time)
                    if task.waiting_time is None:
                        raw_waiting_time = current_t - task.arrival_time
                        task.waiting_time = min(max(0.0, raw_waiting_time), 300.0)
                    
                    # Set start time
                    task.start_time = self.current_time if simulation else (time.time() - start_time)
                    self.current_task = task
            
            if task:
                logger.info(f"Starting execution of {task.id} at time {task.start_time:.2f}")
                
                # Extract features outside critical section
                task.features = self._extract_features(task)[0]
                
                # Execute task
                try:
                    service_time = min(task.service_time, 20.0)  # Cap execution time
                    if simulation:
                        self.current_time += service_time
                        time.sleep(service_time / max(0.1, speed_factor))
                    else:
                        time.sleep(service_time)
                        
                    completion_time = self.current_time if simulation else (time.time() - start_time)
                    task.completion_time = completion_time
                    
                    # Update metrics and history with fine-grained locking
                    with self.lock:
                        if task.deadline is not None:
                            self.metrics['deadline_tasks'] += 1
                            if completion_time > task.deadline:
                                self.metrics['deadline_misses'] += 1
                            else:
                                self.metrics['deadline_met'] += 1
                        
                        self.completed_tasks.append(task)
                        
                        # Only add to history if execution was successful
                        if task.features is not None:
                            actual_time = round(completion_time - task.start_time, 3)
                            if actual_time > 0:
                                entry = {
                                    'features': task.features,
                                    'actual_time': actual_time
                                }
                                self.local_history.append(entry)
                                
                                # Update shared history with separate lock
                                with self.history_lock:
                                    MLScheduler._shared_history.append(entry)
                                    if len(MLScheduler._shared_history) > 1000:
                                        MLScheduler._shared_history = MLScheduler._shared_history[-1000:]
                        
                        # Update prediction error metrics
                        if hasattr(task, 'predicted_time') and task.predicted_time:
                            error = abs(task.predicted_time - actual_time)
                            with self.metrics_lock:
                                self.metrics['prediction_errors'].append(round(error, 3))
                                
                finally:
                    # Always clear current task
                    with self.lock:
                        self.current_task = None
                        
            else:
                # No task to process - shorter sleep in simulation
                sleep_time = 0.01 if simulation else 0.1
                if simulation:
                    sleep_time /= speed_factor
                time.sleep(sleep_time)
    
    def _periodic_training(self):
        """Periodically retrain the ML model"""
        while self.running:
            # Wait for enough samples before first training
            if len(self.local_history) + len(MLScheduler._shared_history) >= self.min_training_samples:
                self._train_model()
            time.sleep(5)  # Retrain every 5 seconds
    
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
        logger.debug("Starting metrics collection for ML scheduler")
        
        while self.running:
            try:
                current_time = time.time()
                relative_time = round(current_time - self.start_time, 3)
                
                with self.lock:
                    # Only collect real measurements
                    memory_percent = psutil.virtual_memory().percent
                    queue_size = self.task_queue.qsize() + len(self.preempted_tasks)
                    
                    # Only append if we got valid measurements
                    self.metrics['timestamp'].append(relative_time)
                    self.metrics['memory_usage'].append(memory_percent)
                    self.metrics['queue_length'].append(queue_size)
                    
                    # Collect ML-specific metrics if available
                    if hasattr(self, 'prediction_errors') and self.prediction_errors:
                        self.metrics['prediction_errors'] = self.prediction_errors.copy()
                    if hasattr(self, 'feature_importances') and self.feature_importances:
                        self.metrics['feature_importances'] = self.feature_importances.copy()
            
            except Exception as e:
                logger.error(f"Error collecting metrics: {str(e)}")
            
            time.sleep(0.5)
    
    def get_metrics(self):
        """Get execution metrics with thread-safe access"""
        with self.metrics_lock:
            # Make a copy of metrics to prevent modification during access
            metrics_copy = {
                'prediction_errors': self.metrics['prediction_errors'].copy() if self.metrics['prediction_errors'] else [],
                'training_events': self.metrics['training_events'].copy(),
                'feature_importances': self.metrics['feature_importances'].copy(),
                'queue_length': self.metrics['queue_length'].copy(),
                'memory_usage': self.metrics['memory_usage'].copy(),
                'timestamp': self.metrics['timestamp'].copy(),
                'deadline_misses': self.metrics['deadline_misses']
            }
        
        with self.lock:
            # Calculate metrics that need scheduler lock
            waiting_times = [task.waiting_time for task in self.completed_tasks 
                        if task.waiting_time is not None]
            avg_waiting_time = sum(waiting_times) / len(waiting_times) if waiting_times else 0
            
            # Calculate waiting times by priority
            waiting_by_priority = {'HIGH': [], 'MEDIUM': [], 'LOW': []}
            tasks_by_priority = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
            
            for task in self.completed_tasks:
                if task.waiting_time is not None and hasattr(task, 'priority'):
                    priority_name = task.priority.name if hasattr(task.priority, 'name') else str(task.priority)
                    if priority_name in waiting_by_priority:
                        waiting_by_priority[priority_name].append(task.waiting_time)
                        tasks_by_priority[priority_name] += 1
            
            avg_wait_by_priority = {}
            for priority, times in waiting_by_priority.items():
                avg_wait_by_priority[priority] = round(sum(times) / len(times), 3) if times else 0
        
        return {
            'completed_tasks': len(self.completed_tasks),
            'avg_waiting_time': round(avg_waiting_time, 3),
            'avg_waiting_by_priority': avg_wait_by_priority,
            'tasks_by_priority': tasks_by_priority,
            'prediction_errors': metrics_copy['prediction_errors'],
            'training_events': metrics_copy['training_events'],
            'feature_importances': metrics_copy['feature_importances'],
            'model_trained': self.trained,
            'training_samples': len(self.local_history) + len(MLScheduler._shared_history),
            'queue_length': metrics_copy['queue_length'],
            'memory_usage': metrics_copy['memory_usage'],
            'timestamp': metrics_copy['timestamp'],
            'deadline_misses': metrics_copy['deadline_misses'],
            'algorithm': 'decision_tree'
        }