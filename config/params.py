"""
Configuration Parameters

Defines parameters for task generation and scheduling.
"""

from src.task_generator import Priority

# Task Generation Parameters
TASK_CONFIG = {
    Priority.HIGH: {
        'count': 20,
        'lambda': 3,  # Average arrival rate (λ) in seconds
        'service_min': 2,
        'service_max': 5
    },
    Priority.MEDIUM: {
        'count': 20,
        'lambda': 5,
        'service_min': 3,
        'service_max': 7
    },
    Priority.LOW: {
        'count': 10,
        'lambda': 7,
        'service_min': 5,
        'service_max': 10
    }
}

# System Parameters
SINGLE_PROCESSOR = {
    'name': 'Raspberry Pi 3',
    'cores': 1
}

MULTI_PROCESSOR = {
    'name': 'Raspberry Pi 3 (Multi-Core)',
    'cores': 4,
    'strategy': 'priority_based'  # Options: 'round_robin', 'least_loaded', 'priority_based'
}

# Simulation Parameters
SIMULATION = {
    'enabled': True,
    'speed_factor': 10.0,  # Higher values = faster simulation
    'run_time': 120,       # Maximum simulation time in seconds
    'metrics_collection_interval': 0.1,  # 100ms interval for metrics collection
    'min_collection_interval': 0.05,     # 50ms minimum between collections
    'throughput_smoothing': 0.3,         # Smoothing factor for throughput calculation
    'processor_busy_utilization': 100,   # CPU utilization when processor is busy
    'validation_thresholds': {
        'max_waiting_time': 1000000,     # Maximum reasonable waiting time (1000s)
        'max_queue_length': 1000,        # Maximum reasonable queue length
        'max_throughput': 1000,          # Maximum tasks per second
        'min_throughput': 0,             # Minimum tasks per second
        'max_cpu_usage': 100,            # Maximum CPU usage percentage
        'min_cpu_usage': 0,              # Minimum CPU usage percentage
        'max_memory_usage': 100,         # Maximum memory usage percentage
        'min_memory_usage': 0,           # Minimum memory usage percentage
    },
    'metrics_validation': {
        'require_positive_waiting': True,      # Enforce non-negative waiting times
        'require_monotonic_timestamps': True,  # Enforce increasing timestamps
        'normalize_priorities': True,          # Normalize priority ratios
        'validate_throughput': True,          # Validate throughput calculations
        'track_inversions': True,             # Track priority inversions
        'track_inheritance': True,            # Track priority inheritance events
    }
}

# Machine Learning Parameters
ML_SCHEDULER = {
    'history_size': 50,
    'features': [
        'priority',
        'service_time',
        'queue_length',
        'cpu_load',
        'memory_usage'
    ],
    'algorithm': 'decision_tree',  # Changed from 'linear_regression' to 'decision_tree'
    'max_depth': 5  # Added a specific parameter for decision trees
}

# Performance Comparison Parameters
COMPARISON = {
    'platforms': [
        {
            'name': 'Raspberry Pi 4',
            'type': 'embedded'
        },
        {
            'name': 'Desktop PC',
            'type': 'desktop'
        },
        {
            'name': 'Laptop',
            'type': 'laptop'
        }
    ],
    'metrics': [
        'avg_waiting_time',
        'throughput',
        'deadline_miss_rate',
        'priority_inversions',
        'memory_usage'
    ]
}

# Visualisation Parameters
VISUALISATION = {
    'colors': {
        'HIGH': '#FF5252',    # Red
        'MEDIUM': '#FFD740',  # Amber
        'LOW': '#69F0AE',     # Green
        'FCFS': '#2196F3',    # Blue
        'EDF': '#7B1FA2',     # Purple
        'Priority': '#FF5722',# Deep Orange
        'ML-Based': '#009688' # Teal
    },
    'save_path': 'results/',
    'max_throughput': 350.0   # Maximum throughput for chart scaling
}