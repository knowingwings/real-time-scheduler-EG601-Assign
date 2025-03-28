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
    'run_time': 120        # Maximum simulation time in seconds
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
    'algorithm': 'linear_regression'  # Options: 'linear_regression', 'decision_tree'
}

# Performance Comparison Parameters
COMPARISON = {
    'platforms': [
        {
            'name': 'Raspberry Pi 3',
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
    'save_path': 'results/'
}