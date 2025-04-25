"""
Data Collection Utilities

This module provides functions for saving scheduling metrics to CSV files
for later analysis and visualization.
"""

import os
import csv
import pandas as pd
import json
import time
from datetime import datetime
import platform
import psutil
import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

def get_platform_info() -> Dict[str, Any]:
    """Get information about the current platform"""
    system_info = {
        'system': platform.system(),
        'node': platform.node(),
        'release': platform.release(),
        'version': platform.version(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'cpu_count': psutil.cpu_count(logical=False),
        'cpu_count_logical': psutil.cpu_count(logical=True),
        'memory_total': psutil.virtual_memory().total,
        'memory_available': psutil.virtual_memory().available,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S')
    }
    
    # Determine platform type
    if 'raspberry' in system_info['node'].lower():
        system_info['type'] = 'embedded'
    elif system_info['system'] == 'Darwin':
        system_info['type'] = 'laptop' if 'MacBook' in platform.node() else 'desktop'
    elif system_info['system'] == 'Windows':
        system_info['type'] = 'laptop' if hasattr(psutil, 'sensors_battery') and psutil.sensors_battery() else 'desktop'
    else:
        system_info['type'] = 'unknown'
    
    return system_info

def ensure_output_dir(base_path='results'):
    """
    Ensure the output directory structure exists
    
    Args:
        base_path: Base path for results directory
        
    Returns:
        Timestamp string used for file naming
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create base directories
    data_dir = f"{base_path}/data/{timestamp}"
    os.makedirs(data_dir, exist_ok=True)
    
    # Create subdirectories
    directories = [
        f"{data_dir}/single_processor",
        f"{data_dir}/multi_processor",
        f"{data_dir}/system_info"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    logger.info(f"Created output directories for timestamp: {timestamp}")
    return timestamp

def save_system_info(base_path='results', timestamp=None):
    """
    Save system information to a JSON file
    
    Args:
        base_path: Base path for results directory
        timestamp: Timestamp string for directory naming
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    system_info = get_platform_info()
    
    # Create the file path
    file_path = f"{base_path}/data/{timestamp}/system_info/platform_info.json"
    
    # Save as JSON
    with open(file_path, 'w') as f:
        json.dump(system_info, f, indent=4)
    
    logger.info(f"Saved system information to {file_path}")
    return system_info

def save_task_metrics(completed_tasks, scheduler_name, processor_name, 
                      base_path='results', timestamp=None, processor_type='single'):
    """
    Save task metrics to CSV files
    
    Args:
        completed_tasks: List of completed Task objects
        scheduler_name: Name of the scheduler used
        processor_name: Name/identifier of the processor
        base_path: Base path for results directory
        timestamp: Timestamp string for directory naming
        processor_type: 'single' or 'multi'
    """
    if not completed_tasks:
        logger.warning(f"No completed tasks for {scheduler_name} on {processor_name}")
        return
    
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create data for CSV
    task_data = []
    for task in completed_tasks:
        task_data.append({
            'id': task.id,
            'priority': task.priority.name if hasattr(task.priority, 'name') else str(task.priority),
            'arrival_time': task.arrival_time,
            'start_time': task.start_time,
            'completion_time': task.completion_time,
            'waiting_time': task.waiting_time,
            'service_time': task.service_time,
            'deadline': task.deadline if hasattr(task, 'deadline') else None,
            'deadline_met': (task.completion_time <= task.deadline) if hasattr(task, 'deadline') and task.deadline is not None else None
        })
    
    # Create directory path based on processor type
    dir_path = f"{base_path}/data/{timestamp}/{processor_type}_processor"
    os.makedirs(dir_path, exist_ok=True)
    
    # Create CSV file name
    file_path = f"{dir_path}/{scheduler_name}_{processor_name.replace(' ', '_')}_tasks.csv"
    
    # Save as CSV
    with open(file_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=task_data[0].keys())
        writer.writeheader()
        writer.writerows(task_data)
    
    logger.info(f"Saved task metrics to {file_path}")

def save_time_series_metrics(metrics, scheduler_name, processor_name, 
                            base_path='results', timestamp=None, processor_type='single'):
    """
    Save time series metrics to CSV files
    
    Args:
        metrics: Dictionary containing time series metrics
        scheduler_name: Name of the scheduler used
        processor_name: Name/identifier of the processor
        base_path: Base path for results directory
        timestamp: Timestamp string for directory naming
        processor_type: 'single' or 'multi'
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create directory path based on processor type
    dir_path = f"{base_path}/data/{timestamp}/{processor_type}_processor"
    os.makedirs(dir_path, exist_ok=True)
    
    # Get time series data
    timestamp_history = metrics.get('timestamp_history', [])
    queue_length_history = metrics.get('queue_length_history', [])
    memory_usage_history = metrics.get('memory_usage_history', [])
    
    # Create a DataFrame for the time series data
    data = {}
    
    # Add timestamps
    if timestamp_history:
        # Convert absolute timestamps to relative time
        if len(timestamp_history) > 0:
            start_time = timestamp_history[0]
            relative_times = [t - start_time for t in timestamp_history]
            data['time'] = relative_times
        else:
            data['time'] = []
    
    # Add queue length history
    if queue_length_history:
        # Ensure same length as time
        if 'time' in data:
            if len(queue_length_history) > len(data['time']):
                queue_length_history = queue_length_history[:len(data['time'])]
            elif len(queue_length_history) < len(data['time']):
                # Pad with zeros
                queue_length_history.extend([0] * (len(data['time']) - len(queue_length_history)))
        
        data['queue_length'] = queue_length_history
    
    # Add memory usage history
    if memory_usage_history:
        # Ensure same length as time
        if 'time' in data:
            if len(memory_usage_history) > len(data['time']):
                memory_usage_history = memory_usage_history[:len(data['time'])]
            elif len(memory_usage_history) < len(data['time']):
                # Pad with zeros
                memory_usage_history.extend([0] * (len(data['time']) - len(memory_usage_history)))
        
        data['memory_usage'] = memory_usage_history
    
    # Create CSV file name
    file_path = f"{dir_path}/{scheduler_name}_{processor_name.replace(' ', '_')}_timeseries.csv"
    
    # Save as CSV if we have any data
    if data and 'time' in data:
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
        logger.info(f"Saved time series metrics to {file_path}")
    else:
        logger.warning(f"No time series data available for {scheduler_name} on {processor_name}")

def save_scheduler_metrics(metrics, scheduler_name, processor_name, 
                          base_path='results', timestamp=None, processor_type='single'):
    """
    Save scheduler metrics to a JSON file
    
    Args:
        metrics: Dictionary containing scheduler metrics
        scheduler_name: Name of the scheduler used
        processor_name: Name/identifier of the processor
        base_path: Base path for results directory
        timestamp: Timestamp string for directory naming
        processor_type: 'single' or 'multi'
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create directory path based on processor type
    dir_path = f"{base_path}/data/{timestamp}/{processor_type}_processor"
    os.makedirs(dir_path, exist_ok=True)
    
    # Extract scalar metrics (excluding time series and task lists)
    scalar_metrics = {}
    for key, value in metrics.items():
        # Skip time series data and task lists
        if (isinstance(value, list) and 
            (key.endswith('_history') or key == 'completed_tasks' or
             key == 'prediction_errors' or key == 'training_events')):
            continue
        
        # Add scalar values
        scalar_metrics[key] = value
    
    # Create JSON file name
    file_path = f"{dir_path}/{scheduler_name}_{processor_name.replace(' ', '_')}_metrics.json"
    
    # Save as JSON
    with open(file_path, 'w') as f:
        json.dump(scalar_metrics, f, indent=4)
    
    logger.info(f"Saved scheduler metrics to {file_path}")

def save_multi_processor_metrics(system_metrics, scheduler_name, 
                                base_path='results', timestamp=None):
    """
    Save multi-processor system metrics to a JSON file
    
    Args:
        system_metrics: Dictionary containing system-wide metrics
        scheduler_name: Name of the scheduler used
        base_path: Base path for results directory
        timestamp: Timestamp string for directory naming
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create directory path
    dir_path = f"{base_path}/data/{timestamp}/multi_processor"
    os.makedirs(dir_path, exist_ok=True)
    
    # Extract system-wide metrics (excluding time series, task lists, and per-processor details)
    system_metrics_copy = {}
    for key, value in system_metrics.items():
        # Skip time series data, task lists, and per-processor metrics
        if (isinstance(value, list) and 
            (key.endswith('_history') or key == 'completed_tasks' or
             key == 'per_processor_metrics')):
            continue
        
        # Add scalar values
        system_metrics_copy[key] = value
    
    # Create JSON file name
    file_path = f"{dir_path}/{scheduler_name}_system_metrics.json"
    
    # Save as JSON
    with open(file_path, 'w') as f:
        json.dump(system_metrics_copy, f, indent=4)
    
    logger.info(f"Saved multi-processor system metrics to {file_path}")

def save_comparison_results(results, scheduler_names, base_path='results', timestamp=None):
    """
    Save comparison results to a JSON file
    
    Args:
        results: Dictionary containing comparison results
        scheduler_names: List of scheduler names compared
        base_path: Base path for results directory
        timestamp: Timestamp string for directory naming
    """
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create directory path
    dir_path = f"{base_path}/data/{timestamp}"
    os.makedirs(dir_path, exist_ok=True)
    
    # Create a copy of the results to modify
    comparison_data = {
        'schedulers': scheduler_names,
        'timestamp': timestamp,
        'platform': get_platform_info()['type'],
        'results': results
    }
    
    # Create JSON file name
    file_path = f"{dir_path}/comparison_results.json"
    
    # Save as JSON
    with open(file_path, 'w') as f:
        json.dump(comparison_data, f, indent=4)
    
    logger.info(f"Saved comparison results to {file_path}")