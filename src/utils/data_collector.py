"""
Data Collection Utilities

This module provides functions for saving scheduling metrics to CSV files
for later analysis and visualisation, with an improved directory structure
and naming convention.
"""

import os
import csv
import pandas as pd
import json
import time
from datetime import datetime
import logging
from typing import Dict, List, Any, Optional
from src.utils.platform_utils import get_platform_info, extract_platform_from_dir
from src.utils.json_utils import save_json

logger = logging.getLogger(__name__)

def ensure_output_dir(base_path='results', experiment_id=None):
    """
    Ensure the output directory structure exists with updated naming
    
    Args:
        base_path: Base path for results directory
        experiment_id: Optional custom experiment ID (if None, timestamp is used)
        
    Returns:
        Tuple of (timestamp string, directory path) used for file naming
    """
    # Get timestamp and platform info
    timestamp = experiment_id or datetime.now().strftime('%Y%m%d_%H%M%S')
    platform_info = get_platform_info()
    platform_type = platform_info['type']
    
    # Create directory with timestamp and platform type
    data_dir = f"{base_path}/data/{timestamp}_{platform_type}"
    os.makedirs(data_dir, exist_ok=True)
    
    # Create system_info directory for platform data
    os.makedirs(f"{data_dir}/system_info", exist_ok=True)
    
    logger.info(f"Created output directory for experiment: {timestamp}_{platform_type}")
    return timestamp, data_dir

def save_system_info(data_dir):
    """
    Save system information to a JSON file
    
    Args:
        data_dir: Directory path to save the system info
    """
    system_info = get_platform_info()
    
    # Create the file path
    file_path = f"{data_dir}/system_info/platform_info.json"
    
    # Save as JSON
    save_json(system_info, file_path)
    
    logger.info(f"Saved system information to {file_path}")
    return system_info

def save_task_metrics(completed_tasks, scheduler_name, processor_name, 
                      data_dir, scenario=1, run_number=1, processor_type='single'):
    """
    Save task metrics to CSV files with updated naming
    
    Args:
        completed_tasks: List of completed Task objects
        scheduler_name: Name of the scheduler used
        processor_name: Name/identifier of the processor
        data_dir: Directory path to save data files
        scenario: Scenario number
        run_number: Run number
        processor_type: 'single' or 'multi'
    """
    if not completed_tasks:
        logger.warning(f"No completed tasks for {scheduler_name} on {processor_name}")
        return
    
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
    
    # Create scenario directory path
    scenario_dir = f"{data_dir}/scenario_{scenario}"
    os.makedirs(scenario_dir, exist_ok=True)
    
    # Create run directory path within scenario
    run_dir = f"{scenario_dir}/run_{run_number}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Create file name with more concise components since folders provide context
    file_path = f"{run_dir}/{scheduler_name}_{processor_type}_{processor_name.replace(' ', '_')}_tasks.csv"
    
    # Save as CSV
    with open(file_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=task_data[0].keys())
        writer.writeheader()
        writer.writerows(task_data)
    
    logger.info(f"Saved task metrics to {file_path}")

def save_time_series_metrics(metrics, scheduler_name, processor_name, 
                            data_dir, scenario=1, run_number=1, processor_type='single'):
    """
    Save time series metrics to CSV files with updated naming
    
    Args:
        metrics: Dictionary containing time series metrics
        scheduler_name: Name of the scheduler used
        processor_name: Name/identifier of the processor
        data_dir: Directory path to save data files
        scenario: Scenario number
        run_number: Run number
        processor_type: 'single' or 'multi'
    """
    # Create scenario directory path
    scenario_dir = f"{data_dir}/scenario_{scenario}"
    os.makedirs(scenario_dir, exist_ok=True)
    
    # Create run directory path within scenario
    run_dir = f"{scenario_dir}/run_{run_number}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Get time series data
    timestamp_history = metrics.get('timestamp_history', [])
    queue_length_history = metrics.get('queue_length_history', [])
    memory_usage_history = metrics.get('memory_usage_history', [])
    
    # Create a DataFrame for the time series data
    data = {}
    
    # Add timestamps - already relative to simulation start
    if timestamp_history:
        data['time'] = timestamp_history
    
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
    
    # Create file name with more concise components
    file_path = f"{run_dir}/{scheduler_name}_{processor_type}_{processor_name.replace(' ', '_')}_timeseries.csv"
    
    # Save as CSV if we have any data
    if data and 'time' in data:
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)
        logger.info(f"Saved time series metrics to {file_path}")
    else:
        logger.warning(f"No time series data available for {scheduler_name} on {processor_name}")

def save_scheduler_metrics(metrics, scheduler_name, processor_name, 
                          data_dir, scenario=1, run_number=1, processor_type='single'):
    """
    Save scheduler metrics to a JSON file with updated naming
    
    Args:
        metrics: Dictionary containing scheduler metrics
        scheduler_name: Name of the scheduler used
        processor_name: Name/identifier of the processor
        data_dir: Directory path to save data files
        scenario: Scenario number
        run_number: Run number
        processor_type: 'single' or 'multi'
    """
    # Create scenario directory path
    scenario_dir = f"{data_dir}/scenario_{scenario}"
    os.makedirs(scenario_dir, exist_ok=True)
    
    # Create run directory path within scenario
    run_dir = f"{scenario_dir}/run_{run_number}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Add metadata to metrics
    metrics['scheduler_name'] = scheduler_name
    metrics['processor_name'] = processor_name
    metrics['processor_type'] = processor_type
    metrics['scenario'] = scenario
    metrics['run_number'] = run_number
    
    # Extract scalar metrics (excluding time series and task lists)
    scalar_metrics = {}
    for key, value in metrics.items():
        # Skip time series data but keep metadata
        if (isinstance(value, list) and 
            key not in ['scheduler_name', 'processor_name', 'processor_type', 'scenario', 'run_number'] and
            (key.endswith('_history') or key == 'completed_tasks' or
             key == 'prediction_errors' or key == 'training_events')):
            continue
        
        # Add scalar values and metadata
        scalar_metrics[key] = value
    
    # Create file name with more concise components
    file_path = f"{run_dir}/{scheduler_name}_{processor_type}_{processor_name.replace(' ', '_')}_metrics.json"
    
    # Save as JSON
    save_json(scalar_metrics, file_path)
    
    logger.info(f"Saved scheduler metrics to {file_path}")

def save_multi_processor_metrics(system_metrics, scheduler_name, data_dir, scenario=1, run_number=1):
    """
    Save multi-processor system metrics to a JSON file with updated naming
    
    Args:
        system_metrics: Dictionary containing system-wide metrics
        scheduler_name: Name of the scheduler used
        data_dir: Directory path to save data files
        scenario: Scenario number
        run_number: Run number
    """
    # Create scenario directory path
    scenario_dir = f"{data_dir}/scenario_{scenario}"
    os.makedirs(scenario_dir, exist_ok=True)
    
    # Create run directory path within scenario
    run_dir = f"{scenario_dir}/run_{run_number}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Add metadata to metrics
    system_metrics['scheduler_name'] = scheduler_name
    system_metrics['processor_type'] = 'multi'
    system_metrics['scenario'] = scenario
    system_metrics['run_number'] = run_number
    
    # Extract system-wide metrics (excluding time series, task lists, and per-processor details)
    system_metrics_copy = {}
    for key, value in system_metrics.items():
        # Skip time series data but keep metadata
        if (isinstance(value, list) and 
            key not in ['scheduler_name', 'processor_type', 'scenario', 'run_number'] and
            (key.endswith('_history') or key == 'completed_tasks')):
            continue
        
        # Skip per-processor metrics but keep everything else
        if key != 'per_processor_metrics':
            system_metrics_copy[key] = value
    
    # Create file name with more concise components
    file_path = f"{run_dir}/{scheduler_name}_multi_system_metrics.json"
    
    # Save as JSON
    save_json(system_metrics_copy, file_path)
    
    logger.info(f"Saved multi-processor system metrics to {file_path}")

def save_comparison_results(results, scheduler_names, data_dir, experiment_name="comparison"):
    """
    Save comparison results to a JSON file with updated naming
    
    Args:
        results: Dictionary containing comparison results
        scheduler_names: List of scheduler names compared
        data_dir: Directory path to save data files
        experiment_name: Name for this comparison experiment
    
    Note:
        This function is used for cross-algorithm and cross-platform comparisons
        to preserve results for later analysis and reporting.
    """
    # Create analysis directory
    analysis_dir = f"{data_dir}/analysis"
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Get platform information
    platform_info = get_platform_info()
    
    # Create a copy of the results to modify
    comparison_data = {
        'schedulers': scheduler_names,
        'platform': platform_info['type'],
        'results': results,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # File name
    file_path = f"{analysis_dir}/{experiment_name}_results.json"
    
    # Save as JSON
    save_json(comparison_data, file_path)
    
    logger.info(f"Saved comparison results to {file_path}")

# Create scenario descriptions for better context in reports
def create_scenario_descriptions(data_dir):
    """
    Create a JSON file with scenario descriptions
    
    Args:
        data_dir: Directory path to save the descriptions
    """
    scenarios = {
        1: {
            "name": "Baseline",
            "description": "Default task distribution as specified in the configuration",
            "characteristics": "Balanced workload with standard arrival rates and service times"
        },
        2: {
            "name": "High Load",
            "description": "Increased arrival rates (lambda values reduced by 50%)",
            "characteristics": "System under stress with more frequent task arrivals"
        },
        3: {
            "name": "Priority Inversion",
            "description": "Adjusted service times to increase likelihood of priority inversion",
            "characteristics": "Low priority tasks have longer service times, high priority tasks are rarer"
        }
    }
    
    # Save to root of experiment directory
    file_path = f"{data_dir}/scenario_descriptions.json"
    save_json(scenarios, file_path)
        
    logger.info(f"Saved scenario descriptions to {file_path}")