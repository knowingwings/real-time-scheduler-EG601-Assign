"""
Enhanced Data Collection Utilities

This module provides improved functions for saving scheduling metrics to CSV files
and JSON for more reliable analysis and visualisation, with enhanced data validation
and standardisation of metrics.
"""

import os
import csv
import pandas as pd
import json
import time
from datetime import datetime
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict
from src.utils.platform_utils import get_platform_info, extract_platform_from_dir
from src.utils.json_utils import save_json

logger = logging.getLogger(__name__)

# Define default metrics to ensure consistent format
STANDARD_METRICS = {
    'completed_tasks': 0,
    'avg_waiting_time': 0.0,
    'avg_waiting_by_priority': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0},
    'tasks_by_priority': {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0},
    'deadline_tasks': 0,
    'deadline_met': 0,
    'deadline_miss_rate': 0.0
}

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
    
    # Create additional directories for organization
    os.makedirs(f"{data_dir}/raw", exist_ok=True)        # Store raw data files
    os.makedirs(f"{data_dir}/processed", exist_ok=True)  # Store processed metrics
    os.makedirs(f"{data_dir}/analysis", exist_ok=True)   # For analysis results
    
    logger.info(f"Created output directory for experiment: {timestamp}_{platform_type}")
    return timestamp, data_dir

def sanitize_metrics(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize metrics to ensure consistent format and valid values
    
    Args:
        metrics: Raw metrics dictionary
        
    Returns:
        Sanitized metrics with consistent types and values
    """
    # Add a fallback mechanism to ensure all metrics are initialized with defaults
    for key, default_value in STANDARD_METRICS.items():
        if key not in metrics:
            metrics[key] = default_value
            logger.warning(f"Missing metric {key}, initialized with default value {default_value}")

    # Create a copy to avoid modifying original
    sanitized = metrics.copy()
    
    # Ensure all standard metrics exist
    for key, default_value in STANDARD_METRICS.items():
        if key not in sanitized:
            sanitized[key] = default_value
        elif key == 'avg_waiting_by_priority':
            # Ensure all priority levels exist
            for priority, default_wait in STANDARD_METRICS['avg_waiting_by_priority'].items():
                if priority not in sanitized[key]:
                    sanitized[key][priority] = default_wait
        elif key == 'tasks_by_priority':
            # Ensure all priority levels exist
            for priority, default_count in STANDARD_METRICS['tasks_by_priority'].items():
                if priority not in sanitized[key]:
                    sanitized[key][priority] = default_count
    
    # Handle nested values like waiting times by priority
    if 'avg_waiting_by_priority' in sanitized:
        for priority, wait_time in sanitized['avg_waiting_by_priority'].items():
            if not isinstance(wait_time, (int, float)) or np.isnan(wait_time) or np.isinf(wait_time):
                sanitized['avg_waiting_by_priority'][priority] = 0.0
                logger.warning(f"Invalid waiting time for {priority} priority: {wait_time}, set to 0.0")
    
    # Handle special case for deadline metrics
    if 'deadline_miss_rate' in sanitized:
        if not isinstance(sanitized['deadline_miss_rate'], (int, float)) or np.isnan(sanitized['deadline_miss_rate']) or np.isinf(sanitized['deadline_miss_rate']):
            sanitized['deadline_miss_rate'] = 0.0
            logger.warning(f"Invalid deadline_miss_rate: set to 0.0")
    
    # Ensure time series data is valid
    for history_key in ['queue_length_history', 'memory_usage_history', 'timestamp_history']:
        if history_key in sanitized:
            # Ensure it's a list
            if not isinstance(sanitized[history_key], list):
                sanitized[history_key] = []
                logger.warning(f"{history_key} was not a list, replaced with empty list")
            
            # Filter out invalid values
            sanitized[history_key] = [
                value for value in sanitized[history_key] 
                if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value)
            ]
            
            # If empty, add a placeholder
            if not sanitized[history_key]:
                sanitized[history_key] = [0.0]
                logger.warning(f"{history_key} was empty, added placeholder value")
    
    return sanitized

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
    Save task metrics to CSV files with updated naming and enhanced data validation
    
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
        # Create empty placeholder file to indicate the run was attempted
        scenario_dir = f"{data_dir}/raw/scenario_{scenario}"
        os.makedirs(scenario_dir, exist_ok=True)
        run_dir = f"{scenario_dir}/run_{run_number}"
        os.makedirs(run_dir, exist_ok=True)
        placeholder_path = f"{run_dir}/{scheduler_name}_{processor_type}_{processor_name.replace(' ', '_')}_tasks_empty.txt"
        with open(placeholder_path, 'w') as f:
            f.write(f"No completed tasks for {scheduler_name} on {processor_name}\n")
        return
    
    # Create data for CSV
    task_data = []
    for task in completed_tasks:
        # Handle missing attributes safely with defaults
        task_dict = {
            'id': getattr(task, 'id', f"unknown_{len(task_data)}"),
            'priority': getattr(task.priority, 'name', str(getattr(task, 'priority', 'UNKNOWN'))),
            'arrival_time': getattr(task, 'arrival_time', 0.0),
            'start_time': getattr(task, 'start_time', 0.0),
            'completion_time': getattr(task, 'completion_time', 0.0),
            'waiting_time': getattr(task, 'waiting_time', 0.0),
            'service_time': getattr(task, 'service_time', 0.0),
            'deadline': getattr(task, 'deadline', None)
        }
        
        # Calculate deadline_met if possible
        if task_dict['deadline'] is not None and task_dict['completion_time'] is not None:
            task_dict['deadline_met'] = task_dict['completion_time'] <= task_dict['deadline']
        else:
            task_dict['deadline_met'] = None
        
        # Validate numeric values
        for key in ['arrival_time', 'start_time', 'completion_time', 'waiting_time', 'service_time']:
            if task_dict[key] is not None:
                # Replace NaN/Inf with 0.0
                if isinstance(task_dict[key], float) and (np.isnan(task_dict[key]) or np.isinf(task_dict[key])):
                    task_dict[key] = 0.0
                    logger.warning(f"Replaced NaN/Inf value in task {task_dict['id']} {key} with 0.0")
        
        task_data.append(task_dict)
    
    # Create scenario directory path
    scenario_dir = f"{data_dir}/raw/scenario_{scenario}"
    os.makedirs(scenario_dir, exist_ok=True)
    
    # Create run directory path within scenario
    run_dir = f"{scenario_dir}/run_{run_number}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Create file name with more concise components
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
    Save time series metrics to CSV files with enhanced data validation
    
    Args:
        metrics: Dictionary containing time series metrics
        scheduler_name: Name of the scheduler used
        processor_name: Name/identifier of the processor
        data_dir: Directory path to save data files
        scenario: Scenario number
        run_number: Run number
        processor_type: 'single' or 'multi'
    """
    # Create scenario directory path in raw data folder
    scenario_dir = f"{data_dir}/raw/scenario_{scenario}"
    os.makedirs(scenario_dir, exist_ok=True)
    
    # Create run directory path within scenario
    run_dir = f"{scenario_dir}/run_{run_number}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Get time series data with validation
    timestamp_history = metrics.get('timestamp_history', [])
    queue_length_history = metrics.get('queue_length_history', [])
    memory_usage_history = metrics.get('memory_usage_history', [])
    
    # Validate data - remove NaN/inf values
    timestamp_history = [t for t in timestamp_history if isinstance(t, (int, float)) and not np.isnan(t) and not np.isinf(t)]
    queue_length_history = [q for q in queue_length_history if isinstance(q, (int, float)) and not np.isnan(q) and not np.isinf(q)]
    memory_usage_history = [m for m in memory_usage_history if isinstance(m, (int, float)) and not np.isnan(m) and not np.isinf(m)]
    
    # Create a DataFrame for the time series data
    data = {}
    
    # Add timestamps - already relative to simulation start
    if timestamp_history:
        data['time'] = timestamp_history
    else:
        # Generate placeholder timestamps if missing
        logger.warning(f"Missing timestamp_history for {scheduler_name} on {processor_name}, generating placeholders")
        data['time'] = list(range(max(len(queue_length_history), len(memory_usage_history))))
    
    # Add queue length history
    if queue_length_history:
        # Ensure same length as time
        if 'time' in data:
            if len(queue_length_history) > len(data['time']):
                queue_length_history = queue_length_history[:len(data['time'])]
            elif len(queue_length_history) < len(data['time']):
                # Pad with last value or zeros
                if queue_length_history:
                    last_value = queue_length_history[-1]
                    queue_length_history.extend([last_value] * (len(data['time']) - len(queue_length_history)))
                else:
                    queue_length_history = [0] * len(data['time'])
        
        data['queue_length'] = queue_length_history
    
    # Add memory usage history
    if memory_usage_history:
        # Ensure same length as time
        if 'time' in data:
            if len(memory_usage_history) > len(data['time']):
                memory_usage_history = memory_usage_history[:len(data['time'])]
            elif len(memory_usage_history) < len(data['time']):
                # Pad with last value or zeros
                if memory_usage_history:
                    last_value = memory_usage_history[-1]
                    memory_usage_history.extend([last_value] * (len(data['time']) - len(memory_usage_history)))
                else:
                    memory_usage_history = [0] * len(data['time'])
        
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
        # Create empty placeholder file
        with open(file_path.replace('.csv', '_empty.txt'), 'w') as f:
            f.write(f"No time series data available for {scheduler_name} on {processor_name}\n")

def save_scheduler_metrics(metrics, scheduler_name, processor_name, 
                          data_dir, scenario=1, run_number=1, processor_type='single'):
    """
    Save scheduler metrics to a JSON file with enhanced data validation
    
    Args:
        metrics: Dictionary containing scheduler metrics
        scheduler_name: Name of the scheduler used
        processor_name: Name/identifier of the processor
        data_dir: Directory path to save data files
        scenario: Scenario number
        run_number: Run number
        processor_type: 'single' or 'multi'
    """
    # Create scenario directory path in raw and processed folders
    raw_scenario_dir = f"{data_dir}/raw/scenario_{scenario}"
    proc_scenario_dir = f"{data_dir}/processed/scenario_{scenario}"
    os.makedirs(raw_scenario_dir, exist_ok=True)
    os.makedirs(proc_scenario_dir, exist_ok=True)
    
    # Create run directory path within scenario
    raw_run_dir = f"{raw_scenario_dir}/run_{run_number}"
    proc_run_dir = f"{proc_scenario_dir}/run_{run_number}"
    os.makedirs(raw_run_dir, exist_ok=True)
    os.makedirs(proc_run_dir, exist_ok=True)
    
    # Add metadata to metrics
    metrics['scheduler_name'] = scheduler_name
    metrics['processor_name'] = processor_name
    metrics['processor_type'] = processor_type
    metrics['scenario'] = scenario
    metrics['run_number'] = run_number
    
    # Save the raw metrics first (unmodified)
    raw_file_path = f"{raw_run_dir}/{scheduler_name}_{processor_type}_{processor_name.replace(' ', '_')}_metrics_raw.json"
    save_json(metrics, raw_file_path)
    
    # Sanitize metrics for processed version
    sanitized_metrics = sanitize_metrics(metrics)
    
    # Extract scalar metrics (excluding time series and task lists) for the processed version
    scalar_metrics = {}
    for key, value in sanitized_metrics.items():
        # Skip time series data but keep metadata
        if (isinstance(value, list) and 
            key not in ['scheduler_name', 'processor_name', 'processor_type', 'scenario', 'run_number'] and
            (key.endswith('_history') or key == 'completed_tasks' or
             key == 'prediction_errors' or key == 'training_events')):
            continue
        
        # Add scalar values and metadata
        scalar_metrics[key] = value
    
    # Create file name for processed metrics
    proc_file_path = f"{proc_run_dir}/{scheduler_name}_{processor_type}_{processor_name.replace(' ', '_')}_metrics.json"
    
    # Save as JSON
    save_json(scalar_metrics, proc_file_path)
    
    logger.info(f"Saved scheduler metrics to {raw_file_path} and {proc_file_path}")

def save_multi_processor_metrics(system_metrics, scheduler_name, data_dir, scenario=1, run_number=1):
    """
    Save multi-processor system metrics to a JSON file with enhanced data validation
    
    Args:
        system_metrics: Dictionary containing system-wide metrics
        scheduler_name: Name of the scheduler used
        data_dir: Directory path to save data files
        scenario: Scenario number
        run_number: Run number
    """
    # Create scenario directory path in raw and processed folders
    raw_scenario_dir = f"{data_dir}/raw/scenario_{scenario}"
    proc_scenario_dir = f"{data_dir}/processed/scenario_{scenario}"
    os.makedirs(raw_scenario_dir, exist_ok=True)
    os.makedirs(proc_scenario_dir, exist_ok=True)
    
    # Create run directory path within scenario
    raw_run_dir = f"{raw_scenario_dir}/run_{run_number}"
    proc_run_dir = f"{proc_scenario_dir}/run_{run_number}"
    os.makedirs(raw_run_dir, exist_ok=True)
    os.makedirs(proc_run_dir, exist_ok=True)
    
    # Add metadata to metrics
    system_metrics['scheduler_name'] = scheduler_name
    system_metrics['processor_type'] = 'multi'
    system_metrics['scenario'] = scenario
    system_metrics['run_number'] = run_number
    
    # Save the raw metrics first (unmodified)
    raw_file_path = f"{raw_run_dir}/{scheduler_name}_multi_system_metrics_raw.json"
    save_json(system_metrics, raw_file_path)
    
    # Extract system-wide metrics (excluding time series, task lists, and per-processor details)
    # and validate them
    system_metrics_sanitized = {}
    for key, value in system_metrics.items():
        # Skip time series data but keep metadata
        if (isinstance(value, list) and 
            key not in ['scheduler_name', 'processor_type', 'scenario', 'run_number'] and
            (key.endswith('_history') or key == 'completed_tasks')):
            continue
        
        # Skip per-processor metrics but keep everything else
        if key != 'per_processor_metrics':
            # Validate numeric values
            if isinstance(value, (int, float)) and (np.isnan(value) or np.isinf(value)):
                system_metrics_sanitized[key] = 0.0
                logger.warning(f"Replaced NaN/Inf value in {key} with 0.0")
            else:
                system_metrics_sanitized[key] = value
    
    # Ensure all standard metrics exist
    for key, default_value in STANDARD_METRICS.items():
        if key not in system_metrics_sanitized:
            system_metrics_sanitized[key] = default_value
    
    # Create file name for processed metrics
    proc_file_path = f"{proc_run_dir}/{scheduler_name}_multi_system_metrics.json"
    
    # Save as JSON
    save_json(system_metrics_sanitized, proc_file_path)
    
    logger.info(f"Saved multi-processor system metrics to {raw_file_path} and {proc_file_path}")

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

def combine_run_metrics(data_dir):
    """
    Combine metrics from all runs into summary files for easier analysis
    
    Args:
        data_dir: Directory containing experiment data
    """
    logger.info(f"Combining run metrics for data in {data_dir}")
    
    # Get all processed metrics files
    metrics_files = []
    for scenario_dir in os.listdir(f"{data_dir}/processed"):
        if not scenario_dir.startswith("scenario_"):
            continue
            
        scenario_path = os.path.join(data_dir, "processed", scenario_dir)
        if not os.path.isdir(scenario_path):
            continue
            
        for run_dir in os.listdir(scenario_path):
            if not run_dir.startswith("run_"):
                continue
                
            run_path = os.path.join(scenario_path, run_dir)
            if not os.path.isdir(run_path):
                continue
                
            # Find all metrics JSON files in this run
            for filename in os.listdir(run_path):
                if filename.endswith("_metrics.json"):
                    file_path = os.path.join(run_path, filename)
                    metrics_files.append(file_path)
    
    # Group by scheduler and processor type - using a regular dict instead of defaultdict to fix the append error
    metrics_by_scheduler = {}
    
    for file_path in metrics_files:
        try:
            with open(file_path, 'r') as f:
                metrics = json.load(f)
                
            scheduler_name = metrics.get('scheduler_name', 'unknown')
            processor_type = metrics.get('processor_type', 'unknown')
            scenario = metrics.get('scenario', 0)
            
            # Organize by scheduler, processor type, and scenario
            key = f"{scheduler_name}_{processor_type}_scenario_{scenario}"
            
            # Initialize list if key doesn't exist
            if key not in metrics_by_scheduler:
                metrics_by_scheduler[key] = []
                
            metrics_by_scheduler[key].append(metrics)
        except Exception as e:
            logger.error(f"Error loading metrics from {file_path}: {str(e)}")
    
    # Create summary directory
    summary_dir = os.path.join(data_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    # Calculate average metrics for each group
    for key, metrics_list in metrics_by_scheduler.items():
        if not metrics_list:
            logger.warning(f"No metrics found for {key}")
            continue
            
        # Initialize with metadata from first metrics
        summary = {
            'scheduler_name': metrics_list[0].get('scheduler_name', 'unknown'),
            'processor_type': metrics_list[0].get('processor_type', 'unknown'),
            'scenario': metrics_list[0].get('scenario', 0),
            'run_count': len(metrics_list),
            'metrics': {}
        }
        
        # Collect all scalar metrics
        scalar_metrics = {}
        
        for metrics in metrics_list:
            for key, value in metrics.items():
                # Skip metadata and non-scalar values
                if key in ['scheduler_name', 'processor_name', 'processor_type', 'scenario', 'run_number']:
                    continue
                    
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if key not in scalar_metrics:
                        scalar_metrics[key] = []
                    scalar_metrics[key].append(value)
        
        # Calculate statistics for each metric
        for key, values in scalar_metrics.items():
            if not values:
                continue
                
            # Filter out NaN values
            valid_values = [v for v in values if not np.isnan(v) and not np.isinf(v)]
            
            if not valid_values:
                continue
                
            summary['metrics'][key] = {
                'mean': float(np.mean(valid_values)),
                'min': float(np.min(valid_values)),
                'max': float(np.max(valid_values)),
                'std': float(np.std(valid_values)) if len(valid_values) > 1 else 0.0
            }
        
        # Save summary
        summary_path = os.path.join(summary_dir, f"{key}_summary.json")
        save_json(summary, summary_path)
        logger.info(f"Saved metrics summary to {summary_path}")
    
    logger.info(f"Completed combining run metrics into summary files")