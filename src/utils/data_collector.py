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
    'deadline_miss_rate': 0.0,
    'deadline_misses': 0,
    'avg_deadline_margin': 0.0,
    'queue_length_history': [],
    'memory_usage_history': [],
    'cpu_usage_history': [],
    'timestamp_history': [],
    'throughput_history': [],
    'avg_cpu_usage': 0.0,
    'avg_memory_usage': 0.0,
    'avg_throughput': 0.0,
    'priority_inversions': 0,
    'priority_inheritance_events': 0,
    'avg_response_time': 0.0,
    'avg_turnaround_time': 0.0
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
    Save task metrics to CSV files with validation that only includes actual collected data
    """
    if not completed_tasks:
        logger.warning(f"No completed tasks for {scheduler_name} on {processor_name}")
        return
    
    # Create data for CSV, only including tasks with valid data
    task_data = []
    for task in completed_tasks:
        # Only include tasks that have the minimum required valid data
        if (not hasattr(task, 'arrival_time') or 
            not hasattr(task, 'start_time') or 
            not hasattr(task, 'completion_time')):
            continue
            
        # Skip tasks with invalid timing data
        if (np.isnan(task.arrival_time) or np.isinf(task.arrival_time) or
            np.isnan(task.start_time) or np.isinf(task.start_time) or
            np.isnan(task.completion_time) or np.isinf(task.completion_time)):
            continue
            
        task_dict = {
            'id': getattr(task, 'id', None),
            'priority': getattr(task.priority, 'name', None) if hasattr(task, 'priority') else None,
            'arrival_time': task.arrival_time,
            'start_time': task.start_time,
            'completion_time': task.completion_time,
            'waiting_time': task.completion_time - task.arrival_time if hasattr(task, 'completion_time') else None,
            'service_time': task.completion_time - task.start_time if hasattr(task, 'completion_time') else None
        }
        
        # Only include deadline info if it exists and is valid
        if hasattr(task, 'deadline') and task.deadline is not None:
            if not (np.isnan(task.deadline) or np.isinf(task.deadline)):
                task_dict['deadline'] = task.deadline
                task_dict['deadline_met'] = task.completion_time <= task.deadline
        
        task_data.append(task_dict)
    
    if not task_data:
        logger.warning(f"No valid task data for {scheduler_name} on {processor_name}")
        return
        
    # Create scenario directory path
    scenario_dir = f"{data_dir}/raw/scenario_{scenario}"
    os.makedirs(scenario_dir, exist_ok=True)
    run_dir = f"{scenario_dir}/run_{run_number}"
    os.makedirs(run_dir, exist_ok=True)
    
    # Save only valid task data
    file_path = f"{run_dir}/{scheduler_name}_{processor_type}_{processor_name.replace(' ', '_')}_tasks.csv"
    df = pd.DataFrame(task_data)
    df.to_csv(file_path, index=False)
    
    logger.info(f"Saved {len(task_data)} valid task metrics to {file_path}")

def save_time_series_metrics(metrics, scheduler_name, processor_name, 
                            data_dir, scenario=1, run_number=1, processor_type='single'):
    """
    Save time series metrics to CSV files with enhanced data validation
    """
    # Get time series data
    timestamp_history = metrics.get('timestamp_history', [])
    queue_length_history = metrics.get('queue_length_history', [])
    memory_usage_history = metrics.get('memory_usage_history', [])
    cpu_usage_history = metrics.get('cpu_usage_history', [])
    throughput_history = metrics.get('throughput_history', [])
    
    # If we don't have valid timestamp data, we can't create valid time series
    if not timestamp_history:
        logger.warning(f"No timestamp data for {scheduler_name} on {processor_name}, skipping time series metrics")
        return
        
    # Create data frame with only valid data points
    data = {'time': timestamp_history}
    
    # Only include metrics that have data and align with timestamps
    if queue_length_history and len(queue_length_history) == len(timestamp_history):
        data['queue_length'] = queue_length_history
    if memory_usage_history and len(memory_usage_history) == len(timestamp_history):
        data['memory_usage'] = memory_usage_history
    if cpu_usage_history and len(cpu_usage_history) == len(timestamp_history):
        data['cpu_usage'] = cpu_usage_history
    if throughput_history and len(throughput_history) == len(timestamp_history):
        data['throughput'] = throughput_history
    
    # Clean up any NaN/Inf values by dropping those rows
    df = pd.DataFrame(data)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    if df.empty:
        logger.warning(f"No valid time series data for {scheduler_name} on {processor_name}")
        return
    
    # Save to CSV
    scenario_dir = f"{data_dir}/raw/scenario_{scenario}"
    os.makedirs(scenario_dir, exist_ok=True)
    run_dir = f"{scenario_dir}/run_{run_number}"
    os.makedirs(run_dir, exist_ok=True)
    
    file_path = f"{run_dir}/{scheduler_name}_{processor_type}_{processor_name.replace(' ', '_')}_timeseries.csv"
    df.to_csv(file_path, index=False)
    logger.info(f"Saved time series metrics to {file_path}")

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
    Combine metrics from all runs with enhanced validation and error handling
    
    Args:
        data_dir: Directory containing experiment data
    """
    metrics_by_scheduler = {}
    
    # Process all metrics files
    for scenario_dir in os.listdir(f"{data_dir}/processed"):
        if not scenario_dir.startswith("scenario_"):
            continue
            
        for run_dir in os.listdir(os.path.join(data_dir, "processed", scenario_dir)):
            if not run_dir.startswith("run_"):
                continue
                
            run_path = os.path.join(data_dir, "processed", scenario_dir, run_dir)
            for filename in os.listdir(run_path):
                if not filename.endswith("_metrics.json"):
                    continue
                    
                try:
                    file_path = os.path.join(run_path, filename)
                    with open(file_path, 'r') as f:
                        metrics = json.load(f)
                    
                    # Extract metadata
                    scheduler_name = metrics.get('scheduler_name', 'unknown')
                    processor_type = metrics.get('processor_type', 'unknown')
                    scenario = metrics.get('scenario', 0)
                    
                    # Create key for grouping
                    key = f"{scheduler_name}_{processor_type}_scenario_{scenario}"
                    if key not in metrics_by_scheduler:
                        metrics_by_scheduler[key] = []
                    
                    # Validate and standardize metrics before adding
                    validated_metrics = sanitize_metrics(metrics)
                    
                    # Ensure all standard metrics exist
                    for metric_key, default_value in STANDARD_METRICS.items():
                        if metric_key not in validated_metrics:
                            validated_metrics[metric_key] = default_value
                            logger.warning(f"Added missing metric {metric_key} with default value")
                    
                    metrics_by_scheduler[key].append(validated_metrics)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    continue
    
    # Create summaries with enhanced error handling
    summary_dir = os.path.join(data_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    
    for key, metrics_list in metrics_by_scheduler.items():
        if not metrics_list:
            logger.warning(f"No valid metrics found for {key}")
            continue
        
        try:
            # Initialize summary with metadata
            summary = {
                'scheduler_name': metrics_list[0].get('scheduler_name', 'unknown'),
                'processor_type': metrics_list[0].get('processor_type', 'unknown'),
                'scenario': metrics_list[0].get('scenario', 0),
                'run_count': len(metrics_list),
                'metrics': {}
            }
            
            # Calculate statistics for scalar metrics
            scalar_metrics = {}
            for metrics in metrics_list:
                for metric_key, value in metrics.items():
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        if metric_key not in scalar_metrics:
                            scalar_metrics[metric_key] = []
                        scalar_metrics[metric_key].append(value)
            
            # Process scalar metrics with validation
            for metric_key, values in scalar_metrics.items():
                valid_values = [v for v in values if not np.isnan(v) and not np.isinf(v)]
                
                if not valid_values:
                    logger.warning(f"No valid values for {metric_key} in {key}")
                    continue
                
                summary['metrics'][metric_key] = {
                    'mean': float(np.mean(valid_values)),
                    'min': float(np.min(valid_values)),
                    'max': float(np.max(valid_values)),
                    'std': float(np.std(valid_values)) if len(valid_values) > 1 else 0.0,
                    'samples': len(valid_values)
                }
            
            # Save summary
            summary_path = os.path.join(summary_dir, f"{key}_summary.json")
            save_json(summary, summary_path)
            logger.info(f"Saved metrics summary to {summary_path}")
            
        except Exception as e:
            logger.error(f"Error creating summary for {key}: {str(e)}")
    
    logger.info("Completed combining run metrics with validation")