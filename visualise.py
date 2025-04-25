#!/usr/bin/env python3
"""
Visualisation Tool for Task Scheduling Data

This script processes CSV and JSON data files collected during scheduling simulations
and generates comprehensive visualisations for analysis. It includes both basic charts
and advanced visualisations like heatmaps and radar charts.

Usage:
    python visualise.py --data-dir results/data/TIMESTAMP_platform_type
    python visualise.py --data-dir results/data/TIMESTAMP_platform_type --output-dir results/visualisations
    python visualise.py --data-dir results/data/TIMESTAMP_platform_type --scheduler FCFS
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path as mpath   
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import re
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("visualise")

# Constants for visualisation
PRIORITY_COLORS = {
    'HIGH': '#FF5252',    # Red
    'MEDIUM': '#FFD740',  # Amber
    'LOW': '#69F0AE'      # Green
}

ALGORITHM_COLORS = {
    'FCFS': '#2196F3',         # Blue
    'EDF': '#7B1FA2',          # Purple
    'Priority': '#FF5722',     # Deep Orange
    'ML-Based': '#009688'      # Teal
}

def get_priority_color(priority):
    """Get color based on task priority"""
    return PRIORITY_COLORS.get(priority, '#2196F3')  # Default to blue

def get_algorithm_color(algorithm):
    """Get color based on algorithm name"""
    return ALGORITHM_COLORS.get(algorithm, '#607D8B')  # Default to gray

def ensure_output_dir(output_path):
    """Ensure the output directory exists"""
    os.makedirs(output_path, exist_ok=True)
    return output_path

def extract_platform_from_dir(data_dir):
    """
    Extract platform type from the directory name
    
    Args:
        data_dir: Path to the data directory
    
    Returns:
        Extracted platform type or None if not found
    """
    # Extract the basename of the directory
    basename = os.path.basename(data_dir)
    
    # Try to extract platform using regex
    # Pattern: timestamp_platform_type
    match = re.search(r'\d{8}_\d{6}_([\w_]+)', basename)
    if match:
        return match.group(1)  # Return the platform type
    
    return None

# Function to find task and metrics files with the updated naming convention
def find_data_files(data_dir, scheduler_name, processor_type='single'):
    """
    Find data files with the updated naming convention
    
    Args:
        data_dir: Path to the data directory
        scheduler_name: Name of the scheduler
        processor_type: 'single' or 'multi'
    
    Returns:
        Tuple of (tasks_path, metrics_path)
    """
    processor_dir = os.path.join(data_dir, f"{processor_type}_processor")
    
    # Check if directory exists
    if not os.path.exists(processor_dir):
        return None, None
    
    # For single processor, files are named simply by scheduler
    if processor_type == 'single':
        tasks_path = os.path.join(processor_dir, f"{scheduler_name}_tasks.csv")
        metrics_path = os.path.join(processor_dir, f"{scheduler_name}_metrics.json")
        
        if os.path.exists(tasks_path) and os.path.exists(metrics_path):
            return tasks_path, metrics_path
    
    # For multi-processor, try to find system metrics
    elif processor_type == 'multi':
        system_metrics_path = os.path.join(processor_dir, f"{scheduler_name}_system_metrics.json")
        
        # For tasks, we might have per-processor files or a combined file
        all_cpus_path = os.path.join(processor_dir, f"{scheduler_name}_All-CPUs_tasks.csv")
        
        if os.path.exists(all_cpus_path) and os.path.exists(system_metrics_path):
            return all_cpus_path, system_metrics_path
        
        # If we don't have a combined file, try to find individual processor files
        task_files = [f for f in os.listdir(processor_dir) 
                     if f.startswith(f"{scheduler_name}_CPU-") and f.endswith("_tasks.csv")]
        
        if task_files and os.path.exists(system_metrics_path):
            # Return the first processor's task file for now
            # The caller will need to handle combining multiple files if needed
            return os.path.join(processor_dir, task_files[0]), system_metrics_path
    
    # Try fallback to old naming convention (with platform name in the filename)
    platform = extract_platform_from_dir(data_dir)
    if platform:
        old_tasks_path = os.path.join(processor_dir, f"{scheduler_name}_{platform.capitalize()}_tasks.csv")
        old_metrics_path = os.path.join(processor_dir, f"{scheduler_name}_{platform.capitalize()}_metrics.json")
        
        if os.path.exists(old_tasks_path) and os.path.exists(old_metrics_path):
            return old_tasks_path, old_metrics_path
    
    # If we still don't have files, try even more general pattern matching
    tasks_pattern = re.compile(f"{scheduler_name}.*_tasks\.csv")
    metrics_pattern = re.compile(f"{scheduler_name}.*_metrics\.json")
    
    tasks_path = None
    metrics_path = None
    
    for file in os.listdir(processor_dir):
        if tasks_pattern.match(file):
            tasks_path = os.path.join(processor_dir, file)
        if metrics_pattern.match(file):
            metrics_path = os.path.join(processor_dir, file)
    
    if tasks_path and metrics_path:
        return tasks_path, metrics_path
    
    # If no matching files found
    return None, None

# ========================= BASIC VISUALISATION FUNCTIONS ========================= #

def plot_task_completion(tasks_df, scheduler_name, output_path=None):
    """
    Plot task completion times
    
    Args:
        tasks_df: DataFrame containing task data
        scheduler_name: Name of the scheduler
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    if tasks_df.empty:
        plt.title(f"No completed tasks for {scheduler_name}")
        if output_path:
            plt.savefig(output_path)
        return
    
    # Sort by start time
    df = tasks_df.sort_values('start_time')
    
    # Create Gantt chart
    for i, task in df.iterrows():
        plt.barh(y=task['id'], 
                width=task['completion_time'] - task['start_time'],
                left=task['start_time'],
                color=get_priority_color(task['priority']),
                edgecolor='black',
                alpha=0.8)
        
        # Add arrival time markers
        plt.scatter(task['arrival_time'], task['id'], marker='|', color='red', s=100)
    
    # Add legend for priorities
    priority_colors = {'HIGH': get_priority_color('HIGH'),
                       'MEDIUM': get_priority_color('MEDIUM'),
                       'LOW': get_priority_color('LOW')}
    
    for priority, color in priority_colors.items():
        plt.barh(0, 0, color=color, label=priority)
    
    plt.scatter([], [], marker='|', color='red', label='Arrival Time')
    
    plt.legend(loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('Task ID')
    plt.title(f'Task Execution Timeline - {scheduler_name}')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    if output_path:
        ensure_output_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_waiting_times(tasks_df, scheduler_name, output_path=None):
    """
    Plot waiting times by priority
    
    Args:
        tasks_df: DataFrame containing task data
        scheduler_name: Name of the scheduler
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    if tasks_df.empty:
        plt.title(f"No completed tasks for {scheduler_name}")
        if output_path:
            plt.savefig(output_path)
        return

    # Drop rows with null waiting time
    df = tasks_df.dropna(subset=['waiting_time'])
    
    if df.empty:
        plt.title(f"No waiting time data for {scheduler_name}")
        if output_path:
            plt.savefig(output_path)
        return
    
    # Create boxplot grouped by priority
    sns.boxplot(x='priority', y='waiting_time', data=df, 
                palette={'HIGH': get_priority_color('HIGH'),
                         'MEDIUM': get_priority_color('MEDIUM'),
                         'LOW': get_priority_color('LOW')})
    
    # Add individual points
    sns.stripplot(x='priority', y='waiting_time', data=df, 
                 color='black', alpha=0.5, jitter=True, size=4)
    
    plt.title(f'Waiting Times by Priority - {scheduler_name}')
    plt.xlabel('Priority')
    plt.ylabel('Waiting Time (s)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    if output_path:
        ensure_output_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_memory_usage(timeseries_df, scheduler_name, output_path=None):
    """
    Plot memory usage over time
    
    Args:
        timeseries_df: DataFrame containing timeseries data
        scheduler_name: Name of the scheduler
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    if timeseries_df.empty or 'memory_usage' not in timeseries_df.columns:
        plt.title(f"No memory data for {scheduler_name}")
        if output_path:
            plt.savefig(output_path)
        return
    
    plt.plot(timeseries_df['time'], timeseries_df['memory_usage'], 
             marker='o', linestyle='-', markersize=3)
    
    plt.title(f'Memory Usage Over Time - {scheduler_name}')
    plt.xlabel('Time (s)')
    plt.ylabel('Memory Usage (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if output_path:
        ensure_output_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_queue_length(timeseries_df, scheduler_name, output_path=None):
    """
    Plot queue length over time
    
    Args:
        timeseries_df: DataFrame containing timeseries data
        scheduler_name: Name of the scheduler
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    if timeseries_df.empty or 'queue_length' not in timeseries_df.columns:
        plt.title(f"No queue data for {scheduler_name}")
        if output_path:
            plt.savefig(output_path)
        return
    
    plt.plot(timeseries_df['time'], timeseries_df['queue_length'], 
             marker='o', linestyle='-', markersize=3)
    
    plt.title(f'Queue Length Over Time - {scheduler_name}')
    plt.xlabel('Time (s)')
    plt.ylabel('Queue Length (tasks)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if output_path:
        ensure_output_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_algorithm_comparison(metrics_by_algorithm, metric_name, title, ylabel, output_path=None):
    """
    Compare a specific metric across different algorithms
    
    Args:
        metrics_by_algorithm: Dictionary mapping algorithm names to their metrics
        metric_name: Name of the metric to compare
        title: Plot title
        ylabel: Y-axis label
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    algorithms = []
    values = []
    
    # This function assumes all schedulers now use standardised metric key names:
    # - 'avg_waiting_time' for average waiting time
    # - 'avg_waiting_by_priority' for waiting times by priority
    # - 'tasks_by_priority' for task counts by priority
    
    for algo, metrics in metrics_by_algorithm.items():
        # Skip if metrics is None
        if metrics is None:
            continue
        
        metric_value = None
        
        # Handle both direct values and nested dictionaries
        if isinstance(metrics, dict):
            # First try the exact metric name
            if metric_name in metrics:
                metric_value = metrics[metric_name]
            
            # Special handling for waiting time metrics which might have inconsistent naming
            elif metric_name == 'avg_waiting_time':
                # Try alternative keys for average waiting time
                for key in ['average_waiting_time', 'avg_waiting_time', 'mean_waiting_time']:
                    if key in metrics:
                        metric_value = metrics[key]
                        break
                
                # If still no value found, try to calculate from waiting_times_by_priority
                if metric_value is None and 'waiting_times_by_priority' in metrics:
                    priority_data = metrics['waiting_times_by_priority']
                    if priority_data and isinstance(priority_data, dict):
                        # Calculate the overall average
                        metric_value = sum(priority_data.values()) / len(priority_data)
                elif metric_value is None and 'avg_waiting_by_priority' in metrics:
                    priority_data = metrics['avg_waiting_by_priority']
                    if priority_data and isinstance(priority_data, dict):
                        # Calculate the overall average
                        metric_value = sum(priority_data.values()) / len(priority_data)
        elif not isinstance(metrics, dict):
            # Direct value
            metric_value = metrics
        
        # Add to our plotting data if we found a value
        if metric_value is not None:
            algorithms.append(algo)
            values.append(metric_value)
    
    if not algorithms:
        plt.title(f"No data available for {metric_name}")
        if output_path:
            ensure_output_dir(os.path.dirname(output_path))
            plt.savefig(output_path)
        return
    
    colors = [get_algorithm_color(algo) for algo in algorithms]
    
    plt.bar(algorithms, values, color=colors, alpha=0.8, edgecolor='black')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(values):
        plt.text(i, v + max(values) * 0.02, f'{v:.2f}', 
                ha='center', va='bottom', fontweight='bold')
    
    if output_path:
        ensure_output_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

# ========================== HEATMAP VISUALISATIONS ========================== #

def create_cpu_utilisation_heatmap(metrics, scheduler_name, output_path=None):
    """
    Create a heatmap showing CPU utilisation across time for multiple processors
    
    Args:
        metrics: Metrics dictionary containing CPU utilisation data
        scheduler_name: Name of the scheduler
        output_path: Base path for saving the heatmap
    """
    plt.figure(figsize=(12, 8))
    
    # Check for required data in the multi-processor metrics
    per_processor_metrics = metrics.get('per_processor_metrics', [])
    
    if not per_processor_metrics:
        plt.title(f"No CPU utilisation data available for {scheduler_name}")
        if output_path:
            ensure_output_dir(os.path.dirname(output_path))
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        return
    
    # Create data structure for the heatmap
    processor_count = len(per_processor_metrics)
    
    # Find CPU usage history for each processor
    cpu_usage_data = []
    timestamps = []
    
    # First try to find timestamps
    for proc_metrics in per_processor_metrics:
        if 'timestamp' in proc_metrics and isinstance(proc_metrics['timestamp'], list):
            timestamps = proc_metrics['timestamp']
            break
    
    # If no timestamps found, check timestamp_history
    if not timestamps and 'timestamp_history' in metrics:
        timestamps = metrics['timestamp_history']
    
    # Get CPU usage data for each processor
    for i, proc_metrics in enumerate(per_processor_metrics):
        cpu_usage = []
        
        # Try different possible keys for CPU usage history
        if 'cpu_usage_history' in proc_metrics:
            cpu_usage = proc_metrics['cpu_usage_history']
        elif 'cpu_usage' in proc_metrics and isinstance(proc_metrics['cpu_usage'], list):
            cpu_usage = proc_metrics['cpu_usage']
        
        # Ensure data is the same length as timestamps
        if cpu_usage:
            # Trim or extend to match timestamp length
            if len(cpu_usage) > len(timestamps):
                cpu_usage = cpu_usage[:len(timestamps)]
            elif len(cpu_usage) < len(timestamps):
                # Extend with the last value or zero
                last_value = cpu_usage[-1] if cpu_usage else 0
                cpu_usage.extend([last_value] * (len(timestamps) - len(cpu_usage)))
        else:
            # If no data, use zeros
            cpu_usage = [0] * len(timestamps)
        
        cpu_usage_data.append(cpu_usage)
    
    # Convert timestamps to relative time
    if timestamps and timestamps[0] is not None:
        # Convert timestamps to relative time if they're absolute timestamps
        if isinstance(timestamps[0], (int, float)) and timestamps[0] > 1000000000:  # Likely a Unix timestamp
            start_time = timestamps[0]
            relative_times = [t - start_time for t in timestamps]
        else:
            relative_times = timestamps
    else:
        relative_times = list(range(len(cpu_usage_data[0])))
    
    # Create time bins for better visualisation
    max_time = max(relative_times)
    time_bins = np.linspace(0, max_time, 11)
    time_bin_labels = [f"{time_bins[i]:.1f}-{time_bins[i+1]:.1f}s" for i in range(len(time_bins)-1)]
    
    # Assign each timestamp to a bin
    time_bin_indices = np.searchsorted(time_bins, relative_times) - 1
    time_bin_indices = np.clip(time_bin_indices, 0, len(time_bins)-2)
    
    # Calculate average CPU usage for each processor in each time bin
    heatmap_data = np.zeros((processor_count, len(time_bins)-1))
    
    for proc_idx in range(processor_count):
        cpu_usage = cpu_usage_data[proc_idx]
        if not cpu_usage:
            continue
            
        for time_idx, bin_idx in enumerate(time_bin_indices):
            if time_idx < len(cpu_usage):
                # Add CPU usage to the appropriate bin
                heatmap_data[proc_idx, bin_idx] += cpu_usage[time_idx]
        
        # Calculate averages
        for bin_idx in range(len(time_bins)-1):
            # Count how many timestamps fall into this bin
            count = np.sum(time_bin_indices == bin_idx)
            if count > 0:
                heatmap_data[proc_idx, bin_idx] /= count
    
    # Create the heatmap
    ax = sns.heatmap(
        heatmap_data,
        cmap="YlOrRd",
        vmin=0,
        vmax=100,
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        cbar_kws={'label': 'CPU Usage (%)'}
    )
    
    # Set labels
    plt.xlabel('Time (seconds)')
    plt.ylabel('Processor')
    plt.title(f'CPU Utilisation Heatmap - {scheduler_name}')
    
    # Set y-tick labels (processor names)
    processor_labels = [f"CPU-{i+1}" for i in range(processor_count)]
    ax.set_yticklabels(processor_labels, rotation=0)
    
    # Set x-tick labels (time bins)
    ax.set_xticklabels(time_bin_labels, rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save the plot if output path is provided
    if output_path:
        ensure_output_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_task_density_heatmap(metrics, scheduler_name, output_path=None):
    """
    Create a heatmap showing task density patterns over time by priority level
    
    Args:
        metrics: Metrics dictionary containing task execution data
        scheduler_name: Name of the scheduler
        output_path: Base path for saving the heatmap
    """
    plt.figure(figsize=(12, 6))
    
    # First check if we have tasks data in a dataframe format (from CSV read)
    # This is more likely in the visualization script context
    tasks_df = None
    if isinstance(metrics, pd.DataFrame):
        tasks_df = metrics
        
    # If metrics is still a dictionary (original behavior), attempt to extract task data
    completed_tasks = []
    
    if tasks_df is None:
        # Try to find completed tasks directly
        if 'completed_tasks' in metrics and isinstance(metrics['completed_tasks'], list):
            completed_tasks = metrics['completed_tasks']
        # For multi-processor, gather tasks from all processors
        elif 'per_processor_metrics' in metrics:
            for proc_metrics in metrics['per_processor_metrics']:
                if 'completed_tasks' in proc_metrics and isinstance(proc_metrics['completed_tasks'], list):
                    completed_tasks.extend(proc_metrics['completed_tasks'])
    
    # If we have task objects but no dataframe, create a dataframe
    task_data = []
    
    if completed_tasks and not tasks_df:
        for task in completed_tasks:
            if hasattr(task, 'start_time') and hasattr(task, 'completion_time') and \
               task.start_time is not None and task.completion_time is not None:
                task_data.append({
                    'priority': task.priority.name if hasattr(task.priority, 'name') else str(task.priority),
                    'start_time': task.start_time,
                    'completion_time': task.completion_time
                })
        
        if task_data:
            tasks_df = pd.DataFrame(task_data)
    
    # If we still don't have usable data, exit gracefully
    if (not tasks_df or tasks_df.empty) and not task_data:
        plt.title(f"No task execution data available for {scheduler_name}")
        if output_path:
            ensure_output_dir(os.path.dirname(output_path))
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        return
    
    # Create dataframe if we have task_data but no dataframe yet
    if task_data and not tasks_df:
        tasks_df = pd.DataFrame(task_data)
    
    # Ensure we have the necessary columns
    required_columns = ['priority', 'start_time', 'completion_time']
    missing_columns = [col for col in required_columns if col not in tasks_df.columns]
    
    if missing_columns:
        plt.title(f"Missing task data columns for {scheduler_name}: {', '.join(missing_columns)}")
        if output_path:
            ensure_output_dir(os.path.dirname(output_path))
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        return
    
    # If priority is not properly set, try to extract from task ID
    if 'priority' in tasks_df.columns:
        # Check if we have legitimate priority values
        valid_priorities = ['HIGH', 'MEDIUM', 'LOW']
        if not any(p in valid_priorities for p in tasks_df['priority'].unique()):
            # Try to infer priority from task ID (e.g., "H1" -> "HIGH")
            def infer_priority(task_id):
                if isinstance(task_id, str) and task_id:
                    first_char = task_id[0].upper()
                    if first_char == 'H':
                        return 'HIGH'
                    elif first_char == 'M':
                        return 'MEDIUM'
                    elif first_char == 'L':
                        return 'LOW'
                return 'MEDIUM'  # Default
            
            tasks_df['priority'] = tasks_df['id'].apply(infer_priority)
    
    # Create time bins
    min_time = tasks_df['start_time'].min()
    max_time = tasks_df['completion_time'].max()
    
    # Ensure min_time and max_time are valid
    if pd.isna(min_time) or pd.isna(max_time) or min_time == max_time:
        min_time = 0
        max_time = 100 if pd.isna(max_time) else max_time + 10
    
    time_bins = np.linspace(min_time, max_time, 11)
    time_bin_labels = [f"{time_bins[i]:.1f}-{time_bins[i+1]:.1f}s" for i in range(len(time_bins)-1)]
    
    # Create priority bins - using the standard priority levels
    priority_levels = ['HIGH', 'MEDIUM', 'LOW']
    
    # Create the heatmap data structure
    heatmap_data = np.zeros((len(priority_levels), len(time_bins)-1))
    
    # Fill the heatmap - for each task, add its contribution to each time bin it spans
    for _, task in tasks_df.iterrows():
        # Find which time bins this task spans
        start_bin = np.searchsorted(time_bins, task['start_time']) - 1
        end_bin = np.searchsorted(time_bins, task['completion_time']) - 1
        
        # Clip to valid bin indices
        start_bin = max(0, min(start_bin, len(time_bins)-2))
        end_bin = max(0, min(end_bin, len(time_bins)-2))
        
        # Add task to all bins it spans
        priority_idx = priority_levels.index(task['priority']) if task['priority'] in priority_levels else 0
        
        for bin_idx in range(start_bin, end_bin + 1):
            heatmap_data[priority_idx, bin_idx] += 1
    
    # Create the heatmap
    cmap = sns.color_palette("viridis", as_cmap=True)
    ax = sns.heatmap(
        heatmap_data,
        cmap=cmap,
        annot=True,
        fmt="g",
        linewidths=0.5,
        cbar_kws={'label': 'Number of Active Tasks'}
    )
    
    # Set labels
    plt.xlabel('Time (seconds)')
    plt.ylabel('Priority Level')
    plt.title(f'Task Density Heatmap - {scheduler_name}')
    
    # Set y-tick labels
    ax.set_yticklabels(priority_levels, rotation=0)
    
    # Set x-tick labels
    ax.set_xticklabels(time_bin_labels, rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save the plot if output path is provided
    if output_path:
        ensure_output_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_resource_bottleneck_heatmap(metrics, scheduler_name, output_path=None):
    """
    Create a heatmap showing resource bottlenecks (CPU, memory, queue length)
    
    Args:
        metrics: Metrics dictionary containing resource usage data
        scheduler_name: Name of the scheduler
        output_path: Base path for saving the heatmap
    """
    plt.figure(figsize=(14, 8))
    
    # Extract data from metrics with better error handling
    timestamps = metrics.get('timestamp_history', [])
    
    # Try alternative timestamp key if not found
    if not timestamps and 'timestamp' in metrics and isinstance(metrics['timestamp'], list):
        timestamps = metrics['timestamp']
    
    # If still no timestamps, create a default range
    if not timestamps:
        timestamps = list(range(10))  # Default range if no timestamps found
        
    # Log issue for debugging
    if not timestamps or len(timestamps) == 0:
        plt.title(f"No timestamp data available for {scheduler_name}")
        if output_path:
            ensure_output_dir(os.path.dirname(output_path))
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        return
    
    # Gather resource metrics with better error handling
    memory_usage = metrics.get('memory_usage_history', [])
    queue_length = metrics.get('queue_length_history', [])
    
    # Try alternative keys
    if not memory_usage and 'memory_usage' in metrics and isinstance(metrics['memory_usage'], list):
        memory_usage = metrics['memory_usage']
    
    if not queue_length and 'queue_length' in metrics and isinstance(metrics['queue_length'], list):
        queue_length = metrics['queue_length']
    
    # Get CPU usage from processor metrics or directly
    cpu_usage = metrics.get('cpu_usage_history', [])
    if not cpu_usage and 'cpu_usage' in metrics and isinstance(metrics['cpu_usage'], list):
        cpu_usage = metrics['cpu_usage']
    
    # Try to extract from per-processor metrics if available and cpu_usage is still empty
    if not cpu_usage:
        per_processor_metrics = metrics.get('per_processor_metrics', [])
        if per_processor_metrics:
            # Average CPU usage across all processors
            for t_idx in range(len(timestamps)):
                total_cpu = 0
                count = 0
                
                for proc_metrics in per_processor_metrics:
                    if ('cpu_usage_history' in proc_metrics and 
                        isinstance(proc_metrics['cpu_usage_history'], list) and 
                        t_idx < len(proc_metrics['cpu_usage_history'])):
                        total_cpu += proc_metrics['cpu_usage_history'][t_idx]
                        count += 1
                    elif ('cpu_usage' in proc_metrics and 
                          isinstance(proc_metrics['cpu_usage'], list) and 
                          t_idx < len(proc_metrics['cpu_usage'])):
                        total_cpu += proc_metrics['cpu_usage'][t_idx]
                        count += 1
                
                if count > 0:
                    cpu_usage.append(total_cpu / count)
                else:
                    # If no data, use the last known value or zero
                    cpu_usage.append(cpu_usage[-1] if cpu_usage else 0)
    
    # Check if we have adequate data
    if not memory_usage and not queue_length and not cpu_usage:
        plt.title(f"No resource data available for {scheduler_name}")
        if output_path:
            ensure_output_dir(os.path.dirname(output_path))
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        return
    
    # Use dummy data if any metric is missing
    if not memory_usage:
        memory_usage = [0] * len(timestamps)
        print(f"Warning: No memory usage data for {scheduler_name}, using zeros")
    
    if not queue_length:
        queue_length = [0] * len(timestamps)
        print(f"Warning: No queue length data for {scheduler_name}, using zeros")
        
    if not cpu_usage:
        cpu_usage = [0] * len(timestamps)
        print(f"Warning: No CPU usage data for {scheduler_name}, using zeros")
    
    # Ensure all lists are the same length as timestamps
    min_length = len(timestamps)
    
    # Trim or extend memory usage
    if len(memory_usage) > min_length:
        memory_usage = memory_usage[:min_length]
    elif len(memory_usage) < min_length:
        # Extend with the last value or zero
        last_value = memory_usage[-1] if memory_usage else 0
        memory_usage.extend([last_value] * (min_length - len(memory_usage)))
    
    # Trim or extend queue length
    if len(queue_length) > min_length:
        queue_length = queue_length[:min_length]
    elif len(queue_length) < min_length:
        # Extend with the last value or zero
        last_value = queue_length[-1] if queue_length else 0
        queue_length.extend([last_value] * (min_length - len(queue_length)))
    
    # Trim or extend CPU usage
    if len(cpu_usage) > min_length:
        cpu_usage = cpu_usage[:min_length]
    elif len(cpu_usage) < min_length:
        # Extend with the last value or zero
        last_value = cpu_usage[-1] if cpu_usage else 0
        cpu_usage.extend([last_value] * (min_length - len(cpu_usage)))
    
    # Convert timestamps to relative time
    if isinstance(timestamps[0], (int, float)) and timestamps[0] > 1000000000:
        start_time = timestamps[0]
        relative_times = [t - start_time for t in timestamps]
    else:
        relative_times = list(range(len(timestamps)))
    
    # Create time bins for better visualization
    max_time = max(relative_times) if relative_times else 10
    time_bins = np.linspace(0, max_time, 11)
    time_bin_labels = [f"{time_bins[i]:.1f}-{time_bins[i+1]:.1f}s" for i in range(len(time_bins)-1)]
    
    # Normalize data for consistent visualization with safeguards
    # Get max values with a minimum floor to avoid division by zero
    max_cpu = max(cpu_usage) if cpu_usage and max(cpu_usage) > 0 else 1.0
    max_memory = max(memory_usage) if memory_usage and max(memory_usage) > 0 else 1.0
    max_queue = max(queue_length) if queue_length and max(queue_length) > 0 else 1.0
    
    # Normalize values (percentage of maximum for each resource)
    normalized_cpu = [min(value / max_cpu * 100, 100) for value in cpu_usage]
    normalized_memory = [min(value / max_memory * 100, 100) for value in memory_usage]
    normalized_queue = [min(value / max_queue * 100, 100) for value in queue_length]
    
    # Assign each timestamp to a bin
    time_bin_indices = np.searchsorted(time_bins, relative_times) - 1
    time_bin_indices = np.clip(time_bin_indices, 0, len(time_bins)-2)
    
    # Create data for the heatmap
    resources = ['CPU', 'Memory', 'Queue']
    heatmap_data = np.zeros((len(resources), len(time_bins)-1))
    
    # Calculate average value for each resource in each time bin
    for time_idx, bin_idx in enumerate(time_bin_indices):
        # Add values to the appropriate bin
        if time_idx < len(normalized_cpu):
            heatmap_data[0, bin_idx] += normalized_cpu[time_idx]
        
        if time_idx < len(normalized_memory):
            heatmap_data[1, bin_idx] += normalized_memory[time_idx]
        
        if time_idx < len(normalized_queue):
            heatmap_data[2, bin_idx] += normalized_queue[time_idx]
    
    # Calculate bin averages
    for bin_idx in range(len(time_bins)-1):
        # Count timestamps in this bin
        count = np.sum(time_bin_indices == bin_idx)
        if count > 0:
            heatmap_data[:, bin_idx] /= count
    
    # Create a custom colormap that goes from green to yellow to red
    cmap = LinearSegmentedColormap.from_list('GYR', [(0, 'green'), (0.5, 'yellow'), (1, 'red')])
    
    # Create the heatmap
    ax = sns.heatmap(
        heatmap_data,
        cmap=cmap,
        vmin=0,
        vmax=100,
        annot=True,
        fmt=".1f",
        linewidths=0.5,
        cbar_kws={'label': 'Resource Usage (%)'}
    )
    
    # Set labels
    plt.xlabel('Time (seconds)')
    plt.ylabel('Resource')
    plt.title(f'Resource Bottleneck Heatmap - {scheduler_name}')
    
    # Set y-tick labels
    ax.set_yticklabels(resources, rotation=0)
    
    # Set x-tick labels
    ax.set_xticklabels(time_bin_labels, rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save the plot if output path is provided
    if output_path:
        ensure_output_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_enhanced_gantt_chart(tasks_df, scheduler_name, output_path=None):
    """
    Create an enhanced Gantt chart with priority inversion highlighting
    
    Args:
        tasks_df: DataFrame containing task data
        scheduler_name: Name of the scheduler
        output_path: Path to save the plot
    """
    plt.figure(figsize=(14, 8))
    
    if tasks_df.empty:
        plt.title(f"No task data available for {scheduler_name}")
        if output_path:
            ensure_output_dir(os.path.dirname(output_path))
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        return
    
    # Sort tasks by start time and then by priority
    df = tasks_df.sort_values(['start_time', 'priority'], ascending=[True, True])
    
    # Create a mapping of task ID to y-position
    task_positions = {}
    current_position = 0
    
    # Sort tasks by priority for y-position assignment
    for task_id in df['id'].unique():
        task_positions[task_id] = current_position
        current_position += 1
    
    # Create Gantt chart with enhanced visualisation
    for i, task in df.iterrows():
        # Get task color based on priority
        color = PRIORITY_COLORS.get(task['priority'], '#2196F3')
        
        # Plot the task execution bar
        y_pos = task_positions[task['id']]
        plt.barh(y=y_pos, 
                width=task['completion_time'] - task['start_time'],
                left=task['start_time'],
                color=color,
                edgecolor='black',
                alpha=0.7,
                height=0.5)
        
        # Add task ID label
        plt.text(task['start_time'] + (task['completion_time'] - task['start_time'])/2,
                y_pos,
                task['id'],
                ha='center',
                va='center',
                color='black',
                fontweight='bold')
        
        # Mark arrival time
        plt.scatter(task['arrival_time'], y_pos, marker='|', color='red', s=150)
        
        # Mark waiting time with a different color
        waiting_time = task['start_time'] - task['arrival_time']
        if waiting_time > 0:
            plt.barh(y=y_pos, 
                    width=waiting_time,
                    left=task['arrival_time'],
                    color='lightgrey',
                    edgecolor='black',
                    alpha=0.5,
                    height=0.3)
    
    # Add deadline markers if available
    if 'deadline' in df.columns:
        for i, task in df.iterrows():
            if pd.notna(task['deadline']):
                y_pos = task_positions[task['id']]
                plt.axvline(x=task['deadline'], ymin=(y_pos-0.4)/current_position, 
                           ymax=(y_pos+0.4)/current_position, color='black', linestyle='--')
                
                # Add red highlight for missed deadlines
                if task['completion_time'] > task['deadline']:
                    plt.axvspan(task['deadline'], task['completion_time'], 
                               ymin=(y_pos-0.4)/current_position, 
                               ymax=(y_pos+0.4)/current_position,
                               color='red', alpha=0.3)
    
    # Add grid lines
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Set y-ticks to task IDs
    plt.yticks(list(task_positions.values()), list(task_positions.keys()))
    
    # Set labels and title
    plt.xlabel('Time (s)')
    plt.ylabel('Task ID')
    plt.title(f'Enhanced Task Execution Timeline - {scheduler_name}')
    
    # Add legend
    legend_elements = []
    for priority, color in PRIORITY_COLORS.items():
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, color=color, label=f'{priority} Priority'))
    
    legend_elements.append(plt.Rectangle((0, 0), 1, 1, color='lightgrey', alpha=0.5, label='Waiting Time'))
    legend_elements.append(plt.Line2D([0], [0], marker='|', color='red', linestyle='', markersize=10, label='Arrival Time'))
    
    if 'deadline' in df.columns:
        legend_elements.append(plt.Line2D([0], [0], color='black', linestyle='--', label='Deadline'))
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, color='red', alpha=0.3, label='Deadline Miss'))
    
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    # Save the plot if output path is provided
    if output_path:
        ensure_output_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# ================= RADAR CHART FOR ALGORITHM COMPARISON ===================== #
def _radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.
    
    Based on matplotlib examples with fixes for compatibility
    """
    # Calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    
    class RadarAxes(PolarAxes):
        name = 'radar'
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')
        
        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)
        
        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
            return lines
        
        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)
        
        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)
        
        def _gen_axes_patch(self):
            # The Theta=0, r=1 reference point
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)
        
        def _gen_axes_spines(self):
            if frame == 'circle':
                return PolarAxes._gen_axes_spines(self)
            elif frame == 'polygon':
                # Use custom polygon spine
                spine_dicts = {}
                for i in range(num_vars):
                    spine = Spine(self, 'polar', 
                                mpath.Path(np.array([[theta[i], 0.0],
                                                  [theta[i], 1.0]])))
                    spine.set_transform(self.transAxes)
                    spine_dicts[f'polar{i}'] = spine
                return spine_dicts
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)
    
    # Register the new projection
    register_projection(RadarAxes)
    
    return theta
def create_radar_chart_comparison(metrics_by_algorithm, output_path=None):
    """
    Create an improved radar chart comparing different algorithms across multiple metrics
    
    Args:
        metrics_by_algorithm: Dictionary mapping algorithm names to their metrics
        output_path: Path to save the plot
    """
    # Define the metrics to compare - avoid metrics that might be missing in multi-processor
    # Choose metrics that are present in both single and multi-processor metrics
    metrics_to_compare = [
        ('avg_waiting_time', 'Avg Waiting Time', True),  # True = lower is better
        ('system_throughput', 'Throughput', False),  # False = higher is better
        ('avg_cpu_usage', 'CPU Usage', True),  # True = lower is better (efficiency)
        ('avg_memory_usage', 'Memory Usage', True)  # True = lower is better (efficiency)
    ]
    
    # Create a dictionary to map various possible key names to our standardized keys
    key_mappings = {
        'avg_waiting_time': ['avg_waiting_time', 'average_waiting_time', 'mean_waiting_time'],
        'system_throughput': ['system_throughput', 'avg_throughput', 'throughput', 'tasks_per_second'],
        'avg_cpu_usage': ['avg_cpu_usage', 'cpu_usage', 'average_cpu_usage'],
        'avg_memory_usage': ['avg_memory_usage', 'memory_usage', 'average_memory_usage']
    }
    
    # Filter algorithms with valid metrics
    valid_algorithms = {}
    for algo, metrics in metrics_by_algorithm.items():
        if metrics is not None:
            valid_algorithms[algo] = metrics
    
    if not valid_algorithms:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No valid algorithm metrics available for radar chart", 
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        
        if output_path:
            ensure_output_dir(os.path.dirname(output_path))
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        return
    
    # For each metric, check which algorithms have that data
    available_metrics = []
    for metric_key, metric_name, lower_is_better in metrics_to_compare:
        available = False
        for algo_metrics in valid_algorithms.values():
            # Check if the metric exists under any of its possible keys
            for possible_key in key_mappings.get(metric_key, [metric_key]):
                if possible_key in algo_metrics:
                    available = True
                    break
        
        if available:
            available_metrics.append((metric_key, metric_name, lower_is_better))
    
    # Add extra debugging
    print(f"Available metrics for radar chart: {[m[1] for m in available_metrics]}")
    
    if not available_metrics:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "No common metrics available for radar chart comparison", 
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        
        if output_path:
            ensure_output_dir(os.path.dirname(output_path))
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        return
    
    # Gather the data for each algorithm, extracting metric values
    data = []
    all_values_by_metric = {metric_key: [] for metric_key, _, _ in available_metrics}
    
    for algo, metrics in valid_algorithms.items():
        algo_data = []
        
        # For each available metric
        for metric_idx, (metric_key, _, _) in enumerate(available_metrics):
            # Get the value using any of its possible keys
            value = None
            used_key = None
            for possible_key in key_mappings.get(metric_key, [metric_key]):
                if possible_key in metrics:
                    value = metrics[possible_key]
                    used_key = possible_key
                    break
            
            # If value is still None, use a default value (0)
            if value is None:
                print(f"Warning: No value found for {algo}, metric {metric_key}")
                value = 0
            
            # Log the actual values for debugging
            print(f"{algo} - {metric_key} ({used_key}): {value}")
            
            # Special handling for ML scheduler with extreme or missing values
            if algo == 'ML-Based':
                # Handle multi-processor missing metrics
                if metric_key == 'system_throughput' and value == 0:
                    # Try to find avg_throughput instead
                    if 'avg_throughput' in metrics:
                        value = metrics['avg_throughput']
                        print(f"  Using avg_throughput instead: {value}")
                    elif 'throughput' in metrics:
                        value = metrics['throughput']
                        print(f"  Using throughput instead: {value}")
            
            # Collect all values for later normalization
            all_values_by_metric[metric_key].append(value)
            algo_data.append(value)
        
        data.append((algo, algo_data))
    
    # Extract information for the radar chart
    metric_names = [name for _, name, _ in available_metrics]
    lower_is_better = [flag for _, _, flag in available_metrics]
    
    # Normalize data to a 0-1 scale for each metric with better handling
    normalized_data = []
    
    # First, filter extreme values by metric
    sanitized_values_by_metric = {}
    for metric_key, values in all_values_by_metric.items():
        # Filter out extreme or invalid values
        valid_values = [v for v in values if isinstance(v, (int, float)) and 
                        v < 1000000 and v > -1000000 and not np.isnan(v)]
        
        if not valid_values:
            # If no valid values, use the original with a fallback
            valid_values = values
            if not valid_values:
                valid_values = [0]
        
        sanitized_values_by_metric[metric_key] = valid_values
    
    # Now normalize using the sanitized values
    for algo_name, algo_data in data:
        normalized_algo_data = []
        for i, (metric_key, value) in enumerate(zip([m[0] for m in available_metrics], algo_data)):
            # Get sanitized values for this metric
            sanitized_values = sanitized_values_by_metric[metric_key]
            
            # Handle empty or constant metrics
            if not sanitized_values or max(sanitized_values) == min(sanitized_values):
                normalized_value = 0.5  # Neutral value
                print(f"Using neutral value (0.5) for {algo_name}, metric {metric_key}")
            else:
                min_val = min(sanitized_values)
                max_val = max(sanitized_values)
                
                # Clamp value to avoid extreme outliers
                clamped_value = max(min_val, min(max_val, value))
                
                # Normalize: lower is better -> invert
                if lower_is_better[i]:
                    # Ensure we don't divide by zero
                    if max_val > min_val:
                        # Calculate regular normalization first
                        regular_normalized = (clamped_value - min_val) / (max_val - min_val)
                        # Then invert it for "lower is better" metrics
                        normalized_value = 1 - regular_normalized
                        print(f"  INVERSION: {metric_key} (lower is better) - {clamped_value} normalized to {regular_normalized}, inverted to {normalized_value}")
                    else:
                        normalized_value = 0.5
                else:
                    # Higher is better
                    if max_val > min_val:
                        normalized_value = (clamped_value - min_val) / (max_val - min_val)
                        print(f"  NO INVERSION: {metric_key} (higher is better) - {clamped_value} normalized to {normalized_value}")
                    else:
                        normalized_value = 0.5
            
            # Ensure ML scheduler never gets perfect 0 to avoid line effect
            if algo_name == 'ML-Based' and normalized_value < 0.05:
                normalized_value = 0.05
                print(f"Adjusting very low value for ML-Based on {metric_key} to 0.05")
            
            normalized_algo_data.append(normalized_value)
            print(f"Normalized {algo_name} - {metric_key}: {value} -> {normalized_value}")
        
        normalized_data.append((algo_name, normalized_algo_data))
    
    # Create a more reliable radar chart using Matplotlib's basic polar plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Set the angle of the first axis
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    # Draw the shape of the chart
    angles = np.linspace(0, 2*np.pi, len(metric_names), endpoint=False).tolist()
    # Close the loop
    angles += angles[:1]
    
    # Draw grid lines
    ax.set_thetagrids(np.degrees(angles[:-1]), metric_names)
    
    # Draw circles at 0.25, 0.5, 0.75, 1.0
    for level in [0.25, 0.5, 0.75]:
        circle = plt.Circle((0, 0), level, transform=ax.transData._b, fill=False, 
                           color='grey', linewidth=0.5, alpha=0.5)
        ax.add_artist(circle)
    
    # Fill in the outer circle manually
    ax.plot(angles, [1] * len(angles), color='grey', linestyle='-', linewidth=0.5, alpha=0.5)
    
    # Set radial limits
    ax.set_ylim(0, 1)
    ax.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], color='grey')
    
    # Plot each algorithm
    for algo_name, values in normalized_data:
        # Close the loop for each algorithm's values
        values_closed = values + values[:1]
        color = get_algorithm_color(algo_name)
        ax.plot(angles, values_closed, linewidth=2, linestyle='-', label=algo_name, color=color)
        ax.fill(angles, values_closed, color=color, alpha=0.25)
    
    # Add legend with better positioning
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.title('Algorithm Performance Comparison', size=15, y=1.1)
    
    # Add footnote explaining the normalization
    plt.figtext(0.5, 0.01, 
               'Note: All metrics are normalized to 0-1 scale. For metrics where lower is better\n'
               '(waiting time, completion time, CPU/memory usage), the scale is inverted so that higher values\n'
               'on the radar chart always represent better performance. Each axis represents relative performance.',
               ha='center', fontsize=9, bbox=dict(facecolor='lightyellow', alpha=0.5))
    
    # Save the plot if output path is provided
    if output_path:
        ensure_output_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# Function to generate all visualisations for a single scheduler
def generate_scheduler_visualisations(tasks_df, metrics, scheduler_name, output_dir):
    """
    Generate all visualisations for a specific scheduler
    
    Args:
        tasks_df: DataFrame containing task data
        metrics: Dictionary containing scheduler metrics
        scheduler_name: Name of the scheduler
        output_dir: Directory to save visualisation outputs
    """
    # Ensure output directory exists
    scheduler_dir = os.path.join(output_dir, scheduler_name)
    ensure_output_dir(scheduler_dir)
    
    # 1. Basic visualisations
    # 1.1 Task completion timeline
    plot_task_completion(
        tasks_df, 
        scheduler_name,
        os.path.join(scheduler_dir, 'task_completion.png')
    )
    
    # 1.2 Waiting times
    plot_waiting_times(
        tasks_df,
        scheduler_name,
        os.path.join(scheduler_dir, 'waiting_times.png')
    )
    
    # 1.3 Enhanced Gantt Chart
    create_enhanced_gantt_chart(
        tasks_df, 
        scheduler_name,
        os.path.join(scheduler_dir, 'enhanced_gantt_chart.png')
    )
    
    # 2. Advanced visualisations
    # 2.1 Resource Bottleneck Heatmap
    create_resource_bottleneck_heatmap(
        metrics,
        scheduler_name,
        os.path.join(scheduler_dir, 'resource_bottleneck_heatmap.png')
    )
    
    # 2.2 Task Density Heatmap
    create_task_density_heatmap(
        metrics,
        scheduler_name,
        os.path.join(scheduler_dir, 'task_density_heatmap.png')
    )
    
    # 2.3 CPU Utilisation Heatmap (for multi-processor)
    if 'per_processor_metrics' in metrics:
        create_cpu_utilisation_heatmap(
            metrics,
            scheduler_name,
            os.path.join(scheduler_dir, 'cpu_utilisation_heatmap.png')
        )

# Function to generate comparison visualisations
def generate_comparison_visualisations(metrics_by_algorithm, output_dir):
    """
    Generate visualisations comparing different algorithms
    
    Args:
        metrics_by_algorithm: Dictionary mapping algorithm names to their metrics
        output_dir: Directory to save visualisation outputs
    """
    # Ensure output directory exists
    comparison_dir = os.path.join(output_dir, 'comparison')
    ensure_output_dir(comparison_dir)
    
    # 1. Radar Chart Comparison
    create_radar_chart_comparison(
        metrics_by_algorithm,
        os.path.join(comparison_dir, 'radar_chart_comparison.png')
    )
    
    # Extract common metrics for bar chart comparisons
    common_metrics = [
        ('avg_waiting_time', 'Average Waiting Time (s)', 'Waiting Time (s)'),
        ('deadline_misses', 'Deadline Misses', 'Count')
    ]
    
    # 2. Bar chart comparisons for each metric
    for metric_key, title_suffix, ylabel in common_metrics:
        # Check if at least one algorithm has this metric
        has_metric = False
        for metrics in metrics_by_algorithm.values():
            if metrics is not None and metric_key in metrics:
                has_metric = True
                break
            # Also check for alternative key names (e.g., average_waiting_time)
            elif metrics is not None and metric_key == 'avg_waiting_time':
                for alt_key in ['average_waiting_time', 'mean_waiting_time']:
                    if alt_key in metrics:
                        has_metric = True
                        break
        
        if has_metric:
            plot_algorithm_comparison(
                metrics_by_algorithm,
                metric_key,
                f'Algorithm Comparison - {title_suffix}',
                ylabel,
                os.path.join(comparison_dir, f'{metric_key}_comparison.png')
            )

def generate_report(single_metrics, multi_metrics, report_path):
    """
    Generate a comprehensive report of the scheduling performance
    
    Args:
        single_metrics: Dictionary mapping scheduler names to metrics for single processor
        multi_metrics: Dictionary mapping scheduler names to metrics for multi-processor
        report_path: Path to save the report
    """
    report_dir = os.path.dirname(report_path)
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    
    with open(report_path, 'w') as f:
        f.write("# Task Scheduling Performance Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Single processor results
        f.write("## Single Processor Results\n\n")
        for algo_name, metrics in single_metrics.items():
            f.write(f"### {algo_name} Scheduler\n\n")
            
            # Get completed tasks count
            completed_tasks = 0
            if 'completed_tasks' in metrics:
                completed_tasks = metrics['completed_tasks']
            
            f.write(f"- Completed Tasks: {completed_tasks}\n")
            
            # Get average waiting time using different possible key names
            avg_waiting_time = 0
            for key in ['avg_waiting_time', 'average_waiting_time', 'mean_waiting_time']:
                if key in metrics:
                    avg_waiting_time = metrics[key]
                    break
            
            f.write(f"- Average Waiting Time: {avg_waiting_time:.2f} seconds\n")
            
            # Add priority-specific metrics if available
            waiting_by_priority = None
            for key in ['avg_waiting_by_priority', 'waiting_times_by_priority']:
                if key in metrics:
                    waiting_by_priority = metrics[key]
                    break
            
            if waiting_by_priority:
                f.write("- Waiting Times by Priority:\n")
                for priority, waiting_time in waiting_by_priority.items():
                    f.write(f"  - {priority}: {waiting_time:.2f} seconds\n")
            
            # Add deadline misses if available (for EDF)
            if 'deadline_misses' in metrics:
                f.write(f"- Deadline Misses: {metrics['deadline_misses']}\n")
            
            # Add priority inversions if available (for Priority-based)
            if 'priority_inversions' in metrics:
                f.write(f"- Priority Inversions: {metrics['priority_inversions']}\n")
                f.write(f"- Priority Inheritance Events: {metrics.get('priority_inheritance_events', 0)}\n")
            
            # Add ML metrics if available
            if 'average_prediction_error' in metrics:
                f.write(f"- Average Prediction Error: {metrics['average_prediction_error']:.2f} seconds\n")
                f.write(f"- Model Trained: {'Yes' if metrics.get('model_trained', False) else 'No'}\n")
            
            f.write("\n")
        
        # Multi-processor results
        if multi_metrics:
            f.write("## Multi-Processor Results\n\n")
            f.write(f"### System Configuration\n\n")
            
            # Get processor count and strategy from first available metrics
            first_metrics = next(iter(multi_metrics.values()))
            processor_count = first_metrics.get('processor_count', 0)
            strategy = first_metrics.get('strategy', 'Unknown')
            
            f.write(f"- Processor Count: {processor_count}\n")
            f.write(f"- Load Balancing Strategy: {strategy}\n\n")
            
            for algo_name, metrics in multi_metrics.items():
                f.write(f"### {algo_name} Scheduler\n\n")
                
                # Get total completed tasks with different possible key names
                total_completed = 0
                for key in ['total_completed_tasks', 'completed_tasks']:
                    if key in metrics:
                        total_completed = metrics[key]
                        break
                
                f.write(f"- Total Completed Tasks: {total_completed}\n")
                
                # Get average waiting time with different possible key names
                avg_waiting_time = 0
                for key in ['avg_waiting_time', 'average_waiting_time', 'mean_waiting_time']:
                    if key in metrics:
                        avg_waiting_time = metrics[key]
                        break
                
                f.write(f"- Average Waiting Time: {avg_waiting_time:.2f} seconds\n")
                
                # Get throughput with different possible key names
                throughput = 0
                for key in ['system_throughput', 'throughput', 'tasks_per_second']:
                    if key in metrics:
                        throughput = metrics[key]
                        break
                
                f.write(f"- System Throughput: {throughput:.2f} tasks/second\n")
                
                # Get load balance with different possible key names
                load_balance = 0
                for key in ['load_balance_cv', 'load_balance']:
                    if key in metrics:
                        load_balance = metrics[key]
                        break
                
                f.write(f"- Load Balance CV: {load_balance:.2f}% (lower is better)\n")
                
                # Get CPU and memory usage
                cpu_usage = metrics.get('avg_cpu_usage', 0)
                memory_usage = metrics.get('avg_memory_usage', 0)
                
                f.write(f"- Average CPU Usage: {cpu_usage:.2f}%\n")
                f.write(f"- Average Memory Usage: {memory_usage:.2f}%\n\n")
                
                # Add tasks by priority if available
                tasks_by_priority = metrics.get('tasks_by_priority')
                if tasks_by_priority:
                    f.write("- Tasks by Priority:\n")
                    for priority, count in tasks_by_priority.items():
                        f.write(f"  - {priority}: {count}\n")
                
                # Add waiting times by priority if available
                waiting_by_priority = None
                for key in ['avg_waiting_by_priority', 'waiting_times_by_priority']:
                    if key in metrics:
                        waiting_by_priority = metrics[key]
                        break
                
                if waiting_by_priority:
                    f.write("- Average Waiting Time by Priority:\n")
                    for priority, waiting_time in waiting_by_priority.items():
                        f.write(f"  - {priority}: {waiting_time:.2f} seconds\n")
                
                f.write("\n")
        
        # Comparative analysis
        f.write("## Comparative Analysis\n\n")
        
        # Compare waiting times across algorithms (single processor)
        f.write("### Single Processor: Average Waiting Time Comparison\n\n")
        f.write("| Algorithm | Average Waiting Time (s) |\n")
        f.write("|-----------|-------------------------|\n")
        for algo_name, metrics in single_metrics.items():
            # Get average waiting time with different possible key names
            waiting_time = 0
            for key in ['avg_waiting_time', 'average_waiting_time', 'mean_waiting_time']:
                if key in metrics:
                    waiting_time = metrics[key]
                    break
            
            f.write(f"| {algo_name} | {waiting_time:.2f} |\n")
        f.write("\n")
        
        # Compare throughput between single and multi-processor
        if single_metrics and multi_metrics:
            f.write("### Single vs. Multi-Processor Throughput\n\n")
            f.write("| System | Average Throughput (tasks/s) |\n")
            f.write("|--------|------------------------------|\n")
            
            # Use first algorithm's metrics for comparison
            first_single_algo = next(iter(single_metrics))
            first_multi_algo = next(iter(multi_metrics))
            
            # Get single processor throughput
            single_throughput = 0
            for key in ['avg_throughput', 'throughput', 'tasks_per_second']:
                if key in single_metrics[first_single_algo]:
                    single_throughput = single_metrics[first_single_algo][key]
                    break
            
            # Get multi-processor throughput
            multi_throughput = 0
            for key in ['system_throughput', 'throughput', 'tasks_per_second']:
                if key in multi_metrics[first_multi_algo]:
                    multi_throughput = multi_metrics[first_multi_algo][key]
                    break
            
            f.write(f"| Single Processor | {single_throughput:.2f} |\n")
            
            processor_count = multi_metrics[first_multi_algo].get('processor_count', 0)
            f.write(f"| Multi-Processor ({processor_count} CPUs) | {multi_throughput:.2f} |\n")
            
            # Calculate speedup
            if single_throughput > 0:
                speedup = multi_throughput / single_throughput
                f.write(f"\nSpeedup factor: {speedup:.2f}x\n")
            
            f.write("\n")
        
        # Conclusion
        f.write("## Conclusion\n\n")
        f.write("Based on the metrics, the following observations can be made:\n\n")
        
        # Find best algorithm for waiting time
        if single_metrics:
            waiting_times = {}
            for name, metrics in single_metrics.items():
                for key in ['avg_waiting_time', 'average_waiting_time', 'mean_waiting_time']:
                    if key in metrics:
                        waiting_times[name] = metrics[key]
                        break
            
            if waiting_times:
                best_waiting = min(waiting_times.items(), key=lambda x: x[1])
                f.write(f"- **Best for Waiting Time**: {best_waiting[0]} scheduler had the lowest average waiting time ({best_waiting[1]:.2f}s).\n")
        
        # Compare deadline misses for EDF
        edf_single = single_metrics.get('EDF', {})
        edf_multi = multi_metrics.get('EDF', {}) if multi_metrics else {}
        
        if 'deadline_misses' in edf_single and 'deadline_misses' in edf_multi:
            single_misses = edf_single['deadline_misses']
            multi_misses = edf_multi['deadline_misses']
            
            if single_misses != multi_misses:
                better_system = "Multi-Processor" if single_misses > multi_misses else "Single Processor"
                f.write(f"- **Deadline Handling**: {better_system} was better at meeting deadlines with the EDF scheduler.\n")
        
        # General observations for multi-processor if available
        if multi_metrics:
            first_metrics = next(iter(multi_metrics.values()))
            
            # Get throughputs 
            single_throughput = 0
            multi_throughput = 0
            
            if single_metrics:
                first_single = next(iter(single_metrics))
                for key in ['avg_throughput', 'throughput', 'tasks_per_second']:
                    if key in single_metrics[first_single]:
                        single_throughput = single_metrics[first_single][key]
                        break
            
            for key in ['system_throughput', 'throughput', 'tasks_per_second']:
                if key in first_metrics:
                    multi_throughput = first_metrics[key]
                    break
            
            if single_throughput > 0:
                speedup = multi_throughput / single_throughput
                
                processor_count = first_metrics.get('processor_count', 0)
                if processor_count > 0:
                    ideal_speedup = processor_count
                    efficiency = (speedup / ideal_speedup) * 100
                    
                    f.write(f"- **Parallelisation Efficiency**: The multi-processor system achieved {efficiency:.1f}% of ideal speedup.\n")
            
            # Check load balance
            load_balance = 0
            for key in ['load_balance_cv', 'load_balance']:
                if key in first_metrics:
                    load_balance = first_metrics[key]
                    break
            
            if load_balance > 0:
                if load_balance < 10:
                    f.write("- **Load Balancing**: Excellent load distribution across processors.\n")
                elif load_balance < 20:
                    f.write("- **Load Balancing**: Good load distribution, but some processors were underutilised.\n")
                else:
                    f.write("- **Load Balancing**: Poor load distribution, significant processor imbalance.\n")
        
        f.write(f"- **Resource Bottlenecks**: The heatmap analysis reveals patterns in resource utilisation that can guide optimisation efforts.\n")
        
        f.write("\nThis report provides a quantitative analysis of different scheduling algorithms on both single and multi-processor systems. The accompanying visualisations provide additional insights into task scheduling behavior and resource utilization patterns.")
    
    return report_path

# Main function to process data and generate visualisations
def process_directory(data_dir, output_dir=None, schedulers=None):
    """
    Process data files in a directory and generate visualisations
    
    Args:
        data_dir: Path to the directory containing data files
        output_dir: Path to save visualisations (if None, uses data_dir/visualisations)
        schedulers: List of schedulers to process (if None, processes all found)
    """
    data_dir = Path(data_dir)
    
    # Default output directory
    if output_dir is None:
        output_dir = data_dir.parent / f"visualisations_{data_dir.name}"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Processing data from {data_dir}")
    logger.info(f"Saving visualisations to {output_dir}")
    
    # Extract platform type from directory
    platform = extract_platform_from_dir(str(data_dir))
    if platform:
        logger.info(f"Detected platform: {platform}")
    
    # Create subdirectories for visualisations
    vis_single_dir = os.path.join(output_dir, "single_processor")
    vis_multi_dir = os.path.join(output_dir, "multi_processor")
    vis_compare_dir = os.path.join(output_dir, "comparison")
    
    os.makedirs(vis_single_dir, exist_ok=True)
    os.makedirs(vis_multi_dir, exist_ok=True)
    os.makedirs(vis_compare_dir, exist_ok=True)
    
    # Process single processor data
    single_processor_dir = data_dir / "single_processor"
    single_metrics = {}
    
    if single_processor_dir.exists():
        # Find all scheduler types from metrics files
        scheduler_types = set()
        
        # Find scheduler types using the pattern SCHEDULER_*.json
        pattern = re.compile(r'(.+?)_.*\.json')
        for file_path in os.listdir(single_processor_dir):
            if file_path.endswith(".json"):
                match = pattern.match(file_path)
                if match:
                    scheduler_types.add(match.group(1))
        
        # Filter by requested schedulers if provided
        if schedulers:
            scheduler_types = [s for s in scheduler_types if s in schedulers]
        
        logger.info(f"Found single processor data for schedulers: {', '.join(scheduler_types)}")
        
        # Process each scheduler
        for scheduler in scheduler_types:
            # Load data files with updated naming convention support
            tasks_path, metrics_path = find_data_files(str(data_dir), scheduler, 'single')
            
            if tasks_path and metrics_path:
                # Load metrics
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                single_metrics[scheduler] = metrics
                
                # Load tasks data
                tasks_df = pd.read_csv(tasks_path)
                
                # Generate enhanced visualisations
                generate_scheduler_visualisations(
                    tasks_df,
                    metrics,
                    scheduler,
                    vis_single_dir
                )
            else:
                logger.warning(f"Could not find data files for {scheduler} in single processor mode")
    
    # Process multi-processor data
    multi_processor_dir = data_dir / "multi_processor"
    multi_metrics = {}
    
    if multi_processor_dir.exists():
        # Find all scheduler types from metrics files
        scheduler_types = set()
        
        # Find scheduler types using the pattern SCHEDULER_*.json
        pattern = re.compile(r'(.+?)_.*\.json')
        for file_path in os.listdir(multi_processor_dir):
            if file_path.endswith(".json"):
                match = pattern.match(file_path)
                if match:
                    scheduler_types.add(match.group(1))
        
        # Filter by requested schedulers if provided
        if schedulers:
            scheduler_types = [s for s in scheduler_types if s in schedulers]
        
        logger.info(f"Found multi-processor data for schedulers: {', '.join(scheduler_types)}")
        
        # Process each scheduler
        for scheduler in scheduler_types:
            # Try to find task files from CPUs
            task_files = [f for f in os.listdir(multi_processor_dir) 
                        if f.startswith(f"{scheduler}_CPU-") and f.endswith("_tasks.csv")]
            
            # Also look for system metrics file
            system_metrics_path = os.path.join(multi_processor_dir, f"{scheduler}_system_metrics.json")
            
            if os.path.exists(system_metrics_path):
                # Load system metrics
                with open(system_metrics_path, 'r') as f:
                    metrics = json.load(f)
                multi_metrics[scheduler] = metrics
                
                # Process tasks if available
                if task_files:
                    # Combine all task files
                    all_tasks = []
                    for task_file in task_files:
                        df = pd.read_csv(os.path.join(multi_processor_dir, task_file))
                        all_tasks.append(df)
                    
                    if all_tasks:
                        combined_df = pd.concat(all_tasks)
                        
                        # Generate enhanced visualisations
                        generate_scheduler_visualisations(
                            combined_df,
                            metrics,
                            scheduler,
                            vis_multi_dir
                        )
                else:
                    logger.warning(f"No task data found for {scheduler} in multi-processor mode, generating metrics-only visualizations")
                    generate_scheduler_visualisations(
                        pd.DataFrame(),  # Empty DataFrame for tasks
                        metrics,
                        scheduler,
                        vis_multi_dir
                    )
            else:
                logger.warning(f"Could not find metrics file for {scheduler} in multi-processor mode")
    
    # Generate comparisons if we have data
    if single_metrics:
        # Compare single processor schedulers
        generate_comparison_visualisations(
            single_metrics,
            vis_single_dir
        )
    
    if multi_metrics:
        # Compare multi-processor schedulers
        generate_comparison_visualisations(
            multi_metrics,
            vis_multi_dir
        )
    
    # Generate comprehensive report
    if single_metrics or multi_metrics:
        report_path = os.path.join(output_dir, "performance_report.md")
        generate_report(single_metrics, multi_metrics, report_path)
        logger.info(f"Generated performance report: {report_path}")
    
    logger.info("Visualisation generation complete")

# If this file is run directly, process command line arguments
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualise task scheduling data')
    parser.add_argument('--data-dir', required=True, help='Directory containing data files')
    parser.add_argument('--output-dir', help='Directory to save visualisations')
    parser.add_argument('--scheduler', action='append', dest='schedulers', 
                        help='Specific scheduler to visualise (can be used multiple times)')
    
    args = parser.parse_args()
    
    process_directory(args.data_dir, args.output_dir, args.schedulers)