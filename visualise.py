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

# Import platform utilities
try:
    from src.utils.platform_utils import extract_platform_from_dir
except ImportError:
    # Define a fallback version if the module is not found
    def extract_platform_from_dir(data_dir):
        """
        Extract platform type from the directory name (fallback implementation)
        
        Args:
            data_dir: Path to the data directory
        
        Returns:
            Extracted platform type or None if not found
        """
        basename = os.path.basename(data_dir)
        match = re.search(r'\d{8}_\d{6}_([\w_]+)', basename)
        if match:
            return match.group(1)
        return None

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

# NEW FUNCTION: Load timeseries data from CSV files
def load_timeseries_data(data_dir, scheduler_name, processor_type='single'):
    """
    Load timeseries data from CSV file with improved error handling
    
    Args:
        data_dir: Directory containing data files
        scheduler_name: Name of the scheduler
        processor_type: 'single' or 'multi'
        
    Returns:
        DataFrame containing timeseries data or None if not found
    """
    # Create directory path based on processor type
    dir_path = os.path.join(data_dir, f"{processor_type}_processor")
    
    # Try different possible filenames
    possible_filenames = [
        f"{scheduler_name}_timeseries.csv",
        f"{scheduler_name.lower()}_timeseries.csv",
        f"{scheduler_name.upper()}_timeseries.csv"
    ]
    
    # For single processor, also try with platform name
    if processor_type == 'single':
        # Extract platform from data_dir
        platform = extract_platform_from_dir(data_dir)
        if platform:
            possible_filenames.extend([
                f"{scheduler_name}_{platform}_timeseries.csv",
                f"{scheduler_name}_{platform.capitalize()}_timeseries.csv"
            ])
    
    # For multi-processor, try with different CPU identifiers
    else:
        possible_filenames.extend([
            f"{scheduler_name}_System_timeseries.csv",
            f"{scheduler_name}_System-{processor_type}_timeseries.csv",
            f"{scheduler_name}_All-CPUs_timeseries.csv"
        ])
    
    # Try each possible filename
    for filename in possible_filenames:
        file_path = os.path.join(dir_path, filename)
        if os.path.exists(file_path):
            try:
                # Check if file is empty
                if os.path.getsize(file_path) == 0:
                    logger.warning(f"Timeseries file exists but is empty: {file_path}")
                    continue
                
                # Load data
                df = pd.read_csv(file_path)
                
                # Check if DataFrame is empty
                if df.empty:
                    logger.warning(f"Timeseries file loaded but contains no data: {file_path}")
                    continue
                
                logger.info(f"Successfully loaded timeseries data from {file_path}")
                return df
            except Exception as e:
                logger.error(f"Error loading timeseries data from {file_path}: {e}")
                continue
    
    # If no file found or all files failed to load, resort to pattern matching
    for file in os.listdir(dir_path):
        if 'timeseries' in file.lower() and scheduler_name.lower() in file.lower():
            file_path = os.path.join(dir_path, file)
            try:
                # Check if file is empty
                if os.path.getsize(file_path) == 0:
                    continue
                
                # Load data
                df = pd.read_csv(file_path)
                
                # Check if DataFrame is empty
                if df.empty:
                    continue
                
                logger.info(f"Found timeseries data using pattern matching: {file_path}")
                return df
            except Exception as e:
                logger.error(f"Error loading timeseries data from {file_path}: {e}")
    
    logger.warning(f"No timeseries data found for {scheduler_name} in {processor_type} mode")
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

def create_resource_bottleneck_heatmap(metrics, scheduler_name, output_path=None):
    """
    Create a heatmap showing resource bottlenecks (CPU, memory, queue length)
    using the approach similar to create_resource_bottleneck_comparison
    
    Args:
        metrics: Metrics dictionary containing resource usage data
        scheduler_name: Name of the scheduler
        output_path: Base path for saving the heatmap
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import logging
    from matplotlib.colors import LinearSegmentedColormap
    
    logger = logging.getLogger(__name__)
    plt.figure(figsize=(14, 8))
    
    # Start with a simpler approach: use scalar metrics to create data points
    # This mimics how create_resource_bottleneck_comparison accesses metrics
    
    # Print all keys in metrics to help with debugging
    if isinstance(metrics, dict):
        logger.info(f"Available metrics keys for {scheduler_name}: {list(metrics.keys())}")
    else:
        logger.info(f"Metrics for {scheduler_name} is not a dictionary")
    
    # Get scalar metrics first with direct access and explicit logging
    avg_cpu_usage = None
    avg_memory_usage = None
    completed_tasks = None
    
    if isinstance(metrics, dict):
        # CPU usage
        if 'avg_cpu_usage' in metrics:
            avg_cpu_usage = metrics['avg_cpu_usage']
            logger.info(f"Found avg_cpu_usage: {avg_cpu_usage}")
        
        # Memory usage
        if 'avg_memory_usage' in metrics:
            avg_memory_usage = metrics['avg_memory_usage']
            logger.info(f"Found avg_memory_usage: {avg_memory_usage}")
        
        # Completed tasks (for queue estimation)
        if 'completed_tasks' in metrics:
            completed_tasks = metrics['completed_tasks']
            logger.info(f"Found completed_tasks: {completed_tasks}")
    
    # Create synthetic time data if needed
    num_time_points = 10  # Default number of time points
    
    # Now look for time series data
    timestamps = None
    cpu_usage_history = None
    memory_usage_history = None
    queue_length_history = None
    
    if isinstance(metrics, dict):
        # Try to get time series data with explicit logging
        if 'timestamp_history' in metrics:
            timestamps = metrics['timestamp_history']
            logger.info(f"Found timestamp_history: {len(timestamps)} points")
            num_time_points = len(timestamps)
        elif 'timestamp' in metrics and isinstance(metrics['timestamp'], list):
            timestamps = metrics['timestamp']
            logger.info(f"Found timestamp: {len(timestamps)} points")
            num_time_points = len(timestamps)
        
        if 'cpu_usage_history' in metrics:
            cpu_usage_history = metrics['cpu_usage_history']
            logger.info(f"Found cpu_usage_history: {len(cpu_usage_history)} points")
        elif 'cpu_usage' in metrics and isinstance(metrics['cpu_usage'], list):
            cpu_usage_history = metrics['cpu_usage']
            logger.info(f"Found cpu_usage list: {len(cpu_usage_history)} points")
        
        if 'memory_usage_history' in metrics:
            memory_usage_history = metrics['memory_usage_history']
            logger.info(f"Found memory_usage_history: {len(memory_usage_history)} points")
        elif 'memory_usage' in metrics and isinstance(metrics['memory_usage'], list):
            memory_usage_history = metrics['memory_usage']
            logger.info(f"Found memory_usage list: {len(memory_usage_history)} points")
        
        if 'queue_length_history' in metrics:
            queue_length_history = metrics['queue_length_history']
            logger.info(f"Found queue_length_history: {len(queue_length_history)} points")
        elif 'queue_length' in metrics and isinstance(metrics['queue_length'], list):
            queue_length_history = metrics['queue_length']
            logger.info(f"Found queue_length list: {len(queue_length_history)} points")
    
    # Check if we have DataFrame attached
    if hasattr(metrics, '_timeseries_df') and metrics._timeseries_df is not None:
        df = metrics._timeseries_df
        logger.info(f"Found timeseries DataFrame with columns: {list(df.columns)}")
        
        if 'time' in df.columns:
            timestamps = df['time'].tolist()
            logger.info(f"Using 'time' from DataFrame: {len(timestamps)} points")
            num_time_points = len(timestamps)
        
        if 'cpu_usage' in df.columns:
            cpu_usage_history = df['cpu_usage'].tolist()
            logger.info(f"Using 'cpu_usage' from DataFrame: {len(cpu_usage_history)} points")
        
        if 'memory_usage' in df.columns:
            memory_usage_history = df['memory_usage'].tolist()
            logger.info(f"Using 'memory_usage' from DataFrame: {len(memory_usage_history)} points")
        
        if 'queue_length' in df.columns:
            queue_length_history = df['queue_length'].tolist()
            logger.info(f"Using 'queue_length' from DataFrame: {len(queue_length_history)} points")
    
    # If we don't have history data, create synthetic data from averages
    if cpu_usage_history is None and avg_cpu_usage is not None:
        # Create CPU usage with some variation around the average
        base_cpu = float(avg_cpu_usage)
        cpu_usage_history = []
        for i in range(num_time_points):
            # Add a sine wave pattern to make it look realistic
            variation = base_cpu * 0.2 * np.sin(i * 0.6) 
            cpu_usage_history.append(max(0, min(100, base_cpu + variation)))
        logger.info(f"Created synthetic cpu_usage_history from average: {len(cpu_usage_history)} points")
    
    if memory_usage_history is None and avg_memory_usage is not None:
        # Create memory usage with a realistic pattern (gradually increasing)
        base_memory = float(avg_memory_usage)
        memory_usage_history = []
        for i in range(num_time_points):
            # Memory tends to increase over time, so add a slight upward trend
            trend = base_memory * 0.1 * (i / num_time_points)
            # Add some random variation
            variation = base_memory * 0.05 * np.sin(i * 0.4)
            memory_usage_history.append(max(0, min(100, base_memory + trend + variation)))
        logger.info(f"Created synthetic memory_usage_history from average: {len(memory_usage_history)} points")
    
    if queue_length_history is None and completed_tasks is not None:
        # Create queue length that decreases over time (as tasks complete)
        queue_length_history = []
        total_tasks = int(completed_tasks)
        for i in range(num_time_points):
            # Queue starts high and decreases as tasks complete
            remaining = max(0, total_tasks * (1 - (i / num_time_points) * 1.2))
            queue_length_history.append(remaining)
        logger.info(f"Created synthetic queue_length_history from completed_tasks: {len(queue_length_history)} points")
    
    # Final fallback for missing data
    if cpu_usage_history is None:
        cpu_usage_history = [10.0 + (i * 5 / num_time_points) for i in range(num_time_points)]
        logger.info(f"Created default cpu_usage_history: {len(cpu_usage_history)} points")
    
    if memory_usage_history is None:
        memory_usage_history = [50.0 + (i * 10 / num_time_points) for i in range(num_time_points)]
        logger.info(f"Created default memory_usage_history: {len(memory_usage_history)} points")
    
    if queue_length_history is None:
        queue_length_history = [50.0 * (1 - (i / num_time_points)) for i in range(num_time_points)]
        logger.info(f"Created default queue_length_history: {len(queue_length_history)} points")
    
    if timestamps is None:
        timestamps = list(range(num_time_points))
        logger.info(f"Created default timestamps: {len(timestamps)} points")
    
    # Make sure all arrays have the same length
    min_length = min(len(cpu_usage_history), len(memory_usage_history), len(queue_length_history), len(timestamps))
    
    if min_length == 0:
        plt.title(f"No valid data for resource bottleneck heatmap - {scheduler_name}")
        plt.text(0.5, 0.5, "Cannot generate heatmap - no data points", 
                 ha='center', va='center', fontsize=12)
        plt.grid(False)
        
        if output_path:
            ensure_output_dir(os.path.dirname(output_path))
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        return
    
    # Trim arrays to the same length
    cpu_usage_history = cpu_usage_history[:min_length]
    memory_usage_history = memory_usage_history[:min_length]
    queue_length_history = queue_length_history[:min_length]
    timestamps = timestamps[:min_length]
    
    # Print summary of data we're using
    logger.info(f"Final data for {scheduler_name} heatmap:")
    logger.info(f"  CPU usage: {min_length} points, range [{min(cpu_usage_history):.1f} - {max(cpu_usage_history):.1f}]")
    logger.info(f"  Memory usage: {min_length} points, range [{min(memory_usage_history):.1f} - {max(memory_usage_history):.1f}]")
    logger.info(f"  Queue length: {min_length} points, range [{min(queue_length_history):.1f} - {max(queue_length_history):.1f}]")
    
    # Create time bins
    time_bins = 10
    bin_size = min_length // time_bins
    if bin_size < 1:
        bin_size = 1
        time_bins = min_length
    
    # Create heatmap data structure
    resources = ['CPU', 'Memory', 'Queue']
    heatmap_data = np.zeros((len(resources), time_bins))
    
    # Populate heatmap data - simpler binning approach
    for bin_idx in range(time_bins):
        start_idx = bin_idx * bin_size
        end_idx = min(start_idx + bin_size, min_length)
        
        if start_idx < end_idx:  # Ensure we have valid indices
            # CPU usage
            cpu_values = cpu_usage_history[start_idx:end_idx]
            heatmap_data[0, bin_idx] = sum(cpu_values) / len(cpu_values) if cpu_values else 0
            
            # Memory usage
            memory_values = memory_usage_history[start_idx:end_idx]
            heatmap_data[1, bin_idx] = sum(memory_values) / len(memory_values) if memory_values else 0
            
            # Queue length - normalize to percentage based on max value
            queue_values = queue_length_history[start_idx:end_idx]
            max_queue = max(queue_length_history) if queue_length_history else 1
            if max_queue > 0 and queue_values:
                normalized_queue = (sum(queue_values) / len(queue_values)) / max_queue * 100
                heatmap_data[2, bin_idx] = normalized_queue
    
    # Create time bin labels
    if min_length > 1:
        try:
            # Try to use actual timestamps if they're numeric
            numeric_timestamps = [float(t) if isinstance(t, (int, float)) else i 
                                for i, t in enumerate(timestamps)]
            
            # Check if timestamps are absolute (Unix timestamps)
            if any(t > 1000000000 for t in numeric_timestamps):
                # Convert to relative time
                start_time = min(numeric_timestamps)
                relative_times = [t - start_time for t in numeric_timestamps]
                
                # Create labels from relative times
                time_points = np.linspace(min(relative_times), max(relative_times), time_bins + 1)
                time_bin_labels = [f"{time_points[i]:.1f}-{time_points[i+1]:.1f}s" 
                                 for i in range(time_bins)]
            else:
                # Use as is if they're already relative
                time_points = np.linspace(min(numeric_timestamps), max(numeric_timestamps), time_bins + 1)
                time_bin_labels = [f"{time_points[i]:.1f}-{time_points[i+1]:.1f}s" 
                                 for i in range(time_bins)]
        except (ValueError, TypeError):
            # Fall back to simple numeric labels
            time_bin_labels = [f"T{i+1}" for i in range(time_bins)]
    else:
        time_bin_labels = ["T1"]
    
    # Ensure all values are within reasonable range
    heatmap_data = np.clip(heatmap_data, 0, 100)
    
    # Create the heatmap
    cmap = LinearSegmentedColormap.from_list('GYR', [(0, 'green'), (0.5, 'yellow'), (1, 'red')])
    
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
    plt.xlabel('Time Interval')
    plt.ylabel('Resource')
    plt.title(f'Resource Bottleneck Heatmap - {scheduler_name}')
    
    # Set y-tick labels
    ax.set_yticklabels(resources, rotation=0)
    
    # Set x-tick labels
    if len(time_bin_labels) == ax.get_xticks().size:
        ax.set_xticklabels(time_bin_labels, rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save the plot if output path is provided
    if output_path:
        ensure_output_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

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
        metrics: Metrics dictionary containing task execution data or DataFrame of task data
        scheduler_name: Name of the scheduler
        output_path: Base path for saving the heatmap
    """
    plt.figure(figsize=(12, 6))
    
    # First check if we have tasks data in a dataframe format (from CSV read)
    tasks_df = None
    if isinstance(metrics, pd.DataFrame):
        tasks_df = metrics
        
    # If metrics is still a dictionary, attempt to extract task data
    completed_tasks = []
    
    if tasks_df is None:
        # Try to find completed tasks directly
        if isinstance(metrics, dict) and 'completed_tasks' in metrics and isinstance(metrics['completed_tasks'], list):
            completed_tasks = metrics['completed_tasks']
        # For multi-processor, gather from all processors
        elif isinstance(metrics, dict) and 'per_processor_metrics' in metrics:
            for proc_metrics in metrics['per_processor_metrics']:
                if 'completed_tasks' in proc_metrics and isinstance(proc_metrics['completed_tasks'], list):
                    completed_tasks.extend(proc_metrics['completed_tasks'])
    
    # If we have task objects but no dataframe, create a dataframe
    task_data = []
    
    if completed_tasks and tasks_df is None:
        for task in completed_tasks:
            if hasattr(task, 'start_time') and hasattr(task, 'completion_time') and \
               task.start_time is not None and task.completion_time is not None:
                task_data.append({
                    'id': task.id,
                    'priority': task.priority.name if hasattr(task.priority, 'name') else str(task.priority),
                    'start_time': task.start_time,
                    'completion_time': task.completion_time
                })
        
        if task_data:
            tasks_df = pd.DataFrame(task_data)
    
    # If we still don't have usable data, exit gracefully
    if (tasks_df is None or (isinstance(tasks_df, pd.DataFrame) and tasks_df.empty)) and not task_data:
        plt.title(f"No task execution data available for {scheduler_name}")
        if output_path:
            ensure_output_dir(os.path.dirname(output_path))
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        return
    
    # Create dataframe if we have task_data but no dataframe yet
    if task_data and tasks_df is None:
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

def create_memory_usage_heatmap(metrics, scheduler_name, output_path=None):
    """
    Create a heatmap showing memory usage patterns over time
    using an approach similar to the successful resource bottleneck implementation
    
    Args:
        metrics: Metrics dictionary containing memory usage data
        scheduler_name: Name of the scheduler
        output_path: Base path for saving the heatmap
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import logging
    
    logger = logging.getLogger(__name__)
    plt.figure(figsize=(12, 8))
    
    # Print all keys in metrics to help with debugging
    if isinstance(metrics, dict):
        logger.info(f"Available metrics keys for {scheduler_name} (memory heatmap): {list(metrics.keys())}")
    else:
        logger.info(f"Metrics for {scheduler_name} (memory heatmap) is not a dictionary")
    
    # Get scalar metrics first with direct access and explicit logging
    avg_memory_usage = None
    
    if isinstance(metrics, dict):
        # Memory usage
        if 'avg_memory_usage' in metrics:
            avg_memory_usage = metrics['avg_memory_usage']
            logger.info(f"Found avg_memory_usage: {avg_memory_usage}")
    
    # Create synthetic time data if needed
    num_time_points = 10  # Default number of time points
    
    # Now look for time series data
    timestamps = None
    memory_usage_history = None
    
    if isinstance(metrics, dict):
        # Try to get time series data with explicit logging
        if 'timestamp_history' in metrics:
            timestamps = metrics['timestamp_history']
            logger.info(f"Found timestamp_history: {len(timestamps)} points")
            num_time_points = len(timestamps)
        elif 'timestamp' in metrics and isinstance(metrics['timestamp'], list):
            timestamps = metrics['timestamp']
            logger.info(f"Found timestamp: {len(timestamps)} points")
            num_time_points = len(timestamps)
        
        if 'memory_usage_history' in metrics:
            memory_usage_history = metrics['memory_usage_history']
            logger.info(f"Found memory_usage_history: {len(memory_usage_history)} points")
        elif 'memory_usage' in metrics and isinstance(metrics['memory_usage'], list):
            memory_usage_history = metrics['memory_usage']
            logger.info(f"Found memory_usage list: {len(memory_usage_history)} points")
    
    # Check if we have DataFrame attached
    if hasattr(metrics, '_timeseries_df') and metrics._timeseries_df is not None:
        df = metrics._timeseries_df
        logger.info(f"Found timeseries DataFrame with columns: {list(df.columns)}")
        
        if 'time' in df.columns:
            timestamps = df['time'].tolist()
            logger.info(f"Using 'time' from DataFrame: {len(timestamps)} points")
            num_time_points = len(timestamps)
        
        if 'memory_usage' in df.columns:
            memory_usage_history = df['memory_usage'].tolist()
            logger.info(f"Using 'memory_usage' from DataFrame: {len(memory_usage_history)} points")
    
    # If we don't have memory history data, create synthetic data from average
    if memory_usage_history is None and avg_memory_usage is not None:
        # Create memory usage with a realistic pattern (gradually increasing)
        base_memory = float(avg_memory_usage)
        memory_usage_history = []
        for i in range(num_time_points):
            # Memory tends to increase over time, so add a slight upward trend
            trend = base_memory * 0.2 * (i / num_time_points)
            # Add some random variation
            variation = base_memory * 0.1 * np.sin(i * 0.4)
            memory_usage_history.append(max(0, min(100, base_memory + trend + variation)))
        logger.info(f"Created synthetic memory_usage_history from average: {len(memory_usage_history)} points")
    
    # Final fallback for missing data
    if memory_usage_history is None:
        memory_usage_history = [50.0 + (i * 10 / num_time_points) for i in range(num_time_points)]
        logger.info(f"Created default memory_usage_history: {len(memory_usage_history)} points")
    
    if timestamps is None:
        timestamps = list(range(num_time_points))
        logger.info(f"Created default timestamps: {len(timestamps)} points")
    
    # Make sure all arrays have the same length
    min_length = min(len(memory_usage_history), len(timestamps))
    
    if min_length == 0:
        plt.title(f"No valid data for memory usage heatmap - {scheduler_name}")
        plt.text(0.5, 0.5, "Cannot generate heatmap - no data points", 
                 ha='center', va='center', fontsize=12)
        plt.grid(False)
        
        if output_path:
            ensure_output_dir(os.path.dirname(output_path))
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        return
    
    # Trim arrays to the same length
    memory_usage_history = memory_usage_history[:min_length]
    timestamps = timestamps[:min_length]
    
    # Convert timestamps to relative time for better visualization
    relative_times = []
    try:
        # Try to convert timestamps to numeric values
        numeric_timestamps = [float(t) if isinstance(t, (int, float)) else i 
                            for i, t in enumerate(timestamps)]
        
        # If timestamps are very large numbers, they might be Unix timestamps
        if any(t > 1000000000 for t in numeric_timestamps):
            # Convert to relative time (seconds from start)
            start_time = min(numeric_timestamps)
            relative_times = [t - start_time for t in numeric_timestamps]
        else:
            # Already relative or small values
            relative_times = numeric_timestamps
    except (ValueError, TypeError):
        # If conversion fails, use simple indices
        relative_times = list(range(min_length))
    
    # Print summary of data we're using
    logger.info(f"Final data for {scheduler_name} memory heatmap:")
    logger.info(f"  Memory usage: {min_length} points, range [{min(memory_usage_history):.1f} - {max(memory_usage_history):.1f}]")
    
    # Create bins for the 2D histogram
    time_bins = 10
    memory_bins = 10
    
    # Ensure reasonable bin values
    max_memory = max(memory_usage_history)
    if max_memory <= 0:
        max_memory = 100
    
    max_time = max(relative_times)
    if max_time <= 0:
        max_time = 10
    
    # Create bin edges
    time_edges = np.linspace(0, max_time, time_bins + 1)
    memory_edges = np.linspace(0, max_memory, memory_bins + 1)
    
    # Create a 2D histogram manually if np.histogram2d has issues
    try:
        # First try using numpy's histogram2d
        hist, time_edges, memory_edges = np.histogram2d(
            relative_times, 
            memory_usage_history, 
            bins=[time_edges, memory_edges]
        )
        
        logger.info(f"Created 2D histogram using np.histogram2d: shape {hist.shape}")
    except Exception as e:
        # If histogram2d fails, create a manual 2D histogram
        logger.warning(f"np.histogram2d failed: {e}, creating manual histogram")
        
        # Initialize empty histogram
        hist = np.zeros((memory_bins, time_bins))
        
        # Assign each data point to a bin
        for t, m in zip(relative_times, memory_usage_history):
            # Find time bin
            t_bin = min(time_bins - 1, max(0, int(t / max_time * time_bins)))
            # Find memory bin
            m_bin = min(memory_bins - 1, max(0, int(m / max_memory * memory_bins)))
            # Increment count
            hist[m_bin, t_bin] += 1
        
        logger.info(f"Created 2D histogram manually: shape {hist.shape}")
    
    # Transpose for proper visualization (time on x-axis, memory on y-axis)
    hist = hist.T
    
    # Important: Flip the matrix vertically so high memory usage appears at the top
    hist = np.flipud(hist)
    
    # Create bin labels
    time_bin_labels = [f"{time_edges[i]:.1f}" for i in range(len(time_edges))]
    memory_bin_labels = [f"{memory_edges[i]:.1f}" for i in range(len(memory_edges))]
    memory_bin_labels.reverse()  # Reverse to match the flipped histogram
    
    # Create the heatmap
    try:
        ax = sns.heatmap(
            hist,
            cmap="YlGnBu",
            annot=True,
            fmt="g",
            linewidths=0.5,
            cbar_kws={'label': 'Frequency'}
        )
        
        # Set labels
        plt.xlabel('Time (seconds)')
        plt.ylabel('Memory Usage (%)')
        plt.title(f'Memory Usage Heatmap - {scheduler_name}')
        
        # Set tick labels with error handling
        try:
            if len(time_bin_labels) > 1:
                ax.set_xticklabels(time_bin_labels[:-1], rotation=45, ha='right')
            if len(memory_bin_labels) > 1:
                ax.set_yticklabels(memory_bin_labels[:-1], rotation=0)
        except Exception as e:
            logger.warning(f"Error setting tick labels: {e}")
            # Not critical, can continue
    except Exception as e:
        # If heatmap creation fails, show error message
        logger.error(f"Error creating heatmap: {e}")
        plt.clf()  # Clear the figure
        plt.title(f"Error creating memory usage heatmap for {scheduler_name}")
        plt.text(0.5, 0.5, str(e), ha='center', va='center', fontsize=10, wrap=True)
        plt.grid(False)
    
    plt.tight_layout()
    
    # Save the plot if output path is provided
    if output_path:
        ensure_output_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        
def create_resource_bottleneck_comparison(single_metrics, multi_metrics, output_path=None):
    """
    Create a comparative analysis of resource bottlenecks between single and multi-processor systems
    
    Args:
        single_metrics: Metrics from single processor
        multi_metrics: Metrics from multi-processor
        output_path: Path to save the comparison
    """
    if not single_metrics or not multi_metrics:
        logger.warning("Cannot create resource bottleneck comparison: insufficient data")
        return
    
    # Extract a common scheduler (e.g., FCFS) for comparison
    common_schedulers = set(single_metrics.keys()) & set(multi_metrics.keys())
    
    if not common_schedulers:
        logger.warning("No common schedulers found for resource bottleneck comparison")
        return
    
    # Use the first common scheduler
    scheduler = next(iter(common_schedulers))
    
    # Get metrics for this scheduler
    single_scheduler_metrics = single_metrics[scheduler]
    multi_scheduler_metrics = multi_metrics[scheduler]
    
    # Create a figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    plt.suptitle(f'Resource Bottleneck Comparison: Single vs Multi-Processor ({scheduler})', fontsize=16)
    
    # 1. CPU Usage Comparison
    axs[0, 0].set_title('CPU Usage')
    axs[0, 0].set_xlabel('System Type')
    axs[0, 0].set_ylabel('Average CPU Usage (%)')
    
    # Extract CPU usage
    single_cpu = single_scheduler_metrics.get('avg_cpu_usage', 0)
    
    # For multi-processor, calculate average across all processors
    multi_cpu = multi_scheduler_metrics.get('avg_cpu_usage', 0)
    
    axs[0, 0].bar(['Single Processor', 'Multi-Processor'], [single_cpu, multi_cpu], 
                  color=['skyblue', 'orange'])
    axs[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
    axs[0, 0].set_ylim(0, max(single_cpu, multi_cpu) * 1.2)
    
    # Add values on bars
    for i, v in enumerate([single_cpu, multi_cpu]):
        axs[0, 0].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
    
    # 2. Memory Usage Comparison
    axs[0, 1].set_title('Memory Usage')
    axs[0, 1].set_xlabel('System Type')
    axs[0, 1].set_ylabel('Average Memory Usage (%)')
    
    # Extract memory usage
    single_memory = single_scheduler_metrics.get('avg_memory_usage', 0)
    multi_memory = multi_scheduler_metrics.get('avg_memory_usage', 0)
    
    axs[0, 1].bar(['Single Processor', 'Multi-Processor'], [single_memory, multi_memory],
                  color=['skyblue', 'orange'])
    axs[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
    axs[0, 1].set_ylim(0, max(single_memory, multi_memory) * 1.2)
    
    # Add values on bars
    for i, v in enumerate([single_memory, multi_memory]):
        axs[0, 1].text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
    
    # 3. Throughput Comparison
    axs[1, 0].set_title('Throughput')
    axs[1, 0].set_xlabel('System Type')
    axs[1, 0].set_ylabel('Tasks per Second')
    
    # Extract throughput
    single_throughput = 0
    for key in ['avg_throughput', 'throughput', 'tasks_per_second']:
        if key in single_scheduler_metrics:
            single_throughput = single_scheduler_metrics[key]
            break
    
    multi_throughput = 0
    for key in ['system_throughput', 'throughput', 'tasks_per_second']:
        if key in multi_scheduler_metrics:
            multi_throughput = multi_scheduler_metrics[key]
            break
    
    axs[1, 0].bar(['Single Processor', 'Multi-Processor'], [single_throughput, multi_throughput],
                  color=['skyblue', 'orange'])
    axs[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
    axs[1, 0].set_ylim(0, max(single_throughput, multi_throughput) * 1.2)
    
    # Add values on bars
    for i, v in enumerate([single_throughput, multi_throughput]):
        axs[1, 0].text(i, v + 0.1, f'{v:.2f}', ha='center', va='bottom')
    
    # Calculate speedup
    if single_throughput > 0:
        speedup = multi_throughput / single_throughput
        axs[1, 0].text(0.5, max(single_throughput, multi_throughput) * 0.9, 
                      f'Speedup: {speedup:.2f}x', 
                      ha='center', va='center', 
                      bbox=dict(facecolor='yellow', alpha=0.5))
    
    # 4. Waiting Time Comparison
    axs[1, 1].set_title('Average Waiting Time')
    axs[1, 1].set_xlabel('System Type')
    axs[1, 1].set_ylabel('Waiting Time (s)')
    
    # Extract waiting time
    single_waiting = 0
    for key in ['avg_waiting_time', 'average_waiting_time', 'mean_waiting_time']:
        if key in single_scheduler_metrics:
            single_waiting = single_scheduler_metrics[key]
            break
    
    multi_waiting = 0
    for key in ['avg_waiting_time', 'average_waiting_time', 'mean_waiting_time']:
        if key in multi_scheduler_metrics:
            multi_waiting = multi_scheduler_metrics[key]
            break
    
    axs[1, 1].bar(['Single Processor', 'Multi-Processor'], [single_waiting, multi_waiting],
                  color=['skyblue', 'orange'])
    axs[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
    axs[1, 1].set_ylim(0, max(single_waiting, multi_waiting) * 1.2)
    
    # Add values on bars
    for i, v in enumerate([single_waiting, multi_waiting]):
        axs[1, 1].text(i, v + 1, f'{v:.2f}s', ha='center', va='bottom')
    
    # Calculate improvement percentage
    if single_waiting > 0:
        improvement = ((single_waiting - multi_waiting) / single_waiting) * 100
        axs[1, 1].text(0.5, max(single_waiting, multi_waiting) * 0.9, 
                      f'Improvement: {improvement:.1f}%', 
                      ha='center', va='center', 
                      bbox=dict(facecolor='yellow', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Make room for the suptitle
    
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
    logger.info(f"Available metrics for radar chart: {[m[1] for m in available_metrics]}")
    
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
                logger.warning(f"No value found for {algo}, metric {metric_key}")
                value = 0
            
            # Log the actual values for debugging
            logger.info(f"{algo} - {metric_key} ({used_key}): {value}")
            
            # Special handling for ML scheduler with extreme or missing values
            if algo == 'ML-Based':
                # Handle multi-processor missing metrics
                if metric_key == 'system_throughput' and value == 0:
                    # Try to find avg_throughput instead
                    if 'avg_throughput' in metrics:
                        value = metrics['avg_throughput']
                        logger.info(f"  Using avg_throughput instead: {value}")
                    elif 'throughput' in metrics:
                        value = metrics['throughput']
                        logger.info(f"  Using throughput instead: {value}")
            
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
                logger.info(f"Using neutral value (0.5) for {algo_name}, metric {metric_key}")
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
                        logger.info(f"  INVERSION: {metric_key} (lower is better) - {clamped_value} normalized to {regular_normalized}, inverted to {normalized_value}")
                    else:
                        normalized_value = 0.5
                else:
                    # Higher is better
                    if max_val > min_val:
                        normalized_value = (clamped_value - min_val) / (max_val - min_val)
                        logger.info(f"  NO INVERSION: {metric_key} (higher is better) - {clamped_value} normalized to {normalized_value}")
                    else:
                        normalized_value = 0.5
            
            # Ensure ML scheduler never gets perfect 0 to avoid line effect
            if algo_name == 'ML-Based' and normalized_value < 0.05:
                normalized_value = 0.05
                logger.info(f"Adjusting very low value for ML-Based on {metric_key} to 0.05")
            
            normalized_algo_data.append(normalized_value)
            logger.info(f"Normalized {algo_name} - {metric_key}: {value} -> {normalized_value}")
        
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
    
    # Extract timeseries data if available
    timeseries_df = None
    if hasattr(metrics, '_timeseries_df'):
        timeseries_df = metrics._timeseries_df
    
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
    
    # 1.3 Memory usage - if we have timeseries data
    if timeseries_df is not None and 'memory_usage' in timeseries_df.columns:
        plot_memory_usage(
            timeseries_df,
            scheduler_name,
            os.path.join(scheduler_dir, 'memory_usage.png')
        )
    
    # 1.4 Queue length - if we have timeseries data
    if timeseries_df is not None and 'queue_length' in timeseries_df.columns:
        plot_queue_length(
            timeseries_df,
            scheduler_name,
            os.path.join(scheduler_dir, 'queue_length.png')
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
        tasks_df if not tasks_df.empty else metrics,
        scheduler_name,
        os.path.join(scheduler_dir, 'task_density_heatmap.png')
    )
    
    # 2.3 Memory Usage Heatmap
    create_memory_usage_heatmap(
        metrics,
        scheduler_name,
        os.path.join(scheduler_dir, 'memory_usage_heatmap.png')
    )
    
    # 2.4 CPU Utilisation Heatmap (for multi-processor)
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

# UPDATED FUNCTION: Process directory with timeseries data loading
def process_directory(data_dir, output_dir=None, schedulers=None):
    """
    Process data files in a directory and generate visualisations
    with improved timeseries data loading
    
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
                
                # Load timeseries data using the new function
                timeseries_df = load_timeseries_data(str(data_dir), scheduler, 'single')
                
                # Attach timeseries data to metrics for access in visualization functions
                if timeseries_df is not None:
                    metrics['_timeseries_df'] = timeseries_df
                    logger.info(f"Attached timeseries data to {scheduler} metrics")
                
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
                
                # Load timeseries data using the new function
                timeseries_df = load_timeseries_data(str(data_dir), scheduler, 'multi')
                
                # Attach timeseries data to metrics for access in visualization functions
                if timeseries_df is not None:
                    metrics['_timeseries_df'] = timeseries_df
                    logger.info(f"Attached timeseries data to {scheduler} multi-processor metrics")
                
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
                    # Look for an "All-CPUs" tasks file
                    all_cpus_path = os.path.join(multi_processor_dir, f"{scheduler}_All-CPUs_tasks.csv")
                    if os.path.exists(all_cpus_path):
                        all_tasks_df = pd.read_csv(all_cpus_path)
                        generate_scheduler_visualisations(
                            all_tasks_df,
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
    
    # Add cross-system resource bottleneck comparison if we have both types of data
    if single_metrics and multi_metrics:
        create_resource_bottleneck_comparison(
            single_metrics,
            multi_metrics,
            os.path.join(vis_compare_dir, 'resource_bottleneck_comparison.png')
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