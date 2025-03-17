"""
Visualisation Utilities

This module provides functions for visualising task scheduling results,
including line plots, bar charts, heatmaps, and report generation.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import os
from matplotlib.colors import LinearSegmentedColormap

def ensure_output_dir(path='results'):
    """Ensure the output directory exists"""
    if not os.path.exists(path):
        os.makedirs(path)
    return path

def plot_task_completion(completed_tasks, scheduler_name, output_path=None):
    """
    Plot task completion times
    
    Args:
        completed_tasks: List of completed Task objects
        scheduler_name: Name of the scheduler
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Extract task data
    if not completed_tasks:
        plt.title(f"No completed tasks for {scheduler_name}")
        if output_path:
            plt.savefig(output_path)
        return
    
    # Convert to DataFrame for easier plotting
    task_data = []
    for task in completed_tasks:
        task_data.append({
            'id': task.id,
            'priority': task.priority.name,
            'arrival_time': task.arrival_time,
            'start_time': task.start_time,
            'completion_time': task.completion_time,
            'waiting_time': task.waiting_time,
            'service_time': task.service_time
        })
    
    df = pd.DataFrame(task_data)
    
    # Sort by start time
    df = df.sort_values('start_time')
    
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
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_waiting_times(completed_tasks, scheduler_name, output_path=None):
    """
    Plot waiting times by priority
    
    Args:
        completed_tasks: List of completed Task objects
        scheduler_name: Name of the scheduler
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    if not completed_tasks:
        plt.title(f"No completed tasks for {scheduler_name}")
        if output_path:
            plt.savefig(output_path)
        return
    
    # Convert to DataFrame
    task_data = []
    for task in completed_tasks:
        if task.waiting_time is not None:  # Ensure waiting time is available
            task_data.append({
                'id': task.id,
                'priority': task.priority.name,
                'waiting_time': task.waiting_time
            })
    
    df = pd.DataFrame(task_data)
    
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
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_memory_usage(metrics, scheduler_name, output_path=None):
    """
    Plot memory usage over time
    
    Args:
        metrics: Metrics dictionary from scheduler
        scheduler_name: Name of the scheduler
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Check if memory_usage_history exists in metrics
    memory_usage = metrics.get('memory_usage_history', [])
    timestamps = metrics.get('timestamp_history', [])
    
    # If not found, check nested in metrics history
    if not memory_usage and 'memory_usage' in metrics:
        if isinstance(metrics['memory_usage'], list):
            memory_usage = metrics['memory_usage']
        
    if not timestamps and 'timestamp' in metrics:
        if isinstance(metrics['timestamp'], list):
            timestamps = metrics['timestamp']
    
    if not memory_usage or not timestamps:
        plt.title(f"No memory data for {scheduler_name}")
        if output_path:
            plt.savefig(output_path)
        return
    
    # Make sure arrays have the same length
    min_length = min(len(memory_usage), len(timestamps))
    memory_usage = memory_usage[:min_length]
    timestamps = timestamps[:min_length]
    
    # Convert timestamps to relative time in seconds
    if min_length > 0:
        start_time = timestamps[0]
        relative_times = [t - start_time for t in timestamps]
        
        plt.plot(relative_times, memory_usage, marker='o', linestyle='-', markersize=3)
        plt.title(f'Memory Usage Over Time - {scheduler_name}')
        plt.xlabel('Time (s)')
        plt.ylabel('Memory Usage (%)')
        plt.grid(True, linestyle='--', alpha=0.7)
    else:
        plt.title(f"Insufficient memory data for {scheduler_name}")
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_queue_length(metrics, scheduler_name, output_path=None):
    """
    Plot queue length over time
    
    Args:
        metrics: Metrics dictionary from scheduler
        scheduler_name: Name of the scheduler
        output_path: Path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Check both possible locations for queue length history
    queue_lengths = metrics.get('queue_length_history', [])
    timestamps = metrics.get('timestamp_history', [])
    
    # If not found, check if nested in metrics history
    if not queue_lengths and 'queue_length' in metrics:
        if isinstance(metrics['queue_length'], list):
            queue_lengths = metrics['queue_length']
    
    if not timestamps and 'timestamp' in metrics:
        if isinstance(metrics['timestamp'], list):
            timestamps = metrics['timestamp']
    
    if not queue_lengths or not timestamps:
        plt.title(f"No queue data for {scheduler_name}")
        if output_path:
            plt.savefig(output_path)
        return
    
    # Make sure the arrays have the same length
    min_length = min(len(queue_lengths), len(timestamps))
    queue_lengths = queue_lengths[:min_length]
    timestamps = timestamps[:min_length]
    
    if min_length > 0:
        # Convert timestamps to relative time in seconds
        start_time = timestamps[0]
        relative_times = [t - start_time for t in timestamps]
        
        plt.plot(relative_times, queue_lengths, marker='o', linestyle='-', markersize=3)
        plt.title(f'Queue Length Over Time - {scheduler_name}')
        plt.xlabel('Time (s)')
        plt.ylabel('Queue Length (tasks)')
        plt.grid(True, linestyle='--', alpha=0.7)
    else:
        plt.title(f"Insufficient queue data for {scheduler_name}")
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_priority_distribution(metrics, scheduler_name, output_path=None):
    """
    Plot distribution of completed tasks by priority
    
    Args:
        metrics: Metrics dictionary from scheduler
        scheduler_name: Name of the scheduler
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Check if tasks_by_priority exists in metrics
    tasks_by_priority = metrics.get('tasks_by_priority', {})
    
    # If not found, try to calculate from completed tasks
    if not tasks_by_priority and 'completed_tasks' in metrics:
        completed_tasks = metrics['completed_tasks']
        tasks_by_priority = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
        
        for task in completed_tasks:
            priority = task.priority.name if hasattr(task, 'priority') else 'Unknown'
            if priority in tasks_by_priority:
                tasks_by_priority[priority] += 1
    
    if not tasks_by_priority:
        plt.title(f"No priority data for {scheduler_name}")
        if output_path:
            plt.savefig(output_path)
        return
    
    # Extract priorities and counts
    priorities = list(tasks_by_priority.keys())
    counts = list(tasks_by_priority.values())
    
    # Create color map
    colors = [get_priority_color(priority) for priority in priorities]
    
    plt.bar(priorities, counts, color=colors, alpha=0.8, edgecolor='black')
    plt.title(f'Completed Tasks by Priority - {scheduler_name}')
    plt.xlabel('Priority')
    plt.ylabel('Number of Tasks')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(counts):
        plt.text(i, v + max(counts) * 0.02, str(v), 
                ha='center', va='bottom', fontweight='bold')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_waiting_times_by_priority(metrics, scheduler_name, output_path=None):
    """
    Plot average waiting times by priority
    
    Args:
        metrics: Metrics dictionary from scheduler
        scheduler_name: Name of the scheduler
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Check if avg_waiting_by_priority exists in metrics
    waiting_by_priority = metrics.get('avg_waiting_by_priority', {})
    
    if not waiting_by_priority:
        plt.title(f"No waiting time by priority data for {scheduler_name}")
        if output_path:
            plt.savefig(output_path)
        return
    
    # Extract priorities and waiting times
    priorities = list(waiting_by_priority.keys())
    waiting_times = list(waiting_by_priority.values())
    
    # Create color map
    colors = [get_priority_color(priority) for priority in priorities]
    
    plt.bar(priorities, waiting_times, color=colors, alpha=0.8, edgecolor='black')
    plt.title(f'Average Waiting Time by Priority - {scheduler_name}')
    plt.xlabel('Priority')
    plt.ylabel('Average Waiting Time (s)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(waiting_times):
        plt.text(i, v + max(waiting_times) * 0.02, f'{v:.2f}', 
                ha='center', va='bottom', fontweight='bold')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_algorithm_comparison(metrics_dict, metric_name, title, ylabel, output_path=None):
    """
    Compare a specific metric across different algorithms
    
    Args:
        metrics_dict: Dictionary mapping algorithm names to their metrics
        metric_name: Name of the metric to compare
        title: Plot title
        ylabel: Y-axis label
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    algorithms = []
    values = []
    
    for algo_name, metrics in metrics_dict.items():
        # Handle both direct values and nested metrics
        if isinstance(metrics, (int, float, np.number)):
            # If metrics_dict contains direct values
            algorithms.append(algo_name)
            values.append(float(metrics))  # Convert to Python float
        elif isinstance(metrics, dict):
            # For metrics that are already scalar values in a dict
            if metric_name in metrics and isinstance(metrics[metric_name], (int, float, np.number)):
                algorithms.append(algo_name)
                values.append(float(metrics[metric_name]))  # Convert to Python float
            # For metrics that are dictionaries (e.g., waiting times by priority)
            elif metric_name.startswith('avg_') and metric_name[4:] in metrics:
                # Compute average of the dictionary values
                avg_value = np.mean(list(metrics[metric_name[4:]].values()))
                algorithms.append(algo_name)
                values.append(float(avg_value))  # Convert to Python float
    
    if not algorithms:
        plt.title(f"No data available for {metric_name}")
        if output_path:
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
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_processor_comparison(single_metrics, multi_metrics, metric_name, title, ylabel, output_path=None):
    """
    Compare metrics between single and multi-processor systems
    
    Args:
        single_metrics: Metrics from single processor
        multi_metrics: Metrics from multi-processor
        metric_name: Name of the metric to compare
        title: Plot title
        ylabel: Y-axis label
        output_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    
    systems = ['Single Processor', 'Multi-Processor']
    values = []
    
    # Extract metric from single processor
    single_value = 0
    if isinstance(single_metrics, (int, float, np.number)):
        # Direct value
        single_value = float(single_metrics)
    elif metric_name in single_metrics and isinstance(single_metrics[metric_name], (int, float, np.number)):
        # Value in dictionary
        single_value = float(single_metrics[metric_name])
    values.append(single_value)
    
    # Extract metric from multi-processor
    multi_value = 0
    if isinstance(multi_metrics, (int, float, np.number)):
        # Direct value
        multi_value = float(multi_metrics)
    elif metric_name in multi_metrics and isinstance(multi_metrics[metric_name], (int, float, np.number)):
        # Value in dictionary
        multi_value = float(multi_metrics[metric_name])
    values.append(multi_value)
    
    plt.bar(systems, values, color=['skyblue', 'orange'], alpha=0.8, edgecolor='black')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(values):
        plt.text(i, v + max(values) * 0.02, f'{v:.2f}', 
                ha='center', va='bottom', fontweight='bold')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

# ===== Heatmap Visualization Functions =====

def create_performance_heatmaps(metrics, scheduler_name, output_path=None):
    """
    Create heatmaps for various system performance metrics
    
    Args:
        metrics: Metrics dictionary containing time-series performance data
        scheduler_name: Name of the scheduler
        output_path: Base path for saving the heatmaps
    """
    # Create output directory if it doesn't exist
    if output_path and not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Create heatmaps for different metrics
    create_cpu_utilization_heatmap(metrics, scheduler_name, output_path)
    create_memory_usage_heatmap(metrics, scheduler_name, output_path)
    create_resource_bottleneck_heatmap(metrics, scheduler_name, output_path)


def create_cpu_utilization_heatmap(metrics, scheduler_name, output_path=None):
    """
    Create a heatmap showing CPU utilization across time for multiple processors
    
    Args:
        metrics: Metrics dictionary containing CPU utilization data
        scheduler_name: Name of the scheduler
        output_path: Base path for saving the heatmap
    """
    plt.figure(figsize=(12, 8))
    
    # Check for required data
    per_processor_metrics = metrics.get('per_processor_metrics', [])
    
    if not per_processor_metrics:
        plt.title(f"No CPU utilization data available for {scheduler_name}")
        if output_path:
            plt.savefig(f"{output_path}/cpu_utilization_heatmap.png", dpi=300, bbox_inches='tight')
        return
    
    # Create data structure for the heatmap
    processor_count = len(per_processor_metrics)
    
    # Try to find CPU usage history
    cpu_usage_data = []
    timestamps = []
    
    # First try to find common timestamps
    for proc_metrics in per_processor_metrics:
        if 'timestamp' in proc_metrics and isinstance(proc_metrics['timestamp'], list):
            timestamps = proc_metrics['timestamp']
            break
    
    # If no timestamps found, try timestamp_history
    if not timestamps and 'timestamp_history' in metrics:
        timestamps = metrics['timestamp_history']
    
    # If still no timestamps, create a default range
    if not timestamps:
        timestamps = list(range(10))  # Default range if no timestamps found
    
    # Get CPU usage data
    for i, proc_metrics in enumerate(per_processor_metrics):
        cpu_usage = []
        
        # Try different possible keys for CPU usage history
        if 'cpu_usage_history' in proc_metrics:
            cpu_usage = proc_metrics['cpu_usage_history']
        elif 'cpu_usage' in proc_metrics and isinstance(proc_metrics['cpu_usage'], list):
            cpu_usage = proc_metrics['cpu_usage']
        
        # If no CPU usage history, use average CPU usage to create a constant line
        if not cpu_usage and 'avg_cpu_usage' in proc_metrics:
            cpu_usage = [proc_metrics['avg_cpu_usage']] * len(timestamps)
        
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
    
    # Create a DataFrame for the heatmap
    if timestamps and timestamps[0] is not None:
        # Convert timestamps to relative time if they're absolute timestamps
        if isinstance(timestamps[0], (int, float)) and timestamps[0] > 1000000000:  # Likely a Unix timestamp
            start_time = timestamps[0]
            relative_times = [t - start_time for t in timestamps]
        else:
            relative_times = timestamps
    else:
        relative_times = list(range(len(cpu_usage_data[0])))
    
    # Create DataFrame with time bins for better visualization
    df = pd.DataFrame()
    
    # Create time bins (10 bins)
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
    plt.title(f'CPU Utilization Heatmap - {scheduler_name}')
    
    # Set y-tick labels (processor names)
    processor_labels = [f"CPU-{i+1}" for i in range(processor_count)]
    ax.set_yticklabels(processor_labels, rotation=0)
    
    # Set x-tick labels (time bins)
    ax.set_xticklabels(time_bin_labels, rotation=45, ha='right')
    
    plt.tight_layout()
    
    # Save the plot if output path is provided
    if output_path:
        plt.savefig(f"{output_path}/cpu_utilization_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_memory_usage_heatmap(metrics, scheduler_name, output_path=None):
    """
    Create a heatmap showing memory usage patterns over time
    
    Args:
        metrics: Metrics dictionary containing memory usage data
        scheduler_name: Name of the scheduler
        output_path: Base path for saving the heatmap
    """
    plt.figure(figsize=(12, 6))
    
    # Check for required data
    memory_usage = metrics.get('memory_usage_history', [])
    timestamps = metrics.get('timestamp_history', [])
    
    # Try alternative keys if not found
    if not memory_usage and 'memory_usage' in metrics and isinstance(metrics['memory_usage'], list):
        memory_usage = metrics['memory_usage']
    
    if not timestamps and 'timestamp' in metrics and isinstance(metrics['timestamp'], list):
        timestamps = metrics['timestamp']
    
    if not memory_usage or not timestamps:
        plt.title(f"No memory usage data available for {scheduler_name}")
        if output_path:
            plt.savefig(f"{output_path}/memory_usage_heatmap.png", dpi=300, bbox_inches='tight')
        return
    
    # Ensure same length
    min_length = min(len(memory_usage), len(timestamps))
    memory_usage = memory_usage[:min_length]
    timestamps = timestamps[:min_length]
    
    if min_length == 0:
        plt.title(f"No memory usage data available for {scheduler_name}")
        if output_path:
            plt.savefig(f"{output_path}/memory_usage_heatmap.png", dpi=300, bbox_inches='tight')
        return
    
    # Convert timestamps to relative time
    if isinstance(timestamps[0], (int, float)) and timestamps[0] > 1000000000:  # Likely a Unix timestamp
        start_time = timestamps[0]
        relative_times = [t - start_time for t in timestamps]
    else:
        relative_times = timestamps
    
    # Create time bins (10 bins)
    max_time = max(relative_times)
    time_bins = np.linspace(0, max_time, 11)
    
    # Create memory usage bins (10 bins) - reversed to have higher values at the top
    max_memory = max(memory_usage)
    memory_bins = np.linspace(0, max_memory if max_memory > 0 else 100, 11)
    
    # Create a 2D histogram of memory usage over time
    hist, x_edges, y_edges = np.histogram2d(
        relative_times, 
        memory_usage, 
        bins=[time_bins, memory_bins]
    )
    
    # Create a heatmap
    plt.figure(figsize=(12, 8))
    
    # Transpose to have time on x-axis and memory usage on y-axis
    hist = hist.T
    
    # IMPORTANT: Flip the matrix vertically so higher values appear at the top
    # This reverses the rows of the histogram
    hist = np.flipud(hist)
    
    # Create readable bin labels
    time_bin_labels = [f"{time_bins[i]:.1f}" for i in range(len(time_bins))]
    
    # Create memory bin labels, but reverse them to match the flipped histogram
    memory_bin_labels = [f"{memory_bins[i]:.1f}" for i in range(len(memory_bins))]
    memory_bin_labels.reverse()  # Reverse to have higher values at the top
    
    # Plot the heatmap
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
    
    # Set tick labels
    ax.set_xticklabels(time_bin_labels[:-1], rotation=45, ha='right')
    ax.set_yticklabels(memory_bin_labels[:-1], rotation=0)
    
    plt.tight_layout()
    
    # Save the plot if output path is provided
    if output_path:
        plt.savefig(f"{output_path}/memory_usage_heatmap.png", dpi=300, bbox_inches='tight')
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
    
    # Extract data from metrics
    timestamps = metrics.get('timestamp_history', [])
    
    # Try alternative timestamp key if not found
    if not timestamps and 'timestamp' in metrics and isinstance(metrics['timestamp'], list):
        timestamps = metrics['timestamp']
    
    # If still no timestamps, create a default range
    if not timestamps:
        timestamps = list(range(10))  # Default range if no timestamps found
    
    # Gather resource metrics
    memory_usage = metrics.get('memory_usage_history', [])
    queue_length = metrics.get('queue_length_history', [])
    
    # Try alternative keys
    if not memory_usage and 'memory_usage' in metrics and isinstance(metrics['memory_usage'], list):
        memory_usage = metrics['memory_usage']
    
    if not queue_length and 'queue_length' in metrics and isinstance(metrics['queue_length'], list):
        queue_length = metrics['queue_length']
    
    # Get CPU usage from processor metrics
    per_processor_metrics = metrics.get('per_processor_metrics', [])
    cpu_usage = []
    
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
    
    # Ensure all lists are the same length as timestamps
    min_length = len(timestamps)
    
    # Trim or extend memory usage
    if memory_usage:
        if len(memory_usage) > min_length:
            memory_usage = memory_usage[:min_length]
        elif len(memory_usage) < min_length:
            # Extend with the last value or zero
            last_value = memory_usage[-1] if memory_usage else 0
            memory_usage.extend([last_value] * (min_length - len(memory_usage)))
    else:
        memory_usage = [0] * min_length
    
    # Trim or extend queue length
    if queue_length:
        if len(queue_length) > min_length:
            queue_length = queue_length[:min_length]
        elif len(queue_length) < min_length:
            # Extend with the last value or zero
            last_value = queue_length[-1] if queue_length else 0
            queue_length.extend([last_value] * (min_length - len(queue_length)))
    else:
        queue_length = [0] * min_length
    
    # Trim or extend CPU usage
    if cpu_usage:
        if len(cpu_usage) > min_length:
            cpu_usage = cpu_usage[:min_length]
        elif len(cpu_usage) < min_length:
            # Extend with the last value or zero
            last_value = cpu_usage[-1] if cpu_usage else 0
            cpu_usage.extend([last_value] * (min_length - len(cpu_usage)))
    else:
        cpu_usage = [0] * min_length
    
    # Convert timestamps to relative time
    if timestamps and isinstance(timestamps[0], (int, float)) and timestamps[0] > 1000000000:
        start_time = timestamps[0]
        relative_times = [t - start_time for t in timestamps]
    else:
        relative_times = list(range(len(timestamps)))
    
    # Create time bins (10 bins)
    max_time = max(relative_times) if relative_times else 10
    time_bins = np.linspace(0, max_time, 11)
    time_bin_labels = [f"{time_bins[i]:.1f}-{time_bins[i+1]:.1f}s" for i in range(len(time_bins)-1)]
    
    # Normalize data for consistent visualization
    max_cpu = max(cpu_usage) if cpu_usage else 100
    max_memory = max(memory_usage) if memory_usage else 100
    max_queue = max(queue_length) if queue_length else 1
    
    # Avoid division by zero
    max_cpu = max(max_cpu, 0.1)
    max_memory = max(max_memory, 0.1)
    max_queue = max(max_queue, 0.1)
    
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
    
    # Create a custom colormap that goes from green to red
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
        plt.savefig(f"{output_path}/resource_bottleneck_heatmap.png", dpi=300, bbox_inches='tight')
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
    
    # Check for required data
    per_processor_metrics = metrics.get('per_processor_metrics', [])
    
    # Gather all tasks across all processors
    all_tasks = []
    for proc_idx, proc_metrics in enumerate(per_processor_metrics):
        if 'completed_tasks' in proc_metrics and isinstance(proc_metrics['completed_tasks'], list):
            # Add processor index to track which CPU processed each task
            for task in proc_metrics['completed_tasks']:
                if hasattr(task, 'start_time') and task.start_time is not None:
                    all_tasks.append({
                        'processor': proc_idx,
                        'start_time': task.start_time,
                        'completion_time': task.completion_time if hasattr(task, 'completion_time') else task.start_time + task.service_time,
                        'priority': task.priority.name if hasattr(task, 'priority') else 'UNKNOWN'
                    })
    
    if not all_tasks:
        plt.title(f"No task execution data available for {scheduler_name}")
        if output_path:
            plt.savefig(f"{output_path}/task_density_heatmap.png", dpi=300, bbox_inches='tight')
        return
    
    # Create a DataFrame for easier analysis
    df = pd.DataFrame(all_tasks)
    
    # Create a time range
    min_time = df['start_time'].min()
    max_time = df['completion_time'].max()
    
    # Create time bins (10 bins)
    time_bins = np.linspace(min_time, max_time, 11)
    time_bin_labels = [f"{time_bins[i]:.1f}-{time_bins[i+1]:.1f}s" for i in range(len(time_bins)-1)]
    
    # Count tasks in each time bin for each priority level
    priority_levels = ['HIGH', 'MEDIUM', 'LOW']
    heatmap_data = np.zeros((len(priority_levels), len(time_bins)-1))
    
    for _, task in df.iterrows():
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
    ax = sns.heatmap(
        heatmap_data,
        cmap="viridis",
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
        plt.savefig(f"{output_path}/task_density_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def get_priority_color(priority):
    """Get color based on task priority"""
    colors = {
        'HIGH': '#FF5252',    # Red
        'MEDIUM': '#FFD740',  # Amber
        'LOW': '#69F0AE'      # Green
    }
    return colors.get(priority, '#2196F3')  # Default to blue

def get_algorithm_color(algorithm):
    """Get color based on algorithm name"""
    colors = {
        'FCFS': '#2196F3',         # Blue
        'EDF': '#7B1FA2',          # Purple
        'Priority': '#FF5722',     # Deep Orange
        'ML-Based': '#009688'      # Teal
    }
    return colors.get(algorithm, '#607D8B')  # Default to gray

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
                if isinstance(metrics['completed_tasks'], list):
                    completed_tasks = len(metrics['completed_tasks'])
                else:
                    completed_tasks = metrics['completed_tasks']
            
            f.write(f"- Completed Tasks: {completed_tasks}\n")
            
            # Get average waiting time using different possible key names
            avg_waiting_time = 0
            for key in ['average_waiting_time', 'avg_waiting_time', 'mean_waiting_time']:
                if key in metrics:
                    avg_waiting_time = metrics[key]
                    break
            
            f.write(f"- Average Waiting Time: {avg_waiting_time:.2f} seconds\n")
            
            # Add priority-specific metrics if available
            waiting_by_priority = None
            for key in ['waiting_times_by_priority', 'avg_waiting_by_priority']:
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
        f.write("## Multi-Processor Results\n\n")
        f.write(f"### System Configuration\n\n")
        
        # Get processor count and strategy from first available metrics
        if multi_metrics:
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
            for key in ['average_waiting_time', 'avg_waiting_time', 'mean_waiting_time']:
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
                for key in ['average_waiting_time', 'avg_waiting_time', 'mean_waiting_time']:
                    if key in metrics:
                        waiting_times[name] = metrics[key]
                        break
            
            if waiting_times:
                best_waiting = min(waiting_times.items(), key=lambda x: x[1])
                f.write(f"- **Best for Waiting Time**: {best_waiting[0]} scheduler had the lowest average waiting time ({best_waiting[1]:.2f}s).\n")
        
        # Compare deadline misses for EDF
        edf_single = single_metrics.get('EDF', {})
        edf_multi = multi_metrics.get('EDF', {})
        
        if 'deadline_misses' in edf_single and 'deadline_misses' in edf_multi:
            single_misses = edf_single['deadline_misses']
            multi_misses = edf_multi.get('deadline_misses', 0)  # Might need to sum across processors
            
            if single_misses != multi_misses:
                better_system = "Multi-Processor" if single_misses > multi_misses else "Single Processor"
                f.write(f"- **Deadline Handling**: {better_system} was better at meeting deadlines with the EDF scheduler.\n")
        
        # General observations
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
                    
                    f.write(f"- **Parallelization Efficiency**: The multi-processor system achieved {efficiency:.1f}% of ideal speedup.\n")
            
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
                    f.write("- **Load Balancing**: Good load distribution, but some processors were underutilized.\n")
                else:
                    f.write("- **Load Balancing**: Poor load distribution, significant processor imbalance.\n")
        
        f.write(f"- **Resource Bottlenecks**: The heatmap analysis reveals patterns in resource utilization that can guide optimization efforts.\n")
        
        f.write("\nThis report provides a quantitative analysis of different scheduling algorithms on both single and multi-processor systems. The accompanying heatmaps provide additional insights into resource utilization and potential bottlenecks.")
    
    return report_path