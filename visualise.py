"""
Enhanced Visualisation Tool for Real-Time Task Scheduling

This script generates comprehensive visualisations for analysing the performance
of different scheduling algorithms in both single and multi-processor configurations. 
It processes data files collected during scheduling simulations and produces
informative charts, heatmaps, and comparative visualisations. 
Usage:
    python visualise.py --data-dir results/data/TIMESTAMP_platform_type
    python visualise.py --data-dir results/data/TIMESTAMP_platform_type --output-dir custom_output
    python visualise.py --data-dir results/data/TIMESTAMP_platform_type --scheduler FCFS
    python visualise.py --cross-platform --platform-dirs dir1 dir2 dir3 
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
from pathlib import Path
import logging
import re
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
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

# Fallback implementation of extract_platform_from_dir if module import fails
def extract_platform_from_dir(data_dir):
    """
    Extract platform type from
    the directory name 

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

# Try importing from src.utils if available
try:
    from src.utils.platform_utils import extract_platform_from_dir
except ImportError:
    logger.info("Using fallback implementation of extract_platform_from_dir")

# ===== Utility Functions =====

def ensure_output_dir(output_path):
    """Ensure the output directory exists"""
    os.makedirs(output_path, exist_ok=True)
    return output_path

def get_priority_color(priority):
    """Get color based on task priority"""
    return PRIORITY_COLORS.get(priority, '#2196F3')  # Default to blue

def get_algorithm_color(algorithm):
    """Get color based on algorithm name"""
    return ALGORITHM_COLORS.get(algorithm, '#607D8B')  # Default to gray

def find_data_files(data_dir, scheduler_name, processor_type='single'):
    """
    Find data files with consistent naming convention

    Args:
        data_dir: Path to the data directory
        scheduler_name: Name of the scheduler
        processor_type: 'single' or 'multi'

    Returns:
        Tuple of (tasks_path, metrics_path, timeseries_path)
    """
    processor_dir = os.path.join(data_dir, f"{processor_type}_processor")

    # Check if directory exists
    if not os.path.exists(processor_dir):
        logger.warning(f"Directory not found: {processor_dir}")
        return None, None, None

    # Define possible file patterns for tasks, metrics, and timeseries
    tasks_patterns = [
        f"{scheduler_name}_tasks.csv",
        f"{scheduler_name.lower()}_tasks.csv",
        f"{scheduler_name.upper()}_tasks.csv"
    ]

    metrics_patterns = [
        f"{scheduler_name}_metrics.json",
        f"{scheduler_name.lower()}_metrics.json",
        f"{scheduler_name.upper()}_metrics.json"
    ]

    timeseries_patterns = [
        f"{scheduler_name}_timeseries.csv",
        f"{scheduler_name.lower()}_timeseries.csv",
        f"{scheduler_name.upper()}_timeseries.csv"
    ]

    # For multi-processor, add patterns for system metrics and All-CPUs tasks
    if processor_type == 'multi':
        metrics_patterns.append(f"{scheduler_name}_system_metrics.json")
        tasks_patterns.append(f"{scheduler_name}_All-CPUs_tasks.csv")
        timeseries_patterns.append(f"{scheduler_name}_System_timeseries.csv")

    # Extract platform from data_dir for legacy file naming
    platform = extract_platform_from_dir(data_dir)
    if platform:
        # Add platform-specific patterns
        tasks_patterns.extend([
            f"{scheduler_name}_{platform}_tasks.csv",
            f"{scheduler_name}_{platform.capitalize()}_tasks.csv"
        ]) 
        metrics_patterns.extend([
            f"{scheduler_name}_{platform}_metrics.json",
            f"{scheduler_name}_{platform.capitalize()}_metrics.json"
        ]) 
        timeseries_patterns.extend([
            f"{scheduler_name}_{platform}_timeseries.csv",
            f"{scheduler_name}_{platform.capitalize()}_timeseries.csv"
        ]) 

    # Find files that match patterns
    tasks_path = None
    metrics_path = None
    timeseries_path = None 

    # Check for task files
    for pattern in tasks_patterns:
        path = os.path.join(processor_dir, pattern)
        if os.path.exists(path): 
            tasks_path = path
            break 

    # Check for metrics files
    for pattern in metrics_patterns:
        path = os.path.join(processor_dir, pattern)
        if os.path.exists(path):
            metrics_path = path
            break 

    # Check for timeseries files
    for pattern in timeseries_patterns: 
        path = os.path.join(processor_dir, pattern)
        if os.path.exists(path):
            timeseries_path = path
            break 

    # For multi-processor, if All-CPUs tasks not found, check for individual CPU tasks
    if processor_type == 'multi' and tasks_path is None:
        task_files = [f for f in os.listdir(processor_dir) 
                     if f.startswith(f"{scheduler_name}_CPU-") and f.endswith("_tasks.csv")]
        if task_files:
            tasks_path = os.path.join(processor_dir, task_files[0])
            logger.info(f"Using {task_files[0]} as representative task file for multi-processor")

    # Log results
    if tasks_path:
        logger.info(f"Found tasks file: {os.path.basename(tasks_path)}")
    else:
        logger.warning(f"No tasks file found for {scheduler_name} in {processor_type} mode")

    if metrics_path:
        logger.info(f"Found metrics file: {os.path.basename(metrics_path)}")
    else:
        logger.warning(f"No metrics file found for {scheduler_name} in {processor_type} mode")

    if timeseries_path:
        logger.info(f"Found timeseries file: {os.path.basename(timeseries_path)}")
    else:
        logger.warning(f"No timeseries file found for {scheduler_name} in {processor_type} mode")

    return tasks_path, metrics_path, timeseries_path

def load_data_files(data_dir, scheduler_name, processor_type='single'):
    """
    Load data files for a specific scheduler and processor type

    Args:
        data_dir: Path to the data directory
        scheduler_name: Name of the scheduler
        processor_type: 'single' or 'multi'

    Returns:
        Tuple of (tasks_df, metrics_dict, timeseries_df)
    """
    tasks_path, metrics_path, timeseries_path = find_data_files(
        data_dir, scheduler_name, processor_type
    )

    tasks_df = None
    metrics_dict = None
    timeseries_df = None

    # Load tasks data if available
    if tasks_path:
        try:
            tasks_df = pd.read_csv(tasks_path)
            logger.info(f"Loaded {len(tasks_df)} task records from {os.path.basename(tasks_path)}") 
        except Exception as e:
            logger.error(f"Error loading tasks file {os.path.basename(tasks_path)}: {e}") 

    # Load metrics data if available
    if metrics_path:
        try:
            with open(metrics_path, 'r') as f:
                metrics_dict = json.load(f)
            logger.info(f"Loaded metrics from {os.path.basename(metrics_path)}") 
        except Exception as e:
            logger.error(f"Error loading metrics file {os.path.basename(metrics_path)}: {e}") 

    # Load timeseries data if available
    if timeseries_path:
        try:
            timeseries_df = pd.read_csv(timeseries_path)
            logger.info(f"Loaded {len(timeseries_df)} timeseries records from {os.path.basename(timeseries_path)}")
        except Exception as e: 
            logger.error(f"Error loading timeseries file {os.path.basename(timeseries_path)}: {e}") 

    # For multi-processor with multiple CPU task files
    if processor_type == 'multi' and tasks_df is None and metrics_dict is not None:
        processor_dir = os.path.join(data_dir, f"{processor_type}_processor")
        task_files = [f for f in os.listdir(processor_dir)
                     if f.startswith(f"{scheduler_name}_CPU-") and f.endswith("_tasks.csv")] 

        if task_files:
            # Combine all task files
            all_tasks = []
            for task_file in task_files: 
                try:
                    df = pd.read_csv(os.path.join(processor_dir, task_file))
                    all_tasks.append(df) 
                except Exception as e:
                    logger.error(f"Error loading task file {task_file}: {e}") 

            if all_tasks:
                tasks_df = pd.concat(all_tasks)
                logger.info(f"Combined {len(tasks_df)} task records from {len(all_tasks)} CPU files") 

    return tasks_df, metrics_dict, timeseries_df 

# ===== Core Visualisation Functions =====

def plot_task_gantt_chart(tasks_df, scheduler_name, output_path=None):
    """
    Create a Gantt chart showing task execution timeline

    Args:
        tasks_df: DataFrame containing task data
        scheduler_name: Name of the scheduler
        output_path: Path to save the plot 
    """
    if tasks_df is None or tasks_df.empty:
        logger.warning(f"No task data available for {scheduler_name} Gantt chart")
        return 

    plt.figure(figsize=(12, 8))

    # Ensure required columns exist
    required_cols = ['id', 'priority', 'arrival_time', 'start_time', 'completion_time']
    missing_cols = [col for col in required_cols if col not in tasks_df.columns] 

    if missing_cols:
        logger.warning(f"Missing columns for Gantt chart: {missing_cols}") 
        return

    # Sort by start time and then by priority value for better visualization
    priority_values = {'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}

    # Add numeric priority for sorting if it doesn't exist
    if 'priority_value' not in tasks_df.columns:
        tasks_df['priority_value'] = tasks_df['priority'].map(priority_values) 

    # Sort tasks for better visualization
    df = tasks_df.sort_values(['start_time', 'priority_value']) 

    # Create Gantt chart with enhanced styling
    for i, task in df.iterrows(): 
        # Plot task execution bar
        plt.barh(
            y=task['id'],
            width=task['completion_time'] - task['start_time'],
            left=task['start_time'],
            color=get_priority_color(task['priority']), 
            edgecolor='black',
            alpha=0.7,
            height=0.5 
        )

        # Add arrival markers with vertical lines
        plt.axvline(
            x=task['arrival_time'],
            ymin=(df['id'].tolist().index(task['id']) - 0.25) / len(df), 
            ymax=(df['id'].tolist().index(task['id']) + 0.25) / len(df),
            color='red',
            linestyle='--',
            alpha=0.6,
            linewidth=1 
        )

    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color=get_priority_color('HIGH'), alpha=0.7, label='HIGH Priority'), 
        plt.Rectangle((0, 0), 1, 1, color=get_priority_color('MEDIUM'), alpha=0.7, label='MEDIUM Priority'),
        plt.Rectangle((0, 0), 1, 1, color=get_priority_color('LOW'), alpha=0.7, label='LOW Priority'),
        plt.Line2D([0], [0], color='red', linestyle='--', label='Arrival Time') 
    ]

    plt.legend(handles=legend_elements, loc='upper right')

    # Set labels and title
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Task ID', fontsize=12)
    plt.title(f'Task Execution Timeline - {scheduler_name}', fontsize=14, fontweight='bold') 

    # Customize grid and styling 
    plt.grid(axis='x', linestyle='--', alpha=0.3)
    plt.tight_layout()

    # Calculate better limit based on data
    x_max = max(tasks_df['completion_time']) * 1.05
    plt.xlim(0, x_max) 

    # Dynamically determine y-axis limits based on data
    task_ids = df['id'].unique()
    if len(task_ids) > 20:
        # If too many tasks, show a subset
        plt.ylim(task_ids[-20], task_ids[0])
        plt.title(f'Task Execution Timeline - {scheduler_name} (Showing 20 of {len(task_ids)} tasks)', 
                 fontsize=14, fontweight='bold') 

    # Save or show the plot
    if output_path:
        ensure_output_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved Gantt chart to {output_path}")
    else:
        plt.show() 

def plot_waiting_time_distribution(tasks_df, scheduler_name, output_path=None):
    """
    Create a distribution plot of waiting times by priority

    Args:
        tasks_df: DataFrame containing task data
        scheduler_name: Name of the scheduler
        output_path: Path to save the plot 
    """
    if tasks_df is None or tasks_df.empty:
        logger.warning(f"No task data available for {scheduler_name} waiting time distribution")
        return 

    # Check for required columns
    if 'waiting_time' not in tasks_df.columns or 'priority' not in tasks_df.columns: 
        logger.warning(f"Missing required columns for waiting time distribution")
        return

    # Drop rows with missing waiting times
    df = tasks_df.dropna(subset=['waiting_time'])

    if df.empty:
        logger.warning(f"No valid waiting time data for {scheduler_name}")
        return 

    plt.figure(figsize=(10, 8)) 

    # Create violin plots grouped by priority
    ax = sns.violinplot(
        x='priority',
        y='waiting_time',
        data=df,
        order=['HIGH', 'MEDIUM', 'LOW'],
        palette={
            'HIGH': get_priority_color('HIGH'),
            'MEDIUM': get_priority_color('MEDIUM'), 
            'LOW': get_priority_color('LOW')
        },
        inner='box',
        cut=0 
    )

    # Add individual points with jitter
    sns.stripplot(
        x='priority',
        y='waiting_time',
        data=df,
        order=['HIGH', 'MEDIUM', 'LOW'],
        color='black', 
        alpha=0.4,
        jitter=True,
        size=3 
    )

    # Calculate and display average waiting time
    avg_waiting_time = df['waiting_time'].mean()
    plt.axhline(
        y=avg_waiting_time,
        color='red',
        linestyle='--',
        linewidth=1,
        label=f'Average: {avg_waiting_time:.2f}s' 
    )

    # Calculate and display average waiting time per priority
    for i, priority in enumerate(['HIGH', 'MEDIUM', 'LOW']): 
        avg = df[df['priority'] == priority]['waiting_time'].mean()
        if not pd.isna(avg):
            plt.text(
                i,
                avg + 0.1, 
                f'{avg:.2f}s',
                ha='center',
                va='bottom',
                fontweight='bold',
                color='darkred' 
            )

    # Set labels and title
    plt.xlabel('Task Priority', fontsize=12) 
    plt.ylabel('Waiting Time (s)', fontsize=12)
    plt.title(f'Waiting Time Distribution by Priority - {scheduler_name}',
             fontsize=14, fontweight='bold')

    # Add legend and customize styling
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout() 

    # Save or show the plot
    if output_path:
        ensure_output_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close() 
        logger.info(f"Saved waiting time distribution to {output_path}")
    else:
        plt.show() 

def plot_resource_utilization(timeseries_df, scheduler_name, output_path=None):
    """
    Create a dual-axis plot showing memory usage and queue length over time

    Args:
        timeseries_df: DataFrame containing timeseries data
        scheduler_name: Name of the scheduler
        output_path: Path to save the plot 
    """
    if timeseries_df is None or timeseries_df.empty:
        logger.warning(f"No timeseries data available for {scheduler_name} resource utilization")
        return 

    # Check for required columns
    has_memory = 'memory_usage' in timeseries_df.columns
    has_queue = 'queue_length' in timeseries_df.columns
    has_time = 'time' in timeseries_df.columns

    if not has_time:
        logger.warning(f"Missing time column for resource utilization plot") 
        return

    if not has_memory and not has_queue: 
        logger.warning(f"Missing both memory_usage and queue_length columns")
        return

    plt.figure(figsize=(12, 6))

    # Create primary axis for memory usage
    ax1 = plt.gca() 

    if has_memory:
        memory_line = ax1.plot(
            timeseries_df['time'], 
            timeseries_df['memory_usage'],
            color='#FF5722',  # Orange
            linewidth=2,
            marker='o',
            markersize=4,
            label='Memory Usage (%)' 
        )

        ax1.set_xlabel('Time (s)', fontsize=12) 
        ax1.set_ylabel('Memory Usage (%)', fontsize=12, color='#FF5722')
        ax1.tick_params(axis='y', labelcolor='#FF5722')
        ax1.set_ylim(bottom=0) 

    # Create secondary axis for queue length if available
    if has_queue:
        ax2 = ax1.twinx() if has_memory else ax1

        queue_line = ax2.plot(
            timeseries_df['time'], 
            timeseries_df['queue_length'],
            color='#2196F3',  # Blue
            linewidth=2,
            marker='s',
            markersize=4,
            label='Queue Length' 
        )

        if has_memory: 
            ax2.set_ylabel('Queue Length (tasks)', fontsize=12, color='#2196F3')
            ax2.tick_params(axis='y', labelcolor='#2196F3') 
        else:
            ax2.set_xlabel('Time (s)', fontsize=12)
            ax2.set_ylabel('Queue Length (tasks)', fontsize=12)

        ax2.set_ylim(bottom=0) 

        # Force y-axis to use integers for queue length 
        ax2.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Add title and grid
    plt.title(f'Resource Utilization Over Time - {scheduler_name}',
             fontsize=14, fontweight='bold')

    ax1.grid(True, linestyle='--', alpha=0.3) 

    # Add legend
    if has_memory and has_queue:
        lines = memory_line + queue_line
        labels = [line.get_label() for line in lines] 
        ax1.legend(lines, labels, loc='upper right')
    elif has_memory:
        ax1.legend(loc='upper right')
    elif has_queue:
        ax2.legend(loc='upper right') 

    plt.tight_layout()

    # Save or show the plot
    if output_path:
        ensure_output_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved resource utilization plot to {output_path}") 
    else:
        plt.show() 

def plot_priority_pie_chart(tasks_df, scheduler_name, output_path=None):
    """
    Create a pie chart showing the distribution of tasks by priority

    Args:
        tasks_df: DataFrame containing task data
        scheduler_name: Name of the scheduler
        output_path: Path to save the plot 
    """
    if tasks_df is None or tasks_df.empty:
        logger.warning(f"No task data available for {scheduler_name} priority pie chart") 
        return

    # Check for required column
    if 'priority' not in tasks_df.columns:
        logger.warning(f"Missing priority column for pie chart")
        return 

    # Count tasks by priority
    priority_counts = tasks_df['priority'].value_counts()

    # Ensure we have HIGH, MEDIUM, LOW priorities (even if they have zero tasks)
    for priority in ['HIGH', 'MEDIUM', 'LOW']: 
        if priority not in priority_counts:
            priority_counts[priority] = 0

    # Sort by priority for consistent order
    priority_order = ['HIGH', 'MEDIUM', 'LOW']
    priority_counts = priority_counts.reindex(priority_order).fillna(0) 

    plt.figure(figsize=(8, 8))

    # Create pie chart with enhanced styling
    wedges, texts, autotexts = plt.pie(
        priority_counts, 
        labels=priority_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=[get_priority_color(p) for p in priority_counts.index],
        wedgeprops={'edgecolor': 'w', 'linewidth': 1},
        shadow=True,
        explode=(0.05, 0, 0)  # Slightly explode the HIGH priority slice 
    )

    # Customize text appearance
    for text in texts:
        text.set_fontsize(12) 

    for autotext in autotexts:
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')
        autotext.set_color('white') 

    # Add task counts to labels
    for i, wedge in enumerate(wedges):
        priority = priority_counts.index[i]
        count = priority_counts[priority]
        wedge.set_label(f"{priority} - {count} tasks") 

    # Add title and legend 
    plt.title(f'Task Distribution by Priority - {scheduler_name}',
             fontsize=14, fontweight='bold')

    plt.legend(
        title="Priorities",
        loc="best",
        fontsize=10 
    )

    plt.tight_layout()

    # Save or show the plot
    if output_path:
        ensure_output_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight') 
        plt.close()
        logger.info(f"Saved priority pie chart to {output_path}")
    else:
        plt.show() 

def create_resource_heatmap(timeseries_df, scheduler_name, output_path=None):
    """
    Create a heatmap showing resource usage patterns over time

    Args:
        timeseries_df: DataFrame containing timeseries data
        scheduler_name: Name of the scheduler
        output_path: Path to save the plot 
    """
    if timeseries_df is None or timeseries_df.empty:
        logger.warning(f"No timeseries data available for {scheduler_name} resource heatmap")
        return 

    # Check for required columns
    required_cols = ['time']
    resource_cols = []

    if 'memory_usage' in timeseries_df.columns:
        resource_cols.append('memory_usage') 

    if 'queue_length' in timeseries_df.columns:
        resource_cols.append('queue_length') 

    if not resource_cols:
        logger.warning(f"No resource metrics found for heatmap")
        return

    # If we have cpu_usage in the metrics, add it
    if 'cpu_usage' in timeseries_df.columns:
        resource_cols.append('cpu_usage') 

    # Normalize the time scale by creating time bins
    max_time = timeseries_df['time'].max()
    time_bins = 10  # Number of time bins 

    # Create time bin edges
    time_edges = np.linspace(0, max_time, time_bins + 1) 

    # Assign each time point to a bin
    timeseries_df['time_bin'] = pd.cut(
        timeseries_df['time'],
        bins=time_edges,
        labels=range(time_bins),
        include_lowest=True 
    )

    # Create a matrix for the heatmap with shape (len(resource_cols), time_bins)
    heatmap_data = np.zeros((len(resource_cols), time_bins)) 

    # Fill the heatmap data
    for bin_idx in range(time_bins):
        bin_data = timeseries_df[timeseries_df['time_bin'] == bin_idx]

        for res_idx, resource in enumerate(resource_cols): 
            if not bin_data.empty:
                # Calculate the average value for this resource in this time bin
                avg_value = bin_data[resource].mean() 
                heatmap_data[res_idx, bin_idx] = avg_value

    # Create bin labels for the x-axis
    time_bin_labels = [f"{time_edges[i]:.1f}-{time_edges[i+1]:.1f}s" for i in range(time_bins)] 

    # Nice human-readable labels for resources
    resource_labels = {
        'memory_usage': 'Memory Usage',
        'queue_length': 'Queue Length', 
        'cpu_usage': 'CPU Usage'
    } 

    # Get the proper resource labels
    y_labels = [resource_labels.get(res, res) for res in resource_cols]

    plt.figure(figsize=(12, len(resource_cols) * 2)) 

    # Choose an appropriate color map based on the data
    if 'memory_usage' in resource_cols and 'queue_length' in resource_cols:
        # Use a custom colormap that goes from green to yellow to red
        cmap = LinearSegmentedColormap.from_list( 
            'GYR', [(0, 'green'), (0.5, 'yellow'), (1, 'red')]
        ) 
    else:
        # For memory usage alone, use a blue to red colormap
        cmap = 'YlOrRd'

    # Create the heatmap
    ax = sns.heatmap(
        heatmap_data,
        annot=True, 
        fmt=".1f",
        cmap=cmap,
        linewidths=0.5,
        cbar_kws={'label': 'Resource Usage (%)'} 
    )

    # Set labels and title
    plt.xlabel('Time Period', fontsize=12)
    plt.ylabel('Resource', fontsize=12)
    plt.title(f'Resource Usage Heatmap - {scheduler_name}',
             fontsize=14, fontweight='bold') 

    # Set axis labels
    ax.set_yticklabels(y_labels, rotation=0)
    ax.set_xticklabels(time_bin_labels, rotation=45, ha='right') 

    plt.tight_layout()

    # Save or show the plot
    if output_path:
        ensure_output_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved resource heatmap to {output_path}")
    else:
        plt.show() 

def plot_cpu_utilization_heatmap(metrics_dict, scheduler_name, output_path=None):
    """
    Create a heatmap showing CPU utilization across multiple processors over time
    Args:
        metrics_dict: Dictionary containing metrics including per_processor_metrics
        scheduler_name: Name of the scheduler
        output_path: Path to save the plot 
    """
    if metrics_dict is None or 'per_processor_metrics' not in metrics_dict:
        logger.warning(f"No per-processor metrics available for {scheduler_name} CPU heatmap")
        return

    per_processor_metrics = metrics_dict['per_processor_metrics']
    if not per_processor_metrics: 
        logger.warning(f"Empty per-processor metrics for {scheduler_name}")
        return 

    # Determine the number of processors
    processor_count = len(per_processor_metrics)
    logger.info(f"Found {processor_count} processors in metrics") 

    # Extract CPU usage data for each processor
    cpu_usage_data = []
    timestamps = []

    # Try to find timestamps first
    for proc_metrics in per_processor_metrics:
        if 'timestamp' in proc_metrics and isinstance(proc_metrics['timestamp'], list): 
            timestamps = proc_metrics['timestamp']
            break

    if not timestamps and 'timestamp_history' in metrics_dict:
        timestamps = metrics_dict['timestamp_history']

    if not timestamps:
        logger.warning(f"No timestamp data found for {scheduler_name} CPU heatmap")
        return 

    # Convert timestamps to relative time 
    if isinstance(timestamps[0], (int, float)) and timestamps[0] > 1000000000:  # Unix timestamp
        start_time = timestamps[0]
        relative_times = [t - start_time for t in timestamps]
    else:
        relative_times = timestamps

    # Extract CPU usage for each processor
    for proc_idx, proc_metrics in enumerate(per_processor_metrics): 
        cpu_usage = []

        # Try different possible keys for CPU usage
        if 'cpu_usage_history' in proc_metrics: 
            cpu_usage = proc_metrics['cpu_usage_history']
        elif 'cpu_usage' in proc_metrics and isinstance(proc_metrics['cpu_usage'], list):
            cpu_usage = proc_metrics['cpu_usage']

        # Ensure data is the same length as timestamps
        if cpu_usage: 
            # Truncate or pad to match timestamps length
            if len(cpu_usage) > len(relative_times): 
                cpu_usage = cpu_usage[:len(relative_times)]
            elif len(cpu_usage) < len(relative_times):
                # Extend with the last value or zero
                last_value = cpu_usage[-1] if cpu_usage else 0 
                cpu_usage.extend([last_value] * (len(relative_times) - len(cpu_usage)))
        else:
            # If no data, use zeros
            cpu_usage = [0] * len(relative_times)

        cpu_usage_data.append(cpu_usage) 

    # Create time bins for better visualization 
    max_time = max(relative_times)
    time_bins = 10

    # Create time bin edges
    time_edges = np.linspace(0, max_time, time_bins + 1)
    time_bin_labels = [f"{time_edges[i]:.1f}-{time_edges[i+1]:.1f}s" for i in range(time_bins)] 

    # Create a matrix for the heatmap
    heatmap_data = np.zeros((processor_count, time_bins))

    # Fill the heatmap data
    for proc_idx in range(processor_count):
        for time_idx, time_val in enumerate(relative_times): 
            # Determine which bin this time belongs to
            bin_idx = min(time_bins - 1, int(time_val / max_time * time_bins)) 

            # Add CPU usage to the appropriate bin
            if time_idx < len(cpu_usage_data[proc_idx]):
                heatmap_data[proc_idx, bin_idx] += cpu_usage_data[proc_idx][time_idx] 

        # Calculate average for each bin
        for bin_idx in range(time_bins):
            # Count points in this bin
            bin_count = sum(1 for t in relative_times if time_edges[bin_idx] <= t < time_edges[bin_idx + 1])
            if bin_count > 0:
                heatmap_data[proc_idx, bin_idx] /= bin_count 

    # Create the heatmap
    plt.figure(figsize=(12, processor_count * 0.8 + 2))

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

    # Set labels and title
    plt.xlabel('Time Period', fontsize=12)
    plt.ylabel('Processor', fontsize=12)
    plt.title(f'CPU Utilization Heatmap - {scheduler_name}',
             fontsize=14, fontweight='bold') 

    # Set y-tick labels (processor names)
    processor_labels = [f"CPU-{i+1}" for i in range(processor_count)]
    ax.set_yticklabels(processor_labels, rotation=0)

    # Set x-tick labels
    ax.set_xticklabels(time_bin_labels, rotation=45, ha='right') 

    plt.tight_layout()

    # Save or show the plot
    if output_path:
        ensure_output_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved CPU utilization heatmap to {output_path}")
    else:
        plt.show() 

def plot_task_density_heatmap(tasks_df, scheduler_name, output_path=None):
    """
    Create a heatmap showing task density by priority level over time 

    Args:
        tasks_df: DataFrame containing task data
        scheduler_name: Name of the scheduler
        output_path: Path to save the plot 
    """
    if tasks_df is None or tasks_df.empty:
        logger.warning(f"No task data available for {scheduler_name} task density heatmap")
        return

    # Check for required columns
    required_cols = ['priority', 'start_time', 'completion_time'] 
    missing_cols = [col for col in required_cols if col not in tasks_df.columns] 

    if missing_cols:
        logger.warning(f"Missing columns for task density heatmap: {missing_cols}")
        return

    # Filter out invalid data
    df = tasks_df.dropna(subset=['start_time', 'completion_time']) 

    if df.empty:
        logger.warning(f"No valid task execution data for {scheduler_name}")
        return 

    # Define priority levels and ensure they exist in the data
    priority_levels = ['HIGH', 'MEDIUM', 'LOW']

    # Create time bins based on task execution range
    min_time = df['start_time'].min()
    max_time = df['completion_time'].max() 

    time_bins = 10
    time_edges = np.linspace(min_time, max_time, time_bins + 1)
    time_bin_labels = [f"{time_edges[i]:.1f}-{time_edges[i+1]:.1f}s" for i in range(time_bins)]

    # Create a matrix for the heatmap
    heatmap_data = np.zeros((len(priority_levels), time_bins)) 

    # Fill the heatmap data - count active tasks in each bin by priority
    for _, task in df.iterrows():
        # Skip if priority not in our list
        if task['priority'] not in priority_levels:
            continue

        priority_idx = priority_levels.index(task['priority']) 

        # Determine which bins this task spans
        start_bin = np.searchsorted(time_edges, task['start_time']) - 1
        end_bin = np.searchsorted(time_edges, task['completion_time']) - 1 

        # Clip to valid bin indices
        start_bin = max(0, min(start_bin, time_bins - 1))
        end_bin = max(0, min(end_bin, time_bins - 1))

        # Increment counts for all bins this task spans
        for bin_idx in range(start_bin, end_bin + 1): 
            heatmap_data[priority_idx, bin_idx] += 1

    plt.figure(figsize=(12, 6))

    # Create the heatmap
    ax = sns.heatmap(
        heatmap_data,
        cmap="viridis",
        annot=True,
        fmt="g", 
        linewidths=0.5,
        cbar_kws={'label': 'Number of Active Tasks'} 
    )

    # Set labels and title
    plt.xlabel('Time Period', fontsize=12)
    plt.ylabel('Priority Level', fontsize=12)
    plt.title(f'Task Density Heatmap - {scheduler_name}',
             fontsize=14, fontweight='bold') 

    # Set y-tick labels
    ax.set_yticklabels(priority_levels, rotation=0)

    # Set x-tick labels
    ax.set_xticklabels(time_bin_labels, rotation=45, ha='right') 

    plt.tight_layout()

    # Save or show the plot
    if output_path:
        ensure_output_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight') 
        plt.close()
        logger.info(f"Saved task density heatmap to {output_path}")
    else:
        plt.show() 

# ===== Comparison Visualisation Functions =====

def plot_performance_comparison(metrics_by_algorithm, metric_name, title, ylabel, output_path=None):
    """
    Create a bar chart comparing a specific metric across different algorithms

    Args:
        metrics_by_algorithm: Dictionary mapping algorithm names to their metrics
        metric_name: Name of the metric to compare
        title: Plot title
        ylabel: Y-axis label
        output_path: Path to save the plot 
    """
    if not metrics_by_algorithm:
        logger.warning(f"No metrics data available for comparison: {metric_name}")
        return

    # Standardized key mappings for common metrics
    key_mappings = {
        'avg_waiting_time': ['avg_waiting_time', 'average_waiting_time', 'mean_waiting_time'],
        'throughput': ['system_throughput', 'avg_throughput', 'throughput', 'tasks_per_second'],
        'deadline_misses': ['deadline_misses'],
        'priority_inversions': ['priority_inversions'], 
        'cpu_usage': ['avg_cpu_usage'],
        'memory_usage': ['avg_memory_usage']
    } 

    # Extract values for each algorithm
    algorithm_names = []
    metric_values = []
    colors = []

    for algo, metrics in metrics_by_algorithm.items(): 
        if metrics is not None:
            # Try different possible keys for this metric
            value = None 
            for key in key_mappings.get(metric_name, [metric_name]):
                if key in metrics:
                    value = metrics[key]
                    break

            if value is not None: 
                algorithm_names.append(algo)
                metric_values.append(float(value))
                colors.append(get_algorithm_color(algo))

    if not algorithm_names:
        logger.warning(f"No data found for metric: {metric_name}")
        return 

    plt.figure(figsize=(10, 6)) 

    # Create bar chart
    bars = plt.bar(
        algorithm_names,
        metric_values,
        color=colors,
        alpha=0.8,
        edgecolor='black',
        linewidth=1.5 
    )

    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height() 
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.02 * max(metric_values),
            f'{metric_values[i]:.2f}',
            ha='center',
            va='bottom',
            fontweight='bold' 
        )

    # Set labels and title
    plt.xlabel('Scheduling Algorithm', fontsize=12) 
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')

    # Customize grid and styling
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # Determine if lower values are better for this metric
    lower_is_better = metric_name in ['avg_waiting_time', 'deadline_misses', 'priority_inversions', 
                                      'cpu_usage', 'memory_usage'] 

    # Highlight the best algorithm
    if lower_is_better:
        best_idx = metric_values.index(min(metric_values))
    else:
        best_idx = metric_values.index(max(metric_values))

    bars[best_idx].set_edgecolor('green')
    bars[best_idx].set_linewidth(2.5) 

    # Add 'best' annotation
    plt.annotate(
        'BEST',
        xy=(best_idx, metric_values[best_idx]),
        xytext=(best_idx, metric_values[best_idx] + 0.1 * max(metric_values)), 
        ha='center',
        fontweight='bold',
        color='green',
        arrowprops=dict(facecolor='green', shrink=0.05, width=2, headwidth=10)
    )

    plt.tight_layout()

    # Save or show the plot
    if output_path:
        ensure_output_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved performance comparison for {metric_name} to {output_path}") 
    else:
        plt.show() 

def create_radar_chart(metrics_by_algorithm, output_path=None):
    """
    Create a radar chart comparing algorithms across multiple metrics

    Args:
        metrics_by_algorithm: Dictionary mapping algorithm names to their metrics
        output_path: Path to save the plot 
    """
    if not metrics_by_algorithm:
        logger.warning(f"No metrics data available for radar chart") 
        return

    # Define metrics to include in the radar chart
    metrics_config = [
        ('avg_waiting_time', 'Waiting Time', True),  # True = lower is better
        ('throughput', 'Throughput', False),  # False = higher is better
        ('avg_cpu_usage', 'CPU Usage', True),
        ('avg_memory_usage', 'Memory Usage', True)
    ] 

    # Key mappings for metric names
    key_mappings = {
        'avg_waiting_time': ['avg_waiting_time', 'average_waiting_time', 'mean_waiting_time'],
        'throughput': ['system_throughput', 'avg_throughput', 'throughput', 'tasks_per_second'],
        'avg_cpu_usage': ['avg_cpu_usage', 'cpu_usage', 'average_cpu_usage'],
        'avg_memory_usage': ['avg_memory_usage', 'memory_usage', 'average_memory_usage']
    } 

    # Collect data for each algorithm
    data = {}
    metric_ranges = {metric_name: {'min': float('inf'), 'max': float('-inf')}
                     for metric_name, _, _ in metrics_config} 

    for algo, metrics in metrics_by_algorithm.items():
        if metrics is None:
            continue

        algo_data = {} 

        for metric_name, _, _ in metrics_config:
            # Try different possible keys for this metric
            value = None 
            for key in key_mappings.get(metric_name, [metric_name]):
                if key in metrics:
                    value = metrics[key]
                    break 

            if value is not None:
                try:
                    value = float(value)
                    algo_data[metric_name] = value 

                    # Update min/max for normalization
                    metric_ranges[metric_name]['min'] = min(metric_ranges[metric_name]['min'], value)
                    metric_ranges[metric_name]['max'] = max(metric_ranges[metric_name]['max'], value) 
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert {value} to float for {algo}, {metric_name}") 
                    continue

        if algo_data:
            data[algo] = algo_data

    if not data:
        logger.warning("No valid data for radar chart") 
        return

    # Extract metric names and labels
    metric_names = [name for name, _, _ in metrics_config]
    metric_labels = [label for _, label, _ in metrics_config]
    lower_is_better = [flag for _, _, flag in metrics_config] 

    # Normalize values to range [0, 1] where higher is always better
    normalized_data = {}

    for algo, algo_data in data.items():
        normalized_algo_data = {} 

        for i, (metric_name, _, lower_better) in enumerate(metrics_config):
            if metric_name in algo_data:
                value = algo_data[metric_name]
                min_val = metric_ranges[metric_name]['min']
                max_val = metric_ranges[metric_name]['max'] 

                # Skip normalization if min == max
                if min_val == max_val: 
                    normalized_value = 0.5  # Middle value
                else:
                    # Normalize to [0, 1]
                    normalized_value = (value - min_val) / (max_val - min_val) 

                    # Invert if lower is better
                    if lower_better: 
                        normalized_value = 1 - normalized_value

                normalized_algo_data[metric_name] = normalized_value
            else:
                normalized_algo_data[metric_name] = 0  # Default if missing 

        normalized_data[algo] = normalized_algo_data

    # Create the radar chart
    plt.figure(figsize=(10, 10)) 

    # Setup radar chart
    ax = plt.subplot(111, polar=True)

    # Number of metrics
    num_metrics = len(metric_names)
    angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()

    # Make the plot circular by appending the first angle 
    angles += angles[:1]
    metric_labels += [metric_labels[0]]

    # Draw the background circles
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels[:-1]) 

    # Set y limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], color='grey')

    # Draw the grid lines
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.grid(True, linestyle='--', alpha=0.5) 

    # Plot each algorithm
    for algo, algo_data in normalized_data.items(): 
        # Get values for each metric
        values = [algo_data.get(metric, 0) for metric in metric_names]

        # Close the polygon
        values += values[:1]

        # Plot values
        ax.plot(angles, values, linewidth=2, linestyle='-', label=algo, color=get_algorithm_color(algo)) 
        ax.fill(angles, values, alpha=0.25, color=get_algorithm_color(algo)) 

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    plt.title('Algorithm Performance Comparison', fontsize=15, y=1.1) 

    # Add explanation of the normalization
    plt.figtext(0.5, 0.01,
               'Note: All metrics are normalized to 0-1 scale. For metrics where lower is better\n' 
               '(waiting time, CPU/memory usage), the scale is inverted so that higher values\n'
               'on the radar chart always represent better performance.',
               ha='center', fontsize=9, bbox=dict(facecolor='lightyellow', alpha=0.5)) 

    # Save or show the plot
    if output_path:
        ensure_output_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight') 
        plt.close()
        logger.info(f"Saved radar chart to {output_path}")
    else:
        plt.show() 

def plot_processor_comparison(single_metrics, multi_metrics, output_path=None):
    """
    Create a comparison of single vs multi-processor performance

    Args:
        single_metrics: Dictionary with single processor metrics by algorithm
        multi_metrics: Dictionary with multi-processor metrics by algorithm
        output_path: Path to save the plot 
    """
    if not single_metrics or not multi_metrics:
        logger.warning("Insufficient data for processor comparison")
        return

    # Find common algorithms across both processor types
    common_algos = set(single_metrics.keys()) & set(multi_metrics.keys()) 

    if not common_algos:
        logger.warning("No common algorithms for processor comparison")
        return

    # Choose a representative algorithm (prefer Priority or EDF)
    chosen_algo = None 
    for algo in ['Priority', 'EDF', 'FCFS', 'ML-Based']:
        if algo in common_algos:
            chosen_algo = algo
            break

    if chosen_algo is None:
        chosen_algo = next(iter(common_algos)) 

    # Get metrics for the chosen algorithm
    single_data = single_metrics[chosen_algo] 
    multi_data = multi_metrics[chosen_algo]

    if single_data is None or multi_data is None:
        logger.warning(f"Missing data for {chosen_algo} processor comparison")
        return 

    # Create a figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Throughput Comparison (top left)
    ax1 = axs[0, 0] 

    # Extract throughput values 
    single_throughput = 0
    multi_throughput = 0

    for key in ['system_throughput', 'avg_throughput', 'throughput', 'tasks_per_second']:
        if key in single_data:
            single_throughput = single_data[key]
            break 

    for key in ['system_throughput', 'avg_throughput', 'throughput', 'tasks_per_second']:
        if key in multi_data:
            multi_throughput = multi_data[key] 
            break

    bars1 = ax1.bar(
        ['Single Processor', 'Multi-Processor'],
        [single_throughput, multi_throughput],
        color=['skyblue', 'orange'],
        alpha=0.8,
        edgecolor='black' 
    )

    # Add value labels
    for bar in bars1:
        height = bar.get_height() 
        ax1.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.01 * max(single_throughput, multi_throughput),
            f'{height:.2f}',
            ha='center',
            va='bottom',
            fontweight='bold' 
        )

    # Calculate speedup
    if single_throughput > 0: 
        speedup = multi_throughput / single_throughput
        ax1.text(
            0.5,
            0.8 * max(single_throughput, multi_throughput),
            f'Speedup: {speedup:.2f}x',
            ha='center',
            fontweight='bold', 
            bbox=dict(facecolor='yellow', alpha=0.3) 
        )

    ax1.set_ylabel('Tasks per second')
    ax1.set_title('Throughput Comparison')
    ax1.grid(axis='y', linestyle='--', alpha=0.3) 

    # 2. Waiting Time Comparison (top right)
    ax2 = axs[0, 1]

    # Extract waiting time values
    single_wait = 0
    multi_wait = 0 

    for key in ['avg_waiting_time', 'average_waiting_time', 'mean_waiting_time']:
        if key in single_data: 
            single_wait = single_data[key]
            break

    for key in ['avg_waiting_time', 'average_waiting_time', 'mean_waiting_time']:
        if key in multi_data:
            multi_wait = multi_data[key]
            break 

    bars2 = ax2.bar(
        ['Single Processor', 'Multi-Processor'], 
        [single_wait, multi_wait],
        color=['skyblue', 'orange'],
        alpha=0.8,
        edgecolor='black' 
    )

    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width()/2., 
            height + 0.01 * max(single_wait, multi_wait), 
            f'{height:.2f}s',
            ha='center',
            va='bottom',
            fontweight='bold'
        )

    # Calculate improvement percentage
    if single_wait > 0:
        improvement = ((single_wait - multi_wait) / single_wait) * 100 
        ax2.text(
            0.5,
            0.8 * max(single_wait, multi_wait),
            f'Improvement: {improvement:.1f}%',
            ha='center',
            fontweight='bold',
            bbox=dict(facecolor='yellow', alpha=0.3)
        )

    ax2.set_ylabel('Waiting Time (s)') 
    ax2.set_title('Average Waiting Time Comparison')
    ax2.grid(axis='y', linestyle='--', alpha=0.3)

    # 3. Resource Usage Comparison (bottom left)
    ax3 = axs[1, 0]

    # Extract CPU and memory usage
    single_cpu = single_data.get('avg_cpu_usage', 0)
    multi_cpu = multi_data.get('avg_cpu_usage', 0)
    single_mem = single_data.get('avg_memory_usage', 0)
    multi_mem = multi_data.get('avg_memory_usage', 0) 

    # Bar positions
    x = np.arange(2)
    width = 0.35 

    # Create grouped bars
    bars3_1 = ax3.bar(x - width/2, [single_cpu, single_mem], width, label='Single Processor', color='skyblue')
    bars3_2 = ax3.bar(x + width/2, [multi_cpu, multi_mem], width, label='Multi-Processor', color='orange') 

    # Add value labels
    for bars in [bars3_1, bars3_2]:
        for bar in bars:
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width()/2., 
                height + 1,
                f'{height:.1f}%',
                ha='center',
                va='bottom'
            )

    ax3.set_ylabel('Usage (%)') 
    ax3.set_title('Resource Usage Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['CPU Usage', 'Memory Usage'])
    ax3.legend()
    ax3.grid(axis='y', linestyle='--', alpha=0.3)

    # 4. Waiting Time by Priority (bottom right)
    ax4 = axs[1, 1] 

    # Extract waiting times by priority
    single_wait_by_priority = {}
    multi_wait_by_priority = {}

    for key in ['avg_waiting_by_priority', 'waiting_times_by_priority']:
        if key in single_data:
            single_wait_by_priority = single_data[key] 
            break

    for key in ['avg_waiting_by_priority', 'waiting_times_by_priority']:
        if key in multi_data:
            multi_wait_by_priority = multi_data[key]
            break 

    # If we have priority data, plot it
    if single_wait_by_priority and multi_wait_by_priority:
        priorities = ['HIGH', 'MEDIUM', 'LOW'] 
        single_values = [single_wait_by_priority.get(p, 0) for p in priorities]
        multi_values = [multi_wait_by_priority.get(p, 0) for p in priorities]

        x = np.arange(len(priorities))
        width = 0.35

        bars4_1 = ax4.bar(x - width/2, single_values, width, label='Single Processor', color='skyblue')
        bars4_2 = ax4.bar(x + width/2, multi_values, width, label='Multi-Processor', color='orange') 

        # Add value labels
        for bars in [bars4_1, bars4_2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0: 
                    ax4.text( 
                        bar.get_x() + bar.get_width()/2.,
                        height + 0.1,
                        f'{height:.1f}s',
                        ha='center', 
                        va='bottom'
                    )

        ax4.set_ylabel('Waiting Time (s)')
        ax4.set_title('Waiting Time by Priority')
        ax4.set_xticks(x)
        ax4.set_xticklabels(priorities)
        ax4.legend() 
        ax4.grid(axis='y', linestyle='--', alpha=0.3)
    else:
        ax4.text(
            0.5, 0.5,
            'No priority-specific data available',
            ha='center',
            va='center',
            fontsize=12,
            transform=ax4.transAxes 
        )

    # Add overall title
    plt.suptitle(
        f'Single vs Multi-Processor Performance Comparison - {chosen_algo} Scheduler',
        fontsize=16,
        fontweight='bold' 
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for suptitle

    # Save or show the plot
    if output_path:
        ensure_output_dir(os.path.dirname(output_path)) 
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved processor comparison to {output_path}")
    else:
        plt.show() 

def plot_cross_platform_comparison(platform_data, metric_name, processor_type, title, ylabel, output_path=None):
    """
    Create a bar chart comparing a specific metric across different platforms
    Args: 
        platform_data: Dictionary mapping platform names to metrics by algorithm
        metric_name: Name of the metric to compare
        processor_type: 'single' or 'multi'
        title: Plot title
        ylabel: Y-axis label
        output_path: Path to save the plot 
    """
    if not platform_data:
        logger.warning(f"No platform data available for comparison: {metric_name}") 
        return

    # Standardized key mappings for common metrics
    key_mappings = {
        'avg_waiting_time': ['avg_waiting_time', 'average_waiting_time', 'mean_waiting_time'],
        'throughput': ['system_throughput', 'avg_throughput', 'throughput', 'tasks_per_second'],
        'deadline_misses': ['deadline_misses'],
        'priority_inversions': ['priority_inversions'],
        'cpu_usage': ['avg_cpu_usage'],
        'memory_usage': ['avg_memory_usage']
    } 

    # Find common algorithms across all platforms
    common_algos = set() 
    first = True

    for platform, metrics in platform_data.items():
        if processor_type not in metrics or not metrics[processor_type]:
            continue

        # Get algorithms available for this platform
        platform_algos = set(metrics[processor_type].keys()) 

        if first: 
            common_algos = platform_algos
            first = False
        else:
            common_algos = common_algos.intersection(platform_algos)

    if not common_algos:
        logger.warning(f"No common algorithms across platforms for {processor_type} processor")
        return 

    # Select the algorithm to compare (prioritize EDF, then Priority, then FCFS)
    selected_algo = None 
    for algo in ['EDF', 'Priority', 'FCFS', 'ML-Based']:
        if algo in common_algos:
            selected_algo = algo
            break

    if not selected_algo:
        selected_algo = next(iter(common_algos))

    logger.info(f"Using {selected_algo} algorithm for cross-platform comparison") 

    # Extract metric values for each platform
    platform_names = []
    metric_values = []

    for platform, metrics in platform_data.items(): 
        if processor_type not in metrics or not metrics[processor_type]:
            continue

        if selected_algo not in metrics[processor_type]:
            continue 

        algo_metrics = metrics[processor_type][selected_algo]

        # Try different possible keys for this metric
        value = None
        for key in key_mappings.get(metric_name, [metric_name]):
            if key in algo_metrics:
                value = algo_metrics[key] 
                break

        if value is not None:
            platform_names.append(platform)
            metric_values.append(float(value)) 

    if not platform_names:
        logger.warning(f"No data found for metric: {metric_name}")
        return

    plt.figure(figsize=(12, 6)) 

    # Create bar chart with platform colors
    colors = plt.cm.viridis(np.linspace(0, 1, len(platform_names)))

    bars = plt.bar(
        platform_names,
        metric_values,
        color=colors,
        alpha=0.8,
        edgecolor='black',
        linewidth=1.5 
    )

    # Add value labels on top of bars 
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.02 * max(metric_values),
            f'{metric_values[i]:.2f}',
            ha='center',
            va='bottom', 
            fontweight='bold' 
        )

    # Set labels and title
    plt.xlabel('Platform', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f'{title} - {selected_algo} Scheduler ({processor_type.capitalize()} Processor)',
             fontsize=14, fontweight='bold')

    # Customize grid and styling
    plt.grid(axis='y', linestyle='--', alpha=0.3) 

    # Determine if lower values are better for this metric
    lower_is_better = metric_name in ['avg_waiting_time', 'deadline_misses', 'priority_inversions', 
                                     'cpu_usage', 'memory_usage']

    # Highlight the best platform
    if lower_is_better:
        best_idx = metric_values.index(min(metric_values))
    else:
        best_idx = metric_values.index(max(metric_values)) 

    bars[best_idx].set_edgecolor('green')
    bars[best_idx].set_linewidth(2.5)

    # Add 'best' annotation
    plt.annotate( 
        'BEST',
        xy=(best_idx, metric_values[best_idx]),
        xytext=(best_idx, metric_values[best_idx] + 0.1 * max(metric_values)),
        ha='center',
        fontweight='bold',
        color='green',
        arrowprops=dict(facecolor='green', shrink=0.05, width=2, headwidth=10)
    )

    plt.tight_layout()

    # Save or show the plot 
    if output_path:
        ensure_output_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved cross-platform comparison for {metric_name} to {output_path}")
    else:
        plt.show() 

def create_platform_radar_chart(platform_data, processor_type, output_path=None):
    """
    Create a radar chart comparing platforms across multiple metrics

    Args:
        platform_data: Dictionary mapping platform names to metrics by algorithm 
        processor_type: 'single' or 'multi'
        output_path: Path to save the plot 
    """
    if not platform_data:
        logger.warning(f"No platform data available for radar chart")
        return

    # Define metrics to include in the radar chart
    metrics_config = [
        ('avg_waiting_time', 'Waiting Time', True),  # True = lower is better 
        ('throughput', 'Throughput', False),  # False = higher is better
        ('cpu_usage', 'CPU Usage', True),
        ('memory_usage', 'Memory Usage', True) 
    ]

    # Key mappings for metric names
    key_mappings = {
        'avg_waiting_time': ['avg_waiting_time', 'average_waiting_time', 'mean_waiting_time'],
        'throughput': ['system_throughput', 'avg_throughput', 'throughput', 'tasks_per_second'],
        'cpu_usage': ['avg_cpu_usage', 'cpu_usage', 'average_cpu_usage'], 
        'memory_usage': ['avg_memory_usage', 'memory_usage', 'average_memory_usage']
    } 

    # Find common algorithms across all platforms
    common_algos = set()
    first = True

    for platform, metrics in platform_data.items():
        if processor_type not in metrics or not metrics[processor_type]:
            continue

        # Get algorithms available for this platform 
        platform_algos = set(metrics[processor_type].keys())

        if first:
            common_algos = platform_algos
            first = False
        else:
            common_algos = common_algos.intersection(platform_algos) 

    if not common_algos:
        logger.warning(f"No common algorithms across platforms for {processor_type} processor") 
        return

    # Select the algorithm to compare (prioritize EDF, then Priority, then FCFS)
    selected_algo = None
    for algo in ['EDF', 'Priority', 'FCFS', 'ML-Based']:
        if algo in common_algos:
            selected_algo = algo
            break 

    if not selected_algo:
        selected_algo = next(iter(common_algos)) 

    logger.info(f"Using {selected_algo} algorithm for platform radar chart")

    # Collect data for each platform
    data = {}
    metric_ranges = {metric_name: {'min': float('inf'), 'max': float('-inf')}
                    for metric_name, _, _ in metrics_config} 

    for platform, metrics in platform_data.items():
        if processor_type not in metrics or not metrics[processor_type]:
            continue 

        if selected_algo not in metrics[processor_type]:
            continue

        algo_metrics = metrics[processor_type][selected_algo]
        platform_data = {} 

        for metric_name, _, _ in metrics_config:
            # Try different possible keys for this metric
            value = None 
            for key in key_mappings.get(metric_name, [metric_name]):
                if key in algo_metrics:
                    value = algo_metrics[key]
                    break 

            if value is not None:
                try:
                    value = float(value)
                    platform_data[metric_name] = value 

                    # Update min/max for normalization
                    metric_ranges[metric_name]['min'] = min(metric_ranges[metric_name]['min'], value)
                    metric_ranges[metric_name]['max'] = max(metric_ranges[metric_name]['max'], value) 
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert {value} to float for {platform}, {metric_name}") 
                    continue

        if platform_data:
            data[platform] = platform_data 

    if not data:
        logger.warning("No valid data for platform radar chart") 
        return

    # Extract metric names and labels
    metric_names = [name for name, _, _ in metrics_config]
    metric_labels = [label for _, label, _ in metrics_config]
    lower_is_better = [flag for _, _, flag in metrics_config] 

    # Normalize values to range [0, 1] where higher is always better
    normalized_data = {}

    for platform, platform_data in data.items(): 
        normalized_platform_data = {}

        for i, (metric_name, _, lower_better) in enumerate(metrics_config):
            if metric_name in platform_data:
                value = platform_data[metric_name]
                min_val = metric_ranges[metric_name]['min']
                max_val = metric_ranges[metric_name]['max'] 

                # Skip normalization if min == max
                if min_val == max_val:
                    normalized_value = 0.5  # Middle value 
                else:
                    # Normalize to [0, 1]
                    normalized_value = (value - min_val) / (max_val - min_val) 

                    # Invert if lower is better
                    if lower_better: 
                        normalized_value = 1 - normalized_value

                normalized_platform_data[metric_name] = normalized_value
            else:
                normalized_platform_data[metric_name] = 0  # Default if missing 

        normalized_data[platform] = normalized_platform_data

    # Create the radar chart
    plt.figure(figsize=(10, 10)) 

    # Setup radar chart
    ax = plt.subplot(111, polar=True)

    # Number of metrics
    num_metrics = len(metric_names)
    angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist() 

    # Make the plot circular by appending the first angle
    angles += angles[:1]
    metric_labels += [metric_labels[0]] 

    # Draw the background circles
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels[:-1], fontsize=12)

    # Set y limits
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], color='grey') 

    # Draw the grid lines
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.grid(True, linestyle='--', alpha=0.5) 

    # Use a colormap for platforms
    colors = plt.cm.tab10(np.linspace(0, 1, len(normalized_data)))

    # Plot each platform
    for i, (platform, platform_data) in enumerate(normalized_data.items()): 
        # Get values for each metric
        values = [platform_data.get(metric, 0) for metric in metric_names]

        # Close the polygon
        values += values[:1] 

        # Plot values with colormap
        ax.plot(angles, values, linewidth=2, linestyle='-', label=platform, color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i]) 

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=10)

    plt.title(f'Platform Performance Comparison - {selected_algo} Scheduler ({processor_type.capitalize()} Processor)',
             fontsize=15, y=1.1) 

    # Add explanation of the normalization
    plt.figtext(0.5, 0.01, 
               'Note: All metrics are normalized to 0-1 scale. For metrics where lower is better\n' 
               '(waiting time, CPU/memory usage), the scale is inverted so that higher values\n'
               'on the radar chart always represent better performance.',
               ha='center', fontsize=9, bbox=dict(facecolor='lightyellow', alpha=0.5))

    # Save or show the plot
    if output_path:
        ensure_output_dir(os.path.dirname(output_path))
        plt.savefig(output_path, dpi=300, bbox_inches='tight') 
        plt.close()
        logger.info(f"Saved platform radar chart to {output_path}")
    else:
        plt.show() 

# ===== Report Generation Functions =====

def generate_performance_report(single_metrics, multi_metrics, output_path):
    """
    Generate a comprehensive Markdown report on scheduling performance

    Args:
        single_metrics: Dictionary with single processor metrics by algorithm
        multi_metrics: Dictionary with multi-processor metrics by algorithm
        output_path: Path to save the report 
    """
    ensure_output_dir(os.path.dirname(output_path))

    with open(output_path, 'w') as f:
        # Report header
        f.write("# Real-Time Task Scheduling Performance Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n") 

        # Single processor results
        f.write("## 1. Single Processor Results\n\n") 

        for algo_name, metrics in single_metrics.items(): 
            if metrics is None:
                continue

            f.write(f"### 1.{list(single_metrics.keys()).index(algo_name) + 1}. {algo_name} Scheduler\n\n") 

            # Extract key metrics with fallback for different naming conventions
            completed_tasks = metrics.get('completed_tasks', 0)

            avg_waiting_time = None
            for key in ['avg_waiting_time', 'average_waiting_time', 'mean_waiting_time']: 
                if key in metrics:
                    avg_waiting_time = metrics[key]
                    break 

            if avg_waiting_time is None:
                avg_waiting_time = 0 

            f.write(f"- **Completed Tasks:** {completed_tasks}\n")
            f.write(f"- **Average Waiting Time:** {avg_waiting_time:.2f} seconds\n")

            # Add waiting times by priority if available
            waiting_by_priority = None 
            for key in ['avg_waiting_by_priority', 'waiting_times_by_priority']: 
                if key in metrics:
                    waiting_by_priority = metrics[key]
                    break

            if waiting_by_priority: 
                f.write("- **Waiting Times by Priority:**\n") 
                for priority, waiting_time in waiting_by_priority.items():
                    f.write(f"  - {priority}: {waiting_time:.2f} seconds\n")

            # Add algorithm-specific metrics
            if algo_name == 'EDF' and 'deadline_misses' in metrics: 
                f.write(f"- **Deadline Misses:** {metrics['deadline_misses']}\n")

            if algo_name == 'Priority':
                if 'priority_inversions' in metrics:
                    f.write(f"- **Priority Inversions:** {metrics['priority_inversions']}\n") 
                if 'priority_inheritance_events' in metrics:
                    f.write(f"- **Priority Inheritance Events:** {metrics['priority_inheritance_events']}\n")

            if algo_name == 'ML-Based':
                if 'average_prediction_error' in metrics:
                    f.write(f"- **Average Prediction Error:** {metrics['average_prediction_error']:.2f} seconds\n") 
                if 'model_trained' in metrics:
                    f.write(f"- **Model Trained:** {'Yes' if metrics['model_trained'] else 'No'}\n")
                if 'feature_importances' in metrics and metrics['feature_importances']:
                    f.write("- **Feature Importances:**\n") 
                    for feature, importance in metrics['feature_importances'].items():
                        f.write(f"  - {feature}: {float(importance):.4f}\n")

            # Add resource utilization metrics
            if 'avg_cpu_usage' in metrics: 
                f.write(f"- **Average CPU Usage:** {metrics['avg_cpu_usage']:.2f}%\n")
            if 'avg_memory_usage' in metrics:
                f.write(f"- **Average Memory Usage:** {metrics['avg_memory_usage']:.2f}%\n")

            f.write("\n") 

        # Multi-processor results
        if multi_metrics: 
            f.write("## 2. Multi-Processor Results\n\n")

            # Extract common system configuration if available
            first_metrics = next((m for m in multi_metrics.values() if m is not None), None)

            if first_metrics: 
                f.write("### 2.1. System Configuration\n\n") 

                processor_count = first_metrics.get('processor_count', 0)
                strategy = first_metrics.get('strategy', 'Unknown')

                f.write(f"- **Processor Count:** {processor_count}\n")
                f.write(f"- **Load Balancing Strategy:** {strategy}\n\n") 

            # Results for each algorithm
            algo_idx = 2
            for algo_name, metrics in multi_metrics.items():
                if metrics is None:
                    continue 

                f.write(f"### 2.{algo_idx}. {algo_name} Scheduler\n\n")
                algo_idx += 1

                # Extract key metrics
                total_completed = None 
                for key in ['total_completed_tasks', 'completed_tasks']:
                    if key in metrics:
                        total_completed = metrics[key]
                        break 
                if total_completed is None:
                    total_completed = 0

                avg_waiting_time = None 
                for key in ['avg_waiting_time', 'average_waiting_time', 'mean_waiting_time']: 
                    if key in metrics:
                        avg_waiting_time = metrics[key]
                        break
                if avg_waiting_time is None:
                    avg_waiting_time = 0 

                throughput = None
                for key in ['system_throughput', 'throughput', 'tasks_per_second']:
                    if key in metrics:
                        throughput = metrics[key] 
                        break
                if throughput is None:
                    throughput = 0

                load_balance = None 
                for key in ['load_balance_cv', 'load_balance']:
                    if key in metrics:
                        load_balance = metrics[key]
                        break 
                if load_balance is None:
                    load_balance = 0

                f.write(f"- **Total Completed Tasks:** {total_completed}\n")
                f.write(f"- **Average Waiting Time:** {avg_waiting_time:.2f} seconds\n") 
                f.write(f"- **System Throughput:** {throughput:.2f} tasks/second\n")
                f.write(f"- **Load Balance CV:** {load_balance:.2f}% (lower is better)\n")

                # Add resource utilization
                if 'avg_cpu_usage' in metrics: 
                    f.write(f"- **Average CPU Usage:** {metrics['avg_cpu_usage']:.2f}%\n")
                if 'avg_memory_usage' in metrics:
                    f.write(f"- **Average Memory Usage:** {metrics['avg_memory_usage']:.2f}%\n")

                # Add tasks by priority if available
                if 'tasks_by_priority' in metrics: 
                    f.write("- **Tasks by Priority:**\n")
                    for priority, count in metrics['tasks_by_priority'].items():
                        f.write(f"  - {priority}: {count}\n") 

                # Add waiting times by priority if available
                waiting_by_priority = None
                for key in ['avg_waiting_by_priority', 'waiting_times_by_priority']: 
                    if key in metrics:
                        waiting_by_priority = metrics[key]
                        break 

                if waiting_by_priority:
                    f.write("- **Average Waiting Time by Priority:**\n") 
                    for priority, waiting_time in waiting_by_priority.items():
                        f.write(f"  - {priority}: {waiting_time:.2f} seconds\n")

                f.write("\n") 

        # Comparative analysis
        f.write("## 3. Comparative Analysis\n\n")

        # Compare waiting times across algorithms (single processor)
        f.write("### 3.1. Single Processor: Average Waiting Time Comparison\n\n") 
        f.write("| Algorithm | Average Waiting Time (s) |\n")
        f.write("|-----------|-------------------------|\n")

        for algo_name, metrics in single_metrics.items():
            if metrics is None:
                continue

            avg_waiting_time = None 
            for key in ['avg_waiting_time', 'average_waiting_time', 'mean_waiting_time']:
                if key in metrics:
                    avg_waiting_time = metrics[key]
                    break

            if avg_waiting_time is not None: 
                f.write(f"| {algo_name} | {avg_waiting_time:.2f} |\n")

        f.write("\n")

        # Compare single vs multi-processor
        if single_metrics and multi_metrics: 
            # Find common algorithms across both processor types 
            common_algos = [algo for algo in single_metrics.keys()
                          if algo in multi_metrics and
                          single_metrics[algo] is not None and 
                          multi_metrics[algo] is not None]

            if common_algos:
                f.write("### 3.2. Single vs. Multi-Processor Throughput\n\n") 
                f.write("| System | Average Throughput (tasks/s) |\n")
                f.write("|--------|------------------------------|\n")

                # Use first common algorithm for comparison
                algo_name = common_algos[0] 

                # Extract throughput for single processor
                single_throughput = 0 
                for key in ['avg_throughput', 'throughput', 'tasks_per_second']:
                    if key in single_metrics[algo_name]:
                        single_throughput = single_metrics[algo_name][key] 
                        break

                # Extract throughput for multi-processor
                multi_throughput = 0 
                for key in ['system_throughput', 'throughput', 'tasks_per_second']:
                    if key in multi_metrics[algo_name]:
                        multi_throughput = multi_metrics[algo_name][key]
                        break 

                processor_count = multi_metrics[algo_name].get('processor_count', 0) 

                f.write(f"| Single Processor | {single_throughput:.2f} |\n") 
                f.write(f"| Multi-Processor ({processor_count} CPUs) | {multi_throughput:.2f} |\n")

                # Calculate and add speedup
                if single_throughput > 0:
                    speedup = multi_throughput / single_throughput 
                    f.write(f"\nSpeedup factor: {speedup:.2f}x\n")

                f.write("\n")

                # Compare waiting times between single and multi-processor
                f.write("### 3.3. Single vs. Multi-Processor Waiting Time\n\n") 
                f.write("| System | Average Waiting Time (s) |\n")
                f.write("|--------|-------------------------|\n")

                # Extract waiting time for single processor
                single_waiting = 0 
                for key in ['avg_waiting_time', 'average_waiting_time', 'mean_waiting_time']:
                    if key in single_metrics[algo_name]:
                        single_waiting = single_metrics[algo_name][key]
                        break 

                # Extract waiting time for multi-processor
                multi_waiting = 0
                for key in ['avg_waiting_time', 'average_waiting_time', 'mean_waiting_time']:
                    if key in multi_metrics[algo_name]: 
                        multi_waiting = multi_metrics[algo_name][key]
                        break

                f.write(f"| Single Processor | {single_waiting:.2f} |\n") 
                f.write(f"| Multi-Processor ({processor_count} CPUs) | {multi_waiting:.2f} |\n")

                # Calculate and add improvement
                if single_waiting > 0:
                    improvement = ((single_waiting - multi_waiting) / single_waiting) * 100 
                    f.write(f"\nWaiting time improvement: {improvement:.2f}%\n")

                f.write("\n") 

        # Conclusion
        f.write("## 4. Conclusion\n\n")
        f.write("Based on the metrics, the following observations can be made:\n\n") 

        # Find best algorithm for waiting time (single processor)
        if single_metrics:
            waiting_times = {}
            for name, metrics in single_metrics.items():
                if metrics is None:
                    continue 

                for key in ['avg_waiting_time', 'average_waiting_time', 'mean_waiting_time']:
                    if key in metrics:
                        waiting_times[name] = metrics[key] 
                        break

            if waiting_times:
                best_waiting = min(waiting_times.items(), key=lambda x: x[1])
                f.write(f"- **Best for Waiting Time**: {best_waiting[0]} scheduler had the lowest average waiting time ({best_waiting[1]:.2f}s).\n") 

        # Compare deadline misses for EDF between single and multi-processor
        if 'EDF' in single_metrics and single_metrics['EDF'] is not None and \
           'EDF' in multi_metrics and multi_metrics['EDF'] is not None and \
           'deadline_misses' in single_metrics['EDF'] and \
           'deadline_misses' in multi_metrics['EDF']: 

            single_misses = single_metrics['EDF']['deadline_misses'] 
            multi_misses = multi_metrics['EDF']['deadline_misses']

            if single_misses != multi_misses:
                better_system = "Multi-Processor" if single_misses > multi_misses else "Single Processor"
                f.write(f"- **Deadline Handling**: {better_system} was better at meeting deadlines with the EDF scheduler.\n") 

        # Evaluate multi-processor efficiency if data available
        if single_metrics and multi_metrics:
            common_algos = [algo for algo in single_metrics.keys()
                          if algo in multi_metrics and
                          single_metrics[algo] is not None and 
                          multi_metrics[algo] is not None]

            if common_algos:
                algo_name = common_algos[0] 

                # Extract throughputs
                single_throughput = 0
                multi_throughput = 0 

                for key in ['avg_throughput', 'throughput', 'tasks_per_second']: 
                    if key in single_metrics[algo_name]:
                        single_throughput = single_metrics[algo_name][key]
                        break

                for key in ['system_throughput', 'throughput', 'tasks_per_second']: 
                    if key in multi_metrics[algo_name]:
                        multi_throughput = multi_metrics[algo_name][key]
                        break 

                if single_throughput > 0: 
                    speedup = multi_throughput / single_throughput

                    processor_count = multi_metrics[algo_name].get('processor_count', 0)
                    if processor_count > 0: 
                        ideal_speedup = processor_count
                        efficiency = (speedup / ideal_speedup) * 100

                        f.write(f"- **Parallelisation Efficiency**: The multi-processor system achieved {efficiency:.1f}% of ideal speedup.\n") 

                # Evaluate load balance if available
                if 'load_balance_cv' in multi_metrics[algo_name]:
                    load_balance = multi_metrics[algo_name]['load_balance_cv'] 

                    if load_balance < 10:
                        f.write("- **Load Balancing**: Excellent load distribution across processors.\n")
                    elif load_balance < 20: 
                        f.write("- **Load Balancing**: Good load distribution, but some processors were underutilised.\n")
                    else:
                        f.write("- **Load Balancing**: Poor load distribution, significant processor imbalance.\n") 

        # Evaluate ML-based scheduler if data available
        if 'ML-Based' in single_metrics and single_metrics['ML-Based'] is not None: 
            metrics = single_metrics['ML-Based']

            if 'average_prediction_error' in metrics:
                error = metrics['average_prediction_error'] 

                if error < 0.5: 
                    f.write("- **ML-Based Scheduling**: Excellent prediction accuracy with very low error.\n")
                elif error < 1.0:
                    f.write("- **ML-Based Scheduling**: Good prediction accuracy with reasonable error levels.\n") 
                else:
                    f.write("- **ML-Based Scheduling**: Moderate prediction accuracy with significant error margins.\n") 

        # Add note about visualisations
        f.write("\n- **Visualisations**: The accompanying charts and heatmaps provide visual insights into resource utilisation, task execution patterns, and comparative performance across scheduling algorithms.\n") 

        # Final summary
        f.write("\nThis report demonstrates the effectiveness of different scheduling algorithms and processor configurations for real-time task scheduling. \n The results can guide the selection of appropriate scheduling strategies based on specific requirements such as minimizing waiting time, meeting deadlines, or optimizing resource utilization.\n") 

    logger.info(f"Generated performance report: {output_path}") 

def generate_platform_comparison_report(platform_data, output_path):
    """
    Generate a comprehensive Markdown report comparing different platforms

    Args:
        platform_data: Dictionary mapping platform names to metrics by algorithm
        output_path: Path to save the report 
    """
    ensure_output_dir(os.path.dirname(output_path))

    with open(output_path, 'w') as f:
        # Report header
        f.write("# Cross-Platform Performance Comparison Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n") 

        # Extract all platforms
        platforms = list(platform_data.keys())

        # Find common algorithms across all platforms for single processor
        common_algos_single = set() 
        first = True

        for platform, metrics in platform_data.items():
            if 'single' not in metrics or not metrics['single']:
                continue 

            # Get algorithms available for this platform
            platform_algos = set(metrics['single'].keys()) 

            if first:
                common_algos_single = platform_algos
                first = False 
            else:
                common_algos_single = common_algos_single.intersection(platform_algos) 

        # Find common algorithms across all platforms for multi-processor
        common_algos_multi = set()
        first = True

        for platform, metrics in platform_data.items(): 
            if 'multi' not in metrics or not metrics['multi']: 
                continue

            # Get algorithms available for this platform
            platform_algos = set(metrics['multi'].keys())

            if first: 
                common_algos_multi = platform_algos
                first = False
            else:
                common_algos_multi = common_algos_multi.intersection(platform_algos)

        # Platform overview
        f.write("## 1. Platform Overview\n\n") 
        f.write("| Platform | Single Processor Data | Multi-Processor Data |\n") 
        f.write("|----------|------------------------|-----------------------|\n")

        for platform in platforms:
            has_single = 'single' in platform_data[platform] and platform_data[platform]['single']
            has_multi = 'multi' in platform_data[platform] and platform_data[platform]['multi'] 

            f.write(f"| {platform} | {'Yes' if has_single else 'No'} | {'Yes' if has_multi else 'No'} |\n") 

        f.write("\n")

        # Single processor comparison
        if common_algos_single: 
            f.write("## 2. Single Processor Comparison\n\n")

            # Select an algorithm to use (prioritize EDF, then Priority, then FCFS)
            selected_algo = None 
            for algo in ['EDF', 'Priority', 'FCFS', 'ML-Based']:
                if algo in common_algos_single:
                    selected_algo = algo
                    break 

            if not selected_algo: 
                selected_algo = next(iter(common_algos_single))

            f.write(f"Using {selected_algo} scheduler for comparison:\n\n")

            # Waiting time comparison
            f.write("### 2.1. Average Waiting Time\n\n") 
            f.write("| Platform | Average Waiting Time (s) |\n")
            f.write("|----------|-------------------------|\n")

            waiting_times = {}
            for platform in platforms:
                if ('single' in platform_data[platform] and
                    platform_data[platform]['single'] and 
                    selected_algo in platform_data[platform]['single']):

                    metrics = platform_data[platform]['single'][selected_algo]
                    waiting_time = None 

                    for key in ['avg_waiting_time', 'average_waiting_time', 'mean_waiting_time']:
                        if key in metrics:
                            waiting_time = metrics[key] 
                            break

                    if waiting_time is not None:
                        waiting_times[platform] = waiting_time 
                        f.write(f"| {platform} | {waiting_time:.2f} |\n") 
                    else:
                        f.write(f"| {platform} | N/A |\n")

            f.write("\n") 

            # Determine best platform for waiting time 
            if waiting_times:
                best_platform = min(waiting_times.items(), key=lambda x: x[1])
                f.write(f"**Best Platform for Waiting Time**: {best_platform[0]} with {best_platform[1]:.2f}s\n\n")

            # Throughput comparison
            f.write("### 2.2. Processing Throughput\n\n") 
            f.write("| Platform | Throughput (tasks/s) |\n")
            f.write("|----------|---------------------|\n")

            throughputs = {}
            for platform in platforms:
                if ('single' in platform_data[platform] and
                    platform_data[platform]['single'] and selected_algo in platform_data[platform]['single']): 
                        metrics = platform_data[platform]['single'][selected_algo]
                        throughput = None                     
                for key in ['avg_throughput', 'throughput', 'tasks_per_second']:
                        if key in metrics:
                            throughput = metrics[key]                            
                            break
                if throughput is not None:
                    throughputs[platform] = throughput
                    f.write(f"| {platform} | {throughput:.2f} |\n")
                else:
                    f.write(f"| {platform} | N/A |\n")

            f.write("\n")             # Determine best platform for throughput            
            if throughputs:
                best_platform = max(throughputs.items(), key=lambda x: x[1])
                f.write(f"**Best Platform for Throughput**: {best_platform[0]} with {best_platform[1]:.2f} tasks/s\n\n")

            # Resource utilization comparison
            f.write("### 2.3. Resource Utilization\n\n")
            f.write("| Platform | CPU Usage (%) | Memory Usage (%) |\n")
            f.write("|----------|---------------|------------------|\n")

            for platform in platforms:
                if ('single' in platform_data[platform] and                    platform_data[platform]['single'] and                     selected_algo in platform_data[platform]['single']):

                    metrics = platform_data[platform]['single'][selected_algo]
                    cpu_usage = metrics.get('avg_cpu_usage', 'N/A')
                    memory_usage = metrics.get('avg_memory_usage', 'N/A') 
                    if cpu_usage != 'N/A' and memory_usage != 'N/A':
                        f.write(f"| {platform} | {cpu_usage:.2f} | {memory_usage:.2f} |\n")                     
                    else:
                        cpu_str = f"{cpu_usage:.2f}" if cpu_usage != 'N/A' else 'N/A'
                        mem_str = f"{memory_usage:.2f}" if memory_usage != 'N/A' else 'N/A'
                        f.write(f"| {platform} | {cpu_str} | {mem_str} |\n") 
            f.write("\n")

        # Multi-processor comparison
        if common_algos_multi:             
            f.write("## 3. Multi-Processor Comparison\n\n")
            # Select an algorithm to use (prioritize EDF, then Priority, then FCFS)
            selected_algo = None             
            for algo in ['EDF', 'Priority', 'FCFS', 'ML-Based']:
                if algo in common_algos_multi:
                    selected_algo = algo
                    break 
            if not selected_algo:
                selected_algo = next(iter(common_algos_multi))

            f.write(f"Using {selected_algo} scheduler for comparison:\n\n") 
            # Waiting time comparison
            f.write("### 3.1. Average Waiting Time\n\n")             
            f.write("| Platform | Average Waiting Time (s) |\n")
            f.write("|----------|-------------------------|\n")

            waiting_times = {}
            for platform in platforms:
                if ('multi' in platform_data[platform] and
                    platform_data[platform]['multi'] and                     selected_algo in platform_data[platform]['multi']):

                    metrics = platform_data[platform]['multi'][selected_algo]
                    waiting_time = None 
                    for key in ['avg_waiting_time', 'average_waiting_time', 'mean_waiting_time']:
                        if key in metrics:
                            waiting_time = metrics[key]                             
                            break

                    if waiting_time is not None:
                        waiting_times[platform] = waiting_time                         
                        f.write(f"| {platform} | {waiting_time:.2f} |\n")                    
                    else:
                        f.write(f"| {platform} | N/A |\n")

            f.write("\n") 
            # Determine best platform for waiting time             
            if waiting_times:
                best_platform = min(waiting_times.items(), key=lambda x: x[1])
                f.write(f"**Best Platform for Waiting Time**: {best_platform[0]} with {best_platform[1]:.2f}s\n\n")

            # Throughput comparison
            f.write("### 3.2. Processing Throughput\n\n") 
            f.write("| Platform | Throughput (tasks/s) |\n")
            f.write("|----------|---------------------|\n")

            throughputs = {}
            for platform in platforms:
                if ('multi' in platform_data[platform] and
                    platform_data[platform]['multi'] and                     selected_algo in platform_data[platform]['multi']):

                    metrics = platform_data[platform]['multi'][selected_algo]
                    throughput = None 
                    for key in ['system_throughput', 'throughput', 'tasks_per_second']:
                        if key in metrics:
                            throughput = metrics[key]                       
                            break

                    if throughput is not None:
                        throughputs[platform] = throughput                        
                        f.write(f"| {platform} | {throughput:.2f} |\n")                
                    else:
                        f.write(f"| {platform} | N/A |\n")

            f.write("\n") 
            # Determine best platform for throughput             
            if throughputs:
                best_platform = max(throughputs.items(), key=lambda x: x[1])
                f.write(f"**Best Platform for Throughput**: {best_platform[0]} with {best_platform[1]:.2f} tasks/s\n\n")

            # Processor count and load balancing
            f.write("### 3.3. System Configuration\n\n")            
            f.write("| Platform | Processor Count | Load Balancing Strategy | Load Balance CV (%) |\n")
            f.write("|----------|----------------|--------------------------|---------------------|\n")

            for platform in platforms:
                if ('multi' in platform_data[platform] and
                    platform_data[platform]['multi'] and                     selected_algo in platform_data[platform]['multi']):

                    metrics = platform_data[platform]['multi'][selected_algo]
                    processor_count = metrics.get('processor_count', 'N/A')          
                    strategy = metrics.get('strategy', 'N/A')

                    load_balance = None
                    for key in ['load_balance_cv', 'load_balance']:            
                        if key in metrics:                             
                            load_balance = metrics[key]
                            break

                    load_str = f"{load_balance:.2f}" if load_balance is not None else 'N/A'
                    f.write(f"| {platform} | {processor_count} | {strategy} | {load_str} |\n") 
            f.write("\n")

        # Speed up comparison (Single vs Multi across platforms)
        f.write("## 4. Speedup Analysis\n\n") 
        # Find algorithms common to both single and multi across platforms
        common_to_both = common_algos_single.intersection(common_algos_multi) 
        if common_to_both:
            # Select an algorithm to use (prioritize EDF, then Priority, then FCFS)
            selected_algo = None
            for algo in ['EDF', 'Priority', 'FCFS', 'ML-Based']:
                if algo in common_to_both:
                    selected_algo = algo                    
                    break

            if not selected_algo:
                selected_algo = next(iter(common_to_both))

            f.write(f"Using {selected_algo} scheduler for comparison:\n\n") 
            f.write("| Platform | Single-Processor (tasks/s) | Multi-Processor (tasks/s) | Speedup |\n")
            f.write("|----------|-----------------------------|----------------------------|--------|\n")

            speedups = {}

            for platform in platforms:                 
                if ('single' in platform_data[platform] and
                    'multi' in platform_data[platform] and
                    platform_data[platform]['single'] and
                    platform_data[platform]['multi'] and
                    selected_algo in platform_data[platform]['single'] and                     selected_algo in platform_data[platform]['multi']):

                    single_metrics = platform_data[platform]['single'][selected_algo]
                    multi_metrics = platform_data[platform]['multi'][selected_algo] 
                    # Extract single processor throughput
                    single_throughput = None
                    for key in ['avg_throughput', 'throughput', 'tasks_per_second']:                        
                         if key in single_metrics:                            
                            single_throughput = single_metrics[key]
                            break

                    # Extract multi-processor throughput
                    multi_throughput = None                    
                    for key in ['system_throughput', 'throughput', 'tasks_per_second']:
                        if key in multi_metrics:
                            multi_throughput = multi_metrics[key]                             
                            break

                    if single_throughput is not None and multi_throughput is not None and single_throughput > 0:      
                        speedup = multi_throughput / single_throughput
                        speedups[platform] = speedup
                        f.write(f"| {platform} | {single_throughput:.2f} | {multi_throughput:.2f} | {speedup:.2f}x |\n")        
                    else:
                        single_str = f"{single_throughput:.2f}" if single_throughput is not None else 'N/A'
                        multi_str = f"{multi_throughput:.2f}" if multi_throughput is not None else 'N/A'
                        f.write(f"| {platform} | {single_str} | {multi_str} | N/A |\n") 
            f.write("\n")

            # Determine best platform for speedup
            if speedups:                 best_platform = max(speedups.items(), key=lambda x: x[1])                 
            f.write(f"**Best Platform for Parallelization**: {best_platform[0]} with {best_platform[1]:.2f}x speedup\n\n")

            # Add processor counts for context
            f.write("Processor counts for multi-processor systems:\n\n")                
            for platform in platforms:                     
                if ('multi' in platform_data[platform] and
                        platform_data[platform]['multi'] and
                        selected_algo in platform_data[platform]['multi']):

                        metrics = platform_data[platform]['multi'][selected_algo]                       
                        processor_count = metrics.get('processor_count', 'Unknown')
                        f.write(f"- {platform}: {processor_count} processors\n")

                f.write("\n") 
        # Conclusion
        f.write("## 5. Conclusion\n\n")
        f.write("Based on the cross-platform analysis, the following observations can be made:\n\n")

        # Summarize best platforms for different metrics
        best_platforms = {} 
        # Single-processor waiting time
        if common_algos_single:
            selected_algo = next(iter(common_algos_single))
            waiting_times = {}

            for platform in platforms:                
                if ('single' in platform_data[platform] and
                    platform_data[platform]['single'] and
                    selected_algo in platform_data[platform]['single']): 
                    metrics = platform_data[platform]['single'][selected_algo]
                    waiting_time = None 
                    for key in ['avg_waiting_time', 'average_waiting_time', 'mean_waiting_time']:
                        if key in metrics:
                            waiting_time = metrics[key]                           
                            break

                    if waiting_time is not None:
                        waiting_times[platform] = waiting_time 
            if waiting_times:
                best_platform = min(waiting_times.items(), key=lambda x: x[1])
                best_platforms['single_waiting'] = best_platform[0]
                f.write(f"- **Waiting Time (Single-Processor)**: {best_platform[0]} provides the lowest waiting time.\n") 
        # Multi-processor throughput
        if common_algos_multi:
            selected_algo = next(iter(common_algos_multi))
            throughputs = {}

            for platform in platforms:                 
                if ('multi' in platform_data[platform] and
                    platform_data[platform]['multi'] and
                    selected_algo in platform_data[platform]['multi']): 
                    metrics = platform_data[platform]['multi'][selected_algo]
                    throughput = None 
                    for key in ['system_throughput', 'throughput', 'tasks_per_second']:
                        if key in metrics:
                            throughput = metrics[key]                             
                            break

                    if throughput is not None:
                        throughputs[platform] = throughput 
            if throughputs:
                best_platform = max(throughputs.items(), key=lambda x: x[1])
                best_platforms['multi_throughput'] = best_platform[0]
                f.write(f"- **Throughput (Multi-Processor)**: {best_platform[0]} achieves the highest task processing rate.\n") 
        # Speedup
        if speedups:
            best_platform = max(speedups.items(), key=lambda x: x[1])
            best_platforms['speedup'] = best_platform[0]
            f.write(f"- **Parallelization Efficiency**: {best_platform[0]} demonstrates the best performance scaling from single to multi-processor configurations.\n") 
        # Overall recommendation
        if best_platforms:             # Count occurrences of each platform in the best performers
            platform_counts = {}
            for platform in best_platforms.values():
                if platform in platform_counts:
                    platform_counts[platform] += 1                 
                else:
                    platform_counts[platform] = 1

            # Find the most frequently occurring platform
            overall_best = max(platform_counts.items(), key=lambda x: x[1])[0] 
            f.write(f"\n**Overall Recommendation**: {overall_best} appears to be the most suitable platform for real-time task scheduling, offering an optimal balance of waiting time, throughput, and scalability.\n") 
        # Add notes about platform-specific considerations
        f.write("\n### Platform-Specific Considerations:\n\n")

        # Raspberry Pi considerations
        if any('raspberry' in platform.lower() for platform in platforms):             f.write("**Raspberry Pi**: Embedded platform with lower resources but potentially more efficient power usage for long-running deployments.\n") 
        # Desktop/Laptop considerations
        if any(platform.lower() in ['windows_desktop', 'linux_desktop', 'mac_desktop'] for platform in platforms):
            f.write("**Desktop Systems**: Higher computational power but increased power consumption, suitable for task scheduling scenarios requiring more intensive processing.\n") 
        # Add note about visualization
        f.write("\nThe radar charts and comparison visualizations provide additional insights into the relative strengths and weaknesses of each platform across multiple performance dimensions.\n") 
    logger.info(f"Generated platform comparison report: {output_path}")

# ===== Processing Logic Functions =====

def generate_scheduler_visualisations(tasks_df, metrics_dict, timeseries_df, scheduler_name, output_dir):
    """
    Generate all visualisations for a specific scheduler

    Args:
        tasks_df: DataFrame containing task data         metrics_dict: Dictionary containing scheduler metrics
        timeseries_df: DataFrame containing timeseries data
        scheduler_name: Name of the scheduler
        output_dir: Directory to save visualisation outputs     """
    # Create scheduler-specific output directory
    scheduler_dir = os.path.join(output_dir, scheduler_name.replace(' ', '_'))
    ensure_output_dir(scheduler_dir)

    logger.info(f"Generating visualisations for {scheduler_name} scheduler") 
    # 1. Task execution visualisations     
    if tasks_df is not None and not tasks_df.empty:
        # 1.1 Task Gantt chart
        plot_task_gantt_chart(
            tasks_df,
            scheduler_name,
            os.path.join(scheduler_dir, 'task_execution_timeline.png')
        ) 
        # 1.2 Waiting time distribution
        plot_waiting_time_distribution(             tasks_df,
            scheduler_name,
            os.path.join(scheduler_dir, 'waiting_time_distribution.png')
        )

        # 1.3 Priority pie chart
        plot_priority_pie_chart(             tasks_df,
            scheduler_name,             os.path.join(scheduler_dir, 'priority_distribution.png')
        )

        # 1.4 Task density heatmap
        plot_task_density_heatmap(             tasks_df,
            scheduler_name,
            os.path.join(scheduler_dir, 'task_density_heatmap.png')
        )

    # 2. Resource utilisation visualisations     
    if timeseries_df is not None and not timeseries_df.empty:
        # 2.1 Resource utilisation over time
        plot_resource_utilization(
            timeseries_df,
            scheduler_name,
            os.path.join(scheduler_dir, 'resource_utilization.png')
        )

        # 2.2 Resource heatmap
        create_resource_heatmap(             timeseries_df,
            scheduler_name,
            os.path.join(scheduler_dir, 'resource_heatmap.png')
        )

    # 3. CPU utilisation heatmap (multi-processor)
    if metrics_dict is not None and 'per_processor_metrics' in metrics_dict:         plot_cpu_utilization_heatmap(
            metrics_dict,             scheduler_name,
            os.path.join(scheduler_dir, 'cpu_utilization_heatmap.png')
        )

    logger.info(f"Completed visualisations for {scheduler_name} scheduler") 
def process_directory(data_dir, output_dir=None, schedulers=None):
    """
    Process data files in a directory and generate visualisations

    Args:
        data_dir: Path to the directory containing data files
        output_dir: Path to save visualisations (if None, uses data_dir/visualisations)         schedulers: List of schedulers to process (if None, processes all found)     """
    # Convert to Path object for easier handling
    data_dir = Path(data_dir)

    # Default output directory
    if output_dir is None:
        output_dir = data_dir.parent / f"visualisations_{data_dir.name}"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True) 
    logger.info(f"Processing data from {data_dir}")
    logger.info(f"Saving visualisations to {output_dir}")

    # Check for single and multi-processor directories
    single_processor_dir = data_dir / "single_processor"
    multi_processor_dir = data_dir / "multi_processor"

    # Dictionary to collect metrics for all schedulers
    single_metrics = {}
    multi_metrics = {} 
    # Process single processor data
    if single_processor_dir.exists():
        logger.info("Processing single processor data")

        # Determine available schedulers
        available_schedulers = [] 
        # Find all JSON files that might contain metrics
        for file_path in single_processor_dir.glob("*_metrics.json"):
            file_name = file_path.name
            match = re.match(r'([^_]+)_.*', file_name)
            if match:                
                scheduler_name = match.group(1)
                if scheduler_name not in available_schedulers:
                    available_schedulers.append(scheduler_name)

        logger.info(f"Found available schedulers: {available_schedulers}") 
        # Filter by requested schedulers if provided         
        if schedulers:
            scheduler_list = [s for s in available_schedulers if s in schedulers]
        else:
            scheduler_list = available_schedulers

        # Process each scheduler
        for scheduler in scheduler_list:             # Load data files
            tasks_df, metrics_dict, timeseries_df = load_data_files(                 str(data_dir), scheduler, 'single'
            )

            # Store metrics
            if metrics_dict:
                single_metrics[scheduler] = metrics_dict 
            # Generate visualisations
            generate_scheduler_visualisations(
                tasks_df, metrics_dict, timeseries_df, scheduler, str(output_dir)
            ) 
    # Process multi-processor data
    if multi_processor_dir.exists():
        logger.info("Processing multi-processor data") 
        # Determine available schedulers
        available_schedulers = [] 
        # Find all JSON files that might contain metrics
        for file_path in multi_processor_dir.glob("*_metrics.json"):
            file_name = file_path.name
            match = re.match(r'([^_]+)_.*', file_name)
            if match:                
                scheduler_name = match.group(1)
                if scheduler_name not in available_schedulers:
                    available_schedulers.append(scheduler_name)

        logger.info(f"Found available schedulers: {available_schedulers}") 
        # Filter by requested schedulers if provided         
        if schedulers:
            scheduler_list = [s for s in available_schedulers if s in schedulers]
        else:
            scheduler_list = available_schedulers

        # Process each scheduler
        for scheduler in scheduler_list:
            # Load data files             
            tasks_df, metrics_dict, timeseries_df = load_data_files(
                str(data_dir), scheduler, 'multi'
            )

            # Store metrics
            if metrics_dict:
                multi_metrics[scheduler] = metrics_dict 
            # Generate visualisations
            generate_scheduler_visualisations(
                tasks_df, metrics_dict, timeseries_df, scheduler, str(output_dir)
            ) 
    # Generate comparison visualisations if we have data
    if single_metrics:         # Compare single processor schedulers
        comparison_dir = os.path.join(output_dir, 'comparison')
        ensure_output_dir(comparison_dir) 
        # Waiting time comparison
        plot_performance_comparison(
            single_metrics,
            'avg_waiting_time',
            'Average Waiting Time Comparison (Single Processor)',
            'Waiting Time (s)',             os.path.join(comparison_dir, 'single_waiting_time_comparison.png')
        )

        # Throughput comparison
        plot_performance_comparison(
            single_metrics,
            'throughput',
            'Throughput Comparison (Single Processor)',
            'Tasks per Second',             os.path.join(comparison_dir, 'single_throughput_comparison.png')
        )

        # Radar chart
        create_radar_chart(
            single_metrics,
            os.path.join(comparison_dir, 'single_processor_radar_chart.png')
        ) 
    if multi_metrics:
        # Compare multi-processor schedulers
        comparison_dir = os.path.join(output_dir, 'comparison')         
        ensure_output_dir(comparison_dir)

        # Waiting time comparison
        plot_performance_comparison(
            multi_metrics,
            'avg_waiting_time',
            'Average Waiting Time Comparison (Multi-Processor)',
            'Waiting Time (s)', 
            os.path.join(comparison_dir, 'multi_waiting_time_comparison.png')
        )

        # Throughput comparison
        plot_performance_comparison(
            multi_metrics,
            'throughput',
            'Throughput Comparison (Multi-Processor)',
            'Tasks per Second', 
            os.path.join(comparison_dir, 'multi_throughput_comparison.png')
        )

        # Radar chart
        create_radar_chart(
            multi_metrics,
            os.path.join(comparison_dir, 'multi_processor_radar_chart.png')
        ) 

    # Compare single vs multi-processor if we have both 
    if single_metrics and multi_metrics:
        comparison_dir = os.path.join(output_dir, 'comparison')
        ensure_output_dir(comparison_dir)

        plot_processor_comparison(
            single_metrics,
            multi_metrics,
            os.path.join(comparison_dir, 'processor_comparison.png')
        ) 

    # Generate performance report 
    if single_metrics or multi_metrics:
        report_path = os.path.join(output_dir, "performance_report.md")
        generate_performance_report(single_metrics, multi_metrics, report_path)
        logger.info(f"Generated performance report: {report_path}")

    logger.info("Completed processing data directory") 

def load_cross_platform_data(platform_dirs):
    """
    Load data from multiple platform directories for cross-platform comparison

    Args:
        platform_dirs: List of directories containing data for different platforms

    Returns:
        Dictionary mapping platform names to metrics by algorithm 
    """
    platform_data = {}

    for platform_dir in platform_dirs: 
        platform_path = Path(platform_dir)

        # Extract platform name from directory
        platform_name = extract_platform_from_dir(str(platform_path))
        if not platform_name:
            # Use directory name as fallback
            platform_name = platform_path.name 

        logger.info(f"Loading data for platform: {platform_name}")

        # Check for single and multi-processor directories
        single_processor_dir = platform_path / "single_processor"
        multi_processor_dir = platform_path / "multi_processor" 

        # Initialize metrics dictionaries
        single_metrics = {} 
        multi_metrics = {}

        # Process single processor data
        if single_processor_dir.exists():
            logger.info(f"Processing single processor data for {platform_name}")

            # Find all metric files
            metric_files = list(single_processor_dir.glob("*_metrics.json")) 

            for metric_file in metric_files: 
                # Extract scheduler name
                match = re.match(r'([^_]+)_.*', metric_file.name)
                if match:
                    scheduler_name = match.group(1) 

                    try:
                        # Load metrics
                        with open(metric_file, 'r') as f:
                            metrics_dict = json.load(f) 

                        single_metrics[scheduler_name] = metrics_dict
                        logger.info(f"Loaded {scheduler_name} metrics for {platform_name}") 
                    except Exception as e:
                        logger.error(f"Error loading {metric_file}: {e}")

        # Process multi-processor data if needed
        if multi_processor_dir.exists():
            logger.info(f"Processing multi-processor data for {platform_name}") 

            # Find all system metric files
            metric_files = list(multi_processor_dir.glob("*_system_metrics.json"))

            for metric_file in metric_files: 
                # Extract scheduler name
                match = re.match(r'([^_]+)_.*', metric_file.name) 
                if match:
                    scheduler_name = match.group(1)

                    try:
                        # Load metrics
                        with open(metric_file, 'r') as f: 
                            metrics_dict = json.load(f)

                        multi_metrics[scheduler_name] = metrics_dict 
                        logger.info(f"Loaded {scheduler_name} system metrics for {platform_name}")
                    except Exception as e:
                        logger.error(f"Error loading {metric_file}: {e}") 

        # Store platform data
        platform_data[platform_name] = {
            'single': single_metrics,
            'multi': multi_metrics
        } 

    return platform_data

def generate_cross_platform_visualisations(platform_data, output_dir):
    """
    Generate visualisations comparing performance across different platforms

    Args:
        platform_data: Dictionary mapping platform names to metrics by algorithm
        output_dir: Directory to save visualisation outputs     """
    # Create platform comparison output directory
    platform_dir = os.path.join(output_dir, 'platform_comparison')
    ensure_output_dir(platform_dir)

    logger.info("Generating cross-platform comparison visualisations") 
    # 1. Single processor comparisons
    # 1.1 Waiting time comparison
    plot_cross_platform_comparison(
        platform_data,
        'avg_waiting_time',
        'single',
        'Average Waiting Time Comparison',
        'Waiting Time (s)',         os.path.join(platform_dir, 'single_waiting_time_comparison.png')
    )

    # 1.2 Throughput comparison
    plot_cross_platform_comparison(
        platform_data,
        'throughput',
        'single',
        'Throughput Comparison',
        'Tasks per Second',
        os.path.join(platform_dir, 'single_throughput_comparison.png')
    ) 
    # 1.3 CPU usage comparison
    plot_cross_platform_comparison(         platform_data,
        'cpu_usage',
        'single',
        'CPU Usage Comparison',
        'CPU Usage (%)',
        os.path.join(platform_dir, 'single_cpu_usage_comparison.png')
    )

    # 1.4 Memory usage comparison
    plot_cross_platform_comparison(
        platform_data,
        'memory_usage',         'single',         'Memory Usage Comparison',
        'Memory Usage (%)',
        os.path.join(platform_dir, 'single_memory_usage_comparison.png')
    )

    # 1.5 Radar chart for single processor
    create_platform_radar_chart(
        platform_data,
        'single',
        os.path.join(platform_dir, 'single_processor_radar_chart.png')
    ) 
    # 2. Multi-processor comparisons
    # 2.1 Waiting time comparison
    plot_cross_platform_comparison(         platform_data,
        'avg_waiting_time',
        'multi',
        'Average Waiting Time Comparison',
        'Waiting Time (s)',
        os.path.join(platform_dir, 'multi_waiting_time_comparison.png')
    )

    # 2.2 Throughput comparison
    plot_cross_platform_comparison(
        platform_data,
        'throughput',         'multi',         'Throughput Comparison',
        'Tasks per Second',
        os.path.join(platform_dir, 'multi_throughput_comparison.png')
    )

    # 2.3 CPU usage comparison
    plot_cross_platform_comparison(
        platform_data,
        'cpu_usage',
        'multi',
        'CPU Usage Comparison',
        'CPU Usage (%)',         os.path.join(platform_dir, 'multi_cpu_usage_comparison.png')     )

    # 2.4 Memory usage comparison
    plot_cross_platform_comparison(
        platform_data,
        'memory_usage',
        'multi',
        'Memory Usage Comparison',
        'Memory Usage (%)',
        os.path.join(platform_dir, 'multi_memory_usage_comparison.png')     )

    # 2.5 Radar chart for multi-processor
    create_platform_radar_chart(         platform_data,
        'multi',
        os.path.join(platform_dir, 'multi_processor_radar_chart.png')
    )

    # 3. Generate platform comparison report
    report_path = os.path.join(platform_dir, 'platform_comparison_report.md')
    generate_platform_comparison_report(platform_data, report_path) 
    logger.info("Completed cross-platform comparison visualisations")

def process_cross_platform(platform_dirs, output_dir=None):
    """
    Process data from multiple platforms for cross-platform comparison

    Args:
        platform_dirs: List of directories containing data for different platforms
        output_dir: Directory to save visualisations (if None, creates a new directory) 
    """
    # Convert all paths to Path objects for easier handling
    platform_paths = [Path(d) for d in platform_dirs]

    # Default output directory (based on current time)
    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"cross_platform_comparison_{timestamp}"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True) 

    logger.info(f"Processing data from {len(platform_dirs)} platforms") 
    logger.info(f"Saving cross-platform visualisations to {output_dir}")

    # Load data from all platforms
    platform_data = load_cross_platform_data(platform_dirs)

    if not platform_data:
        logger.error("No valid platform data found. Exiting.") 
        return

    # Generate cross-platform visualisations
    generate_cross_platform_visualisations(platform_data, output_dir)

    logger.info("Completed cross-platform analysis") 

# ===== Main Execution Block =====

def main():
    """Main function for command-line execution"""
    parser = argparse.ArgumentParser(description='Enhanced visualisation tool for task scheduling data')

    # Create mutually exclusive group for single directory vs cross-platform
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--data-dir', help='Directory containing data files')
    group.add_argument('--cross-platform', action='store_true',
                      help='Enable cross-platform comparison mode') 

    # Common options
    parser.add_argument('--output-dir', help='Directory to save visualisations') 

    # Options for single directory mode
    parser.add_argument('--scheduler', action='append', dest='schedulers',
                      help='Specific scheduler to visualise (can be used multiple times)')     # Options for cross-platform mode
    parser.add_argument('--platform-dirs', nargs='+',
                      help='List of directories containing data for different platforms')     
    args = parser.parse_args()

    if args.cross_platform: 
        if not args.platform_dirs:           
            parser.error('--cross-platform requires --platform-dirs')
        process_cross_platform(args.platform_dirs, args.output_dir)
    else:
        process_directory(args.data_dir, args.output_dir, args.schedulers) 
if __name__ == "__main__":
    main() 