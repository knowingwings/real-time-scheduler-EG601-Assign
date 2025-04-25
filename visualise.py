#!/usr/bin/env python3
"""
Visualisation Tool for Task Scheduling Data

This script processes CSV and JSON data files collected during scheduling simulations
and generates visualizations for analysis. It can be run after simulations to create
charts and graphs for the report.

Usage:
    python visualise.py --data-dir results/data/TIMESTAMP
    python visualise.py --data-dir results/data/TIMESTAMP --output-dir results/visualizations
    python visualise.py --data-dir results/data/TIMESTAMP --scheduler FCFS
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
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("visualise")

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

def ensure_output_dir(output_path):
    """Ensure the output directory exists"""
    os.makedirs(output_path, exist_ok=True)
    return output_path

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
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.tight_layout()
        plt.show()

def plot_priority_distribution(tasks_df, scheduler_name, output_path=None):
    """
    Plot distribution of completed tasks by priority
    
    Args:
        tasks_df: DataFrame containing task data
        scheduler_name: Name of the scheduler
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    if tasks_df.empty:
        plt.title(f"No priority data for {scheduler_name}")
        if output_path:
            plt.savefig(output_path)
        return
    
    # Count tasks by priority
    priority_counts = tasks_df['priority'].value_counts()
    
    # Extract priorities and counts
    priorities = priority_counts.index.tolist()
    counts = priority_counts.values.tolist()
    
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

def plot_waiting_times_by_priority(tasks_df, scheduler_name, output_path=None):
    """
    Plot average waiting times by priority
    
    Args:
        tasks_df: DataFrame containing task data
        scheduler_name: Name of the scheduler
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    if tasks_df.empty:
        plt.title(f"No waiting time by priority data for {scheduler_name}")
        if output_path:
            plt.savefig(output_path)
        return
    
    # Calculate average waiting time by priority
    avg_waiting = tasks_df.groupby('priority')['waiting_time'].mean()
    
    # Extract priorities and waiting times
    priorities = avg_waiting.index.tolist()
    waiting_times = avg_waiting.values.tolist()
    
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
    
    for algo, metrics in metrics_by_algorithm.items():
        # Handle both direct values and nested dictionaries
        if isinstance(metrics, dict) and metric_name in metrics:
            # metrics is a dictionary with metric_name as a key
            algorithms.append(algo)
            values.append(metrics[metric_name])
        elif not isinstance(metrics, dict):
            # metrics is a direct value
            algorithms.append(algo)
            values.append(metrics)
    
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
        single_metrics: Dictionary with metrics from single processor
        multi_metrics: Dictionary with metrics from multi-processor
        metric_name: Name of the metric to compare
        title: Plot title
        ylabel: Y-axis label
        output_path: Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    
    systems = ['Single Processor', 'Multi-Processor']
    values = [0, 0]  # Default values
    
    # Handle different key names for the same metrics
    # For single processor
    if isinstance(single_metrics, dict):
        # Try different possible keys for throughput
        if metric_name == 'avg_throughput':
            for key in ['avg_throughput', 'throughput', 'tasks_per_second']:
                if key in single_metrics:
                    values[0] = single_metrics[key]
                    break
        else:
            # For other metrics, try the exact key name first
            if metric_name in single_metrics:
                values[0] = single_metrics[metric_name]
    elif isinstance(single_metrics, (int, float)):
        # Direct value
        values[0] = single_metrics
    
    # For multi-processor
    if isinstance(multi_metrics, dict):
        # Try different possible keys for throughput
        if metric_name == 'avg_throughput':
            for key in ['system_throughput', 'avg_throughput', 'throughput', 'tasks_per_second']:
                if key in multi_metrics:
                    values[1] = multi_metrics[key]
                    break
        else:
            # For other metrics, try the exact key name first
            if metric_name in multi_metrics:
                values[1] = multi_metrics[metric_name]
    elif isinstance(multi_metrics, (int, float)):
        # Direct value
        values[1] = multi_metrics
    
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

def generate_report(single_metrics, multi_metrics, report_path):
    """
    Generate a comprehensive report of the scheduling performance
    
    Args:
        single_metrics: Dictionary with metrics from single processor by scheduler
        multi_metrics: Dictionary with metrics from multi-processor by scheduler
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
            completed_tasks = metrics.get('completed_tasks', 0)
            f.write(f"- Completed Tasks: {completed_tasks}\n")
            
            # Get average waiting time
            avg_waiting_time = metrics.get('avg_waiting_time', 0)
            f.write(f"- Average Waiting Time: {avg_waiting_time:.2f} seconds\n")
            
            # Add priority-specific metrics if available
            waiting_by_priority = metrics.get('avg_waiting_by_priority', {})
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
        f.write("### System Configuration\n\n")
        
        # Get processor count and strategy from first available metrics
        if multi_metrics:
            first_metrics = next(iter(multi_metrics.values()))
            processor_count = first_metrics.get('processor_count', 0)
            strategy = first_metrics.get('strategy', 'Unknown')
            
            f.write(f"- Processor Count: {processor_count}\n")
            f.write(f"- Load Balancing Strategy: {strategy}\n\n")
        
        for algo_name, metrics in multi_metrics.items():
            f.write(f"### {algo_name} Scheduler\n\n")
            
            # Get total completed tasks
            total_completed = metrics.get('total_completed_tasks', 0)
            f.write(f"- Total Completed Tasks: {total_completed}\n")
            
            # Get average waiting time
            avg_waiting_time = metrics.get('avg_waiting_time', 0)
            f.write(f"- Average Waiting Time: {avg_waiting_time:.2f} seconds\n")
            
            # Get throughput
            throughput = metrics.get('system_throughput', 0)
            f.write(f"- System Throughput: {throughput:.2f} tasks/second\n")
            
            # Get load balance
            load_balance = metrics.get('load_balance_cv', 0)
            f.write(f"- Load Balance CV: {load_balance:.2f}% (lower is better)\n")
            
            # Get CPU and memory usage
            cpu_usage = metrics.get('avg_cpu_usage', 0)
            memory_usage = metrics.get('avg_memory_usage', 0)
            
            f.write(f"- Average CPU Usage: {cpu_usage:.2f}%\n")
            f.write(f"- Average Memory Usage: {memory_usage:.2f}%\n\n")
            
            # Add tasks by priority if available
            tasks_by_priority = metrics.get('tasks_by_priority', {})
            if tasks_by_priority:
                f.write("- Tasks by Priority:\n")
                for priority, count in tasks_by_priority.items():
                    f.write(f"  - {priority}: {count}\n")
            
            # Add waiting times by priority if available
            waiting_by_priority = metrics.get('avg_waiting_by_priority', {})
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
            waiting_time = metrics.get('avg_waiting_time', 0)
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
            single_throughput = single_metrics[first_single_algo].get('avg_throughput', 0)
            
            # Get multi-processor throughput
            multi_throughput = multi_metrics[first_multi_algo].get('system_throughput', 0)
            
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
            waiting_times = {name: metrics.get('avg_waiting_time', float('inf')) 
                           for name, metrics in single_metrics.items()}
            
            if waiting_times:
                best_waiting = min(waiting_times.items(), key=lambda x: x[1])
                f.write(f"- **Best for Waiting Time**: {best_waiting[0]} scheduler had the lowest average waiting time ({best_waiting[1]:.2f}s).\n")
        
        # Compare deadline misses for EDF
        edf_single = single_metrics.get('EDF', {})
        edf_multi = multi_metrics.get('EDF', {})
        
        if 'deadline_misses' in edf_single and 'deadline_misses' in edf_multi:
            single_misses = edf_single['deadline_misses']
            multi_misses = edf_multi['deadline_misses']
            
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
                single_throughput = single_metrics[first_single].get('avg_throughput', 0)
            
            multi_throughput = first_metrics.get('system_throughput', 0)
            
            if single_throughput > 0:
                speedup = multi_throughput / single_throughput
                
                processor_count = first_metrics.get('processor_count', 0)
                if processor_count > 0:
                    ideal_speedup = processor_count
                    efficiency = (speedup / ideal_speedup) * 100
                    
                    f.write(f"- **Parallelisation Efficiency**: The multi-processor system achieved {efficiency:.1f}% of ideal speedup.\n")
            
            # Check load balance
            load_balance = first_metrics.get('load_balance_cv', 0)
            
            if load_balance > 0:
                if load_balance < 10:
                    f.write("- **Load Balancing**: Excellent load distribution across processors.\n")
                elif load_balance < 20:
                    f.write("- **Load Balancing**: Good load distribution, but some processors were underutilised.\n")
                else:
                    f.write("- **Load Balancing**: Poor load distribution, significant processor imbalance.\n")
        
        f.write("\nThis report provides a quantitative analysis of different scheduling algorithms on both single and multi-processor systems.")
    
    return report_path

def process_directory(data_dir, output_dir=None, schedulers=None):
    """
    Process data files in a directory and generate visualizations
    
    Args:
        data_dir: Path to the directory containing data files
        output_dir: Path to save visualizations (if None, uses data_dir/visualizations)
        schedulers: List of schedulers to process (if None, processes all found)
    """
    data_dir = Path(data_dir)
    
    # Default output directory
    if output_dir is None:
        output_dir = data_dir.parent / f"visualizations_{data_dir.name}"
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Processing data from {data_dir}")
    logger.info(f"Saving visualizations to {output_dir}")
    
    # Load system information
    system_info_path = data_dir / "system_info" / "platform_info.json"
    system_info = {}
    if system_info_path.exists():
        with open(system_info_path, 'r') as f:
            system_info = json.load(f)
        logger.info(f"Loaded system info: {system_info['system']} {system_info['node']}")
    
    # Create subdirectories for visualizations
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
        for file_path in single_processor_dir.glob("*_metrics.json"):
            scheduler_name = file_path.name.split('_')[0]
            scheduler_types.add(scheduler_name)
        
        # Filter by requested schedulers if provided
        if schedulers:
            scheduler_types = [s for s in scheduler_types if s in schedulers]
        
        logger.info(f"Found single processor data for schedulers: {', '.join(scheduler_types)}")
        
        # Process each scheduler
        for scheduler in scheduler_types:
            scheduler_dir = os.path.join(vis_single_dir, scheduler)
            os.makedirs(scheduler_dir, exist_ok=True)
            
            # Load metrics
            metrics_path = single_processor_dir / f"{scheduler}_Raspberry_Pi_3_metrics.json"
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                single_metrics[scheduler] = metrics
            
            # Load tasks data
            tasks_path = single_processor_dir / f"{scheduler}_Raspberry_Pi_3_tasks.csv"
            if tasks_path.exists():
                tasks_df = pd.read_csv(tasks_path)
                
                # Generate task visualizations
                plot_task_completion(
                    tasks_df, 
                    f"{scheduler} on Single Processor",
                    os.path.join(scheduler_dir, "task_completion.png")
                )
                
                plot_waiting_times(
                    tasks_df,
                    f"{scheduler} on Single Processor",
                    os.path.join(scheduler_dir, "waiting_times.png")
                )
                
                plot_priority_distribution(
                    tasks_df,
                    f"{scheduler} on Single Processor",
                    os.path.join(scheduler_dir, "priority_distribution.png")
                )
                
                plot_waiting_times_by_priority(
                    tasks_df,
                    f"{scheduler} on Single Processor",
                    os.path.join(scheduler_dir, "waiting_times_by_priority.png")
                )
            
            # Load timeseries data
            timeseries_path = single_processor_dir / f"{scheduler}_Raspberry_Pi_3_timeseries.csv"
            if timeseries_path.exists():
                timeseries_df = pd.read_csv(timeseries_path)
                
                # Generate timeseries visualizations
                plot_memory_usage(
                    timeseries_df,
                    f"{scheduler} on Single Processor",
                    os.path.join(scheduler_dir, "memory_usage.png")
                )
                
                plot_queue_length(
                    timeseries_df,
                    f"{scheduler} on Single Processor",
                    os.path.join(scheduler_dir, "queue_length.png")
                )
    
    # Process multi-processor data
    multi_processor_dir = data_dir / "multi_processor"
    multi_metrics = {}
    
    if multi_processor_dir.exists():
        # Find all scheduler types from metrics files
        scheduler_types = set()
        for file_path in multi_processor_dir.glob("*_system_metrics.json"):
            scheduler_name = file_path.name.split('_')[0]
            scheduler_types.add(scheduler_name)
        
        # Filter by requested schedulers if provided
        if schedulers:
            scheduler_types = [s for s in scheduler_types if s in schedulers]
        
        logger.info(f"Found multi-processor data for schedulers: {', '.join(scheduler_types)}")
        
        # Process each scheduler
        for scheduler in scheduler_types:
            scheduler_dir = os.path.join(vis_multi_dir, scheduler)
            os.makedirs(scheduler_dir, exist_ok=True)
            
            # Load system metrics
            metrics_path = multi_processor_dir / f"{scheduler}_system_metrics.json"
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                multi_metrics[scheduler] = metrics
            
            # Load tasks data if available (might be per-processor)
            tasks_pattern = f"{scheduler}_CPU-*_tasks.csv"
            task_files = list(multi_processor_dir.glob(tasks_pattern))
            
            if task_files:
                # Process each processor's tasks
                for task_file in task_files:
                    processor_name = task_file.name.split('_')[1]
                    processor_dir = os.path.join(scheduler_dir, processor_name)
                    os.makedirs(processor_dir, exist_ok=True)
                    
                    tasks_df = pd.read_csv(task_file)
                    
                    # Generate per-processor visualizations
                    plot_task_completion(
                        tasks_df, 
                        f"{scheduler} on {processor_name}",
                        os.path.join(processor_dir, "task_completion.png")
                    )
                    
                    plot_waiting_times(
                        tasks_df,
                        f"{scheduler} on {processor_name}",
                        os.path.join(processor_dir, "waiting_times.png")
                    )
                    
                    plot_priority_distribution(
                        tasks_df,
                        f"{scheduler} on {processor_name}",
                        os.path.join(processor_dir, "priority_distribution.png")
                    )
                
                # Combine all tasks for a system-wide view
                all_tasks = []
                for task_file in task_files:
                    df = pd.read_csv(task_file)
                    all_tasks.append(df)
                
                if all_tasks:
                    combined_df = pd.concat(all_tasks)
                    
                    # Generate combined visualizations
                    plot_task_completion(
                        combined_df, 
                        f"{scheduler} on Multi-Processor (All CPUs)",
                        os.path.join(scheduler_dir, "task_completion_all_cpus.png")
                    )
                    
                    plot_waiting_times(
                        combined_df,
                        f"{scheduler} on Multi-Processor (All CPUs)",
                        os.path.join(scheduler_dir, "waiting_times_all_cpus.png")
                    )
                    
                    plot_priority_distribution(
                        combined_df,
                        f"{scheduler} on Multi-Processor (All CPUs)",
                        os.path.join(scheduler_dir, "priority_distribution.png")
                    )
                    
                    plot_waiting_times_by_priority(
                        combined_df,
                        f"{scheduler} on Multi-Processor (All CPUs)",
                        os.path.join(scheduler_dir, "waiting_times_by_priority.png")
                    )
            
            # Load timeseries data
            timeseries_path = multi_processor_dir / f"{scheduler}_system_timeseries.csv"
            if timeseries_path.exists():
                timeseries_df = pd.read_csv(timeseries_path)
                
                # Generate timeseries visualizations
                plot_memory_usage(
                    timeseries_df,
                    f"{scheduler} on Multi-Processor",
                    os.path.join(scheduler_dir, "memory_usage.png")
                )
                
                plot_queue_length(
                    timeseries_df,
                    f"{scheduler} on Multi-Processor",
                    os.path.join(scheduler_dir, "queue_length.png")
                )
    
    # Generate comparisons if we have both single and multi-processor data
    if single_metrics and multi_metrics:
        common_schedulers = set(single_metrics.keys()) & set(multi_metrics.keys())
        
        logger.info(f"Generating comparisons for schedulers: {', '.join(common_schedulers)}")
        
        # Compare waiting times across algorithms (single processor)
        waiting_times = {name: metrics.get('avg_waiting_time', 0) 
                       for name, metrics in single_metrics.items()}
        
        if waiting_times:
            plot_algorithm_comparison(
                waiting_times,
                'avg_waiting_time',
                'Average Waiting Time Comparison (Single Processor)',
                'Waiting Time (s)',
                os.path.join(vis_compare_dir, "waiting_time_comparison_single.png")
            )
        
        # Compare waiting times across algorithms (multi-processor)
        multi_waiting_times = {name: metrics.get('avg_waiting_time', 0)
                             for name, metrics in multi_metrics.items()}
        
        if multi_waiting_times:
            plot_algorithm_comparison(
                multi_waiting_times,
                'avg_waiting_time',
                'Average Waiting Time Comparison (Multi-Processor)',
                'Waiting Time (s)',
                os.path.join(vis_compare_dir, "waiting_time_comparison_multi.png")
            )
        
        # Compare throughput between single and multi-processor for each scheduler
        for scheduler in common_schedulers:
            if scheduler in single_metrics and scheduler in multi_metrics:
                single_throughput = single_metrics[scheduler].get('avg_throughput', 0)
                multi_throughput = multi_metrics[scheduler].get('system_throughput', 0)
                
                if single_throughput > 0 or multi_throughput > 0:
                    plot_processor_comparison(
                        {'avg_throughput': single_throughput},
                        {'system_throughput': multi_throughput},
                        'avg_throughput',
                        f'Throughput Comparison - {scheduler}',
                        'Tasks/second',
                        os.path.join(vis_compare_dir, f"throughput_comparison_{scheduler}.png")
                    )
        
        # Generate comprehensive report
        try:
            generate_report(
                single_metrics,
                multi_metrics,
                os.path.join(vis_compare_dir, "performance_report.md")
            )
            logger.info(f"Generated performance report: {os.path.join(vis_compare_dir, 'performance_report.md')}")
        except Exception as e:
            logger.error(f"Error generating report: {e}")
    
    logger.info("Visualization generation complete")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Visualise task scheduling data')
    parser.add_argument('--data-dir', required=True, help='Directory containing data files')
    parser.add_argument('--output-dir', help='Directory to save visualizations')
    parser.add_argument('--scheduler', action='append', dest='schedulers', 
                        help='Specific scheduler to visualise (can be used multiple times)')
    
    args = parser.parse_args()
    
    process_directory(args.data_dir, args.output_dir, args.schedulers)

if __name__ == "__main__":
    main()