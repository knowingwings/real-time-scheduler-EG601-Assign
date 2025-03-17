"""
Visualisation Utilities

This module provides functions for visualising task scheduling results.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import os

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
        
        f.write("\nThis report provides a quantitative analysis of different scheduling algorithms on both single and multi-processor systems.")
    
    return report_path