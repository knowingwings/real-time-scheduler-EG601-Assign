"""
Real-Time Task Scheduling on Raspberry Pi 3

Main execution script for the task scheduling project.
"""

import time
import logging
import os
import argparse
import platform
import sys
import psutil
from datetime import datetime

# Import project modules
from src.task_generator import TaskGenerator, Priority
from src.schedulers.fcfs import FCFSScheduler
from src.schedulers.edf import EDFScheduler
from src.schedulers.priority_based import PriorityScheduler
from src.schedulers.ml_scheduler import MLScheduler
from src.processors.single_processor import SingleProcessor
from src.processors.multi_processor import MultiProcessor
from src.utils.data_collector import (
    ensure_output_dir, save_system_info, save_task_metrics,
    save_time_series_metrics, save_scheduler_metrics,
    save_multi_processor_metrics, save_comparison_results
)
from src.utils.metrics import MetricsCalculator
from config.params import (
    TASK_CONFIG, SINGLE_PROCESSOR, MULTI_PROCESSOR,
    SIMULATION, ML_SCHEDULER, COMPARISON
)

def setup_logging():
    """Set up logging to write to results/logs directory"""
    # Create logs directory if it doesn't exist
    logs_dir = 'results/logs'
    os.makedirs(logs_dir, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(logs_dir, f'scheduler_{timestamp}.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Get the root logger and add a message about where logs are being saved
    logger = logging.getLogger()
    logger.info(f"Logging to file: {log_file}")
    
    return logger, timestamp

# Global logger will be set up when this module is imported
logger = None
timestamp = None

def get_platform_info():
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
        'memory_available': psutil.virtual_memory().available
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

def run_single_processor(tasks, timestamp, simulation=False):
    """
    Run scheduling on a single processor
    
    Args:
        tasks: List of Task objects
        timestamp: Timestamp for saving files
        simulation: Whether to run in simulation mode
        
    Returns:
        Dictionary containing metrics for each scheduler
    """
    logger.info("Starting single processor scheduling")
    
    # Initialise schedulers
    schedulers = {
        'FCFS': FCFSScheduler(),
        'EDF': EDFScheduler(),
        'Priority': PriorityScheduler(),
        'ML-Based': MLScheduler(history_size=ML_SCHEDULER['history_size'])
    }
    
    # Run each scheduler and collect metrics
    results = {}
    
    for name, scheduler in schedulers.items():
        logger.info(f"Running {name} scheduler on single processor")
        
        # Create processor with this scheduler
        processor = SingleProcessor(scheduler, name=f"{SINGLE_PROCESSOR['name']}-{name}")
        
        # Add tasks
        processor.add_tasks(tasks)
        
        # Run processor
        start_time = time.time()
        processor.run(simulation=simulation, speed_factor=SIMULATION['speed_factor'])
        end_time = time.time()
        
        # Get metrics
        metrics = processor.get_metrics()
        metrics['execution_time'] = end_time - start_time
        
        # Store results
        results[name] = metrics
        
        # Save data instead of visualizing
        if len(scheduler.completed_tasks) > 0:
            # Save task metrics
            save_task_metrics(
                scheduler.completed_tasks,
                name,
                SINGLE_PROCESSOR['name'],
                timestamp=timestamp,
                processor_type='single'
            )
            
            # Save time series metrics
            save_time_series_metrics(
                metrics,
                name,
                SINGLE_PROCESSOR['name'],
                timestamp=timestamp,
                processor_type='single'
            )
            
            # Save scheduler metrics
            save_scheduler_metrics(
                metrics,
                name,
                SINGLE_PROCESSOR['name'],
                timestamp=timestamp,
                processor_type='single'
            )
            
            # Calculate and save tasks by priority
            tasks_by_priority = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
            for task in scheduler.completed_tasks:
                tasks_by_priority[task.priority.name] += 1
            
            metrics['tasks_by_priority'] = tasks_by_priority
            
            # Calculate waiting times by priority
            waiting_by_priority = {'HIGH': [], 'MEDIUM': [], 'LOW': []}
            for task in scheduler.completed_tasks:
                if task.waiting_time is not None:
                    waiting_by_priority[task.priority.name].append(task.waiting_time)
            
            avg_waiting_by_priority = {}
            for priority, times in waiting_by_priority.items():
                avg_waiting_by_priority[priority] = sum(times) / len(times) if times else 0
            
            metrics['avg_waiting_by_priority'] = avg_waiting_by_priority
        
        logger.info(f"Completed {name} scheduler on single processor")
    
    logger.info("Completed all single processor scheduling")
    return results

def run_multi_processor(tasks, timestamp, simulation=False):
    """
    Run scheduling on multiple processors
    
    Args:
        tasks: List of Task objects
        timestamp: Timestamp for saving files
        simulation: Whether to run in simulation mode
        
    Returns:
        Dictionary containing metrics for each scheduler
    """
    logger.info("Starting multi-processor scheduling")
    
    # Initialise schedulers for each core
    scheduler_classes = {
        'FCFS': FCFSScheduler,
        'EDF': EDFScheduler,
        'Priority': PriorityScheduler,
        'ML-Based': MLScheduler
    }
    
    # Run each scheduler type and collect metrics
    results = {}
    
    for name, scheduler_class in scheduler_classes.items():
        logger.info(f"Running {name} scheduler on multi-processor")
        
        try:
            # Create schedulers for each core
            schedulers = [scheduler_class() for _ in range(MULTI_PROCESSOR['cores'])]
            
            # Create multi-processor system
            processor = MultiProcessor(
                schedulers, 
                processor_count=MULTI_PROCESSOR['cores'],
                strategy=MULTI_PROCESSOR['strategy']
            )
            
            # Add tasks
            processor.add_tasks(tasks)
            
            # Run processor
            start_time = time.time()
            processor.run(simulation=simulation, speed_factor=SIMULATION['speed_factor'])
            end_time = time.time()
            
            # Get metrics
            metrics = processor.get_metrics()
            metrics['execution_time'] = end_time - start_time
            
            # Store results
            results[name] = metrics
            
            # Save data for multi-processor system
            save_multi_processor_metrics(
                metrics,
                name,
                timestamp=timestamp
            )
            
            # Save time series metrics
            save_time_series_metrics(
                metrics,
                name,
                f"System-{MULTI_PROCESSOR['strategy']}",
                timestamp=timestamp,
                processor_type='multi'
            )
            
            # Save data for individual processors
            if processor.processors and any(len(p.scheduler.completed_tasks) > 0 for p in processor.processors):
                for i, proc in enumerate(processor.processors):
                    if len(proc.scheduler.completed_tasks) > 0:
                        # Save task metrics for this processor
                        save_task_metrics(
                            proc.scheduler.completed_tasks,
                            name,
                            f"CPU-{i+1}",
                            timestamp=timestamp,
                            processor_type='multi'
                        )
                
                # Save aggregate data for all tasks
                all_completed_tasks = []
                for proc in processor.processors:
                    all_completed_tasks.extend(proc.scheduler.completed_tasks)
                
                if all_completed_tasks:
                    # Save combined task data
                    save_task_metrics(
                        all_completed_tasks,
                        name,
                        "All-CPUs",
                        timestamp=timestamp,
                        processor_type='multi'
                    )
                    
        except Exception as e:
            logger.error(f"Error running {name} scheduler on multi-processor: {e}")
            # Still add an entry to the results, but with None value
            results[name] = None
    
    logger.info("Completed multi-processor scheduling")
    return results


def compare_algorithms(single_metrics, multi_metrics, timestamp):
    """
    Compare performance across different algorithms
    
    Args:
        single_metrics: Metrics from single processor
        multi_metrics: Metrics from multi-processor
        timestamp: Timestamp for saving files
        
    Returns:
        None (saves comparison results)
    """
    logger.info("Comparing algorithms")
    
    # Ensure we have valid dictionaries
    single_metrics = {} if single_metrics is None else single_metrics
    multi_metrics = {} if multi_metrics is None else multi_metrics
    
    # Early return if both metrics are empty
    if not single_metrics and not multi_metrics:
        logger.warning("No metrics available for comparison. Skipping algorithm comparison.")
        return
    
    # Prepare metrics calculator
    calculator = MetricsCalculator()
    
    # Create comparison data structure
    comparison_results = {
        'single_processor': {},
        'multi_processor': {},
        'cross_system': {}
    }
    
    # Compare waiting times across algorithms (single processor)
    if single_metrics:
        avg_waiting_times = {}
        for name, metrics in single_metrics.items():
            if metrics is None:
                continue  # Skip None metrics
                
            # Try different possible key names for waiting time
            waiting_time = None
            for key in ['average_waiting_time', 'avg_waiting_time', 'mean_waiting_time']:
                if key in metrics:
                    waiting_time = metrics[key]
                    break
            
            # Default to 0 if we still don't have a waiting time
            avg_waiting_times[name] = waiting_time if waiting_time is not None else 0
        
        if avg_waiting_times:
            comparison_results['single_processor']['waiting_times'] = avg_waiting_times
    
    # Compare waiting times across algorithms (multi-processor)
    if multi_metrics:
        multi_waiting_times = {}
        for name, metrics in multi_metrics.items():
            if metrics is None:
                continue  # Skip None metrics
                
            # Try different possible key names for waiting time
            waiting_time = None
            for key in ['avg_waiting_time', 'average_waiting_time', 'mean_waiting_time']:
                if key in metrics:
                    waiting_time = metrics[key]
                    break
            
            # Default to 0 if we still don't have a waiting time
            multi_waiting_times[name] = waiting_time if waiting_time is not None else 0
        
        if multi_waiting_times:
            comparison_results['multi_processor']['waiting_times'] = multi_waiting_times
    
    # Compare throughput between single and multi-processor
    if single_metrics and multi_metrics:
        throughput_comparison = {}
        
        for scheduler in ['FCFS', 'EDF', 'Priority', 'ML-Based']:
            # Check if this scheduler exists in both metrics
            if (scheduler in single_metrics and 
                scheduler in multi_metrics and 
                single_metrics[scheduler] is not None and
                multi_metrics[scheduler] is not None):
                
                # Try to extract throughput metrics with different possible key names
                single_throughput = 0
                s_metrics = single_metrics[scheduler]
                if s_metrics:  # Check if not None
                    for key in ['avg_throughput', 'throughput', 'tasks_per_second']:
                        if key in s_metrics:
                            single_throughput = s_metrics[key]
                            break
                
                multi_throughput = 0
                m_metrics = multi_metrics[scheduler]
                if m_metrics:  # Check if not None
                    for key in ['system_throughput', 'throughput', 'tasks_per_second']:
                        if key in m_metrics:
                            multi_throughput = m_metrics[key]
                            break
                
                if single_throughput > 0 or multi_throughput > 0:
                    throughput_comparison[scheduler] = {
                        'single': single_throughput,
                        'multi': multi_throughput,
                        'speedup': multi_throughput / single_throughput if single_throughput > 0 else 0
                    }
        
        if throughput_comparison:
            comparison_results['cross_system']['throughput'] = throughput_comparison
    
    # Compare waiting times by priority
    if single_metrics and multi_metrics:
        priority_waiting_comparison = {}
        
        for scheduler in ['FCFS', 'EDF', 'Priority', 'ML-Based']:
            # Check if this scheduler exists in both metrics
            if (scheduler in single_metrics and 
                scheduler in multi_metrics and 
                single_metrics[scheduler] is not None and
                multi_metrics[scheduler] is not None):
                
                # Try different key names for waiting times by priority
                single_waiting_by_priority = {}
                s_metrics = single_metrics[scheduler]
                if s_metrics:  # Check if not None
                    for key in ['avg_waiting_by_priority', 'waiting_times_by_priority']:
                        if key in s_metrics:
                            single_waiting_by_priority = s_metrics[key]
                            break
                
                # For multi-processor
                multi_waiting_by_priority = {}
                m_metrics = multi_metrics[scheduler]
                if m_metrics:  # Check if not None
                    for key in ['avg_waiting_by_priority', 'waiting_times_by_priority']:
                        if key in m_metrics:
                            multi_waiting_by_priority = m_metrics[key]
                            break
                
                if single_waiting_by_priority and multi_waiting_by_priority:
                    priority_comparison = {}
                    
                    for priority in ['HIGH', 'MEDIUM', 'LOW']:
                        if priority in single_waiting_by_priority and priority in multi_waiting_by_priority:
                            single_time = single_waiting_by_priority[priority]
                            multi_time = multi_waiting_by_priority[priority]
                            
                            priority_comparison[priority] = {
                                'single': single_time,
                                'multi': multi_time,
                                'improvement': ((single_time - multi_time) / single_time * 100) if single_time > 0 else 0
                            }
                    
                    if priority_comparison:
                        priority_waiting_comparison[scheduler] = priority_comparison
        
        if priority_waiting_comparison:
            comparison_results['cross_system']['waiting_by_priority'] = priority_waiting_comparison
    
    # Save comparison results
    save_comparison_results(
        comparison_results,
        list(set([*single_metrics.keys(), *multi_metrics.keys()])),
        timestamp=timestamp
    )
    
    logger.info("Completed algorithm comparison")

def compare_platforms(platform_info, current_metrics, timestamp):
    """
    Compare performance across different platforms
    
    Args:
        platform_info: Information about the current platform
        current_metrics: Metrics from the current platform
        timestamp: Timestamp for saving files
        
    Returns:
        None (saves metrics for later comparison)
    """
    # Determine platform type
    platform_type = platform_info.get('type', 'unknown')
    platform_name = None
    
    for platform_config in COMPARISON['platforms']:
        if platform_config['type'] == platform_type:
            platform_name = platform_config['name']
            break
    
    if not platform_name:
        logger.warning(f"Unknown platform type: {platform_type}")
        return
    
    logger.info(f"Saving metrics for platform: {platform_name}")
    
    # We already saved all the metrics through the data_collector functions
    # Just log the platform used
    with open(f"results/data/{timestamp}/platform_used.txt", 'w') as f:
        f.write(f"Platform: {platform_name}\n")
        f.write(f"Type: {platform_type}\n")
        f.write(f"System: {platform_info['system']}\n")
        f.write(f"Node: {platform_info['node']}\n")
    
    logger.info(f"Saved platform information")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Real-Time Task Scheduling on Raspberry Pi 3')
    
    parser.add_argument('--single', action='store_true', help='Run single processor tests')
    parser.add_argument('--multi', action='store_true', help='Run multi-processor tests')
    parser.add_argument('--compare', action='store_true', help='Compare algorithms')
    parser.add_argument('--platforms', action='store_true', help='Compare platforms')
    parser.add_argument('--simulation', action='store_true', help='Run in simulation mode')
    parser.add_argument('--speed', type=float, default=SIMULATION['speed_factor'], help='Simulation speed factor')
    
    # Use a safer way to access task count - either through the total across all priorities
    # or falling back to a default value of 50
    total_tasks = 0
    if 'high_priority' in TASK_CONFIG and 'count' in TASK_CONFIG['high_priority']:
        total_tasks += TASK_CONFIG['high_priority']['count']
    if 'medium_priority' in TASK_CONFIG and 'count' in TASK_CONFIG['medium_priority']:
        total_tasks += TASK_CONFIG['medium_priority']['count']
    if 'low_priority' in TASK_CONFIG and 'count' in TASK_CONFIG['low_priority']:
        total_tasks += TASK_CONFIG['low_priority']['count']
    
    parser.add_argument('--tasks', type=int, default=total_tasks if total_tasks > 0 else 50, 
                        help='Total number of tasks to generate')
    
    return parser.parse_args()

def main():
    """Main execution function"""
    # Initialise logging
    global logger, timestamp
    logger, timestamp = setup_logging()
    
    # Initialize run_timestamp - will be set properly later
    run_timestamp = timestamp
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Update simulation speed if provided
    if args.speed:
        SIMULATION['speed_factor'] = args.speed
    
    # Update task count if provided - distribute proportionally across priority levels
    if args.tasks and args.tasks > 0:
        # Calculate the original distribution ratio
        high_count = TASK_CONFIG.get('high_priority', {}).get('count', 20)
        medium_count = TASK_CONFIG.get('medium_priority', {}).get('count', 20)
        low_count = TASK_CONFIG.get('low_priority', {}).get('count', 10)
        
        total_original = high_count + medium_count + low_count
        if total_original > 0:
            # Calculate new counts maintaining the same ratio
            high_ratio = high_count / total_original
            medium_ratio = medium_count / total_original
            low_ratio = low_count / total_original
            
            # Update the counts
            if 'high_priority' in TASK_CONFIG:
                TASK_CONFIG['high_priority']['count'] = int(args.tasks * high_ratio)
            if 'medium_priority' in TASK_CONFIG:
                TASK_CONFIG['medium_priority']['count'] = int(args.tasks * medium_ratio)
            if 'low_priority' in TASK_CONFIG:
                TASK_CONFIG['low_priority']['count'] = int(args.tasks * low_ratio)
            
            # Adjust for any rounding errors to ensure total equals args.tasks
            current_total = (TASK_CONFIG.get('high_priority', {}).get('count', 0) + 
                             TASK_CONFIG.get('medium_priority', {}).get('count', 0) + 
                             TASK_CONFIG.get('low_priority', {}).get('count', 0))
            
            if current_total != args.tasks and 'high_priority' in TASK_CONFIG:
                TASK_CONFIG['high_priority']['count'] += (args.tasks - current_total)
    
    # Set up output directories and get timestamp if needed
    run_timestamp = ensure_output_dir()
    # If we already have a timestamp from logging, use that instead
    if timestamp:
        run_timestamp = timestamp
    save_system_info(timestamp=run_timestamp)
    
    # Get platform information
    platform_info = get_platform_info()
    logger.info(f"Running on platform: {platform_info['system']} - {platform_info['node']}")
    
    # Generate tasks
    generator = TaskGenerator(config=TASK_CONFIG)
    tasks = generator.generate_tasks()
    logger.info(f"Generated {len(tasks)} tasks")
    
    # Store results
    results = {}
    
    # Run single processor tests
    if args.single or not (args.single or args.multi):
        try:
            single_results = run_single_processor(tasks, run_timestamp, simulation=args.simulation or SIMULATION['enabled'])
            results['single'] = single_results
        except Exception as e:
            logger.error(f"Error running single processor simulation: {e}")
            results['single'] = None
    
    # Run multi-processor tests
    if args.multi or not (args.single or args.multi):
        try:
            multi_results = run_multi_processor(tasks, run_timestamp, simulation=args.simulation or SIMULATION['enabled'])
            results['multi'] = multi_results
        except Exception as e:
            logger.error(f"Error running multi-processor simulation: {e}")
            results['multi'] = None
    
    # Compare algorithms
    if args.compare or not args.platforms:
        # Check if we have any results to compare
        if 'single' in results or 'multi' in results:
            # Make sure to pass None if a result doesn't exist
            single_data = results.get('single')
            multi_data = results.get('multi')
            compare_algorithms(single_data, multi_data, run_timestamp)
        else:
            logger.warning("No results available for comparison. Skipping algorithm comparison.")
    
    # Compare platforms - only if both args.platforms is True and we have some valid results
    if args.platforms and results:
        # Check if we have at least one valid result (single or multi)
        has_valid_results = False
        if 'single' in results and results['single'] is not None:
            has_valid_results = True
        if 'multi' in results and results['multi'] is not None:
            has_valid_results = True
            
        if has_valid_results:
            compare_platforms(platform_info, results, run_timestamp)
        else:
            logger.warning("No valid results available for platform comparison. Skipping platform comparison.")
    
    logger.info(f"Task scheduling completed successfully. Data saved with timestamp: {run_timestamp}")
    logger.info(f"To generate visualizations, run: python visualize.py --data-dir results/data/{run_timestamp}")

if __name__ == "__main__":
    main()