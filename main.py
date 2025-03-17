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
from src.utils.visualisation import (
    plot_task_completion, plot_waiting_times, plot_memory_usage,
    plot_queue_length, plot_algorithm_comparison, plot_processor_comparison,
    generate_report
)
from src.utils.metrics import MetricsCalculator
from config.params import (
    TASK_CONFIG, SINGLE_PROCESSOR, MULTI_PROCESSOR,
    SIMULATION, ML_SCHEDULER, COMPARISON, VISUALISATION
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scheduler.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def setup_output_dirs():
    """Set up output directories for results"""
    # Create main results directory
    os.makedirs('results', exist_ok=True)
    
    # Create subdirectories
    directories = [
        'results/single_processor',
        'results/multi_processor',
        'results/comparison',
        'results/logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    # Create platform-specific directories
    for platform_info in COMPARISON['platforms']:
        platform_dir = f"results/comparison/{platform_info['name'].replace(' ', '_')}"
        os.makedirs(platform_dir, exist_ok=True)
    
    logger.info("Created output directories")

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

def run_single_processor(tasks, simulation=False):
    """
    Run scheduling on a single processor
    
    Args:
        tasks: List of Task objects
        simulation: Whether to run in simulation mode
        
    Returns:
        Dictionary containing metrics for each scheduler
    """
    logger.info("Starting single processor scheduling")
    
    # Initialize schedulers
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
        
        # Visualise results
        if len(scheduler.completed_tasks) > 0:
            # Create output directory
            output_dir = f"results/single_processor/{name}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate plots
            plot_task_completion(
                scheduler.completed_tasks, 
                f"{name} on Single Processor",
                f"{output_dir}/task_completion.png"
            )
            
            plot_waiting_times(
                scheduler.completed_tasks,
                f"{name} on Single Processor",
                f"{output_dir}/waiting_times.png"
            )
            
            plot_memory_usage(
                metrics,
                f"{name} on Single Processor",
                f"{output_dir}/memory_usage.png"
            )
            
            plot_queue_length(
                metrics,
                f"{name} on Single Processor",
                f"{output_dir}/queue_length.png"
            )
        
        logger.info(f"Completed {name} scheduler on single processor")
    
    logger.info("Completed all single processor scheduling")
    return results

def run_multi_processor(tasks, simulation=False):
    """
    Run scheduling on multiple processors
    
    Args:
        tasks: List of Task objects
        simulation: Whether to run in simulation mode
        
    Returns:
        Dictionary containing metrics for each scheduler
    """
    logger.info("Starting multi-processor scheduling")
    
    # Initialize schedulers for each core
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
        
        # Visualise results (using data from the first processor for task-specific visualisations)
        if processor.processors and len(processor.processors[0].scheduler.completed_tasks) > 0:
            # Create output directory
            output_dir = f"results/multi_processor/{name}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate plots
            # For task completion and waiting times, we'll use the first processor's data
            first_processor = processor.processors[0]
            
            plot_task_completion(
                first_processor.scheduler.completed_tasks, 
                f"{name} on CPU-1 (Multi-Processor)",
                f"{output_dir}/task_completion_cpu1.png"
            )
            
            plot_waiting_times(
                first_processor.scheduler.completed_tasks,
                f"{name} on CPU-1 (Multi-Processor)",
                f"{output_dir}/waiting_times_cpu1.png"
            )
            
            # For system-wide metrics like memory and queue length, use the multi-processor metrics
            plot_memory_usage(
                metrics,
                f"{name} on Multi-Processor",
                f"{output_dir}/memory_usage.png"
            )
            
            # Generate per-processor visualisations if needed
            # This could be extended to create comparisons between processors
        
        logger.info(f"Completed {name} scheduler on multi-processor")
    
    logger.info("Completed all multi-processor scheduling")
    return results

def compare_algorithms(single_metrics, multi_metrics):
    """
    Compare performance across different algorithms
    
    Args:
        single_metrics: Metrics from single processor
        multi_metrics: Metrics from multi-processor
        
    Returns:
        None (generates visualisations)
    """
    logger.info("Comparing algorithms")
    
    # Create output directory
    output_dir = f"results/comparison"
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare metrics calculator
    calculator = MetricsCalculator()
    
    # Compare waiting times across algorithms (single processor)
    avg_waiting_times = {
        name: metrics['average_waiting_time']
        for name, metrics in single_metrics.items()
    }
    
    plot_algorithm_comparison(
        avg_waiting_times,
        'average_waiting_time',
        'Average Waiting Time Comparison (Single Processor)',
        'Waiting Time (s)',
        f"{output_dir}/waiting_time_comparison_single.png"
    )
    
    # Compare waiting times across algorithms (multi-processor)
    multi_waiting_times = {
        name: metrics['avg_waiting_time']
        for name, metrics in multi_metrics.items()
    }
    
    plot_algorithm_comparison(
        multi_waiting_times,
        'avg_waiting_time',
        'Average Waiting Time Comparison (Multi-Processor)',
        'Waiting Time (s)',
        f"{output_dir}/waiting_time_comparison_multi.png"
    )
    
    # Compare throughput between single and multi-processor
    for scheduler in ['FCFS', 'EDF', 'Priority', 'ML-Based']:
        if scheduler in single_metrics and scheduler in multi_metrics:
            single_throughput = single_metrics[scheduler].get('avg_throughput', 0)
            multi_throughput = multi_metrics[scheduler].get('system_throughput', 0)
            
            if single_throughput > 0 and multi_throughput > 0:
                plot_processor_comparison(
                    {'avg_throughput': single_throughput},
                    {'system_throughput': multi_throughput},
                    'throughput',
                    f'Throughput Comparison - {scheduler}',
                    'Tasks/second',
                    f"{output_dir}/throughput_comparison_{scheduler}.png"
                )
    
    # Generate comprehensive report
    generate_report(
        single_metrics,
        multi_metrics,
        f"{output_dir}/performance_report.md"
    )
    
    logger.info("Completed algorithm comparison")

def compare_platforms(platform_info, current_metrics):
    """
    Compare performance across different platforms
    
    Args:
        platform_info: Information about the current platform
        current_metrics: Metrics from the current platform
        
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
    
    # Create platform-specific directory
    platform_dir = f"results/comparison/{platform_name.replace(' ', '_')}"
    os.makedirs(platform_dir, exist_ok=True)
    
    # Save metrics
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save single processor metrics
    if 'single' in current_metrics:
        with open(f"{platform_dir}/single_processor_{timestamp}.txt", 'w') as f:
            for scheduler, metrics in current_metrics['single'].items():
                f.write(f"Scheduler: {scheduler}\n")
                f.write(f"Completed Tasks: {metrics.get('completed_tasks', 0)}\n")
                f.write(f"Average Waiting Time: {metrics.get('average_waiting_time', 0):.2f}s\n")
                f.write(f"Execution Time: {metrics.get('execution_time', 0):.2f}s\n")
                f.write("\n")
    
    # Save multi-processor metrics
    if 'multi' in current_metrics:
        with open(f"{platform_dir}/multi_processor_{timestamp}.txt", 'w') as f:
            for scheduler, metrics in current_metrics['multi'].items():
                f.write(f"Scheduler: {scheduler}\n")
                f.write(f"Total Completed Tasks: {metrics.get('total_completed_tasks', 0)}\n")
                f.write(f"Average Waiting Time: {metrics.get('avg_waiting_time', 0):.2f}s\n")
                f.write(f"System Throughput: {metrics.get('system_throughput', 0):.2f} tasks/s\n")
                f.write(f"Execution Time: {metrics.get('execution_time', 0):.2f}s\n")
                f.write("\n")
    
    # Save system info
    with open(f"{platform_dir}/system_info_{timestamp}.txt", 'w') as f:
        for key, value in platform_info.items():
            f.write(f"{key}: {value}\n")
    
    logger.info(f"Saved metrics for platform: {platform_name}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Real-Time Task Scheduling on Raspberry Pi 3')
    
    parser.add_argument('--single', action='store_true', help='Run single processor tests')
    parser.add_argument('--multi', action='store_true', help='Run multi-processor tests')
    parser.add_argument('--compare', action='store_true', help='Compare algorithms')
    parser.add_argument('--platforms', action='store_true', help='Compare platforms')
    parser.add_argument('--simulation', action='store_true', help='Run in simulation mode')
    parser.add_argument('--speed', type=float, default=SIMULATION['speed_factor'], help='Simulation speed factor')
    
    return parser.parse_args()

def main():
    """Main execution function"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up output directories
    setup_output_dirs()
    
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
        single_results = run_single_processor(tasks, simulation=args.simulation or SIMULATION['enabled'])
        results['single'] = single_results
    
    # Run multi-processor tests
    if args.multi or not (args.single or args.multi):
        multi_results = run_multi_processor(tasks, simulation=args.simulation or SIMULATION['enabled'])
        results['multi'] = multi_results
    
    # Compare algorithms
    if args.compare and 'single' in results and 'multi' in results:
        compare_algorithms(results['single'], results['multi'])
    
    # Compare platforms
    if args.platforms:
        compare_platforms(platform_info, results)
    
    logger.info("Task scheduling completed successfully")

if __name__ == "__main__":
    main()