#!/usr/bin/env python3
"""
Real-Time Task Scheduling on Raspberry Pi 3

Main execution script for the scheduling simulation.
Handles command-line arguments, runs simulations, and coordinates data collection.
"""

import argparse
import logging
import time
import os
import json
from datetime import datetime
import numpy as np
from typing import Dict, List, Any, Tuple, Optional

# Import project modules
from src.task_generator import TaskGenerator, Priority
from src.schedulers import FCFSScheduler, EDFScheduler, PriorityScheduler, MLScheduler, PriorityBasicScheduler
from src.processors import SingleProcessor, MultiProcessor
from src.utils.data_collector import (
    ensure_output_dir, save_system_info, save_task_metrics,
    save_time_series_metrics, save_scheduler_metrics,
    save_multi_processor_metrics, create_scenario_descriptions,
    combine_run_metrics
)
from src.utils.metrics import MetricsCalculator
from src.utils.json_utils import save_json
from config.params import TASK_CONFIG, SIMULATION

# Import analysis and visualization modules
import analysis
from src.utils import visualize as visualise  # Fixed import to match actual filename

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("simulation.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command-line arguments for simulation control"""
    parser = argparse.ArgumentParser(description="Real-Time Task Scheduling Simulation")
    
    # Simulation mode flags
    parser.add_argument("--single", action="store_true", help="Run only single processor tests")
    parser.add_argument("--multi", action="store_true", help="Run only multi-processor tests")
    parser.add_argument("--compare", action="store_true", help="Compare algorithms (requires both single and multi)")
    
    # Platform comparison
    parser.add_argument("--platforms", action="store_true", help="Enable platform comparison (saves metrics)")
    
    # Simulation parameters
    parser.add_argument("--simulation", action="store_true", help="Force simulation mode")
    parser.add_argument("--speed", type=float, default=SIMULATION['speed_factor'], 
                      help=f"Simulation speed factor (default: {SIMULATION['speed_factor']})")
    parser.add_argument("--tasks", type=int, default=None, 
                      help="Total number of tasks to generate (overrides config)")
    
    # Analysis and visualization
    parser.add_argument("--visualise", action="store_true", help="Generate visualizations during execution")
    parser.add_argument("--analyze-only", action="store_true", help="Only analyze existing data without running simulations")
    parser.add_argument("--data-dir", type=str, help="Directory with previously collected data for analysis")
    
    # Task distribution scenarios
    parser.add_argument("--scenarios", type=int, default=3, 
                      help="Number of task distribution scenarios to run (default: 3)")
    
    # Scheduler selection
    parser.add_argument("--schedulers", type=str, nargs='+', 
                      choices=['fcfs', 'edf', 'priority', 'priority_basic', 'ml', 'all'],
                      default=['all'], help="Schedulers to test")
    
    # Run settings
    parser.add_argument("--runs", type=int, default=3, 
                      help="Number of runs per scenario for statistical validity (default: 3)")
    
    # Output directory
    parser.add_argument("--output-dir", type=str, default="results", 
                      help="Base directory for output files (default: results)")
    
    return parser.parse_args()

def create_scheduler(scheduler_type: str) -> Any:
    """
    Create and return a scheduler instance based on the specified type
    
    Args:
        scheduler_type: Type of scheduler to create ('fcfs', 'edf', 'priority', 'priority_basic', 'ml')
        
    Returns:
        Scheduler instance
    """
    if scheduler_type == 'fcfs':
        return FCFSScheduler()
    elif scheduler_type == 'edf':
        return EDFScheduler()
    elif scheduler_type == 'priority':
        return PriorityScheduler()
    elif scheduler_type == 'priority_basic':
        return PriorityBasicScheduler()
    elif scheduler_type == 'ml':
        return MLScheduler()
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

def get_scheduler_name(scheduler_type: str) -> str:
    """
    Get a human-readable name for a scheduler type
    
    Args:
        scheduler_type: Type of scheduler ('fcfs', 'edf', 'priority', 'priority_basic', 'ml')
        
    Returns:
        Human-readable name (with no spaces or special characters for filenames)
    """
    names = {
        'fcfs': 'FCFS',
        'edf': 'EDF',
        'priority': 'Priority',
        'priority_basic': 'BasicPriority',
        'ml': 'ML'
    }
    return names.get(scheduler_type, scheduler_type.upper())

def create_task_configuration(scenario: int, tasks_override: int = None) -> Dict:
    """
    Create a task configuration based on the scenario
    
    Args:
        scenario: Scenario number (1, 2, 3)
            - Scenario 1: Baseline distribution (default from config)
            - Scenario 2: High load scenario (increased arrival rates)
            - Scenario 3: Priority inversion scenario (adjusted service times)
        tasks_override: Override total task count if specified
        
    Returns:
        Task configuration dictionary
    """
    # Start with the base configuration
    config = TASK_CONFIG.copy()
    
    # Modify based on scenario
    if scenario == 1:
        # Baseline scenario - use default config
        pass
    elif scenario == 2:
        # High load scenario - faster arrivals (reduce lambda values by 50%)
        for priority in config:
            config[priority]['lambda'] = config[priority]['lambda'] * 0.5
    elif scenario == 3:
        # Priority inversion scenario - adjust service times to increase likelihood of priority inversion
        # Increase low priority task service times and make them more variable
        config[Priority.LOW]['service_min'] = 8
        config[Priority.LOW]['service_max'] = 15
        # Decrease high priority arrivals to make them rarer
        config[Priority.HIGH]['lambda'] = config[Priority.HIGH]['lambda'] * 1.5
    else:
        logger.warning(f"Unknown scenario {scenario}, using default configuration")
    
    # Override task counts if specified
    if tasks_override is not None:
        # Calculate proportions from original config
        total_original = sum(cfg['count'] for cfg in config.values())
        proportions = {p: cfg['count'] / total_original for p, cfg in config.items()}
        
        # Apply proportions to new total
        for priority, proportion in proportions.items():
            config[priority]['count'] = int(tasks_override * proportion)
        
        # Adjust for rounding errors
        diff = tasks_override - sum(cfg['count'] for cfg in config.values())
        config[Priority.MEDIUM]['count'] += diff  # Add any difference to MEDIUM priority
    
    return config

def run_simulation(
    scheduler_type: str,
    processor_type: str,
    scenario: int,
    run_number: int,
    simulation_mode: bool,
    speed_factor: float,
    tasks_override: int,
    output_dir: str,
    experiment_dir: Optional[str] = None
) -> Tuple[Dict, str]:
    """
    Run a single simulation with specified parameters
    
    Args:
        scheduler_type: Type of scheduler to use
        processor_type: 'single' or 'multi'
        scenario: Scenario number
        run_number: Run number for this configuration
        simulation_mode: Whether to run in simulation mode
        speed_factor: Simulation speed factor
        tasks_override: Override for task count
        output_dir: Base directory for output files
        experiment_dir: Optional existing experiment directory
        
    Returns:
        Tuple of (metrics dictionary, experiment directory)
    """
    start_time = time.time()
    logger = logging.getLogger(__name__)
    logger.info(f"Starting {processor_type} processor simulation with {scheduler_type} scheduler (Scenario {scenario}, Run {run_number})")
    
    # Create task configuration and generator
    task_config = create_task_configuration(scenario, tasks_override)
    task_generator = TaskGenerator(task_config)
    
    # Generate tasks
    tasks = task_generator.generate_tasks()
    logger.info(f"Generated {len(tasks)} tasks for simulation")
    
    # Create scheduler(s)
    scheduler_name = get_scheduler_name(scheduler_type)
    
    # Get or create experiment directory
    if experiment_dir is None:
        timestamp, data_dir = ensure_output_dir(output_dir)
        # Save system info and scenario descriptions once per experiment
        save_system_info(data_dir)
        create_scenario_descriptions(data_dir)
    else:
        data_dir = experiment_dir
    
    try:
        if processor_type == 'single':
            # Single processor simulation
            scheduler = create_scheduler(scheduler_type)
            processor = SingleProcessor(scheduler)
            processor.add_tasks(tasks)
            
            # Run simulation
            processor.run(simulation=simulation_mode, speed_factor=speed_factor)
            
            # Get metrics with error handling
            try:
                metrics = processor.get_metrics()
                logger.info(f"Successfully collected metrics for {scheduler_name} scheduler")
            except Exception as e:
                logger.error(f"Error collecting metrics from processor: {str(e)}")
                # Create empty metrics to prevent further errors
                metrics = {
                    'completed_tasks': 0,
                    'avg_waiting_time': 0.0,
                    'simulation_duration': time.time() - start_time
                }
            
            # Save metrics and task data with error handling
            try:
                save_task_metrics(scheduler.completed_tasks, scheduler_name, processor.name, 
                                data_dir, scenario, run_number, 'single')
            except Exception as e:
                logger.error(f"Error saving task metrics: {str(e)}")
            
            try:
                save_time_series_metrics(metrics, scheduler_name, processor.name, 
                                    data_dir, scenario, run_number, 'single')
            except Exception as e:
                logger.error(f"Error saving time series metrics: {str(e)}")
            
            try:
                save_scheduler_metrics(metrics, scheduler_name, processor.name, 
                                    data_dir, scenario, run_number, 'single')
            except Exception as e:
                logger.error(f"Error saving scheduler metrics: {str(e)}")
            
            # Add scenario and run metadata
            metrics['scenario'] = scenario
            metrics['run_number'] = run_number
            metrics['scheduler_type'] = scheduler_type
            metrics['processor_type'] = processor_type
            
        else:
            # Multi-processor simulation
            processor_count = 4  # Using 4 cores for Raspberry Pi 3
            schedulers = [create_scheduler(scheduler_type) for _ in range(processor_count)]
            processor = MultiProcessor(schedulers, processor_count=processor_count)
            processor.add_tasks(tasks)
            
            # Run simulation
            processor.run(simulation=simulation_mode, speed_factor=speed_factor)
            
            # Get metrics with error handling
            try:
                metrics = processor.get_metrics()
                logger.info(f"Successfully collected metrics for multi-processor {scheduler_name} scheduler")
            except Exception as e:
                logger.error(f"Error collecting metrics from multi-processor: {str(e)}")
                # Create empty metrics to prevent further errors
                metrics = {
                    'processor_count': processor_count,
                    'strategy': getattr(processor, 'strategy', 'unknown'),
                    'total_completed_tasks': 0,
                    'avg_waiting_time': 0.0,
                    'simulation_duration': time.time() - start_time,
                    'per_processor_metrics': []
                }
            
            # Save metrics with error handling
            try:
                save_multi_processor_metrics(metrics, scheduler_name, data_dir, scenario, run_number)
            except Exception as e:
                logger.error(f"Error saving multi-processor metrics: {str(e)}")
            
            # Also save individual processor metrics
            for i, p_metrics in enumerate(metrics.get('per_processor_metrics', [])):
                try:
                    cpu_name = f"CPU-{i+1}"
                    save_scheduler_metrics(p_metrics, scheduler_name, cpu_name, 
                                        data_dir, scenario, run_number, 'multi')
                except Exception as e:
                    logger.error(f"Error saving processor {i+1} metrics: {str(e)}")
            
            # Add scenario and run metadata
            metrics['scenario'] = scenario
            metrics['run_number'] = run_number
            metrics['scheduler_type'] = scheduler_type
            metrics['processor_type'] = processor_type
    except Exception as e:
        logger.error(f"Error during simulation: {str(e)}")
        # Create minimal metrics to prevent further errors
        metrics = {
            'scenario': scenario,
            'run_number': run_number,
            'scheduler_type': scheduler_type,
            'processor_type': processor_type,
            'error': str(e),
            'simulation_duration': time.time() - start_time
        }
    
    end_time = time.time()
    execution_time = end_time - start_time
    metrics['simulation_duration'] = round(execution_time, 3)
    
    # Round all time-related metrics for clarity
    if 'avg_waiting_time' in metrics:
        metrics['avg_waiting_time'] = round(metrics['avg_waiting_time'], 3)
    
    if 'avg_waiting_by_priority' in metrics:
        for priority, wait_time in metrics['avg_waiting_by_priority'].items():
            metrics['avg_waiting_by_priority'][priority] = round(wait_time, 3)
    
    logger.info(f"Completed simulation in {execution_time:.2f} seconds")
    return metrics, data_dir

def run_simulations(args):
    """
    Run all simulations with specified parameters
    
    Args:
        args: Command-line arguments
        
    Returns:
        Tuple of (results dictionary, experiment directory)
    """
    all_results = {}
    experiment_dir = None
    
    # Get scheduler types
    scheduler_types = []
    if 'all' in args.schedulers:
        scheduler_types = ['fcfs', 'edf', 'priority', 'priority_basic', 'ml']
    else:
        scheduler_types = args.schedulers
    
    # Create experiment directory
    timestamp, data_dir = ensure_output_dir(args.output_dir)
    experiment_dir = data_dir
    
    # Save system info and scenario descriptions
    save_system_info(data_dir)
    create_scenario_descriptions(data_dir)
    
    # Track completion status
    completion_status = {
        'total_runs': len(scheduler_types) * args.scenarios * args.runs * (1 if args.single and not args.multi else 2),
        'completed_runs': 0,
        'failed_runs': 0
    }
    
    # Run simulations
    for scheduler_type in scheduler_types:
        logger.info(f"Starting simulations for {get_scheduler_name(scheduler_type)} scheduler")
        
        # Initialize results structure for this scheduler
        if scheduler_type not in all_results:
            all_results[scheduler_type] = {'single': [], 'multi': []}
        
        # Run all scenarios and runs for this scheduler
        for scenario in range(1, args.scenarios + 1):
            for run in range(1, args.runs + 1):
                logger.info(f"  Running scenario {scenario}, run {run}")
                
                # Single processor simulation if enabled
                if args.single or not args.multi:
                    try:
                        metrics, _ = run_simulation(
                            scheduler_type=scheduler_type,
                            processor_type='single',
                            scenario=scenario,
                            run_number=run,
                            simulation_mode=args.simulation,
                            speed_factor=args.speed,
                            tasks_override=args.tasks,
                            output_dir=args.output_dir,
                            experiment_dir=experiment_dir
                        )
                        
                        all_results[scheduler_type]['single'].append(metrics)
                        completion_status['completed_runs'] += 1
                    except Exception as e:
                        logger.error(f"Error in single processor simulation: {str(e)}")
                        completion_status['failed_runs'] += 1
                
                # Multi-processor simulation if enabled
                if args.multi or not args.single:
                    try:
                        metrics, _ = run_simulation(
                            scheduler_type=scheduler_type,
                            processor_type='multi',
                            scenario=scenario,
                            run_number=run,
                            simulation_mode=args.simulation,
                            speed_factor=args.speed,
                            tasks_override=args.tasks,
                            output_dir=args.output_dir,
                            experiment_dir=experiment_dir
                        )
                        
                        all_results[scheduler_type]['multi'].append(metrics)
                        completion_status['completed_runs'] += 1
                    except Exception as e:
                        logger.error(f"Error in multi-processor simulation: {str(e)}")
                        completion_status['failed_runs'] += 1
        
        logger.info(f"Completed all simulations for {get_scheduler_name(scheduler_type)} scheduler")
    
    # Run post-processing to combine metrics
    try:
        combine_run_metrics(data_dir)
        logger.info("Successfully combined run metrics")
    except Exception as e:
        logger.error(f"Error combining run metrics: {str(e)}")
    
    # Log completion status
    logger.info(f"Simulation complete: {completion_status['completed_runs']} of {completion_status['total_runs']} runs completed successfully, {completion_status['failed_runs']} runs failed")
    
    return all_results, experiment_dir

def main():
    """Main execution function"""
    # Parse command-line arguments
    args = parse_arguments()
    
    # If analyze-only flag is set, skip simulations
    if args.analyze_only:
        if args.data_dir:
            logger.info(f"Analyzing existing data from directory: {args.data_dir}")
            analysis.analyze_existing_data(args.data_dir)
            
            if args.visualise:
                logger.info("Generating visualizations from existing data")
                visualise.create_visualizations(args.data_dir)
            
            return
        else:
            logger.error("--analyze-only requires --data-dir to be specified")
            return
    
    # Run all simulations and collect results
    all_results, experiment_dir = run_simulations(args)
    
    # Save combined results as JSON
    if experiment_dir:
        # Create analysis directory for saving combined results
        analysis_dir = os.path.join(experiment_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Save combined results as JSON
        # Save combined results as JSON
        save_json(all_results, os.path.join(analysis_dir, "all_results.json"))
        
        # Run analysis on results
        analysis.analyze_results(all_results, experiment_dir)
        
        # Generate visualizations if requested
        if args.visualise:
            visualise.create_visualizations(experiment_dir, all_results)
        
        logger.info(f"All simulations complete. Results saved to {experiment_dir}")
    else:
        logger.error("No experiment directory was created. No results to analyze.")

if __name__ == "__main__":
    main()