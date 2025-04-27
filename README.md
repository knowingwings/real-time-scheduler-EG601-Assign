# Real-Time Task Scheduling on Raspberry Pi 3

This project implements and analyses various real-time task scheduling algorithms on a Raspberry Pi 3 using Python. It simulates 50 tasks with varying priorities, service times, and arrival patterns based on a Poisson distribution to evaluate different scheduling approaches. Designed for EG6801: Real-Time Embedded Systems at the University of Gloucestershire.

## Project Overview

The simulation system evaluates the performance of different scheduling algorithms under various conditions, including:
- Priority inversion scenarios
- Varying task arrival rates
- Different task priority distributions
- Single vs multi-processor configurations

The project incorporates theoretical concepts of real-time task scheduling, practical implementation using Python, performance analysis through visualisation, and machine learning-based scheduling optimisation.

## Project Structure

```
real-time-scheduler/
├── src/
│   ├── __init__.py
│   ├── task_generator.py       # Task generation with Poisson distribution
│   ├── schedulers/
│   │   ├── __init__.py
│   │   ├── fcfs.py             # First-Come-First-Served scheduler
│   │   ├── edf.py              # Earliest Deadline First scheduler
│   │   ├── priority_based.py   # Priority-based scheduler with inversion handling
│   │   └── ml_scheduler.py     # Machine learning-based scheduler using decision trees
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── single_processor.py # Single processor implementation
│   │   └── multi_processor.py  # Multi-processor with load balancing strategies
│   └── utils/
│       ├── __init__.py
│       ├── visualisation.py    # Plotting and visualisation tools
│       ├── metrics.py          # Performance metrics calculation
│       ├── data_collector.py   # Data collection utilities
│       └── platform_utils.py   # Platform detection utilities
├── results/
│   ├── data/                   # Raw data from simulation runs
│   ├── logs/                   # Execution logs
│   └── visualisations/         # Generated visualisations
├── config/
│   └── params.py               # Configuration parameters
├── main.py                     # Main execution script
├── visualise.py                # Visualisation tool for post-processing
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

## Features

### Task Generation
- Simulates 50 tasks with Poisson distribution for arrival times
- Configurable priority levels (High, Medium, Low)
- Customisable service times based on priority
- Dynamic deadline assignment

### Scheduling Algorithms
- **First-Come-First-Served (FCFS)**: Non-preemptive scheduling based on arrival order
- **Earliest Deadline First (EDF)**: Preemptive scheduling prioritising tasks with earlier deadlines
- **Priority-Based Scheduling**: Preemptive scheduling with priority inversion handling using Priority Inheritance
- **Machine Learning-Based Scheduling**: Uses Decision Trees to predict execution times and optimise scheduling decisions

### Processor Configurations
- **Single-Processor Execution**: Traditional single CPU execution
- **Multi-Processor Execution**: Tasks distributed across multiple processors with different load-balancing strategies:
  - Round-Robin: Simple cyclic distribution
  - Least-Loaded: Assigns to processor with shortest queue
  - Priority-Based: Distributes tasks based on priority and processor capabilities

### Performance Analysis
- Comprehensive metrics collection and analysis
- Task completion visualisation with Gantt charts
- Memory and CPU utilisation tracking
- Queue length monitoring
- Resource bottleneck identification through heatmaps
- Waiting time analysis by priority level
- Cross-algorithm comparison using radar charts
- Cross-platform performance comparison

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/real-time-scheduler.git
   cd real-time-scheduler
   ```

2. Set up a virtual environment (recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running Simulations

Run the main script with various command-line arguments:

```
python main.py [options]
```

### Command Line Arguments

- `--single`: Run only single processor tests
- `--multi`: Run only multi-processor tests
- `--compare`: Compare algorithms (requires both single and multi-processor tests)
- `--platforms`: Enable platform comparison (saves metrics for comparison)
- `--simulation`: Force simulation mode
- `--speed SPEED`: Set simulation speed factor (default: 10.0)
- `--tasks NUM`: Set total number of tasks to generate
- `--visualise`: Generate visualisations after simulation

### Examples

```
# Run only single processor tests in simulation mode
python main.py --single --simulation

# Run multi-processor tests with a faster simulation speed
python main.py --multi --simulation --speed 20.0

# Run all tests and compare algorithms
python main.py --single --multi --compare

# Run with a custom number of tasks
python main.py --single --tasks 100

# Generate visualisations during execution
python main.py --single --multi --visualise
```

### Visualisation

For post-processing and visualisation of previously generated data:

```
python visualise.py --data-dir results/data/TIMESTAMP_platform_type
```

Additional visualisation options:
```
python visualise.py --data-dir results/data/TIMESTAMP_platform_type --output-dir custom_output
python visualise.py --data-dir results/data/TIMESTAMP_platform_type --scheduler FCFS
```

## Output

The system generates several types of output:

- **Data Files**: Raw CSV and JSON data in the `results/data/` directory
- **Visualisations**: Charts and graphs saved in the `results/visualisations/` directory, including:
  - Task completion timelines
  - Waiting time distributions
  - Memory usage charts
  - Queue length tracking
  - Resource bottleneck heatmaps
  - Task density heatmaps
  - CPU utilisation heatmaps
  - Algorithm comparison radar charts
- **Performance Report**: A comprehensive Markdown report summarising all findings
- **Logs**: Execution logs in the `results/logs/` directory

## Configuration

The simulation behaviour can be customised through the `config/params.py` file:

- Task generation parameters (priority distribution, arrival rates, service times)
- Single and multi-processor configurations
- Simulation speed and run time
- Machine learning parameters for the ML-based scheduler
- Visualisation parameters

## Cross-Platform Testing

The project supports performance comparison across different platforms:

1. Run the same task set on each platform using the `--platforms` flag
2. Results are stored in `results/data/{timestamp}_{platform_type}/`
3. Compare metrics across platforms using `visualise.py`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.