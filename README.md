# Real-Time Task Scheduling on Raspberry Pi 3

This project implements and analyses various real-time task scheduling algorithms on a Raspberry Pi 3 using Python. It simulates 50 tasks with varying priorities, service times, and arrival patterns based on a Poisson distribution. Completed for EG6801 - Real-Time Embedded Systems at the University of Gloucestershire.

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
│   │   └── ml_scheduler.py     # Machine learning-based scheduler
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── single_processor.py # Single processor implementation
│   │   └── multi_processor.py  # Multi-processor with load balancing
│   └── utils/
│       ├── __init__.py
│       ├── visualisation.py    # Plotting and visualisation tools
│       └── metrics.py          # Performance metrics calculation
├── results/
│   ├── single_processor/       # Single processor results
│   ├── multi_processor/        # Multi-processor results
│   └── comparison/             # Cross-platform comparisons
├── config/
│   └── params.py               # Configuration parameters
├── requirements.txt            # Project dependencies
├── main.py                     # Main execution script
└── README.md                   # Project documentation
```

## Features

- **Task Generation**: Simulates 50 tasks with different priorities using Poisson distribution for arrival times
- **Scheduling Algorithms**:
  - First-Come-First-Served (FCFS)
  - Earliest Deadline First (EDF)
  - Priority-Based Scheduling with Priority Inversion Handling
  - Machine Learning-Based Scheduling
- **Processor Configurations**:
  - Single-Processor Execution
  - Multi-Processor Execution with different load-balancing strategies
- **Analysis and Visualisation**:
  - Waiting time analysis
  - Task completion visualisation
  - Memory utilisation monitoring
  - Algorithm comparison

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/real-time-scheduler.git
   cd real-time-scheduler
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the main script:

```
python main.py
```

### Command Line Arguments

- `--single`: Run only single processor tests
- `--multi`: Run only multi-processor tests
- `--compare`: Compare algorithms (requires both single and multi-processor tests)
- `--platforms`: Enable platform comparison (saves metrics for comparison)
- `--simulation`: Force simulation mode
- `--speed SPEED`: Set simulation speed factor (default: 10.0)

Examples:

```
# Run only single processor tests in simulation mode
python main.py --single --simulation

# Run multi-processor tests with a faster simulation speed
python main.py --multi --simulation --speed 20.0

# Run all tests and compare algorithms
python main.py --single --multi --compare

# Run on multiple platforms for comparison
python main.py --platforms
```

## Cross-Platform Testing

The project supports performance comparison across different platforms:

1. Run the same task set on each platform using the `--platforms` flag
2. Results are stored in `results/comparison/{platform_name}/`
3. Compare metrics across platforms manually or using custom scripts

## Output

- **Visualisation**: Charts and graphs are saved in the `results/` directory
- **Metrics**: Performance metrics are calculated and displayed for all schedulers
- **Reports**: A comprehensive report is generated in `results/comparison/performance_report.md`

## License

[MIT License](LICENSE)