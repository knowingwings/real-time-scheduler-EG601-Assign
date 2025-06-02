# Real-Time Task Scheduling System for Raspberry Pi 3

## EG6801 - Real-Time Embedded System (Assignment 001)

A comprehensive real-time task scheduling simulation system implementing multiple scheduling algorithms with performance analysis and machine learning optimisation.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [GUI Mode](#gui-mode)
  - [Headless Mode](#headless-mode)
  - [Comprehensive Testing](#comprehensive-testing)
- [Scheduling Algorithms](#scheduling-algorithms)
- [Output and Results](#output-and-results)
- [Project Structure](#project-structure)
- [Assignment Requirements Coverage](#assignment-requirements-coverage)

## Overview

This project implements a real-time task scheduling system that simulates 50 tasks with varying priorities, service times, and arrival patterns based on Poisson distributions. The system evaluates the performance of different scheduling algorithms under various conditions, including priority inversion, delays, and memory utilisation.

## Features

- **Task Generation**: 50 tasks with Poisson-distributed arrival times and three priority levels
- **Multiple Scheduling Algorithms**: FCFS, EDF, Priority-based, and ML-enhanced schedulers
- **Multi-processor Support**: Single and multiple processor configurations
- **Priority Inversion Handling**: Priority inheritance mechanism
- **Machine Learning Integration**: Linear regression, decision trees, and Q-learning for optimisation
- **Comprehensive Testing Suite**: Automated testing across multiple scenarios
- **Rich Visualisations**: Gantt charts, performance metrics, heatmaps, and comparison plots
- **GUI and Headless Modes**: Flexible operation for different environments

## Requirements

### Hardware
- Raspberry Pi 4 (or compatible system for simulation)
- Minimum 1GB RAM
- 100MB free disk space

## Installation

1. **Clone or extract the project**:
```bash
# If using git
git clone <repository-url>
cd real-time-scheduler
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install numpy matplotlib seaborn scikit-learn pandas psutil scipy
```

3. **Verify installation**:
```bash
python main.py --help
```

## Usage

### GUI Mode

The GUI mode provides an interactive interface for configuration, simulation, and analysis.

#### Starting the GUI
```bash
python main.py --mode gui
# or simply
python main.py
```

#### GUI Workflow

1. **Configuration Tab**:
   - Set random seed for reproducible results
   - Click "Generate Tasks" to create 50 tasks
   - Review task specifications in the display area
   - Configure number of processors (1-4)
   - Select schedulers to compare

2. **Simulation Tab**:
   - Click "Run Simulation" for basic simulation
   - Click "Run Comprehensive Tests" for full test suite
   - Monitor progress in the log window

3. **Results Tab**:
   - View generated reports
   - Save results and visualisations
   - Export data for further analysis

### Headless Mode

Ideal for automated testing, remote execution, or systems without GUI support.

#### Basic Simulation
```bash
# Run with default settings
python main.py --mode headless

# Specify seed for reproducibility
python main.py --mode headless --seed 42

# Multi-processor simulation
python main.py --mode headless --num-processors 4

# Select specific schedulers
python main.py --mode headless --enable-fcfs --enable-priority --enable-ml-linear

# Generate plots
python main.py --mode headless --generate-plots

# Import tasks from file
python main.py --mode headless --import-tasks tasks.json
```

#### Comprehensive Testing
```bash
# Run full test suite
python main.py --mode headless --run-comprehensive-tests
```

### Comprehensive Testing

The comprehensive test suite automatically runs multiple scenarios:

1. **Baseline Test**: Standard configuration
2. **High Load Test**: Increased task arrival rate
3. **Multi-processor Tests**: 2 and 4 processor configurations
4. **Priority Distribution Tests**: Skewed priority distributions
5. **Extreme Service Times**: Extended task durations
6. **Stress Test**: 100 tasks simulation

Results include:
- Detailed performance metrics
- Statistical analysis with t-tests
- Processor scaling analysis
- Priority inversion reports
- Comprehensive visualisations

## Scheduling Algorithms

### Traditional Algorithms

1. **FCFS (First-Come-First-Served)**
   - Tasks executed in arrival order
   - Simple, no starvation
   - No priority consideration

2. **EDF (Earliest Deadline First)**
   - Dynamic priority based on deadlines
   - Optimal for single processor
   - Deadline-driven scheduling

3. **Priority-Based**
   - Static priority levels (High/Medium/Low)
   - Preemptive scheduling
   - Priority inheritance for inversion handling

### Machine Learning Algorithms

4. **ML-Linear (Linear Regression)**
   - Predicts waiting times
   - Learns from completed tasks
   - Optimises queue ordering

5. **ML-DecisionTree**
   - Categorises tasks dynamically
   - Tree-based urgency scoring
   - Adaptive to workload patterns

6. **ML-QLearning**
   - Reinforcement learning approach
   - State-action value learning
   - Balances exploration and exploitation

## Output and Results

### Directory Structure
```
results/
├── basic_sim_[timestamp]/
│   ├── basic_report.txt
│   ├── gantt_*.png
│   └── metrics_*.png
└── comprehensive_test_[timestamp]/
    ├── comprehensive_report.txt
    ├── detailed_results.pkl
    ├── processor_comparison.png
    ├── gantt_charts/
    ├── metrics_plots/
    └── comparisons/
```

### Report Contents
- **Executive Summary**: Best performers by metric
- **Detailed Metrics**: 
  - Average, P90, P99 waiting times
  - CPU utilisation and throughput
  - Queue length statistics
- **Statistical Analysis**: T-test comparisons
- **Priority Analysis**: Performance by priority level
- **Processor Scaling**: Multi-processor efficiency

### Visualisations
1. **Gantt Charts**: Task execution timeline
2. **Metrics Plots**: Queue length, memory, CPU over time
3. **Comparison Heatmaps**: Performance across scenarios
4. **Priority Performance**: Bar charts by priority
5. **Processor Scaling**: Speedup and efficiency graphs

## Project Structure

```
real-time-scheduler/
├── main.py                 # Main application entry point
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── results/              # Output directory (created automatically)
└── docs/                 # Additional documentation
    └── EG6801-Assessment1.pdf
```

### Key Classes

- **Task**: Task representation with timing attributes
- **TaskGenerator**: Poisson-based task generation
- **Scheduler Classes**: Base and specific implementations
- **Simulator**: Main simulation engine
- **Visualizer**: Plotting and reporting
- **ComprehensiveTestSuite**: Automated testing
- **GUI**: Tkinter-based interface

## Assignment Requirements Coverage

### Phase 1: Design ✓
- Task specifications with Poisson distribution
- Mathematical models for timing calculations
- Three scheduling algorithms with flowcharts
- Priority inversion handling design
- ML-based scheduling approach

### Phase 2: Implementation ✓
- Python implementation on Raspberry Pi 3
- Threading and multiprocessing support
- Memory monitoring with psutil
- Complete scheduling systems

### Phase 3: Testing and Analysis ✓
- Multiple test scenarios
- Performance metrics collection
- Comprehensive visualisations
- Statistical comparisons

### Phase 4: Report and Presentation ✓
- Automated report generation
- Complete code documentation
- Performance analysis
- Visual demonstrations

## Troubleshooting

### Common Issues

1. **Import Error for tkinter**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install python3-tk
   
   # Or use headless mode
   python main.py --mode headless
   ```

2. **Memory Issues with Large Tests**:
   - Reduce simulation time in code
   - Run fewer concurrent tests
   - Use headless mode

3. **Matplotlib Backend Issues**:
   ```bash
   export MPLBACKEND=Agg  # For non-GUI systems
   ```

### Performance Tips

- Use consistent seeds for reproducible results
- Close unnecessary applications during testing
- For Raspberry Pi: Ensure adequate cooling
- Consider using swap space for stress tests

## Contact and Support

For questions about this implementation:
- Refer to assignment documentation
- Check code comments for implementation details
- Review the comprehensive report for methodology

---

**Note**: This system is designed for educational purposes as part of EG6801. While it simulates real-time scheduling, actual real-time performance depends on the underlying operating system and hardware capabilities.