"""
Visualization GUI for Real-Time Task Scheduling Analysis

This module provides a graphical interface for viewing and generating visualizations
from the scheduling simulation results.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns
from typing import Dict, List, Any
import pandas as pd

class VisualizationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Scheduler Visualization")
        
        # Set up the main window
        self.root.geometry("1200x800")
        
        # Create the main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create a frame for controls
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N))
        
        # Add data loading button
        self.load_button = ttk.Button(self.control_frame, text="Load Results Directory", command=self.load_results)
        self.load_button.grid(row=0, column=0, padx=5)
        
        # Add visualization type selection
        self.viz_types = [
            "Task Generation",
            "Priority Distribution",
            "Queue Length Over Time",
            "Waiting Time Distribution",
            "Processor Utilization",
            "Deadline Miss Rate",
            "Priority Inversion Analysis",
            "Machine Learning Performance",
            "Resource Utilization",
        ]
        self.selected_viz = tk.StringVar(value=self.viz_types[0])
        self.viz_combo = ttk.Combobox(self.control_frame, textvariable=self.selected_viz, values=self.viz_types)
        self.viz_combo.grid(row=0, column=1, padx=5)
        
        # Add generate button
        self.generate_button = ttk.Button(self.control_frame, text="Generate Visualization", command=self.generate_visualization)
        self.generate_button.grid(row=0, column=2, padx=5)
        
        # Create figure frame
        self.fig_frame = ttk.Frame(self.main_frame)
        self.fig_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.fig_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create toolbar frame
        self.toolbar_frame = ttk.Frame(self.fig_frame)
        self.toolbar_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Add toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbar_frame)
        self.toolbar.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Initialize data storage
        self.results_dir = None
        self.analysis_results = None
        self.raw_data = None
        
        # Configure grid weights
        self.main_frame.columnconfigure(0, weight=1)
        self.main_frame.rowconfigure(1, weight=1)
        self.fig_frame.columnconfigure(0, weight=1)
        self.fig_frame.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

    def load_results(self):
        """Load results from a selected directory"""
        self.results_dir = filedialog.askdirectory(title="Select Results Directory")
        if not self.results_dir:
            return
            
        try:
            # Load analysis results
            analysis_path = os.path.join(self.results_dir, "analysis", "analysis_results.json")
            if os.path.exists(analysis_path):
                with open(analysis_path, 'r') as f:
                    self.analysis_results = json.load(f)
            
            # Enable generate button
            self.generate_button.state(['!disabled'])
            messagebox.showinfo("Success", "Results loaded successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load results: {str(e)}")

    def generate_visualization(self):
        """Generate the selected visualization"""
        if not self.results_dir or not self.analysis_results:
            messagebox.showerror("Error", "Please load results first!")
            return
            
        viz_type = self.selected_viz.get()
        self.fig.clear()
        
        try:
            if viz_type == "Task Generation":
                self.plot_task_generation()
            elif viz_type == "Priority Distribution":
                self.plot_priority_distribution()
            elif viz_type == "Queue Length Over Time":
                self.plot_queue_length()
            elif viz_type == "Waiting Time Distribution":
                self.plot_waiting_time_distribution()
            elif viz_type == "Processor Utilization":
                self.plot_processor_utilization()
            elif viz_type == "Deadline Miss Rate":
                self.plot_deadline_miss_rate()
            elif viz_type == "Priority Inversion Analysis":
                self.plot_priority_inversion()
            elif viz_type == "Machine Learning Performance":
                self.plot_ml_performance()
            elif viz_type == "Resource Utilization":
                self.plot_resource_utilization()
            
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate visualization: {str(e)}")

    def plot_task_generation(self):
        """Plot task generation distribution"""
        ax = self.fig.add_subplot(111)
        
        # Extract task arrival times from results
        tasks_by_scheduler = {}
        for scheduler_type in ['fcfs', 'edf', 'priority']:
            if scheduler_type in self.analysis_results.get('throughput', {}).get('arrival_times', {}):
                tasks_by_scheduler[scheduler_type] = self.analysis_results['throughput']['arrival_times'][scheduler_type]
        
        # Plot histogram for each scheduler
        colors = ['#2196F3', '#7B1FA2', '#FF5722']
        for (scheduler, arrivals), color in zip(tasks_by_scheduler.items(), colors):
            if arrivals:
                ax.hist(arrivals, bins=30, alpha=0.5, label=scheduler.upper(), color=color)
        
        ax.set_title('Task Arrival Time Distribution')
        ax.set_xlabel('Arrival Time (s)')
        ax.set_ylabel('Frequency')
        ax.legend()

    def plot_priority_distribution(self):
        """Plot priority distribution"""
        ax = self.fig.add_subplot(111)
        
        priority_data = {}
        for scheduler, data in self.analysis_results.get('waiting_time', {}).get('by_scheduler_and_priority', {}).items():
            priority_data[scheduler.upper()] = {
                'HIGH': data.get('HIGH', 0),
                'MEDIUM': data.get('MEDIUM', 0),
                'LOW': data.get('LOW', 0)
            }
        
        df = pd.DataFrame(priority_data)
        df.plot(kind='bar', ax=ax)
        
        ax.set_title('Task Distribution by Priority Level')
        ax.set_xlabel('Priority Level')
        ax.set_ylabel('Average Waiting Time (s)')
        ax.legend(title='Scheduler')

    def plot_queue_length(self):
        """Plot queue length over time"""
        ax = self.fig.add_subplot(111)
        
        resource_data = self.analysis_results.get('resource_utilization', {}).get('queue_length', {})
        schedulers = []
        single_proc = []
        multi_proc = []
        
        for scheduler, data in resource_data.items():
            schedulers.append(scheduler.upper())
            single_proc.append(data.get('single', 0))
            multi_proc.append(data.get('multi', 0))
        
        x = np.arange(len(schedulers))
        width = 0.35
        
        ax.bar(x - width/2, single_proc, width, label='Single Processor')
        ax.bar(x + width/2, multi_proc, width, label='Multi Processor')
        
        ax.set_title('Average Queue Length by Scheduler')
        ax.set_xlabel('Scheduler')
        ax.set_ylabel('Average Queue Length')
        ax.set_xticks(x)
        ax.set_xticklabels(schedulers)
        ax.legend()

    def plot_waiting_time_distribution(self):
        """Plot waiting time distribution"""
        ax = self.fig.add_subplot(111)
        waiting_times = self.analysis_results.get('waiting_time', {})
        
        data = []
        labels = []
        for scheduler, times in waiting_times.get('by_scheduler', {}).items():
            if isinstance(times, (int, float)):
                data.append(times)
                labels.append(scheduler.upper())
        
        sns.boxplot(data=data, ax=ax)
        ax.set_xticklabels(labels)
        ax.set_title('Waiting Time Distribution by Scheduler')
        ax.set_xlabel('Scheduler')
        ax.set_ylabel('Waiting Time (s)')

    def plot_processor_utilization(self):
        """Plot processor utilization"""
        ax = self.fig.add_subplot(111)
        
        # Get processor scaling data
        scaling = self.analysis_results.get('processor_scaling', {})
        efficiency = scaling.get('efficiency', {})
        
        schedulers = []
        values = []
        
        for scheduler, eff in efficiency.items():
            schedulers.append(scheduler.upper())
            values.append(eff)
        
        ax.bar(schedulers, values)
        ax.set_title('Processor Utilization Efficiency')
        ax.set_xlabel('Scheduler')
        ax.set_ylabel('Efficiency (%)')
        
        # Add value labels on top of bars
        for i, v in enumerate(values):
            ax.text(i, v, f'{v:.1f}%', ha='center', va='bottom')

    def plot_deadline_miss_rate(self):
        """Plot deadline miss rate comparison"""
        ax = self.fig.add_subplot(111)
        
        deadline_data = self.analysis_results.get('deadline_satisfaction', {})
        scenarios = deadline_data.get('by_scenario', {})
        
        schedulers = []
        normal_load = []
        high_load = []
        
        for scheduler in ['edf', 'priority', 'ml']:
            if scheduler in scenarios.get('1', {}) and scheduler in scenarios.get('2', {}):
                schedulers.append(scheduler.upper())
                normal_load.append(scenarios['1'][scheduler])
                high_load.append(scenarios['2'][scheduler])
        
        x = np.arange(len(schedulers))
        width = 0.35
        
        ax.bar(x - width/2, normal_load, width, label='Normal Load')
        ax.bar(x + width/2, high_load, width, label='High Load')
        
        ax.set_title('Deadline Satisfaction Rate by Load')
        ax.set_xlabel('Scheduler')
        ax.set_ylabel('Satisfaction Rate (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(schedulers)
        ax.legend()

    def plot_priority_inversion(self):
        """Plot priority inversion analysis"""
        ax = self.fig.add_subplot(111)
        
        priority_inv = self.analysis_results.get('priority_inversion', {})
        blocking = priority_inv.get('blocking_incidents', {})
        duration = priority_inv.get('blocking_duration', {})
        
        labels = ['Without Inheritance', 'With Inheritance']
        incidents = [blocking.get('without_inheritance', 0), blocking.get('with_inheritance', 0)]
        durations = [duration.get('without_inheritance', 0), duration.get('with_inheritance', 0)]
        
        x = np.arange(len(labels))
        width = 0.35
        
        ax.bar(x - width/2, incidents, width, label='Blocking Incidents')
        ax.bar(x + width/2, durations, width, label='Avg Duration (s)')
        
        ax.set_title('Priority Inversion Analysis')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

    def plot_ml_performance(self):
        """Plot machine learning performance metrics"""
        ax = self.fig.add_subplot(111)
        
        ml_results = self.analysis_results.get('machine_learning', {})
        feature_imp = ml_results.get('feature_importance', {})
        
        features = []
        importance = []
        
        for feature, imp in sorted(feature_imp.items(), key=lambda x: x[1], reverse=True):
            features.append(feature)
            importance.append(imp)
        
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importance)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_title('ML Feature Importance')
        ax.set_xlabel('Importance')

    def plot_resource_utilization(self):
        """Plot resource utilization metrics"""
        ax = self.fig.add_subplot(111)
        
        resource_data = self.analysis_results.get('resource_utilization', {})
        cpu_usage = resource_data.get('cpu_usage', {})
        
        schedulers = []
        single_cpu = []
        multi_cpu = []
        
        for scheduler, data in cpu_usage.items():
            schedulers.append(scheduler.upper())
            single_cpu.append(data.get('single', 0))
            multi_cpu.append(data.get('multi', 0))
        
        x = np.arange(len(schedulers))
        width = 0.35
        
        ax.bar(x - width/2, single_cpu, width, label='Single Processor')
        ax.bar(x + width/2, multi_cpu, width, label='Multi Processor')
        
        ax.set_title('CPU Usage by Scheduler')
        ax.set_xlabel('Scheduler')
        ax.set_ylabel('CPU Usage (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(schedulers)
        ax.legend()

def main():
    root = tk.Tk()
    app = VisualizationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()