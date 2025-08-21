#!/usr/bin/env python3
"""
Real-time Training Progress Tracker
==================================

This module provides real-time visualization and tracking of model training progress,
including loss curves, metrics monitoring, and performance analytics.

Features:
- Live loss and metric plotting
- Training progress visualization
- Performance prediction
- Resource monitoring
- Interactive dashboards

Author: AI Tutorial by AI
"""

import time
import threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# Remove tkinter dependency for headless environments
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import json
import os
from collections import deque, defaultdict
import warnings
warnings.filterwarnings('ignore')

class TrainingTracker:
    """Real-time training progress tracker with live visualization"""
    
    def __init__(self, 
                 metrics: List[str] = ['loss', 'accuracy'],
                 max_history: int = 1000,
                 update_frequency: int = 10,
                 save_dir: str = "training_logs"):
        """
        Initialize the training tracker
        
        Args:
            metrics: List of metrics to track
            max_history: Maximum number of data points to keep in memory
            update_frequency: How often to update plots (in epochs)
            save_dir: Directory to save training logs
        """
        self.metrics = metrics
        self.max_history = max_history
        self.update_frequency = update_frequency
        self.save_dir = save_dir
        
        # Training data storage
        self.training_data = defaultdict(lambda: deque(maxlen=max_history))
        self.validation_data = defaultdict(lambda: deque(maxlen=max_history))
        self.epochs = deque(maxlen=max_history)
        self.timestamps = deque(maxlen=max_history)
        
        # Training state
        self.start_time = None
        self.current_epoch = 0
        self.is_training = False
        self.total_epochs = 0
        
        # Performance tracking
        self.epoch_times = deque(maxlen=50)  # Keep last 50 epoch times for speed estimation
        self.best_metrics = {}
        self.early_stop_patience = 0
        self.early_stop_counter = 0
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Initialize live plot data
        self.live_fig = None
        self.live_axes = {}
        
    def start_training(self, total_epochs: int = None):
        """Start training session"""
        self.start_time = datetime.now()
        self.is_training = True
        self.current_epoch = 0
        self.total_epochs = total_epochs or 100
        self.epoch_times.clear()
        
        print(f"🚀 Training started at {self.start_time.strftime('%H:%M:%S')}")
        if total_epochs:
            print(f"📊 Target epochs: {total_epochs}")
        
    def log_epoch(self, 
                  epoch: int,
                  train_metrics: Dict[str, float],
                  val_metrics: Dict[str, float] = None,
                  extra_info: Dict[str, Any] = None):
        """
        Log metrics for an epoch
        
        Args:
            epoch: Current epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics (optional)
            extra_info: Additional information to log
        """
        current_time = datetime.now()
        
        # Calculate epoch time
        if hasattr(self, '_last_epoch_time'):
            epoch_time = (current_time - self._last_epoch_time).total_seconds()
            self.epoch_times.append(epoch_time)
        self._last_epoch_time = current_time
        
        # Store data
        self.epochs.append(epoch)
        self.timestamps.append(current_time)
        self.current_epoch = epoch
        
        # Store training metrics
        for metric, value in train_metrics.items():
            self.training_data[metric].append(value)
            
            # Track best metrics
            if metric not in self.best_metrics:
                self.best_metrics[metric] = {'value': value, 'epoch': epoch, 'type': 'train'}
            elif (metric == 'loss' and value < self.best_metrics[metric]['value']) or \
                 (metric != 'loss' and value > self.best_metrics[metric]['value']):
                self.best_metrics[metric] = {'value': value, 'epoch': epoch, 'type': 'train'}
        
        # Store validation metrics
        if val_metrics:
            for metric, value in val_metrics.items():
                self.validation_data[metric].append(value)
                
                # Track best validation metrics
                val_key = f'val_{metric}'
                if val_key not in self.best_metrics:
                    self.best_metrics[val_key] = {'value': value, 'epoch': epoch, 'type': 'val'}
                elif (metric == 'loss' and value < self.best_metrics[val_key]['value']) or \
                     (metric != 'loss' and value > self.best_metrics[val_key]['value']):
                    self.best_metrics[val_key] = {'value': value, 'epoch': epoch, 'type': 'val'}
        
        # Print progress
        if epoch % self.update_frequency == 0 or epoch == 1:
            self._print_progress(train_metrics, val_metrics, extra_info)
    
    def _print_progress(self, 
                       train_metrics: Dict[str, float],
                       val_metrics: Dict[str, float] = None,
                       extra_info: Dict[str, Any] = None):
        """Print training progress to console"""
        elapsed = datetime.now() - self.start_time
        
        # Estimate remaining time
        if len(self.epoch_times) > 5:
            avg_epoch_time = np.mean(list(self.epoch_times))
            remaining_epochs = max(0, self.total_epochs - self.current_epoch)
            eta = timedelta(seconds=avg_epoch_time * remaining_epochs)
        else:
            eta = "calculating..."
        
        # Format metrics
        train_str = " | ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])
        val_str = ""
        if val_metrics:
            val_str = " | " + " | ".join([f"val_{k}: {v:.4f}" for k, v in val_metrics.items()])
        
        # Progress bar
        if self.total_epochs:
            progress = self.current_epoch / self.total_epochs
            bar_length = 20
            filled_length = int(bar_length * progress)
            bar = "█" * filled_length + "░" * (bar_length - filled_length)
            progress_str = f"[{bar}] {progress*100:.1f}%"
        else:
            progress_str = f"Epoch {self.current_epoch}"
        
        print(f"\r{progress_str} | {train_str}{val_str} | Elapsed: {elapsed} | ETA: {eta}", end="")
        
        # New line every 10 epochs for readability
        if self.current_epoch % (self.update_frequency * 10) == 0:
            print()
    
    def create_live_dashboard(self, save_html: str = None) -> go.Figure:
        """Create interactive training dashboard"""
        
        if not self.epochs:
            print("⚠️  No training data available yet")
            return None
        
        # Determine subplot layout
        n_metrics = len(self.metrics)
        rows = (n_metrics + 1) // 2 if n_metrics > 1 else 1
        cols = 2 if n_metrics > 1 else 1
        
        # Create subplots
        subplot_titles = []
        for metric in self.metrics:
            subplot_titles.append(f"{metric.title()} Over Time")
        
        if len(subplot_titles) % 2 == 1:
            subplot_titles.append("Training Speed")
        else:
            subplot_titles.extend(["Training Speed", "Performance Summary"])
        
        fig = make_subplots(
            rows=rows + 1, cols=cols,
            subplot_titles=subplot_titles,
            specs=[[{"secondary_y": False} for _ in range(cols)] for _ in range(rows + 1)]
        )
        
        epochs_list = list(self.epochs)
        
        # Plot metrics
        for i, metric in enumerate(self.metrics):
            row = (i // cols) + 1
            col = (i % cols) + 1
            
            # Training data
            if metric in self.training_data:
                train_values = list(self.training_data[metric])
                fig.add_trace(
                    go.Scatter(
                        x=epochs_list,
                        y=train_values,
                        mode='lines+markers',
                        name=f'Train {metric}',
                        line=dict(color='blue', width=2),
                        marker=dict(size=4)
                    ),
                    row=row, col=col
                )
            
            # Validation data
            if metric in self.validation_data:
                val_values = list(self.validation_data[metric])
                fig.add_trace(
                    go.Scatter(
                        x=epochs_list,
                        y=val_values,
                        mode='lines+markers',
                        name=f'Val {metric}',
                        line=dict(color='red', width=2, dash='dash'),
                        marker=dict(size=4)
                    ),
                    row=row, col=col
                )
        
        # Training speed plot
        if len(self.epoch_times) > 1:
            speed_row = rows + 1
            speed_col = 1
            
            fig.add_trace(
                go.Scatter(
                    x=epochs_list[-len(self.epoch_times):],
                    y=list(self.epoch_times),
                    mode='lines+markers',
                    name='Epoch Time (s)',
                    line=dict(color='green', width=2),
                    marker=dict(size=4)
                ),
                row=speed_row, col=speed_col
            )
        
        # Performance summary (if we have space)
        if n_metrics % 2 == 0 and len(self.best_metrics) > 0:
            summary_row = rows + 1
            summary_col = 2
            
            best_metric_names = list(self.best_metrics.keys())[:5]  # Top 5 metrics
            best_values = [self.best_metrics[m]['value'] for m in best_metric_names]
            
            fig.add_trace(
                go.Bar(
                    x=best_metric_names,
                    y=best_values,
                    name='Best Values',
                    marker_color='lightblue'
                ),
                row=summary_row, col=summary_col
            )
        
        # Update layout
        fig.update_layout(
            title=f"🚀 Live Training Dashboard - Epoch {self.current_epoch}",
            height=400 * (rows + 1),
            showlegend=True,
            template="plotly_white"
        )
        
        # Save HTML if requested
        if save_html:
            html_path = os.path.join(self.save_dir, save_html)
            fig.write_html(html_path)
            print(f"\n📊 Dashboard saved as {html_path}")
        
        return fig
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        if not self.epochs:
            return {"status": "No training data available"}
        
        total_time = datetime.now() - self.start_time if self.start_time else timedelta(0)
        
        summary = {
            "training_status": "active" if self.is_training else "completed",
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "total_time": str(total_time),
            "epochs_completed": len(self.epochs),
            "best_metrics": self.best_metrics,
            "average_epoch_time": np.mean(list(self.epoch_times)) if self.epoch_times else 0,
            "training_efficiency": self.current_epoch / total_time.total_seconds() * 3600 if total_time.total_seconds() > 0 else 0  # epochs per hour
        }
        
        # Add current metrics
        if self.epochs:
            latest_metrics = {}
            for metric in self.metrics:
                if metric in self.training_data and self.training_data[metric]:
                    latest_metrics[f"latest_train_{metric}"] = self.training_data[metric][-1]
                if metric in self.validation_data and self.validation_data[metric]:
                    latest_metrics[f"latest_val_{metric}"] = self.validation_data[metric][-1]
            summary["latest_metrics"] = latest_metrics
        
        return summary
    
    def save_training_log(self, filename: str = None):
        """Save training log to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"training_log_{timestamp}.json"
        
        # Prepare data for JSON serialization
        log_data = {
            "metadata": {
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "current_epoch": self.current_epoch,
                "total_epochs": self.total_epochs,
                "metrics_tracked": self.metrics
            },
            "training_data": {k: list(v) for k, v in self.training_data.items()},
            "validation_data": {k: list(v) for k, v in self.validation_data.items()},
            "epochs": list(self.epochs),
            "timestamps": [t.isoformat() for t in self.timestamps],
            "best_metrics": self.best_metrics,
            "summary": self.get_training_summary()
        }
        
        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        print(f"💾 Training log saved to {filepath}")
        return filepath
    
    def finish_training(self):
        """Mark training as finished and generate final report"""
        self.is_training = False
        total_time = datetime.now() - self.start_time if self.start_time else timedelta(0)
        
        print(f"\n\n🏁 Training completed!")
        print(f"📈 Total epochs: {self.current_epoch}")
        print(f"⏱️  Total time: {total_time}")
        print(f"🚀 Average epoch time: {np.mean(list(self.epoch_times)):.2f}s" if self.epoch_times else "N/A")
        
        # Print best metrics
        if self.best_metrics:
            print("\n🏆 Best Metrics:")
            for metric, info in self.best_metrics.items():
                print(f"  {metric}: {info['value']:.4f} (epoch {info['epoch']})")
        
        # Save final log
        self.save_training_log()
        
        # Create final dashboard
        self.create_live_dashboard("final_training_dashboard.html")
        
        return self.get_training_summary()


def demonstrate_training_tracker():
    """Demonstrate the training tracker with a mock training session"""
    print("🔬 Training Tracker Demonstration")
    print("=" * 50)
    
    # Initialize tracker
    tracker = TrainingTracker(
        metrics=['loss', 'accuracy', 'f1_score'],
        update_frequency=5
    )
    
    # Simulate training
    tracker.start_training(total_epochs=50)
    
    # Mock training loop
    base_loss = 2.0
    base_acc = 0.1
    base_f1 = 0.1
    
    for epoch in range(1, 51):
        # Simulate improving metrics with some noise
        noise = np.random.normal(0, 0.05)
        train_loss = base_loss * np.exp(-epoch * 0.05) + abs(noise)
        train_acc = min(0.98, base_acc + epoch * 0.015 + noise)
        train_f1 = min(0.96, base_f1 + epoch * 0.014 + noise)
        
        val_loss = train_loss + np.random.normal(0, 0.02)
        val_acc = train_acc - np.random.uniform(0, 0.05)
        val_f1 = train_f1 - np.random.uniform(0, 0.03)
        
        # Log metrics
        tracker.log_epoch(
            epoch=epoch,
            train_metrics={'loss': train_loss, 'accuracy': train_acc, 'f1_score': train_f1},
            val_metrics={'loss': val_loss, 'accuracy': val_acc, 'f1_score': val_f1}
        )
        
        # Simulate variable epoch times
        time.sleep(0.01)  # Small delay to simulate actual training
    
    # Finish training
    summary = tracker.finish_training()
    
    print(f"\n📊 Final Summary: {summary['latest_metrics']}")
    
    return tracker


if __name__ == "__main__":
    # Run demonstration
    demo_tracker = demonstrate_training_tracker()