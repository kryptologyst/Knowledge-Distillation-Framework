"""
Visualization Tools for Knowledge Distillation
=============================================

This module provides comprehensive visualization tools for:
- Training progress monitoring
- Model comparison
- Performance analysis
- Knowledge distillation effectiveness
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class DistillationVisualizer:
    """Comprehensive visualization class for knowledge distillation"""
    
    def __init__(self, config: Dict[str, Any], save_dir: str = "plots"):
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up matplotlib
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        
    def plot_training_curves(self, metrics_history: Dict[str, List], save: bool = True) -> None:
        """Plot training curves for both teacher and student"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Knowledge Distillation Training Progress', fontsize=16, fontweight='bold')
        
        # Extract data
        teacher_metrics = metrics_history['teacher']
        student_metrics = metrics_history['student']
        
        epochs_teacher = [m.epoch for m in teacher_metrics]
        epochs_student = [m.epoch for m in student_metrics]
        
        # Plot 1: Training Loss
        axes[0, 0].plot(epochs_teacher, [m.train_loss for m in teacher_metrics], 
                       'b-', label='Teacher', linewidth=2, marker='o')
        axes[0, 0].plot(epochs_student, [m.train_loss for m in student_metrics], 
                       'r-', label='Student', linewidth=2, marker='s')
        axes[0, 0].set_title('Training Loss', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Validation Loss
        axes[0, 1].plot(epochs_teacher, [m.val_loss for m in teacher_metrics], 
                       'b-', label='Teacher', linewidth=2, marker='o')
        axes[0, 1].plot(epochs_student, [m.val_loss for m in student_metrics], 
                       'r-', label='Student', linewidth=2, marker='s')
        axes[0, 1].set_title('Validation Loss', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Training Accuracy
        axes[1, 0].plot(epochs_teacher, [m.train_acc for m in teacher_metrics], 
                       'b-', label='Teacher', linewidth=2, marker='o')
        axes[1, 0].plot(epochs_student, [m.train_acc for m in student_metrics], 
                       'r-', label='Student', linewidth=2, marker='s')
        axes[1, 0].set_title('Training Accuracy', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Validation Accuracy
        axes[1, 1].plot(epochs_teacher, [m.val_acc for m in teacher_metrics], 
                       'b-', label='Teacher', linewidth=2, marker='o')
        axes[1, 1].plot(epochs_student, [m.val_acc for m in student_metrics], 
                       'r-', label='Student', linewidth=2, marker='s')
        axes[1, 1].set_title('Validation Accuracy', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
            logger.info(f"Training curves saved to {self.save_dir / 'training_curves.png'}")
        
        if self.config.get('show_plots', False):
            plt.show()
        else:
            plt.close()
    
    def plot_distillation_loss(self, student_metrics: List, save: bool = True) -> None:
        """Plot distillation loss progression"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        epochs = [m.epoch for m in student_metrics]
        distill_losses = [m.distillation_loss for m in student_metrics]
        
        ax.plot(epochs, distill_losses, 'g-', linewidth=2, marker='o', markersize=6)
        ax.set_title('Knowledge Distillation Loss Progression', fontweight='bold', fontsize=14)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Distillation Loss')
        ax.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(epochs, distill_losses, 1)
        p = np.poly1d(z)
        ax.plot(epochs, p(epochs), "r--", alpha=0.8, linewidth=1)
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.save_dir / 'distillation_loss.png', dpi=300, bbox_inches='tight')
            logger.info(f"Distillation loss plot saved to {self.save_dir / 'distillation_loss.png'}")
        
        if self.config.get('show_plots', False):
            plt.show()
        else:
            plt.close()
    
    def plot_model_comparison(self, teacher_acc: float, student_acc: float, 
                            teacher_params: int, student_params: int, save: bool = True) -> None:
        """Plot model comparison metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        models = ['Teacher', 'Student']
        accuracies = [teacher_acc, student_acc]
        colors = ['skyblue', 'lightcoral']
        
        bars1 = ax1.bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        ax1.set_title('Model Accuracy Comparison', fontweight='bold', fontsize=14)
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Parameter count comparison
        params = [teacher_params, student_params]
        bars2 = ax2.bar(models, params, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        ax2.set_title('Model Parameter Count', fontweight='bold', fontsize=14)
        ax2.set_ylabel('Number of Parameters')
        ax2.set_yscale('log')
        
        # Add value labels on bars
        for bar, param in zip(bars2, params):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                    f'{param:,}', ha='center', va='bottom', fontweight='bold')
        
        # Add compression ratio
        compression_ratio = teacher_params / student_params
        ax2.text(0.5, 0.95, f'Compression Ratio: {compression_ratio:.1f}x', 
                transform=ax2.transAxes, ha='center', va='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.save_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {self.save_dir / 'model_comparison.png'}")
        
        if self.config.get('show_plots', False):
            plt.show()
        else:
            plt.close()
    
    def plot_confusion_matrices(self, teacher_preds: np.ndarray, student_preds: np.ndarray, 
                              true_labels: np.ndarray, save: bool = True) -> None:
        """Plot confusion matrices for both models"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Teacher confusion matrix
        cm_teacher = confusion_matrix(true_labels, teacher_preds)
        sns.heatmap(cm_teacher, annot=True, fmt='d', cmap='Blues', ax=ax1)
        ax1.set_title('Teacher Model Confusion Matrix', fontweight='bold', fontsize=14)
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # Student confusion matrix
        cm_student = confusion_matrix(true_labels, student_preds)
        sns.heatmap(cm_student, annot=True, fmt='d', cmap='Reds', ax=ax2)
        ax2.set_title('Student Model Confusion Matrix', fontweight='bold', fontsize=14)
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.save_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrices saved to {self.save_dir / 'confusion_matrices.png'}")
        
        if self.config.get('show_plots', False):
            plt.show()
        else:
            plt.close()
    
    def plot_knowledge_retention(self, teacher_acc: float, student_acc: float, save: bool = True) -> None:
        """Plot knowledge retention visualization"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        retention_rate = student_acc / teacher_acc
        
        # Create a gauge-like visualization
        categories = ['Knowledge\nRetention']
        values = [retention_rate]
        colors = ['green' if retention_rate > 0.9 else 'orange' if retention_rate > 0.8 else 'red']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
        
        ax.set_title('Knowledge Retention Rate', fontweight='bold', fontsize=16)
        ax.set_ylabel('Retention Rate')
        ax.set_ylim(0, 1)
        
        # Add value label
        ax.text(0, retention_rate + 0.05, f'{retention_rate:.3f}', 
               ha='center', va='bottom', fontsize=14, fontweight='bold')
        
        # Add interpretation text
        if retention_rate > 0.9:
            interpretation = "Excellent Knowledge Transfer!"
        elif retention_rate > 0.8:
            interpretation = "Good Knowledge Transfer"
        else:
            interpretation = "Needs Improvement"
        
        ax.text(0, 0.1, interpretation, ha='center', va='center', 
               fontsize=12, fontweight='bold', 
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.save_dir / 'knowledge_retention.png', dpi=300, bbox_inches='tight')
            logger.info(f"Knowledge retention plot saved to {self.save_dir / 'knowledge_retention.png'}")
        
        if self.config.get('show_plots', False):
            plt.show()
        else:
            plt.close()
    
    def create_interactive_dashboard(self, metrics_history: Dict[str, List], 
                                  teacher_acc: float, student_acc: float,
                                  teacher_params: int, student_params: int) -> None:
        """Create interactive Plotly dashboard"""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Loss', 'Validation Accuracy', 
                          'Model Comparison', 'Knowledge Retention'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Extract data
        teacher_metrics = metrics_history['teacher']
        student_metrics = metrics_history['student']
        
        epochs_teacher = [m.epoch for m in teacher_metrics]
        epochs_student = [m.epoch for m in student_metrics]
        
        # Training Loss
        fig.add_trace(
            go.Scatter(x=epochs_teacher, y=[m.train_loss for m in teacher_metrics],
                      mode='lines+markers', name='Teacher Loss', line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=epochs_student, y=[m.train_loss for m in student_metrics],
                      mode='lines+markers', name='Student Loss', line=dict(color='red')),
            row=1, col=1
        )
        
        # Validation Accuracy
        fig.add_trace(
            go.Scatter(x=epochs_teacher, y=[m.val_acc for m in teacher_metrics],
                      mode='lines+markers', name='Teacher Acc', line=dict(color='blue')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=epochs_student, y=[m.val_acc for m in student_metrics],
                      mode='lines+markers', name='Student Acc', line=dict(color='red')),
            row=1, col=2
        )
        
        # Model Comparison
        fig.add_trace(
            go.Bar(x=['Teacher', 'Student'], y=[teacher_acc, student_acc],
                  name='Accuracy', marker_color=['skyblue', 'lightcoral']),
            row=2, col=1
        )
        
        # Knowledge Retention
        retention_rate = student_acc / teacher_acc
        fig.add_trace(
            go.Bar(x=['Knowledge Retention'], y=[retention_rate],
                  name='Retention Rate', marker_color='green'),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Knowledge Distillation Dashboard",
            showlegend=True,
            height=800,
            width=1200
        )
        
        # Save interactive plot
        fig.write_html(self.save_dir / 'interactive_dashboard.html')
        logger.info(f"Interactive dashboard saved to {self.save_dir / 'interactive_dashboard.html'}")
    
    def generate_summary_report(self, metrics_history: Dict[str, List], 
                             teacher_acc: float, student_acc: float,
                             teacher_params: int, student_params: int) -> str:
        """Generate a comprehensive summary report"""
        retention_rate = student_acc / teacher_acc
        compression_ratio = teacher_params / student_params
        
        report = f"""
# Knowledge Distillation Training Report

## Summary Statistics
- **Teacher Model Accuracy**: {teacher_acc:.4f} ({teacher_acc*100:.2f}%)
- **Student Model Accuracy**: {student_acc:.4f} ({student_acc*100:.2f}%)
- **Knowledge Retention Rate**: {retention_rate:.4f} ({retention_rate*100:.2f}%)
- **Model Compression Ratio**: {compression_ratio:.1f}x
- **Parameter Reduction**: {((teacher_params - student_params) / teacher_params * 100):.1f}%

## Training Configuration
- **Teacher Epochs**: {len(metrics_history['teacher'])}
- **Student Epochs**: {len(metrics_history['student'])}
- **Distillation Temperature**: {self.config['distillation']['temperature']}
- **Alpha (Soft/Hard Loss Weight)**: {self.config['distillation']['alpha']}

## Performance Analysis
"""
        
        if retention_rate > 0.9:
            report += "- ✅ **Excellent knowledge transfer** - Student model retains over 90% of teacher's performance\n"
        elif retention_rate > 0.8:
            report += "- ✅ **Good knowledge transfer** - Student model retains over 80% of teacher's performance\n"
        else:
            report += "- ⚠️ **Knowledge transfer needs improvement** - Consider adjusting hyperparameters\n"
        
        if compression_ratio > 5:
            report += "- ✅ **Significant model compression** - Achieved over 5x parameter reduction\n"
        elif compression_ratio > 2:
            report += "- ✅ **Moderate model compression** - Achieved over 2x parameter reduction\n"
        else:
            report += "- ⚠️ **Limited compression** - Consider reducing student model size\n"
        
        report += f"""
## Recommendations
1. **For Production**: The student model achieves {student_acc*100:.1f}% accuracy with {compression_ratio:.1f}x fewer parameters
2. **For Further Optimization**: Consider experimenting with different temperatures and alpha values
3. **For Deployment**: Student model is ready for deployment in resource-constrained environments

---
*Report generated by Knowledge Distillation Framework*
"""
        
        return report


def create_visualization_suite(metrics_history: Dict[str, List], config: Dict[str, Any],
                             teacher_acc: float, student_acc: float,
                             teacher_params: int, student_params: int,
                             teacher_preds: Optional[np.ndarray] = None,
                             student_preds: Optional[np.ndarray] = None,
                             true_labels: Optional[np.ndarray] = None) -> None:
    """Create complete visualization suite"""
    
    visualizer = DistillationVisualizer(config)
    
    # Generate all visualizations
    visualizer.plot_training_curves(metrics_history)
    visualizer.plot_distillation_loss(metrics_history['student'])
    visualizer.plot_model_comparison(teacher_acc, student_acc, teacher_params, student_params)
    visualizer.plot_knowledge_retention(teacher_acc, student_acc)
    
    if teacher_preds is not None and student_preds is not None and true_labels is not None:
        visualizer.plot_confusion_matrices(teacher_preds, student_preds, true_labels)
    
    # Create interactive dashboard
    visualizer.create_interactive_dashboard(metrics_history, teacher_acc, student_acc, 
                                          teacher_params, student_params)
    
    # Generate summary report
    report = visualizer.generate_summary_report(metrics_history, teacher_acc, student_acc,
                                               teacher_params, student_params)
    
    # Save report
    with open(visualizer.save_dir / 'training_report.md', 'w') as f:
        f.write(report)
    
    logger.info(f"Complete visualization suite saved to {visualizer.save_dir}")
    logger.info("Generated visualizations:")
    logger.info("- training_curves.png")
    logger.info("- distillation_loss.png") 
    logger.info("- model_comparison.png")
    logger.info("- knowledge_retention.png")
    logger.info("- confusion_matrices.png (if predictions provided)")
    logger.info("- interactive_dashboard.html")
    logger.info("- training_report.md")
