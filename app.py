"""
Streamlit Web Interface for Knowledge Distillation
=================================================

This module provides an interactive web interface for:
- Training knowledge distillation models
- Real-time monitoring of training progress
- Model comparison and evaluation
- Hyperparameter tuning
- Results visualization
"""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml
import json
from pathlib import Path
import time
import logging
from typing import Dict, List, Tuple, Optional

# Import our modules
from knowledge_distillation import Config, KnowledgeDistillationTrainer, TrainingMetrics
from visualization import DistillationVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Knowledge Distillation Framework",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .warning-message {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffeaa7;
    }
</style>
""", unsafe_allow_html=True)


class StreamlitKnowledgeDistillationApp:
    """Main Streamlit application class"""
    
    def __init__(self):
        self.config = None
        self.trainer = None
        self.metrics_history = None
        
    def run(self):
        """Run the Streamlit application"""
        # Header
        st.markdown('<h1 class="main-header">🧠 Knowledge Distillation Framework</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Navigate",
            ["🏠 Home", "⚙️ Configuration", "🚀 Training", "📊 Results", "🔍 Model Analysis"]
        )
        
        if page == "🏠 Home":
            self.show_home_page()
        elif page == "⚙️ Configuration":
            self.show_configuration_page()
        elif page == "🚀 Training":
            self.show_training_page()
        elif page == "📊 Results":
            self.show_results_page()
        elif page == "🔍 Model Analysis":
            self.show_analysis_page()
    
    def show_home_page(self):
        """Display home page with overview"""
        st.markdown("""
        ## Welcome to the Knowledge Distillation Framework! 🎓
        
        This interactive web application allows you to:
        - **Train teacher-student models** using knowledge distillation
        - **Monitor training progress** in real-time
        - **Compare model performance** and analyze results
        - **Tune hyperparameters** interactively
        - **Visualize knowledge transfer** effectiveness
        
        ### What is Knowledge Distillation?
        
        Knowledge distillation is a technique for transferring knowledge from a large, 
        accurate model (teacher) to a smaller, faster model (student). The student learns 
        not just from hard labels but also from the soft predictions of the teacher, 
        enabling better generalization even with fewer parameters.
        
        ### Key Benefits:
        - 🎯 **Model Compression**: Reduce model size while maintaining performance
        - ⚡ **Faster Inference**: Smaller models run faster on edge devices
        - 📱 **Mobile Deployment**: Deploy AI models on resource-constrained devices
        - 🔬 **Research Tool**: Study knowledge transfer mechanisms
        
        ### Getting Started:
        1. Go to **Configuration** to set up your training parameters
        2. Navigate to **Training** to start the knowledge distillation process
        3. Check **Results** to see training progress and final metrics
        4. Use **Model Analysis** for detailed performance evaluation
        
        ---
        """)
        
        # Quick stats if models exist
        if Path("models").exists() and any(Path("models").glob("*.pth")):
            st.markdown("### 📈 Recent Training Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Models Available", "2", "Teacher & Student")
            
            with col2:
                st.metric("Training Status", "Complete", "✅")
            
            with col3:
                st.metric("Visualizations", "Generated", "📊")
            
            with col4:
                st.metric("Reports", "Available", "📄")
    
    def show_configuration_page(self):
        """Display configuration page"""
        st.header("⚙️ Training Configuration")
        
        # Create tabs for different configuration sections
        tab1, tab2, tab3, tab4 = st.tabs(["Model Architecture", "Training Parameters", 
                                         "Distillation Settings", "Data & Logging"])
        
        with tab1:
            st.subheader("Model Architecture")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Teacher Model")
                teacher_layers = st.text_input(
                    "Hidden Layers (comma-separated)", 
                    value="512,256",
                    help="Enter layer sizes separated by commas"
                )
                teacher_dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)
                teacher_activation = st.selectbox("Activation Function", ["relu", "tanh"], index=0)
            
            with col2:
                st.markdown("#### Student Model")
                student_layers = st.text_input(
                    "Hidden Layers (comma-separated)", 
                    value="128",
                    help="Enter layer sizes separated by commas"
                )
                student_dropout = st.slider("Dropout Rate", 0.0, 0.5, 0.1, 0.05)
                student_activation = st.selectbox("Activation Function", ["relu", "tanh"], index=0)
        
        with tab2:
            st.subheader("Training Parameters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                batch_size = st.selectbox("Batch Size", [32, 64, 128, 256], index=1)
                teacher_epochs = st.slider("Teacher Epochs", 1, 20, 5)
                student_epochs = st.slider("Student Epochs", 1, 20, 10)
            
            with col2:
                learning_rate = st.number_input("Learning Rate", 0.0001, 0.01, 0.001, 0.0001)
                weight_decay = st.number_input("Weight Decay", 0.0, 0.01, 0.0001, 0.0001)
        
        with tab3:
            st.subheader("Knowledge Distillation Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                temperature = st.slider("Temperature", 1.0, 10.0, 3.0, 0.5)
                st.markdown("""
                **Temperature** controls the softness of the probability distribution.
                Higher values make the distribution softer.
                """)
            
            with col2:
                alpha = st.slider("Alpha (Soft/Hard Loss Weight)", 0.0, 1.0, 0.7, 0.1)
                st.markdown("""
                **Alpha** balances between soft loss (teacher knowledge) and hard loss (ground truth).
                Higher values emphasize teacher knowledge.
                """)
        
        with tab4:
            st.subheader("Data & Logging")
            
            col1, col2 = st.columns(2)
            
            with col1:
                data_dir = st.text_input("Data Directory", value="./data")
                normalize_data = st.checkbox("Normalize Data", value=True)
                save_models = st.checkbox("Save Models", value=True)
            
            with col2:
                log_level = st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"], index=1)
                show_plots = st.checkbox("Show Plots", value=False)
                use_cuda = st.checkbox("Use CUDA (if available)", value=True)
        
        # Save configuration
        if st.button("💾 Save Configuration", type="primary"):
            config_dict = {
                'model': {
                    'teacher': {
                        'hidden_layers': [int(x.strip()) for x in teacher_layers.split(',')],
                        'dropout': teacher_dropout,
                        'activation': teacher_activation
                    },
                    'student': {
                        'hidden_layers': [int(x.strip()) for x in student_layers.split(',')],
                        'dropout': student_dropout,
                        'activation': student_activation
                    }
                },
                'training': {
                    'batch_size': batch_size,
                    'teacher_epochs': teacher_epochs,
                    'student_epochs': student_epochs,
                    'learning_rate': learning_rate,
                    'weight_decay': weight_decay
                },
                'distillation': {
                    'temperature': temperature,
                    'alpha': alpha
                },
                'data': {
                    'dataset': 'MNIST',
                    'data_dir': data_dir,
                    'download': True,
                    'normalize': normalize_data
                },
                'logging': {
                    'log_level': log_level,
                    'log_file': 'logs/training.log',
                    'save_models': save_models,
                    'model_dir': 'models'
                },
                'visualization': {
                    'save_plots': True,
                    'plot_dir': 'plots',
                    'show_plots': show_plots
                },
                'device': {
                    'use_cuda': use_cuda,
                    'device_id': 0
                }
            }
            
            # Save to YAML file
            with open('config.yaml', 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
            
            st.success("✅ Configuration saved successfully!")
            st.session_state.config_saved = True
    
    def show_training_page(self):
        """Display training page"""
        st.header("🚀 Model Training")
        
        # Check if configuration exists
        if not Path("config.yaml").exists():
            st.warning("⚠️ Please configure your training parameters first!")
            st.markdown("Go to the **Configuration** page to set up your training parameters.")
            return
        
        # Load configuration
        try:
            with open('config.yaml', 'r') as f:
                config_dict = yaml.safe_load(f)
            self.config = Config()
            self.config.config = config_dict
        except Exception as e:
            st.error(f"Error loading configuration: {e}")
            return
        
        # Training controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown("### Training Controls")
        
        with col2:
            if st.button("🎓 Train Teacher", type="primary"):
                self.train_teacher()
        
        with col3:
            if st.button("🎓 Train Student", type="secondary"):
                self.train_student()
        
        # Full training pipeline
        st.markdown("### Complete Training Pipeline")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown("""
            Run the complete knowledge distillation pipeline:
            1. Train the teacher model
            2. Train the student model using knowledge distillation
            3. Generate comprehensive visualizations and reports
            """)
        
        with col2:
            if st.button("🚀 Run Full Pipeline", type="primary"):
                self.run_full_training()
        
        # Training progress
        if 'training_progress' in st.session_state:
            st.markdown("### 📊 Training Progress")
            
            progress_data = st.session_state.training_progress
            
            # Create progress charts
            if progress_data:
                self.plot_training_progress(progress_data)
    
    def train_teacher(self):
        """Train teacher model"""
        with st.spinner("Training teacher model..."):
            try:
                self.trainer = KnowledgeDistillationTrainer(self.config)
                train_loader, val_loader = self.trainer._get_data_loaders()
                self.trainer._create_models()
                self.trainer._create_optimizers()
                
                # Train teacher
                teacher_metrics = self.trainer.train_teacher(train_loader, val_loader)
                
                st.success("✅ Teacher model trained successfully!")
                
                # Store metrics
                if 'training_progress' not in st.session_state:
                    st.session_state.training_progress = {}
                st.session_state.training_progress['teacher'] = teacher_metrics
                
            except Exception as e:
                st.error(f"Error training teacher: {e}")
    
    def train_student(self):
        """Train student model"""
        if not hasattr(self, 'trainer') or self.trainer is None:
            st.warning("⚠️ Please train the teacher model first!")
            return
        
        with st.spinner("Training student model with knowledge distillation..."):
            try:
                train_loader, val_loader = self.trainer._get_data_loaders()
                
                # Train student
                student_metrics = self.trainer.train_student_with_distillation(train_loader, val_loader)
                
                st.success("✅ Student model trained successfully!")
                
                # Store metrics
                if 'training_progress' not in st.session_state:
                    st.session_state.training_progress = {}
                st.session_state.training_progress['student'] = student_metrics
                
            except Exception as e:
                st.error(f"Error training student: {e}")
    
    def run_full_training(self):
        """Run complete training pipeline"""
        with st.spinner("Running complete knowledge distillation pipeline..."):
            try:
                self.trainer = KnowledgeDistillationTrainer(self.config)
                metrics_history = self.trainer.run_full_training()
                
                st.success("✅ Complete training pipeline finished!")
                
                # Store metrics
                st.session_state.training_progress = metrics_history
                st.session_state.training_complete = True
                
                # Show final results
                self.show_final_results()
                
            except Exception as e:
                st.error(f"Error in training pipeline: {e}")
    
    def plot_training_progress(self, progress_data):
        """Plot training progress"""
        if 'teacher' in progress_data and 'student' in progress_data:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Training Loss', 'Validation Loss', 
                              'Training Accuracy', 'Validation Accuracy'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            teacher_metrics = progress_data['teacher']
            student_metrics = progress_data['student']
            
            epochs_teacher = [m.epoch for m in teacher_metrics]
            epochs_student = [m.epoch for m in student_metrics]
            
            # Training Loss
            fig.add_trace(
                go.Scatter(x=epochs_teacher, y=[m.train_loss for m in teacher_metrics],
                          mode='lines+markers', name='Teacher', line=dict(color='blue')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=epochs_student, y=[m.train_loss for m in student_metrics],
                          mode='lines+markers', name='Student', line=dict(color='red')),
                row=1, col=1
            )
            
            # Validation Loss
            fig.add_trace(
                go.Scatter(x=epochs_teacher, y=[m.val_loss for m in teacher_metrics],
                          mode='lines+markers', name='Teacher', line=dict(color='blue'), showlegend=False),
                row=1, col=2
            )
            fig.add_trace(
                go.Scatter(x=epochs_student, y=[m.val_loss for m in student_metrics],
                          mode='lines+markers', name='Student', line=dict(color='red'), showlegend=False),
                row=1, col=2
            )
            
            # Training Accuracy
            fig.add_trace(
                go.Scatter(x=epochs_teacher, y=[m.train_acc for m in teacher_metrics],
                          mode='lines+markers', name='Teacher', line=dict(color='blue'), showlegend=False),
                row=2, col=1
            )
            fig.add_trace(
                go.Scatter(x=epochs_student, y=[m.train_acc for m in student_metrics],
                          mode='lines+markers', name='Student', line=dict(color='red'), showlegend=False),
                row=2, col=1
            )
            
            # Validation Accuracy
            fig.add_trace(
                go.Scatter(x=epochs_teacher, y=[m.val_acc for m in teacher_metrics],
                          mode='lines+markers', name='Teacher', line=dict(color='blue'), showlegend=False),
                row=2, col=2
            )
            fig.add_trace(
                go.Scatter(x=epochs_student, y=[m.val_acc for m in student_metrics],
                          mode='lines+markers', name='Student', line=dict(color='red'), showlegend=False),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
    
    def show_final_results(self):
        """Show final training results"""
        st.markdown("### 🎯 Final Results")
        
        if hasattr(self, 'trainer') and self.trainer is not None:
            # Get final accuracies
            train_loader, val_loader = self.trainer._get_data_loaders()
            teacher_val_loss, teacher_val_acc = self.trainer.evaluate_model(
                self.trainer.models['teacher'], val_loader
            )
            student_val_loss, student_val_acc = self.trainer.evaluate_model(
                self.trainer.models['student'], val_loader
            )
            
            # Calculate metrics
            retention_rate = student_val_acc / teacher_val_acc
            teacher_params = sum(p.numel() for p in self.trainer.models['teacher'].parameters())
            student_params = sum(p.numel() for p in self.trainer.models['student'].parameters())
            compression_ratio = teacher_params / student_params
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Teacher Accuracy", f"{teacher_val_acc:.3f}", f"{teacher_val_acc*100:.1f}%")
            
            with col2:
                st.metric("Student Accuracy", f"{student_val_acc:.3f}", f"{student_val_acc*100:.1f}%")
            
            with col3:
                st.metric("Knowledge Retention", f"{retention_rate:.3f}", f"{retention_rate*100:.1f}%")
            
            with col4:
                st.metric("Compression Ratio", f"{compression_ratio:.1f}x", f"{compression_ratio:.1f}x")
    
    def show_results_page(self):
        """Display results page"""
        st.header("📊 Training Results")
        
        # Check if training is complete
        if not st.session_state.get('training_complete', False):
            st.info("ℹ️ No training results available. Please run training first.")
            return
        
        # Load results if available
        if Path("models/training_history.json").exists():
            with open("models/training_history.json", 'r') as f:
                metrics_history = json.load(f)
            
            st.markdown("### 📈 Training Metrics")
            
            # Convert back to TrainingMetrics objects
            teacher_metrics = [TrainingMetrics(**m) for m in metrics_history['teacher']]
            student_metrics = [TrainingMetrics(**m) for m in metrics_history['student']]
            
            # Create comprehensive visualizations
            self.create_results_visualizations(teacher_metrics, student_metrics)
        
        # Show saved plots
        plots_dir = Path("plots")
        if plots_dir.exists():
            st.markdown("### 🖼️ Generated Visualizations")
            
            plot_files = list(plots_dir.glob("*.png"))
            if plot_files:
                cols = st.columns(2)
                for i, plot_file in enumerate(plot_files):
                    with cols[i % 2]:
                        st.image(str(plot_file), caption=plot_file.stem, use_column_width=True)
    
    def create_results_visualizations(self, teacher_metrics, student_metrics):
        """Create comprehensive results visualizations"""
        # Training curves
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Training Loss', 'Validation Loss', 
                          'Training Accuracy', 'Validation Accuracy')
        )
        
        epochs_teacher = [m.epoch for m in teacher_metrics]
        epochs_student = [m.epoch for m in student_metrics]
        
        # Add traces
        fig.add_trace(go.Scatter(x=epochs_teacher, y=[m.train_loss for m in teacher_metrics],
                               mode='lines+markers', name='Teacher Loss'), row=1, col=1)
        fig.add_trace(go.Scatter(x=epochs_student, y=[m.train_loss for m in student_metrics],
                               mode='lines+markers', name='Student Loss'), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=epochs_teacher, y=[m.val_loss for m in teacher_metrics],
                               mode='lines+markers', name='Teacher Val Loss'), row=1, col=2)
        fig.add_trace(go.Scatter(x=epochs_student, y=[m.val_loss for m in student_metrics],
                               mode='lines+markers', name='Student Val Loss'), row=1, col=2)
        
        fig.add_trace(go.Scatter(x=epochs_teacher, y=[m.train_acc for m in teacher_metrics],
                               mode='lines+markers', name='Teacher Train Acc'), row=2, col=1)
        fig.add_trace(go.Scatter(x=epochs_student, y=[m.train_acc for m in student_metrics],
                               mode='lines+markers', name='Student Train Acc'), row=2, col=1)
        
        fig.add_trace(go.Scatter(x=epochs_teacher, y=[m.val_acc for m in teacher_metrics],
                               mode='lines+markers', name='Teacher Val Acc'), row=2, col=2)
        fig.add_trace(go.Scatter(x=epochs_student, y=[m.val_acc for m in student_metrics],
                               mode='lines+markers', name='Student Val Acc'), row=2, col=2)
        
        fig.update_layout(height=600, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    def show_analysis_page(self):
        """Display model analysis page"""
        st.header("🔍 Model Analysis")
        
        # Check if models exist
        if not Path("models").exists() or not any(Path("models").glob("*.pth")):
            st.warning("⚠️ No trained models found. Please train models first.")
            return
        
        st.markdown("### 📊 Model Comparison")
        
        # Load models and compare
        if hasattr(self, 'trainer') and self.trainer is not None:
            # Model architecture comparison
            teacher_params = sum(p.numel() for p in self.trainer.models['teacher'].parameters())
            student_params = sum(p.numel() for p in self.trainer.models['student'].parameters())
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Teacher Model")
                st.metric("Parameters", f"{teacher_params:,}")
                st.metric("Architecture", "Large")
            
            with col2:
                st.markdown("#### Student Model")
                st.metric("Parameters", f"{student_params:,}")
                st.metric("Architecture", "Compact")
            
            # Compression analysis
            compression_ratio = teacher_params / student_params
            reduction_percentage = (teacher_params - student_params) / teacher_params * 100
            
            st.markdown("### 📉 Compression Analysis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Compression Ratio", f"{compression_ratio:.1f}x")
            
            with col2:
                st.metric("Parameter Reduction", f"{reduction_percentage:.1f}%")
            
            with col3:
                st.metric("Size Reduction", f"{100 - (100/compression_ratio):.1f}%")
        
        # Performance analysis
        st.markdown("### 🎯 Performance Analysis")
        
        if Path("plots/training_report.md").exists():
            with open("plots/training_report.md", 'r') as f:
                report_content = f.read()
            
            st.markdown(report_content)


def main():
    """Main function to run the Streamlit app"""
    app = StreamlitKnowledgeDistillationApp()
    app.run()


if __name__ == "__main__":
    main()
