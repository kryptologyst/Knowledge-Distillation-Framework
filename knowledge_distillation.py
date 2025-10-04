"""
Modern Knowledge Distillation Implementation
==========================================

This module implements a comprehensive knowledge distillation framework with:
- Modern PyTorch practices
- Configuration management
- Comprehensive logging
- Visualization tools
- Model checkpointing
- Web interface support
"""

import os
import yaml
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Container for training metrics"""
    epoch: int
    train_loss: float
    val_loss: float
    train_acc: float
    val_acc: float
    distillation_loss: float = 0.0


class Config:
    """Configuration management class"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self._create_directories()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            'model': {
                'teacher': {'hidden_layers': [512, 256], 'dropout': 0.2, 'activation': 'relu'},
                'student': {'hidden_layers': [128], 'dropout': 0.1, 'activation': 'relu'}
            },
            'training': {
                'batch_size': 64, 'teacher_epochs': 5, 'student_epochs': 10,
                'learning_rate': 0.001, 'weight_decay': 1e-4
            },
            'distillation': {'temperature': 3.0, 'alpha': 0.7},
            'data': {'dataset': 'MNIST', 'data_dir': './data', 'download': True, 'normalize': True},
            'logging': {'log_level': 'INFO', 'log_file': 'logs/training.log', 'save_models': True, 'model_dir': 'models'},
            'visualization': {'save_plots': True, 'plot_dir': 'plots', 'show_plots': False},
            'device': {'use_cuda': True, 'device_id': 0}
        }
    
    def _create_directories(self):
        """Create necessary directories"""
        dirs = [
            self.config['data']['data_dir'],
            self.config['logging']['model_dir'],
            self.config['visualization']['plot_dir'],
            'logs'
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


class ModernTeacherNet(nn.Module):
    """Modern Teacher Network with configurable architecture"""
    
    def __init__(self, config: Dict[str, Any], input_size: int = 784, num_classes: int = 10):
        super(ModernTeacherNet, self).__init__()
        self.config = config
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Build network dynamically
        layers = []
        prev_size = input_size
        
        for hidden_size in config['hidden_layers']:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU() if config['activation'] == 'relu' else nn.Tanh(),
                nn.Dropout(config['dropout'])
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(x.view(x.size(0), -1))


class ModernStudentNet(nn.Module):
    """Modern Student Network with configurable architecture"""
    
    def __init__(self, config: Dict[str, Any], input_size: int = 784, num_classes: int = 10):
        super(ModernStudentNet, self).__init__()
        self.config = config
        self.input_size = input_size
        self.num_classes = num_classes
        
        # Build network dynamically
        layers = []
        prev_size = input_size
        
        for hidden_size in config['hidden_layers']:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU() if config['activation'] == 'relu' else nn.Tanh(),
                nn.Dropout(config['dropout'])
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(x.view(x.size(0), -1))


class KnowledgeDistillationTrainer:
    """Modern Knowledge Distillation Trainer"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = self._setup_device()
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
        self.metrics_history = []
        
        logger.info(f"Using device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup computation device"""
        if self.config.config['device']['use_cuda'] and torch.cuda.is_available():
            device_id = self.config.config['device']['device_id']
            device = torch.device(f"cuda:{device_id}")
            logger.info(f"CUDA available. Using GPU {device_id}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")
        return device
    
    def _get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """Load and prepare datasets"""
        data_config = self.config.config['data']
        
        # Define transforms
        transform_list = [transforms.ToTensor()]
        if data_config['normalize']:
            transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
        
        transform = transforms.Compose(transform_list)
        
        # Load datasets
        train_dataset = torchvision.datasets.MNIST(
            root=data_config['data_dir'],
            train=True,
            download=data_config['download'],
            transform=transform
        )
        
        test_dataset = torchvision.datasets.MNIST(
            root=data_config['data_dir'],
            train=False,
            download=data_config['download'],
            transform=transform
        )
        
        # Create data loaders
        batch_size = self.config.config['training']['batch_size']
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        logger.info(f"Loaded MNIST dataset: {len(train_dataset)} train, {len(test_dataset)} test samples")
        return train_loader, test_loader
    
    def _create_models(self):
        """Create teacher and student models"""
        model_config = self.config.config['model']
        
        self.models['teacher'] = ModernTeacherNet(
            model_config['teacher']
        ).to(self.device)
        
        self.models['student'] = ModernStudentNet(
            model_config['student']
        ).to(self.device)
        
        logger.info("Created teacher and student models")
        logger.info(f"Teacher parameters: {sum(p.numel() for p in self.models['teacher'].parameters()):,}")
        logger.info(f"Student parameters: {sum(p.numel() for p in self.models['student'].parameters()):,}")
    
    def _create_optimizers(self):
        """Create optimizers for both models"""
        training_config = self.config.config['training']
        
        self.optimizers['teacher'] = torch.optim.Adam(
            self.models['teacher'].parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay']
        )
        
        self.optimizers['student'] = torch.optim.Adam(
            self.models['student'].parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay']
        )
        
        # Learning rate schedulers
        self.schedulers['teacher'] = torch.optim.lr_scheduler.StepLR(
            self.optimizers['teacher'], step_size=3, gamma=0.7
        )
        
        self.schedulers['student'] = torch.optim.lr_scheduler.StepLR(
            self.optimizers['student'], step_size=3, gamma=0.7
        )
    
    def distillation_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, 
                         labels: torch.Tensor) -> torch.Tensor:
        """Compute knowledge distillation loss"""
        distillation_config = self.config.config['distillation']
        T = distillation_config['temperature']
        alpha = distillation_config['alpha']
        
        # Soft loss (KL divergence between teacher and student softmax)
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1),
            reduction='batchmean'
        ) * (T * T)
        
        # Hard loss (standard cross entropy)
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # Combined loss
        total_loss = alpha * soft_loss + (1 - alpha) * hard_loss
        
        return total_loss, soft_loss, hard_loss
    
    def evaluate_model(self, model: nn.Module, data_loader: DataLoader) -> Tuple[float, float]:
        """Evaluate model on given data loader"""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        avg_loss = total_loss / len(data_loader)
        
        return avg_loss, accuracy
    
    def train_teacher(self, train_loader: DataLoader, val_loader: DataLoader) -> List[TrainingMetrics]:
        """Train the teacher model"""
        logger.info("Starting teacher training...")
        teacher = self.models['teacher']
        optimizer = self.optimizers['teacher']
        scheduler = self.schedulers['teacher']
        
        teacher.train()
        metrics_history = []
        
        epochs = self.config.config['training']['teacher_epochs']
        
        for epoch in range(epochs):
            teacher.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            progress_bar = tqdm(train_loader, desc=f"Teacher Epoch {epoch+1}/{epochs}")
            
            for images, labels in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = teacher(images)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100 * train_correct / train_total:.2f}%'
                })
            
            scheduler.step()
            
            # Evaluate on validation set
            val_loss, val_acc = self.evaluate_model(teacher, val_loader)
            train_acc = train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=avg_train_loss,
                val_loss=val_loss,
                train_acc=train_acc,
                val_acc=val_acc
            )
            metrics_history.append(metrics)
            
            logger.info(f"Teacher Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
                       f"Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        logger.info("Teacher training completed!")
        return metrics_history
    
    def train_student_with_distillation(self, train_loader: DataLoader, val_loader: DataLoader) -> List[TrainingMetrics]:
        """Train student model using knowledge distillation"""
        logger.info("Starting student training with knowledge distillation...")
        
        teacher = self.models['teacher']
        student = self.models['student']
        optimizer = self.optimizers['student']
        scheduler = self.schedulers['student']
        
        teacher.eval()  # Freeze teacher
        student.train()
        metrics_history = []
        
        epochs = self.config.config['training']['student_epochs']
        
        for epoch in range(epochs):
            student.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            distillation_loss_total = 0.0
            
            progress_bar = tqdm(train_loader, desc=f"Student Epoch {epoch+1}/{epochs}")
            
            for images, labels in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Get teacher predictions (no gradients)
                with torch.no_grad():
                    teacher_outputs = teacher(images)
                
                # Get student predictions
                student_outputs = student(images)
                
                # Compute distillation loss
                total_loss, soft_loss, hard_loss = self.distillation_loss(
                    student_outputs, teacher_outputs, labels
                )
                
                total_loss.backward()
                optimizer.step()
                
                train_loss += total_loss.item()
                distillation_loss_total += soft_loss.item()
                
                _, predicted = torch.max(student_outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                progress_bar.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'Distill': f'{soft_loss.item():.4f}',
                    'Acc': f'{100 * train_correct / train_total:.2f}%'
                })
            
            scheduler.step()
            
            # Evaluate on validation set
            val_loss, val_acc = self.evaluate_model(student, val_loader)
            train_acc = train_correct / train_total
            avg_train_loss = train_loss / len(train_loader)
            avg_distillation_loss = distillation_loss_total / len(train_loader)
            
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=avg_train_loss,
                val_loss=val_loss,
                train_acc=train_acc,
                val_acc=val_acc,
                distillation_loss=avg_distillation_loss
            )
            metrics_history.append(metrics)
            
            logger.info(f"Student Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, "
                       f"Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                       f"Distill Loss: {avg_distillation_loss:.4f}")
        
        logger.info("Student training with distillation completed!")
        return metrics_history
    
    def save_models(self):
        """Save trained models"""
        if self.config.config['logging']['save_models']:
            model_dir = Path(self.config.config['logging']['model_dir'])
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save teacher
            torch.save(self.models['teacher'].state_dict(), model_dir / 'teacher_model.pth')
            
            # Save student
            torch.save(self.models['student'].state_dict(), model_dir / 'student_model.pth')
            
            # Save training history
            with open(model_dir / 'training_history.json', 'w') as f:
                json.dump(self.metrics_history, f, indent=2, default=str)
            
            logger.info(f"Models saved to {model_dir}")
    
    def load_models(self):
        """Load pre-trained models"""
        model_dir = Path(self.config.config['logging']['model_dir'])
        
        if (model_dir / 'teacher_model.pth').exists():
            self.models['teacher'].load_state_dict(torch.load(model_dir / 'teacher_model.pth'))
            logger.info("Loaded teacher model")
        
        if (model_dir / 'student_model.pth').exists():
            self.models['student'].load_state_dict(torch.load(model_dir / 'student_model.pth'))
            logger.info("Loaded student model")
    
    def run_full_training(self):
        """Run complete knowledge distillation training pipeline"""
        logger.info("Starting Knowledge Distillation Training Pipeline")
        
        # Load data
        train_loader, val_loader = self._get_data_loaders()
        
        # Create models and optimizers
        self._create_models()
        self._create_optimizers()
        
        # Train teacher
        teacher_metrics = self.train_teacher(train_loader, val_loader)
        
        # Train student with distillation
        student_metrics = self.train_student_with_distillation(train_loader, val_loader)
        
        # Combine metrics
        self.metrics_history = {
            'teacher': teacher_metrics,
            'student': student_metrics
        }
        
        # Save models
        self.save_models()
        
        # Final evaluation
        teacher_val_loss, teacher_val_acc = self.evaluate_model(self.models['teacher'], val_loader)
        student_val_loss, student_val_acc = self.evaluate_model(self.models['student'], val_loader)
        
        logger.info("=" * 50)
        logger.info("FINAL RESULTS:")
        logger.info(f"Teacher Validation Accuracy: {teacher_val_acc:.4f}")
        logger.info(f"Student Validation Accuracy: {student_val_acc:.4f}")
        logger.info(f"Knowledge Retention: {student_val_acc/teacher_val_acc:.4f}")
        logger.info("=" * 50)
        
        return self.metrics_history


def main():
    """Main training function"""
    # Load configuration
    config = Config()
    
    # Create trainer
    trainer = KnowledgeDistillationTrainer(config)
    
    # Run training
    metrics_history = trainer.run_full_training()
    
    return trainer, metrics_history


if __name__ == "__main__":
    trainer, history = main()
