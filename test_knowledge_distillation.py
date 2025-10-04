"""
Unit Tests for Knowledge Distillation Framework
==============================================

This module contains comprehensive unit tests for:
- Model architectures
- Training pipeline
- Knowledge distillation loss
- Configuration management
- Visualization tools
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import shutil
from pathlib import Path
import yaml
import json

# Import modules to test
from knowledge_distillation import (
    Config, ModernTeacherNet, ModernStudentNet, 
    KnowledgeDistillationTrainer, TrainingMetrics
)
from visualization import DistillationVisualizer


class TestConfig(unittest.TestCase):
    """Test configuration management"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.yaml"
        
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_default_config(self):
        """Test default configuration creation"""
        config = Config()
        self.assertIsInstance(config.config, dict)
        self.assertIn('model', config.config)
        self.assertIn('training', config.config)
        self.assertIn('distillation', config.config)
    
    def test_config_loading(self):
        """Test configuration loading from file"""
        test_config = {
            'model': {'teacher': {'hidden_layers': [256, 128]}},
            'training': {'batch_size': 32},
            'distillation': {'temperature': 2.0}
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(test_config, f)
        
        config = Config(str(self.config_path))
        self.assertEqual(config.config['model']['teacher']['hidden_layers'], [256, 128])
        self.assertEqual(config.config['training']['batch_size'], 32)
    
    def test_directory_creation(self):
        """Test automatic directory creation"""
        config = Config()
        self.assertTrue(Path(config.config['data']['data_dir']).exists())
        self.assertTrue(Path(config.config['logging']['model_dir']).exists())


class TestModelArchitectures(unittest.TestCase):
    """Test model architectures"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.device = torch.device("cpu")
        self.input_size = 784
        self.num_classes = 10
        
    def test_teacher_net_creation(self):
        """Test teacher network creation"""
        config = {
            'hidden_layers': [512, 256],
            'dropout': 0.2,
            'activation': 'relu'
        }
        
        model = ModernTeacherNet(config, self.input_size, self.num_classes)
        self.assertIsInstance(model, nn.Module)
        
        # Test forward pass
        x = torch.randn(32, 28, 28)
        output = model(x)
        self.assertEqual(output.shape, (32, self.num_classes))
    
    def test_student_net_creation(self):
        """Test student network creation"""
        config = {
            'hidden_layers': [128],
            'dropout': 0.1,
            'activation': 'relu'
        }
        
        model = ModernStudentNet(config, self.input_size, self.num_classes)
        self.assertIsInstance(model, nn.Module)
        
        # Test forward pass
        x = torch.randn(32, 28, 28)
        output = model(x)
        self.assertEqual(output.shape, (32, self.num_classes))
    
    def test_model_parameter_count(self):
        """Test model parameter counting"""
        teacher_config = {'hidden_layers': [512, 256], 'dropout': 0.2, 'activation': 'relu'}
        student_config = {'hidden_layers': [128], 'dropout': 0.1, 'activation': 'relu'}
        
        teacher = ModernTeacherNet(teacher_config, self.input_size, self.num_classes)
        student = ModernStudentNet(student_config, self.input_size, self.num_classes)
        
        teacher_params = sum(p.numel() for p in teacher.parameters())
        student_params = sum(p.numel() for p in student.parameters())
        
        self.assertGreater(teacher_params, student_params)
        self.assertGreater(teacher_params, 0)
        self.assertGreater(student_params, 0)


class TestKnowledgeDistillationTrainer(unittest.TestCase):
    """Test knowledge distillation trainer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create minimal config
        self.config_dict = {
            'model': {
                'teacher': {'hidden_layers': [64], 'dropout': 0.1, 'activation': 'relu'},
                'student': {'hidden_layers': [32], 'dropout': 0.1, 'activation': 'relu'}
            },
            'training': {
                'batch_size': 32, 'teacher_epochs': 1, 'student_epochs': 1,
                'learning_rate': 0.001, 'weight_decay': 1e-4
            },
            'distillation': {'temperature': 2.0, 'alpha': 0.7},
            'data': {'dataset': 'MNIST', 'data_dir': self.temp_dir, 'download': False, 'normalize': True},
            'logging': {'log_level': 'INFO', 'log_file': 'logs/training.log', 'save_models': True, 'model_dir': 'models'},
            'visualization': {'save_plots': False, 'plot_dir': 'plots', 'show_plots': False},
            'device': {'use_cuda': False, 'device_id': 0}
        }
        
        self.config = Config()
        self.config.config = self.config_dict
        
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_trainer_initialization(self):
        """Test trainer initialization"""
        trainer = KnowledgeDistillationTrainer(self.config)
        self.assertIsNotNone(trainer.device)
        self.assertIsInstance(trainer.device, torch.device)
    
    def test_distillation_loss(self):
        """Test knowledge distillation loss calculation"""
        trainer = KnowledgeDistillationTrainer(self.config)
        
        # Create dummy data
        batch_size = 16
        num_classes = 10
        
        student_logits = torch.randn(batch_size, num_classes)
        teacher_logits = torch.randn(batch_size, num_classes)
        labels = torch.randint(0, num_classes, (batch_size,))
        
        # Test loss calculation
        total_loss, soft_loss, hard_loss = trainer.distillation_loss(
            student_logits, teacher_logits, labels
        )
        
        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertIsInstance(soft_loss, torch.Tensor)
        self.assertIsInstance(hard_loss, torch.Tensor)
        self.assertGreater(total_loss.item(), 0)
        self.assertGreater(soft_loss.item(), 0)
        self.assertGreater(hard_loss.item(), 0)
    
    def test_model_creation(self):
        """Test model creation"""
        trainer = KnowledgeDistillationTrainer(self.config)
        trainer._create_models()
        
        self.assertIn('teacher', trainer.models)
        self.assertIn('student', trainer.models)
        self.assertIsInstance(trainer.models['teacher'], ModernTeacherNet)
        self.assertIsInstance(trainer.models['student'], ModernStudentNet)
    
    def test_optimizer_creation(self):
        """Test optimizer creation"""
        trainer = KnowledgeDistillationTrainer(self.config)
        trainer._create_models()
        trainer._create_optimizers()
        
        self.assertIn('teacher', trainer.optimizers)
        self.assertIn('student', trainer.optimizers)
        self.assertIsInstance(trainer.optimizers['teacher'], torch.optim.Adam)
        self.assertIsInstance(trainer.optimizers['student'], torch.optim.Adam)


class TestTrainingMetrics(unittest.TestCase):
    """Test training metrics"""
    
    def test_training_metrics_creation(self):
        """Test training metrics creation"""
        metrics = TrainingMetrics(
            epoch=0,
            train_loss=0.5,
            val_loss=0.6,
            train_acc=0.8,
            val_acc=0.75,
            distillation_loss=0.3
        )
        
        self.assertEqual(metrics.epoch, 0)
        self.assertEqual(metrics.train_loss, 0.5)
        self.assertEqual(metrics.val_loss, 0.6)
        self.assertEqual(metrics.train_acc, 0.8)
        self.assertEqual(metrics.val_acc, 0.75)
        self.assertEqual(metrics.distillation_loss, 0.3)


class TestVisualization(unittest.TestCase):
    """Test visualization tools"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'distillation': {'temperature': 3.0, 'alpha': 0.7},
            'show_plots': False
        }
        
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_visualizer_initialization(self):
        """Test visualizer initialization"""
        visualizer = DistillationVisualizer(self.config, self.temp_dir)
        self.assertEqual(visualizer.save_dir, Path(self.temp_dir))
        self.assertTrue(Path(self.temp_dir).exists())
    
    def test_summary_report_generation(self):
        """Test summary report generation"""
        visualizer = DistillationVisualizer(self.config, self.temp_dir)
        
        # Create dummy metrics
        teacher_metrics = [TrainingMetrics(0, 0.5, 0.6, 0.8, 0.75)]
        student_metrics = [TrainingMetrics(0, 0.4, 0.5, 0.85, 0.8, 0.3)]
        
        metrics_history = {
            'teacher': teacher_metrics,
            'student': student_metrics
        }
        
        report = visualizer.generate_summary_report(
            metrics_history, 0.75, 0.8, 1000, 500
        )
        
        self.assertIsInstance(report, str)
        self.assertIn('Knowledge Distillation Training Report', report)
        self.assertIn('Teacher Model Accuracy', report)
        self.assertIn('Student Model Accuracy', report)


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create minimal config for integration tests
        self.config_dict = {
            'model': {
                'teacher': {'hidden_layers': [32], 'dropout': 0.1, 'activation': 'relu'},
                'student': {'hidden_layers': [16], 'dropout': 0.1, 'activation': 'relu'}
            },
            'training': {
                'batch_size': 16, 'teacher_epochs': 1, 'student_epochs': 1,
                'learning_rate': 0.01, 'weight_decay': 1e-4
            },
            'distillation': {'temperature': 2.0, 'alpha': 0.7},
            'data': {'dataset': 'MNIST', 'data_dir': self.temp_dir, 'download': False, 'normalize': True},
            'logging': {'log_level': 'INFO', 'log_file': 'logs/training.log', 'save_models': True, 'model_dir': 'models'},
            'visualization': {'save_plots': False, 'plot_dir': 'plots', 'show_plots': False},
            'device': {'use_cuda': False, 'device_id': 0}
        }
        
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_training(self):
        """Test end-to-end training pipeline"""
        config = Config()
        config.config = self.config_dict
        
        trainer = KnowledgeDistillationTrainer(config)
        
        # Create dummy data loaders
        dummy_data = torch.randn(100, 28, 28)
        dummy_labels = torch.randint(0, 10, (100,))
        dummy_dataset = torch.utils.data.TensorDataset(dummy_data, dummy_labels)
        
        train_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=16, shuffle=True)
        val_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=16, shuffle=False)
        
        # Create models and optimizers
        trainer._create_models()
        trainer._create_optimizers()
        
        # Test teacher training
        teacher_metrics = trainer.train_teacher(train_loader, val_loader)
        self.assertIsInstance(teacher_metrics, list)
        self.assertEqual(len(teacher_metrics), 1)  # 1 epoch
        
        # Test student training
        student_metrics = trainer.train_student_with_distillation(train_loader, val_loader)
        self.assertIsInstance(student_metrics, list)
        self.assertEqual(len(student_metrics), 1)  # 1 epoch
        
        # Test model evaluation
        teacher_loss, teacher_acc = trainer.evaluate_model(trainer.models['teacher'], val_loader)
        student_loss, student_acc = trainer.evaluate_model(trainer.models['student'], val_loader)
        
        self.assertIsInstance(teacher_loss, float)
        self.assertIsInstance(teacher_acc, float)
        self.assertIsInstance(student_loss, float)
        self.assertIsInstance(student_acc, float)
        
        self.assertGreaterEqual(teacher_acc, 0)
        self.assertLessEqual(teacher_acc, 1)
        self.assertGreaterEqual(student_acc, 0)
        self.assertLessEqual(student_acc, 1)


def run_tests():
    """Run all tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_classes = [
        TestConfig,
        TestModelArchitectures,
        TestKnowledgeDistillationTrainer,
        TestTrainingMetrics,
        TestVisualization,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        exit(1)
