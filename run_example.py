#!/usr/bin/env python3
"""
Example Usage of Knowledge Distillation Framework
================================================

This script demonstrates how to use the modernized knowledge distillation framework
with different configurations and options.
"""

import argparse
import logging
from pathlib import Path

from knowledge_distillation import Config, KnowledgeDistillationTrainer
from visualization import create_visualization_suite

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description="Knowledge Distillation Framework")
    parser.add_argument("--config", type=str, default="config.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick training with minimal epochs")
    parser.add_argument("--web", action="store_true", 
                       help="Launch web interface")
    parser.add_argument("--test", action="store_true", 
                       help="Run tests")
    
    args = parser.parse_args()
    
    if args.web:
        logger.info("Launching web interface...")
        import subprocess
        subprocess.run(["streamlit", "run", "app.py"])
        return
    
    if args.test:
        logger.info("Running tests...")
        import subprocess
        result = subprocess.run(["python", "-m", "pytest", "test_knowledge_distillation.py", "-v"])
        return result.returncode
    
    # Load configuration
    if not Path(args.config).exists():
        logger.warning(f"Config file {args.config} not found. Using defaults.")
        config = Config()
    else:
        config = Config(args.config)
    
    # Modify config for quick training if requested
    if args.quick:
        logger.info("Running quick training with minimal epochs...")
        config.config['training']['teacher_epochs'] = 1
        config.config['training']['student_epochs'] = 1
        config.config['training']['batch_size'] = 128
    
    # Create trainer
    logger.info("Initializing trainer...")
    trainer = KnowledgeDistillationTrainer(config)
    
    # Run training
    logger.info("Starting knowledge distillation training...")
    metrics_history = trainer.run_full_training()
    
    # Generate visualizations
    logger.info("Generating visualizations...")
    
    # Get final metrics
    train_loader, val_loader = trainer._get_data_loaders()
    teacher_val_loss, teacher_val_acc = trainer.evaluate_model(trainer.models['teacher'], val_loader)
    student_val_loss, student_val_acc = trainer.evaluate_model(trainer.models['student'], val_loader)
    
    teacher_params = sum(p.numel() for p in trainer.models['teacher'].parameters())
    student_params = sum(p.numel() for p in trainer.models['student'].parameters())
    
    # Create comprehensive visualizations
    create_visualization_suite(
        metrics_history,
        config.config,
        teacher_acc=teacher_val_acc,
        student_acc=student_val_acc,
        teacher_params=teacher_params,
        student_params=student_params
    )
    
    logger.info("Training completed successfully!")
    logger.info(f"Teacher Accuracy: {teacher_val_acc:.4f}")
    logger.info(f"Student Accuracy: {student_val_acc:.4f}")
    logger.info(f"Knowledge Retention: {student_val_acc/teacher_val_acc:.4f}")
    logger.info(f"Compression Ratio: {teacher_params/student_params:.1f}x")


if __name__ == "__main__":
    main()
