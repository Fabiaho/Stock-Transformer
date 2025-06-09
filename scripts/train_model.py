#!/usr/bin/env python3
"""
Main training script for stock transformer predictor.
Integrates all components for end-to-end model training.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, EarlyStopping, LearningRateMonitor, RichProgressBar
)
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.datamodule import StockDataModule
from src.models.lightning_module import StockTransformerLightning
from src.utils.logging import setup_logging, log_system_info, log_config, log_model_summary, log_data_info


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML files."""
    configs = {}
    config_dir = Path(config_path)
    
    # Load all config files
    for config_file in ['data_config.yaml', 'model_config.yaml', 'training_config.yaml']:
        file_path = config_dir / config_file
        if file_path.exists():
            with open(file_path, 'r') as f:
                config_name = config_file.replace('_config.yaml', '')
                configs[config_name] = yaml.safe_load(f)
        else:
            logging.warning(f"Config file not found: {file_path}")
    
    return configs


def setup_callbacks(config: Dict[str, Any]) -> list:
    """Setup training callbacks."""
    callbacks = []
    
    training_config = config.get('training', {})
    
    # Model checkpoint
    checkpoint_dir = training_config.get('checkpoint_dir', 'models/checkpoints')
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='{epoch:02d}-{val_loss:.3f}',
        monitor='val/loss',
        mode='min',
        save_top_k=training_config.get('save_top_k', 3),
        save_last=training_config.get('save_last', True),
        auto_insert_metric_name=False
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    if training_config.get('early_stopping', True):
        early_stop_callback = EarlyStopping(
            monitor='val/loss',
            patience=training_config.get('early_stopping_patience', 20),
            mode='min',
            verbose=True
        )
        callbacks.append(early_stop_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Progress bar
    if training_config.get('use_rich_progress', True):
        progress_bar = RichProgressBar()
        callbacks.append(progress_bar)
    
    return callbacks


def setup_logger(config: Dict[str, Any], experiment_name: str) -> Optional[pl.loggers.Logger]:
    """Setup experiment logger."""
    logging_config = config.get('logging', {})
    
    if logging_config.get('use_wandb', False):
        logger = WandbLogger(
            project=logging_config.get('wandb_project', 'stock-transformer'),
            name=logging_config.get('wandb_name') or experiment_name,
            save_dir=logging_config.get('log_dir', 'results/logs')
        )
    elif logging_config.get('use_tensorboard', True):
        logger = TensorBoardLogger(
            save_dir=logging_config.get('log_dir', 'results/logs'),
            name=experiment_name,
            version=None
        )
    else:
        logger = None
    
    return logger


def create_datamodule(config: Dict[str, Any]) -> StockDataModule:
    """Create and configure data module."""
    data_config = config.get('data', {})
    training_config = config.get('training', {})
    
    datamodule = StockDataModule(
        symbols=data_config.get('symbols', ['AAPL', 'MSFT', 'GOOGL']),
        start_date=data_config.get('start_date', '2018-01-01'),
        end_date=data_config.get('end_date', '2023-12-31'),
        sequence_length=data_config.get('sequence_length', 60),
        prediction_horizon=data_config.get('prediction_horizon', 1),
        batch_size=training_config.get('batch_size', 32),
        num_workers=training_config.get('num_workers', 4),
        train_val_test_split=data_config.get('train_val_test_split', [0.7, 0.15, 0.15]),
        target_column=data_config.get('target_column', 'returns'),
        target_type=data_config.get('target_type', 'regression'),
        feature_columns=data_config.get('feature_columns'),
        add_technical_indicators=data_config.get('add_technical_indicators', True),
        add_market_data=data_config.get('add_market_data', False),
        market_indices=data_config.get('market_indices', ['^GSPC', '^VIX']),
        scale_features=data_config.get('scale_features', True),
        cache_dir=data_config.get('cache_dir'),
        gap_days=data_config.get('gap_days', 5),
        classification_bins=data_config.get('classification_bins')
    )
    
    return datamodule


def create_model(config: Dict[str, Any], input_dim: int, n_classes: int = 3) -> StockTransformerLightning:
    """Create and configure the model."""
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    data_config = config.get('data', {})
    
    model = StockTransformerLightning(
        # Model architecture
        input_dim=input_dim,
        d_model=model_config.get('d_model', 512),
        n_heads=model_config.get('n_heads', 8),
        n_layers=model_config.get('n_layers', 6),
        d_ff=model_config.get('d_ff', 2048),
        dropout=model_config.get('dropout', 0.1),
        max_seq_len=model_config.get('max_seq_len', 512),
        use_temporal_conv=model_config.get('use_temporal_conv', True),
        conv_channels=model_config.get('conv_channels', [64, 128, 256]),
        use_regime_bias=model_config.get('use_regime_bias', True),
        
        # Task configuration
        output_type=data_config.get('target_type', 'regression'),
        n_classes=n_classes,
        
        # Optimization
        learning_rate=training_config.get('learning_rate', 1e-4),
        weight_decay=training_config.get('weight_decay', 1e-5),
        warmup_steps=training_config.get('warmup_steps', 1000),
        scheduler_type=training_config.get('scheduler_type', 'cosine'),
        
        # Loss configuration
        loss_type=training_config.get('loss_type', 'mse'),
        loss_weights=training_config.get('loss_weights'),
        
        # Financial metrics
        calculate_financial_metrics=training_config.get('calculate_financial_metrics', True),
        transaction_cost=training_config.get('transaction_cost', 0.001),
        
        # Multi-horizon prediction
        use_multi_horizon=model_config.get('use_multi_horizon', False),
        prediction_horizons=model_config.get('prediction_horizons', [1, 5, 20]),
        
        # Additional features
        use_market_regime=model_config.get('use_market_regime', False),
        log_attention_weights=config.get('logging', {}).get('log_attention_weights', True),
        gradient_clip_val=training_config.get('gradient_clip_val', 1.0)
    )
    
    return model


def setup_trainer(config: Dict[str, Any], callbacks: list, logger: Optional[pl.loggers.Logger]) -> pl.Trainer:
    """Setup PyTorch Lightning trainer."""
    training_config = config.get('training', {})
    
    trainer = pl.Trainer(
        max_epochs=training_config.get('max_epochs', 100),
        accelerator='auto',
        devices='auto',
        precision=training_config.get('precision', 32),
        deterministic=training_config.get('deterministic', False),
        benchmark=training_config.get('benchmark', True),
        gradient_clip_val=training_config.get('gradient_clip_val', 1.0),
        accumulate_grad_batches=training_config.get('accumulate_grad_batches', 1),
        val_check_interval=training_config.get('val_check_interval', 1.0),
        log_every_n_steps=training_config.get('log_every_n_steps', 50),
        callbacks=callbacks,
        logger=logger,
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    return trainer


def save_experiment_info(config: Dict[str, Any], experiment_name: str, model: StockTransformerLightning):
    """Save experiment configuration and model info."""
    experiment_dir = Path('results') / 'experiments' / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(experiment_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Save model summary
    with open(experiment_dir / 'model_summary.txt', 'w') as f:
        f.write(str(model))
    
    # Save hyperparameters
    with open(experiment_dir / 'hyperparameters.yaml', 'w') as f:
        yaml.dump(dict(model.hparams), f, default_flow_style=False)
    
    logging.info(f"Experiment info saved to {experiment_dir}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Stock Transformer Model')
    parser.add_argument('--config', type=str, default='config', 
                       help='Path to config directory')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name (auto-generated if not provided)')
    parser.add_argument('--resume-from-checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--fast-dev-run', action='store_true',
                       help='Run a fast development run')
    parser.add_argument('--overfit-batches', type=int, default=0,
                       help='Number of batches to overfit (for debugging)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Setup everything but don\'t train')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only run validation')
    parser.add_argument('--test-only', action='store_true',
                       help='Only run testing')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Log system information
    log_system_info()
    
    # Generate experiment name if not provided
    if args.experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.experiment_name = f'transformer_experiment_{timestamp}'
    
    logging.info(f"Starting experiment: {args.experiment_name}")
    
    try:
        # Load configuration
        logging.info("Loading configuration...")
        config = load_config(args.config)
        log_config(config)
        
        # Create data module
        logging.info("Setting up data module...")
        datamodule = create_datamodule(config)
        
        # Prepare data to get dimensions
        datamodule.prepare_data()
        datamodule.setup()
        
        # Log data information
        log_data_info(datamodule)
        
        # Get data dimensions
        input_dim = datamodule.get_num_features()
        n_classes = datamodule.get_num_classes()
        
        logging.info(f"Input dimension: {input_dim}")
        logging.info(f"Number of classes: {n_classes}")
        logging.info(f"Target type: {config.get('data', {}).get('target_type', 'regression')}")
        
        # Create model
        logging.info("Creating model...")
        model = create_model(config, input_dim, n_classes)
        
        # Log model information
        log_model_summary(model)
        
        # Setup callbacks and logger
        callbacks = setup_callbacks(config)
        logger = setup_logger(config, args.experiment_name)
        
        # Setup trainer
        trainer_kwargs = {}
        if args.fast_dev_run:
            trainer_kwargs['fast_dev_run'] = True
        if args.overfit_batches > 0:
            trainer_kwargs['overfit_batches'] = args.overfit_batches
        
        trainer = setup_trainer(config, callbacks, logger)
        
        # Apply additional trainer kwargs
        for key, value in trainer_kwargs.items():
            setattr(trainer, key, value)
        
        # Save experiment info
        save_experiment_info(config, args.experiment_name, model)
        
        if args.dry_run:
            logging.info("Dry run completed. Exiting without training.")
            return
        
        # Training, validation, or testing
        if args.validate_only:
            logging.info("Running validation only...")
            trainer.validate(model, datamodule=datamodule)
        elif args.test_only:
            logging.info("Running testing only...")
            trainer.test(model, datamodule=datamodule)
        else:
            # Full training
            logging.info("Starting training...")
            trainer.fit(
                model, 
                datamodule=datamodule,
                ckpt_path=args.resume_from_checkpoint
            )
            
            # Run test if configured
            if config.get('training', {}).get('run_test', True):
                logging.info("Running final test...")
                trainer.test(model, datamodule=datamodule)
        
        # Save final model if configured
        if config.get('training', {}).get('save_final_model', True) and not args.validate_only and not args.test_only:
            final_model_path = Path('models') / 'final' / f'{args.experiment_name}.ckpt'
            final_model_path.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(final_model_path)
            logging.info(f"Final model saved to {final_model_path}")
        
        logging.info("Training completed successfully!")
        
    except Exception as e:
        logging.error(f"Training failed with error: {e}")
        raise
    
    finally:
        # Cleanup
        if logger and hasattr(logger, 'finalize'):
            logger.finalize('success')


if __name__ == '__main__':
    main()