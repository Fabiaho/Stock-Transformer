#!/usr/bin/env python3
"""
Hyperparameter optimization using Optuna with PyTorch Lightning.
Optimizes for both ML metrics and financial performance.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import yaml
import optuna
from optuna.integration import PyTorchLightningPruningCallback
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import torch

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.datamodule import StockDataModule
from src.models.lightning_module import StockTransformerLightning
from src.utils.logging import setup_logging
from src.evaluation.backtester import Backtester, BacktestConfig


class FinancialMetricCallback(pl.Callback):
    """Callback to track financial metrics during validation."""
    
    def __init__(self):
        self.best_sharpe = -np.inf
        self.best_return = -np.inf
        
    def on_validation_epoch_end(self, trainer, pl_module):
        """Extract financial metrics from validation."""
        # Get logged metrics
        logged_metrics = trainer.callback_metrics
        
        # Track best financial metrics
        if 'val/sharpe_ratio' in logged_metrics:
            sharpe = logged_metrics['val/sharpe_ratio'].item()
            if sharpe > self.best_sharpe:
                self.best_sharpe = sharpe
                
        if 'val/annual_return' in logged_metrics:
            ret = logged_metrics['val/annual_return'].item()
            if ret > self.best_return:
                self.best_return = ret


def create_objective(config: dict, data_config: dict, optimize_metric: str = 'sharpe_ratio'):
    """
    Create Optuna objective function.
    
    Args:
        config: Base configuration
        data_config: Data configuration
        optimize_metric: Metric to optimize ('sharpe_ratio', 'annual_return', 'combined')
    """
    
    def objective(trial: optuna.Trial) -> float:
        """Objective function for Optuna."""
        
        # Suggest hyperparameters
        hyperparams = {
            # Model architecture
            'd_model': trial.suggest_categorical('d_model', [256, 512, 768]),
            'n_heads': trial.suggest_categorical('n_heads', [4, 8, 12]),
            'n_layers': trial.suggest_int('n_layers', 2, 8),
            'd_ff': trial.suggest_categorical('d_ff', [1024, 2048, 4096]),
            'dropout': trial.suggest_float('dropout', 0.0, 0.5, step=0.05),
            
            # Temporal convolution
            'use_temporal_conv': trial.suggest_categorical('use_temporal_conv', [True, False]),
            
            # Training
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
            'weight_decay': trial.suggest_loguniform('weight_decay', 1e-6, 1e-3),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            
            # Loss configuration
            'loss_type': trial.suggest_categorical('loss_type', ['mse', 'huber', 'financial', 'combined']),
        }
        
        # Loss weights for combined loss
        if hyperparams['loss_type'] == 'combined':
            hyperparams['loss_weights'] = {
                'mse': trial.suggest_float('loss_weight_mse', 0.3, 0.8, step=0.1),
                'direction': trial.suggest_float('loss_weight_direction', 0.1, 0.5, step=0.1),
                'sharpe': trial.suggest_float('loss_weight_sharpe', 0.0, 0.3, step=0.05),
            }
            # Normalize weights to sum to 1
            total = sum(hyperparams['loss_weights'].values())
            hyperparams['loss_weights'] = {k: v/total for k, v in hyperparams['loss_weights'].items()}
        
        # Data-specific parameters
        if data_config['data']['target_type'] == 'classification':
            hyperparams['classification_bins'] = trial.suggest_categorical(
                'classification_bins',
                [
                    [-0.01, 0, 0.01],  # 1% threshold
                    [-0.02, 0, 0.02],  # 2% threshold
                    [-0.015, -0.005, 0.005, 0.015],  # 4 classes
                ]
            )
        
        # Additional architecture choices
        if hyperparams['use_temporal_conv']:
            n_conv_layers = trial.suggest_int('n_conv_layers', 1, 3)
            conv_channels = []
            prev_channels = 64
            for i in range(n_conv_layers):
                channels = trial.suggest_categorical(f'conv_channels_{i}', [64, 128, 256, 512])
                conv_channels.append(channels)
                prev_channels = channels
            hyperparams['conv_channels'] = conv_channels
        
        # Create data module
        datamodule = StockDataModule(
            symbols=data_config['data']['symbols'],
            start_date=data_config['data']['start_date'],
            end_date=data_config['data']['end_date'],
            sequence_length=data_config['data']['sequence_length'],
            prediction_horizon=data_config['data']['prediction_horizon'],
            batch_size=hyperparams['batch_size'],
            target_type=data_config['data']['target_type'],
            classification_bins=hyperparams.get('classification_bins', data_config['data'].get('classification_bins')),
            add_technical_indicators=data_config['data']['add_technical_indicators'],
            add_market_data=data_config['data']['add_market_data'],
            cache_dir=data_config['data']['cache_dir']
        )
        
        # Prepare data
        datamodule.prepare_data()
        datamodule.setup()
        
        # Create model
        model = StockTransformerLightning(
            input_dim=datamodule.get_num_features(),
            d_model=hyperparams['d_model'],
            n_heads=hyperparams['n_heads'],
            n_layers=hyperparams['n_layers'],
            d_ff=hyperparams['d_ff'],
            dropout=hyperparams['dropout'],
            use_temporal_conv=hyperparams['use_temporal_conv'],
            conv_channels=hyperparams.get('conv_channels', [64, 128, 256]),
            learning_rate=hyperparams['learning_rate'],
            weight_decay=hyperparams['weight_decay'],
            loss_type=hyperparams['loss_type'],
            loss_weights=hyperparams.get('loss_weights'),
            output_type=data_config['data']['target_type'],
            n_classes=datamodule.get_num_classes(),
            calculate_financial_metrics=True
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val/loss',
                patience=10,
                mode='min'
            ),
            PyTorchLightningPruningCallback(trial, monitor='val/loss'),
            FinancialMetricCallback()
        ]
        
        # Trainer
        trainer = pl.Trainer(
            max_epochs=50,  # Reduced for faster search
            callbacks=callbacks,
            enable_progress_bar=False,
            enable_model_summary=False,
            accelerator='auto',
            devices=1,
            logger=False,  # Disable logging during search
            deterministic=True
        )
        
        # Train
        try:
            trainer.fit(model, datamodule)
        except Exception as e:
            logging.warning(f"Trial failed with error: {e}")
            return float('inf')
        
        # Get final metrics
        if optimize_metric == 'sharpe_ratio':
            # Maximize Sharpe ratio
            sharpe = callbacks[-1].best_sharpe
            return -sharpe if not np.isnan(sharpe) else float('inf')
        elif optimize_metric == 'annual_return':
            # Maximize annual return
            ret = callbacks[-1].best_return
            return -ret if not np.isnan(ret) else float('inf')
        elif optimize_metric == 'combined':
            # Combined metric: Sharpe + normalized return
            sharpe = callbacks[-1].best_sharpe
            ret = callbacks[-1].best_return
            
            if np.isnan(sharpe) or np.isnan(ret):
                return float('inf')
                
            # Normalize and combine (60% Sharpe, 40% return)
            combined = -0.6 * sharpe - 0.4 * max(0, ret)
            return combined
        else:
            # Default to validation loss
            return trainer.callback_metrics.get('val/loss', float('inf')).item()
    
    return objective


def run_full_backtest(best_params: dict, config: dict) -> dict:
    """Run full backtest with best parameters."""
    logging.info("Running full backtest with best parameters...")
    
    # Create data module with best batch size
    datamodule = StockDataModule(
        symbols=config['data']['symbols'],
        start_date=config['data']['start_date'], 
        end_date=config['data']['end_date'],
        sequence_length=config['data']['sequence_length'],
        prediction_horizon=config['data']['prediction_horizon'],
        batch_size=best_params.get('batch_size', 32),
        target_type=config['data']['target_type'],
        classification_bins=best_params.get('classification_bins', config['data'].get('classification_bins')),
        add_technical_indicators=config['data']['add_technical_indicators'],
        add_market_data=config['data']['add_market_data'],
        cache_dir=config['data']['cache_dir']
    )
    
    datamodule.prepare_data()
    datamodule.setup()
    
    # Create model with best parameters
    model = StockTransformerLightning(
        input_dim=datamodule.get_num_features(),
        **{k: v for k, v in best_params.items() if k not in ['batch_size', 'classification_bins']},
        output_type=config['data']['target_type'],
        n_classes=datamodule.get_num_classes(),
        calculate_financial_metrics=True
    )
    
    # Train with full epochs
    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[
            EarlyStopping(monitor='val/loss', patience=20, mode='min'),
            ModelCheckpoint(
                dirpath='models/optuna_best',
                filename='best-{epoch:02d}-{val_loss:.3f}',
                monitor='val/loss',
                mode='min',
                save_top_k=1
            )
        ],
        enable_progress_bar=True,
        accelerator='auto',
        devices=1
    )
    
    trainer.fit(model, datamodule)
    
    # Run backtest on test set
    test_results = trainer.test(model, datamodule)
    
    return test_results[0] if test_results else {}


def main():
    """Main function for hyperparameter optimization."""
    parser = argparse.ArgumentParser(description='Hyperparameter optimization for stock transformer')
    parser.add_argument('--config', type=str, default='config',
                       help='Path to config directory')
    parser.add_argument('--n-trials', type=int, default=100,
                       help='Number of optimization trials')
    parser.add_argument('--n-jobs', type=int, default=1,
                       help='Number of parallel jobs')
    parser.add_argument('--study-name', type=str, default=None,
                       help='Optuna study name')
    parser.add_argument('--storage', type=str, default=None,
                       help='Optuna storage URL (e.g., sqlite:///optuna.db)')
    parser.add_argument('--optimize-metric', type=str, default='sharpe_ratio',
                       choices=['sharpe_ratio', 'annual_return', 'combined', 'loss'],
                       help='Metric to optimize')
    parser.add_argument('--timeout', type=int, default=None,
                       help='Timeout in seconds per trial')
    parser.add_argument('--pruning', action='store_true',
                       help='Enable trial pruning')
    parser.add_argument('--run-backtest', action='store_true',
                       help='Run full backtest with best parameters')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level='INFO')
    
    # Load configuration
    config_path = Path(args.config)
    with open(config_path / 'data_config.yaml', 'r') as f:
        data_config = yaml.safe_load(f)
    with open(config_path / 'training_config.yaml', 'r') as f:
        training_config = yaml.safe_load(f)
    
    # Create study name if not provided
    if args.study_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.study_name = f'stock_transformer_{args.optimize_metric}_{timestamp}'
    
    logging.info(f"Starting hyperparameter optimization: {args.study_name}")
    logging.info(f"Optimizing for: {args.optimize_metric}")
    logging.info(f"Number of trials: {args.n_trials}")
    
    # Create Optuna study
    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()
    
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        sampler=sampler,
        pruner=pruner,
        direction='minimize',
        load_if_exists=True
    )
    
    # Create objective function
    objective = create_objective(
        {'data': data_config, 'training': training_config},
        data_config,
        args.optimize_metric
    )
    
    # Run optimization
    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
            timeout=args.timeout,
            show_progress_bar=True
        )
    except KeyboardInterrupt:
        logging.info("Optimization interrupted by user")
    
    # Print results
    logging.info("\nOptimization Results:")
    logging.info(f"Number of finished trials: {len(study.trials)}")
    
    if len(study.trials) > 0:
        logging.info(f"\nBest trial:")
        trial = study.best_trial
        logging.info(f"  Value: {trial.value}")
        logging.info(f"  Params:")
        for key, value in trial.params.items():
            logging.info(f"    {key}: {value}")
        
        # Save best parameters
        output_dir = Path('results/hyperparameter_search')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        best_params_file = output_dir / f'{args.study_name}_best_params.yaml'
        with open(best_params_file, 'w') as f:
            yaml.dump(trial.params, f)
        logging.info(f"\nBest parameters saved to: {best_params_file}")
        
        # Generate optimization report
        report = {
            'study_name': args.study_name,
            'optimize_metric': args.optimize_metric,
            'n_trials': len(study.trials),
            'best_value': trial.value,
            'best_params': trial.params,
            'optimization_history': [
                {
                    'trial': i,
                    'value': t.value,
                    'state': str(t.state),
                    'duration': (t.datetime_complete - t.datetime_start).total_seconds() if t.datetime_complete else None
                }
                for i, t in enumerate(study.trials)
            ]
        }
        
        report_file = output_dir / f'{args.study_name}_report.yaml'
        with open(report_file, 'w') as f:
            yaml.dump(report, f)
        logging.info(f"Optimization report saved to: {report_file}")
        
        # Visualize optimization history
        try:
            import matplotlib.pyplot as plt
            
            # Plot optimization history
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Objective value over trials
            values = [t.value for t in study.trials if t.value != float('inf')]
            if values:
                ax1.plot(values, 'b-', alpha=0.7)
                ax1.axhline(y=min(values), color='r', linestyle='--', label=f'Best: {min(values):.4f}')
                ax1.set_xlabel('Trial')
                ax1.set_ylabel('Objective Value')
                ax1.set_title('Optimization History')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # Parameter importance
            if len(study.trials) > 10:
                importance = optuna.importance.get_param_importances(study)
                if importance:
                    params = list(importance.keys())[:10]  # Top 10
                    values = [importance[p] for p in params]
                    ax2.barh(params, values)
                    ax2.set_xlabel('Importance')
                    ax2.set_title('Parameter Importance (Top 10)')
                    ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'{args.study_name}_optimization_plot.png', dpi=150)
            logging.info(f"Optimization plot saved to: {output_dir / f'{args.study_name}_optimization_plot.png'}")
        except Exception as e:
            logging.warning(f"Failed to create visualization: {e}")
        
        # Run full backtest with best parameters
        if args.run_backtest:
            backtest_results = run_full_backtest(trial.params, data_config)
            logging.info("\nBacktest Results:")
            for metric, value in backtest_results.items():
                logging.info(f"  {metric}: {value}")
    
    else:
        logging.warning("No trials completed successfully")


if __name__ == '__main__':
    main()