"""
PyTorch Lightning module for training the stock transformer model.
Updated for PyTorch Lightning 2.0+ compatibility.
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import wandb

from src.models.transformer import StockTransformer
from src.training.losses import FinancialLoss, DirectionalAccuracy
from src.training.metrics import FinancialMetrics


class StockTransformerLightning(pl.LightningModule):
    """Lightning module for stock price prediction transformer."""
    
    def __init__(
        self,
        # Model architecture
        input_dim: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        use_temporal_conv: bool = True,
        conv_channels: List[int] = [64, 128, 256],
        use_regime_bias: bool = True,
        
        # Task configuration
        output_type: str = 'regression',  # 'regression' or 'classification'
        n_classes: int = 3,
        
        # Optimization
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_steps: int = 1000,
        max_steps: int = 100000,
        scheduler_type: str = 'cosine',  # 'cosine', 'plateau', 'linear'
        
        # Loss configuration
        loss_type: str = 'mse',  # 'mse', 'huber', 'financial', 'combined'
        loss_weights: Optional[Dict[str, float]] = None,
        
        # Financial metrics
        calculate_financial_metrics: bool = True,
        transaction_cost: float = 0.001,  # 10 basis points
        
        # Multi-horizon prediction
        use_multi_horizon: bool = False,
        prediction_horizons: List[int] = [1, 5, 20],
        
        # Additional features
        use_market_regime: bool = False,
        log_attention_weights: bool = False,
        gradient_clip_val: float = 1.0
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        # Initialize model
        self.model = StockTransformer(
            input_dim=input_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len,
            output_dim=1 if output_type == 'regression' else n_classes,
            use_temporal_conv=use_temporal_conv,
            conv_channels=conv_channels,
            use_regime_bias=use_regime_bias,
            output_type=output_type,
            n_classes=n_classes
        )
        
        # Initialize losses
        self._setup_losses()
        
        # Initialize metrics
        if calculate_financial_metrics:
            self.financial_metrics = FinancialMetrics(
                transaction_cost=transaction_cost,
                calculate_sharpe=True,
                calculate_sortino=True,
                calculate_calmar=True
            )
            
        # For storing validation outputs
        self.validation_step_outputs = []
        self.validation_attention_weights = []
        
    def _setup_losses(self):
        """Setup loss functions based on configuration."""
        loss_type = self.hparams.loss_type
        output_type = self.hparams.output_type
        
        if output_type == 'regression':
            if loss_type == 'mse':
                self.criterion = nn.MSELoss()
            elif loss_type == 'huber':
                self.criterion = nn.HuberLoss(delta=1.0)
            elif loss_type == 'financial':
                self.criterion = FinancialLoss(
                    base_loss='mse',
                    sharpe_weight=0.1,
                    downside_weight=0.2
                )
            elif loss_type == 'combined':
                # Combine multiple losses
                self.criterion = self._create_combined_loss()
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
        else:  # classification
            self.criterion = nn.CrossEntropyLoss()
            
        # Additional metrics
        self.directional_accuracy = DirectionalAccuracy()
        
    def _create_combined_loss(self):
        """Create a combined loss function."""
        weights = self.hparams.loss_weights or {
            'mse': 0.7,
            'direction': 0.2,
            'sharpe': 0.1
        }
        
        def combined_loss(pred, target, returns=None):
            total_loss = 0
            
            if 'mse' in weights:
                total_loss += weights['mse'] * F.mse_loss(pred, target)
                
            if 'direction' in weights and returns is not None:
                dir_loss = self.directional_accuracy.loss(pred, returns)
                total_loss += weights['direction'] * dir_loss
                
            if 'sharpe' in weights and returns is not None:
                # Penalize negative Sharpe ratio predictions
                pred_returns = pred * returns.sign()
                sharpe_penalty = -torch.mean(pred_returns) / (torch.std(pred_returns) + 1e-8)
                total_loss += weights['sharpe'] * sharpe_penalty
                
            return total_loss
            
        return combined_loss
    
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass through the model."""
        return self.model(x, **kwargs)
    
    def _compute_loss(self, batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss based on task type and configuration."""
        target = batch['target']
        pred = outputs['output']
        
        if self.hparams.output_type == 'regression':
            target = target.float()
            if self.hparams.loss_type == 'combined':
                # Extract returns if available for combined loss
                returns = batch.get('returns', None)
                loss = self.criterion(pred, target, returns)
            else:
                loss = self.criterion(pred.squeeze(), target.squeeze())
        else:  # classification
            target = target.long().squeeze()
            loss = self.criterion(pred, target)
            
        return loss
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        # Forward pass
        outputs = self(batch['sequence'])
        
        # Compute loss
        loss = self._compute_loss(batch, outputs)
        
        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Additional metrics
        if self.hparams.output_type == 'regression':
            with torch.no_grad():
                mae = F.l1_loss(outputs['output'].squeeze(), batch['target'].squeeze())
                self.log('train/mae', mae, on_step=False, on_epoch=True)
                
                # Directional accuracy
                if 'returns' in batch:
                    dir_acc = self.directional_accuracy(outputs['output'], batch['returns'])
                    self.log('train/directional_accuracy', dir_acc, on_step=False, on_epoch=True)
        else:  # classification
            with torch.no_grad():
                acc = (outputs['output'].argmax(dim=-1) == batch['target'].squeeze()).float().mean()
                self.log('train/accuracy', acc, on_step=False, on_epoch=True, prog_bar=True)
                
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Validation step."""
        # Forward pass
        outputs = self(batch['sequence'])
        
        # Compute loss
        loss = self._compute_loss(batch, outputs)
        
        # Store predictions for financial metrics
        predictions = outputs['output'].detach()
        targets = batch['target'].detach()
        
        # Log basic metrics
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Store attention weights for visualization (only for first batch and if enabled)
        if (batch_idx == 0 and 
            self.hparams.log_attention_weights and 
            'attention_weights' in outputs and 
            outputs['attention_weights']):
            self.validation_attention_weights = outputs['attention_weights']
        
        # Store outputs for epoch-end processing
        step_output = {
            'loss': loss,
            'predictions': predictions,
            'targets': targets,
            'metadata': batch.get('metadata', None)
        }
        self.validation_step_outputs.append(step_output)
        
        return step_output
    
    def on_validation_epoch_end(self):
        """Aggregate validation metrics at epoch end."""
        if not self.validation_step_outputs:
            return
            
        # Gather all predictions and targets
        all_preds = torch.cat([x['predictions'] for x in self.validation_step_outputs])
        all_targets = torch.cat([x['targets'] for x in self.validation_step_outputs])
        
        if self.hparams.output_type == 'regression':
            # Regression metrics
            mae = F.l1_loss(all_preds.squeeze(), all_targets.squeeze())
            rmse = torch.sqrt(F.mse_loss(all_preds.squeeze(), all_targets.squeeze()))
            
            self.log('val/mae', mae)
            self.log('val/rmse', rmse)
            
            # R-squared
            ss_res = torch.sum((all_targets.squeeze() - all_preds.squeeze()) ** 2)
            ss_tot = torch.sum((all_targets.squeeze() - all_targets.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else torch.tensor(0.0)
            self.log('val/r2', r2)
            
            # Financial metrics
            if self.hparams.calculate_financial_metrics:
                returns_pred = all_preds.squeeze().cpu().numpy()
                returns_true = all_targets.squeeze().cpu().numpy()
                
                try:
                    metrics = self.financial_metrics.calculate_all_metrics(
                        returns_true, returns_pred
                    )
                    
                    for name, value in metrics.items():
                        if not np.isnan(value) and not np.isinf(value):
                            self.log(f'val/{name}', value)
                except Exception as e:
                    print(f"Warning: Failed to calculate financial metrics: {e}")
                        
        else:  # classification
            # Classification metrics
            preds_class = all_preds.argmax(dim=-1)
            targets_class = all_targets.squeeze()
            
            accuracy = (preds_class == targets_class).float().mean()
            self.log('val/accuracy', accuracy)
            
            # Per-class accuracy
            for i in range(self.hparams.n_classes):
                mask = targets_class == i
                if mask.sum() > 0:
                    class_acc = (preds_class[mask] == i).float().mean()
                    self.log(f'val/accuracy_class_{i}', class_acc)
                    
        # Log attention visualization if using wandb
        if self.logger and self.hparams.log_attention_weights and self.validation_attention_weights:
            self._log_attention_visualization()
            
        # Clear stored outputs
        self.validation_step_outputs.clear()
        
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step - similar to validation but might include additional metrics."""
        return self.validation_step(batch, batch_idx)
    
    def on_test_epoch_end(self):
        """Aggregate test metrics."""
        self.on_validation_epoch_end()  # Reuse validation logic
        
        # Additional test-specific metrics can be added here
        if self.hparams.calculate_financial_metrics:
            # Calculate and log backtesting metrics
            pass
            
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler
        if self.hparams.scheduler_type == 'cosine':
            scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.hparams.warmup_steps,
                T_mult=2,
                eta_min=1e-7
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            }
        elif self.hparams.scheduler_type == 'plateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                min_lr=1e-7
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val/loss',
                    'interval': 'epoch'
                }
            }
        else:  # linear or none
            return optimizer
            
    def _log_attention_visualization(self):
        """Log attention weight visualizations with proper logger detection."""
        if not self.validation_attention_weights:
            return
            
        try:
            # Take first sample from first layer
            attn_weights = self.validation_attention_weights[0][0].detach().cpu().numpy()
            
            # Average over heads
            attn_avg = attn_weights.mean(axis=0)
            
            # Create heatmap with proper figure management
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            import numpy as np
            
            # Close any existing figures to prevent memory issues
            plt.close('all')
            
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(attn_avg, cmap='Blues', aspect='auto')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
            ax.set_title(f'Average Attention Weights (Layer 1) - Epoch {self.current_epoch}')
            plt.colorbar(im)
            
            # Log based on logger type
            if self.logger:
                if hasattr(self.logger, 'experiment'):
                    # Check if it's wandb or tensorboard
                    if hasattr(self.logger.experiment, 'log'):
                        # Weights & Biases
                        import wandb
                        self.logger.experiment.log({
                            'attention_heatmap': wandb.Image(fig),
                            'epoch': self.current_epoch
                        })
                    elif hasattr(self.logger.experiment, 'add_figure'):
                        # TensorBoard
                        self.logger.experiment.add_figure(
                            'attention_heatmap', 
                            fig, 
                            global_step=self.current_epoch
                        )
                    else:
                        # Fallback: save to file
                        import os
                        os.makedirs('results/attention_plots', exist_ok=True)
                        fig.savefig(f'results/attention_plots/attention_epoch_{self.current_epoch}.png', 
                                dpi=150, bbox_inches='tight')
                        print(f"Attention plot saved to results/attention_plots/attention_epoch_{self.current_epoch}.png")
            
            # Always close the figure to prevent memory leaks
            plt.close(fig)
        
        except Exception as e:
            print(f"Warning: Failed to log attention visualization: {e}")
            # Don't let visualization errors break training
            # Close any figures that might be open
            import matplotlib.pyplot as plt
            plt.close('all')
        
    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        # Log current learning rate
        try:
            current_lr = self.optimizers().param_groups[0]['lr']
            self.log('train/learning_rate', current_lr)
        except:
            pass
        
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Prediction step for inference."""
        outputs = self(batch['sequence'])
        
        # Add metadata to outputs
        outputs['metadata'] = batch.get('metadata', None)
        
        # For multi-horizon prediction
        if self.hparams.use_multi_horizon:
            multi_horizon_preds = self.model.predict_multi_horizon(
                batch['sequence'],
                horizons=self.hparams.prediction_horizons
            )
            outputs.update(multi_horizon_preds)
            
        return outputs