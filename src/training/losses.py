"""
Custom loss functions for financial time series prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Callable
import numpy as np


class FinancialLoss(nn.Module):
    """
    Financial-aware loss function that combines prediction accuracy with financial metrics.
    """
    
    def __init__(
        self,
        base_loss: str = 'mse',
        sharpe_weight: float = 0.1,
        downside_weight: float = 0.2,
        transaction_cost: float = 0.001,
        risk_free_rate: float = 0.02
    ):
        super().__init__()
        
        self.base_loss = base_loss
        self.sharpe_weight = sharpe_weight
        self.downside_weight = downside_weight
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate / 252  # Daily risk-free rate
        
        # Base loss functions
        if base_loss == 'mse':
            self.base_criterion = nn.MSELoss()
        elif base_loss == 'mae':
            self.base_criterion = nn.L1Loss()
        elif base_loss == 'huber':
            self.base_criterion = nn.HuberLoss()
        else:
            raise ValueError(f"Unknown base loss: {base_loss}")
            
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        returns: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate financial loss.
        
        Args:
            predictions: Model predictions
            targets: Target values
            returns: Optional actual returns for Sharpe/downside calculations
            
        Returns:
            Combined loss value
        """
        # Base prediction loss
        base_loss = self.base_criterion(predictions.squeeze(), targets.squeeze())
        
        total_loss = base_loss
        
        if returns is not None and self.sharpe_weight > 0:
            # Sharpe ratio component
            # Use predictions to determine position sizing
            positions = torch.tanh(predictions.squeeze())  # Normalize to [-1, 1]
            strategy_returns = positions * returns.squeeze()
            
            # Account for transaction costs
            position_changes = torch.diff(positions, dim=0)
            transaction_costs = torch.abs(position_changes) * self.transaction_cost
            
            # Net returns after costs
            net_returns = strategy_returns
            if len(transaction_costs) > 0:
                net_returns[1:] = net_returns[1:] - transaction_costs
                
            # Sharpe ratio (negative because we minimize loss)
            excess_returns = net_returns - self.risk_free_rate
            sharpe = -torch.mean(excess_returns) / (torch.std(excess_returns) + 1e-8)
            
            total_loss += self.sharpe_weight * sharpe
            
        if returns is not None and self.downside_weight > 0:
            # Downside deviation component
            positions = torch.tanh(predictions.squeeze())
            strategy_returns = positions * returns.squeeze()
            
            # Only consider negative returns
            downside_returns = torch.where(
                strategy_returns < 0,
                strategy_returns,
                torch.zeros_like(strategy_returns)
            )
            
            downside_dev = torch.sqrt(torch.mean(downside_returns ** 2) + 1e-8)
            total_loss += self.downside_weight * downside_dev
            
        return total_loss


class DirectionalAccuracy(nn.Module):
    """
    Loss based on directional accuracy - penalizes wrong direction predictions.
    """
    
    def __init__(self, threshold: float = 0.0):
        super().__init__()
        self.threshold = threshold
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate directional accuracy.
        
        Args:
            predictions: Model predictions
            targets: Target returns
            
        Returns:
            Directional accuracy (higher is better)
        """
        pred_direction = (predictions.squeeze() > self.threshold).float()
        true_direction = (targets.squeeze() > self.threshold).float()
        
        accuracy = (pred_direction == true_direction).float().mean()
        return accuracy
        
    def loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate directional loss (for minimization).
        
        Args:
            predictions: Model predictions
            targets: Target returns
            
        Returns:
            Directional loss (lower is better)
        """
        accuracy = self.forward(predictions, targets)
        return 1.0 - accuracy  # Convert to loss


class WeightedMSELoss(nn.Module):
    """
    MSE loss with sample weighting based on market conditions or importance.
    """
    
    def __init__(self, volatility_weight: bool = True, recency_weight: bool = True):
        super().__init__()
        self.volatility_weight = volatility_weight
        self.recency_weight = recency_weight
        
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        volatility: Optional[torch.Tensor] = None,
        timestamps: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate weighted MSE loss.
        
        Args:
            predictions: Model predictions
            targets: Target values
            volatility: Optional volatility values for weighting
            timestamps: Optional timestamps for recency weighting
            
        Returns:
            Weighted MSE loss
        """
        mse = (predictions.squeeze() - targets.squeeze()) ** 2
        weights = torch.ones_like(mse)
        
        if self.volatility_weight and volatility is not None:
            # Weight by inverse volatility (more weight on stable periods)
            vol_weights = 1.0 / (volatility.squeeze() + 1e-8)
            vol_weights = vol_weights / vol_weights.mean()  # Normalize
            weights *= vol_weights
            
        if self.recency_weight and timestamps is not None:
            # Exponential decay for older samples
            max_time = timestamps.max()
            time_diff = (max_time - timestamps).float()
            recency_weights = torch.exp(-0.01 * time_diff)  # Decay factor
            weights *= recency_weights.squeeze()
            
        weighted_mse = (mse * weights).mean()
        return weighted_mse


class RankingLoss(nn.Module):
    """
    Loss for learning to rank stocks by expected returns.
    """
    
    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin
        
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        pairs: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Calculate ranking loss.
        
        Args:
            predictions: Model predictions (batch_size,)
            targets: Target returns (batch_size,)
            pairs: Optional pre-computed pairs, otherwise all pairs are used
            
        Returns:
            Ranking loss
        """
        batch_size = predictions.size(0)
        
        if pairs is None:
            # Create all pairs
            pairs = []
            for i in range(batch_size):
                for j in range(i + 1, batch_size):
                    if targets[i] > targets[j]:
                        pairs.append((i, j))
                    elif targets[j] > targets[i]:
                        pairs.append((j, i))
                        
        if len(pairs) == 0:
            return torch.tensor(0.0, device=predictions.device)
            
        loss = 0.0
        for i, j in pairs:
            # We want pred[i] > pred[j] when target[i] > target[j]
            ranking_loss = F.relu(
                self.margin - (predictions[i] - predictions[j])
            )
            loss += ranking_loss
            
        return loss / len(pairs)


class QuantileLoss(nn.Module):
    """
    Quantile loss for predicting different quantiles of the return distribution.
    """
    
    def __init__(self, quantiles: list = [0.1, 0.5, 0.9]):
        super().__init__()
        self.quantiles = torch.tensor(quantiles)
        
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate quantile loss.
        
        Args:
            predictions: Model predictions (batch_size, n_quantiles)
            targets: Target values (batch_size,)
            
        Returns:
            Quantile loss
        """
        if predictions.dim() == 1:
            predictions = predictions.unsqueeze(1)
            
        targets = targets.unsqueeze(1).expand_as(predictions)
        errors = targets - predictions
        
        quantiles = self.quantiles.to(predictions.device)
        quantiles = quantiles.view(1, -1)
        
        loss = torch.where(
            errors >= 0,
            quantiles * errors,
            (quantiles - 1) * errors
        )
        
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Combine multiple loss functions with configurable weights.
    """
    
    def __init__(self, loss_configs: Dict[str, Dict]):
        """
        Args:
            loss_configs: Dictionary mapping loss names to their configs
                Example: {
                    'mse': {'weight': 0.5, 'loss': nn.MSELoss()},
                    'direction': {'weight': 0.3, 'loss': DirectionalAccuracy()},
                    'ranking': {'weight': 0.2, 'loss': RankingLoss()}
                }
        """
        super().__init__()
        self.losses = nn.ModuleDict()
        self.weights = {}
        
        for name, config in loss_configs.items():
            self.losses[name] = config['loss']
            self.weights[name] = config.get('weight', 1.0)
            
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Calculate combined loss.
        
        Args:
            predictions: Model predictions
            targets: Target values
            **kwargs: Additional arguments for specific losses
            
        Returns:
            Dictionary with total loss and individual components
        """
        total_loss = 0.0
        loss_components = {}
        
        for name, loss_fn in self.losses.items():
            # Some losses might need additional arguments
            if name in ['financial', 'weighted_mse']:
                component_loss = loss_fn(predictions, targets, **kwargs)
            else:
                component_loss = loss_fn(predictions, targets)
                
            weighted_loss = self.weights[name] * component_loss
            total_loss += weighted_loss
            loss_components[name] = component_loss
            
        loss_components['total'] = total_loss
        return loss_components