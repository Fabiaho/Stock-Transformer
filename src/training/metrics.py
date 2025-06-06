"""
Financial metrics for evaluating trading strategies and predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Union, Tuple
from scipy import stats


class FinancialMetrics:
    """Calculate various financial metrics for strategy evaluation."""
    
    def __init__(
        self,
        transaction_cost: float = 0.001,
        risk_free_rate: float = 0.02,
        trading_days: int = 252,
        calculate_sharpe: bool = True,
        calculate_sortino: bool = True,
        calculate_calmar: bool = True,
        calculate_information_ratio: bool = True
    ):
        self.transaction_cost = transaction_cost
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
        self.daily_rf = risk_free_rate / trading_days
        
        self.calculate_sharpe = calculate_sharpe
        self.calculate_sortino = calculate_sortino
        self.calculate_calmar = calculate_calmar
        self.calculate_information_ratio = calculate_information_ratio
        
    def calculate_returns(
        self,
        predictions: np.ndarray,
        actual_returns: np.ndarray,
        position_sizing: str = 'binary'
    ) -> np.ndarray:
        """
        Calculate strategy returns based on predictions.
        
        Args:
            predictions: Model predictions
            actual_returns: Actual market returns
            position_sizing: Strategy for position sizing
                - 'binary': +1 or -1 positions
                - 'proportional': Position size proportional to prediction
                - 'kelly': Kelly criterion based sizing
                
        Returns:
            Strategy returns
        """
        if position_sizing == 'binary':
            positions = np.sign(predictions)
        elif position_sizing == 'proportional':
            positions = np.tanh(predictions)  # Bounded [-1, 1]
        elif position_sizing == 'kelly':
            positions = self._kelly_position_size(predictions, actual_returns)
        else:
            raise ValueError(f"Unknown position sizing: {position_sizing}")
            
        # Calculate raw returns
        strategy_returns = positions * actual_returns
        
        # Apply transaction costs
        position_changes = np.diff(positions)
        costs = np.abs(position_changes) * self.transaction_cost
        
        # Adjust returns for costs
        if len(costs) > 0:
            strategy_returns[1:] -= costs
            
        return strategy_returns
        
    def _kelly_position_size(
        self,
        predictions: np.ndarray,
        actual_returns: np.ndarray,
        lookback: int = 60
    ) -> np.ndarray:
        """Calculate position sizes using Kelly criterion."""
        positions = np.zeros_like(predictions)
        
        for i in range(lookback, len(predictions)):
            # Estimate win probability and odds from recent history
            recent_preds = predictions[i-lookback:i]
            recent_returns = actual_returns[i-lookback:i]
            
            # Simple Kelly approximation
            wins = (np.sign(recent_preds) == np.sign(recent_returns))
            p = np.mean(wins)  # Win probability
            
            if p > 0.5 and np.std(recent_returns) > 0:
                # Kelly fraction
                q = 1 - p
                b = np.mean(np.abs(recent_returns[wins])) / np.mean(np.abs(recent_returns[~wins]))
                f = (p * b - q) / b
                
                # Apply with safety factor
                positions[i] = np.clip(f * 0.25, -1, 1) * np.sign(predictions[i])
            else:
                positions[i] = 0
                
        return positions
        
    def sharpe_ratio(self, returns: np.ndarray) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Strategy returns
            
        Returns:
            Annualized Sharpe ratio
        """
        excess_returns = returns - self.daily_rf
        
        if len(returns) < 2 or np.std(excess_returns) == 0:
            return 0.0
            
        sharpe = np.mean(excess_returns) / np.std(excess_returns)
        return sharpe * np.sqrt(self.trading_days)
        
    def sortino_ratio(self, returns: np.ndarray) -> float:
        """
        Calculate Sortino ratio (uses downside deviation).
        
        Args:
            returns: Strategy returns
            
        Returns:
            Annualized Sortino ratio
        """
        excess_returns = returns - self.daily_rf
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) < 2:
            return 0.0
            
        downside_std = np.std(downside_returns)
        if downside_std == 0:
            return 0.0
            
        sortino = np.mean(excess_returns) / downside_std
        return sortino * np.sqrt(self.trading_days)
        
    def calmar_ratio(self, returns: np.ndarray) -> float:
        """
        Calculate Calmar ratio (return over max drawdown).
        
        Args:
            returns: Strategy returns
            
        Returns:
            Calmar ratio
        """
        cumulative_returns = (1 + returns).cumprod()
        annual_return = cumulative_returns[-1] ** (self.trading_days / len(returns)) - 1
        
        max_dd = self.max_drawdown(returns)
        
        if max_dd == 0:
            return 0.0
            
        return annual_return / abs(max_dd)
        
    def max_drawdown(self, returns: np.ndarray) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            returns: Strategy returns
            
        Returns:
            Maximum drawdown (negative value)
        """
        cumulative = (1 + returns).cumprod()
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
        
    def information_ratio(
        self,
        returns: np.ndarray,
        benchmark_returns: np.ndarray
    ) -> float:
        """
        Calculate information ratio.
        
        Args:
            returns: Strategy returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Annualized information ratio
        """
        active_returns = returns - benchmark_returns
        
        if len(active_returns) < 2 or np.std(active_returns) == 0:
            return 0.0
            
        ir = np.mean(active_returns) / np.std(active_returns)
        return ir * np.sqrt(self.trading_days)
        
    def win_rate(self, returns: np.ndarray) -> float:
        """Calculate percentage of positive returns."""
        return np.mean(returns > 0)
        
    def profit_factor(self, returns: np.ndarray) -> float:
        """Calculate ratio of gross profits to gross losses."""
        profits = returns[returns > 0].sum()
        losses = abs(returns[returns < 0].sum())
        
        if losses == 0:
            return np.inf if profits > 0 else 0.0
            
        return profits / losses
        
    def tail_ratio(self, returns: np.ndarray, percentile: float = 0.05) -> float:
        """Calculate ratio of right tail to left tail."""
        right_tail = np.percentile(returns, 100 - percentile * 100)
        left_tail = np.percentile(returns, percentile * 100)
        
        if abs(left_tail) < 1e-8:
            return 0.0
            
        return abs(right_tail / left_tail)
        
    def var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Value at Risk."""
        return np.percentile(returns, (1 - confidence) * 100)
        
    def cvar(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        var_threshold = self.var(returns, confidence)
        return np.mean(returns[returns <= var_threshold])
        
    def calculate_all_metrics(
        self,
        actual_returns: np.ndarray,
        predictions: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None,
        position_sizing: str = 'binary'
    ) -> Dict[str, float]:
        """
        Calculate all configured metrics.
        
        Args:
            actual_returns: Actual market returns
            predictions: Model predictions
            benchmark_returns: Optional benchmark returns
            position_sizing: Position sizing strategy
            
        Returns:
            Dictionary of metric names to values
        """
        # Calculate strategy returns
        strategy_returns = self.calculate_returns(
            predictions, actual_returns, position_sizing
        )
        
        metrics = {}
        
        # Basic return metrics
        metrics['total_return'] = (1 + strategy_returns).prod() - 1
        metrics['annual_return'] = (1 + metrics['total_return']) ** (self.trading_days / len(strategy_returns)) - 1
        metrics['volatility'] = np.std(strategy_returns) * np.sqrt(self.trading_days)
        
        # Risk-adjusted metrics
        if self.calculate_sharpe:
            metrics['sharpe_ratio'] = self.sharpe_ratio(strategy_returns)
            
        if self.calculate_sortino:
            metrics['sortino_ratio'] = self.sortino_ratio(strategy_returns)
            
        if self.calculate_calmar:
            metrics['calmar_ratio'] = self.calmar_ratio(strategy_returns)
            
        # Drawdown metrics
        metrics['max_drawdown'] = self.max_drawdown(strategy_returns)
        
        # Win/loss metrics
        metrics['win_rate'] = self.win_rate(strategy_returns)
        metrics['profit_factor'] = self.profit_factor(strategy_returns)
        metrics['tail_ratio'] = self.tail_ratio(strategy_returns)
        
        # Risk metrics
        metrics['var_95'] = self.var(strategy_returns, 0.95)
        metrics['cvar_95'] = self.cvar(strategy_returns, 0.95)
        
        # Benchmark comparison
        if benchmark_returns is not None and self.calculate_information_ratio:
            metrics['information_ratio'] = self.information_ratio(
                strategy_returns, benchmark_returns
            )
            metrics['beta'] = self._calculate_beta(strategy_returns, benchmark_returns)
            metrics['alpha'] = self._calculate_alpha(
                strategy_returns, benchmark_returns, metrics['beta']
            )
            
        # Prediction accuracy metrics
        metrics['directional_accuracy'] = np.mean(
            np.sign(predictions) == np.sign(actual_returns)
        )
        metrics['hit_rate'] = np.mean(
            (predictions > 0) == (actual_returns > 0)
        )
        
        return metrics
        
    def _calculate_beta(
        self,
        returns: np.ndarray,
        benchmark_returns: np.ndarray
    ) -> float:
        """Calculate beta relative to benchmark."""
        if len(returns) != len(benchmark_returns):
            return 0.0
            
        covariance = np.cov(returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        
        if benchmark_variance == 0:
            return 0.0
            
        return covariance / benchmark_variance
        
    def _calculate_alpha(
        self,
        returns: np.ndarray,
        benchmark_returns: np.ndarray,
        beta: float
    ) -> float:
        """Calculate alpha (excess return)."""
        strategy_mean = np.mean(returns) * self.trading_days
        benchmark_mean = np.mean(benchmark_returns) * self.trading_days
        
        return strategy_mean - (self.risk_free_rate + beta * (benchmark_mean - self.risk_free_rate))
        
    def rolling_metrics(
        self,
        returns: np.ndarray,
        window: int = 60,
        metrics_to_calculate: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Calculate rolling metrics over time.
        
        Args:
            returns: Strategy returns
            window: Rolling window size
            metrics_to_calculate: List of metrics to calculate
            
        Returns:
            DataFrame with rolling metrics
        """
        if metrics_to_calculate is None:
            metrics_to_calculate = ['sharpe_ratio', 'volatility', 'max_drawdown']
            
        rolling_data = []
        
        for i in range(window, len(returns)):
            window_returns = returns[i-window:i]
            metrics = {}
            
            if 'sharpe_ratio' in metrics_to_calculate:
                metrics['sharpe_ratio'] = self.sharpe_ratio(window_returns)
                
            if 'volatility' in metrics_to_calculate:
                metrics['volatility'] = np.std(window_returns) * np.sqrt(self.trading_days)
                
            if 'max_drawdown' in metrics_to_calculate:
                metrics['max_drawdown'] = self.max_drawdown(window_returns)
                
            if 'win_rate' in metrics_to_calculate:
                metrics['win_rate'] = self.win_rate(window_returns)
                
            rolling_data.append(metrics)
            
        return pd.DataFrame(rolling_data, index=range(window, len(returns)))