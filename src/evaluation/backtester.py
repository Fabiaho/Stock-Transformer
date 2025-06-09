"""
Comprehensive backtesting engine for evaluating trading strategies.
Includes realistic simulation with transaction costs, slippage, and position limits.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json

from src.training.metrics import FinancialMetrics

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters."""
    initial_capital: float = 100000.0
    position_size_method: str = 'fixed_fraction'  # 'fixed_fraction', 'kelly', 'risk_parity', 'equal_weight'
    max_position_size: float = 0.2  # Maximum 20% per position
    min_position_size: float = 0.01  # Minimum 1% per position
    transaction_cost: float = 0.001  # 10 basis points
    slippage: float = 0.0005  # 5 basis points
    short_selling_allowed: bool = True
    max_positions: int = 10  # Maximum number of concurrent positions
    rebalance_frequency: str = 'daily'  # 'daily', 'weekly', 'monthly'
    stop_loss: Optional[float] = 0.05  # 5% stop loss
    take_profit: Optional[float] = 0.10  # 10% take profit
    use_trailing_stop: bool = False
    trailing_stop_distance: float = 0.03  # 3% trailing stop
    margin_requirement: float = 0.5  # 50% margin for shorts
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    side: str  # 'long' or 'short'
    current_price: float = 0.0
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    trailing_stop_price: Optional[float] = None
    highest_price: float = 0.0  # For trailing stop
    pnl: float = 0.0
    return_pct: float = 0.0
    holding_period: int = 0
    
    def update_price(self, current_price: float):
        """Update position with current price."""
        self.current_price = current_price
        self.highest_price = max(self.highest_price, current_price)
        
        if self.side == 'long':
            self.pnl = (current_price - self.entry_price) * self.shares
            self.return_pct = (current_price - self.entry_price) / self.entry_price
        else:  # short
            self.pnl = (self.entry_price - current_price) * self.shares
            self.return_pct = (self.entry_price - current_price) / self.entry_price
            
    def should_exit(self, config: BacktestConfig) -> Tuple[bool, str]:
        """Check if position should be exited."""
        if self.side == 'long':
            # Stop loss
            if self.stop_loss and self.current_price <= self.stop_loss:
                return True, 'stop_loss'
            
            # Take profit
            if self.take_profit and self.current_price >= self.take_profit:
                return True, 'take_profit'
                
            # Trailing stop
            if config.use_trailing_stop and self.trailing_stop_price:
                if self.current_price <= self.trailing_stop_price:
                    return True, 'trailing_stop'
                    
        else:  # short
            # Stop loss (price goes up for shorts)
            if self.stop_loss and self.current_price >= self.stop_loss:
                return True, 'stop_loss'
            
            # Take profit (price goes down for shorts)
            if self.take_profit and self.current_price <= self.take_profit:
                return True, 'take_profit'
                
        return False, ''


@dataclass
class Trade:
    """Represents a completed trade."""
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: int
    side: str
    pnl: float
    return_pct: float
    holding_period: int
    exit_reason: str
    commission: float


class Portfolio:
    """Manages portfolio state during backtesting."""
    
    def __init__(self, initial_capital: float, config: BacktestConfig):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.closed_trades: List[Trade] = []
        self.equity_curve: List[float] = [initial_capital]
        self.cash_curve: List[float] = [initial_capital]
        self.dates: List[pd.Timestamp] = []
        
    @property
    def equity(self) -> float:
        """Calculate total portfolio equity."""
        positions_value = sum(
            pos.shares * pos.current_price for pos in self.positions.values()
        )
        return self.cash + positions_value
        
    @property
    def num_positions(self) -> int:
        """Number of open positions."""
        return len(self.positions)
        
    def can_open_position(self) -> bool:
        """Check if new position can be opened."""
        return self.num_positions < self.config.max_positions
        
    def position_size(self, price: float, volatility: Optional[float] = None) -> int:
        """Calculate position size based on method."""
        if self.config.position_size_method == 'fixed_fraction':
            position_value = self.equity * self.config.max_position_size
        elif self.config.position_size_method == 'risk_parity' and volatility:
            # Size inversely proportional to volatility
            target_risk = 0.02  # 2% portfolio risk
            position_value = (self.equity * target_risk) / volatility
        elif self.config.position_size_method == 'equal_weight':
            position_value = self.equity / self.config.max_positions
        else:
            position_value = self.equity * 0.1  # Default 10%
            
        # Apply constraints
        position_value = min(position_value, self.equity * self.config.max_position_size)
        position_value = max(position_value, self.equity * self.config.min_position_size)
        
        shares = int(position_value / price)
        return max(shares, 1)  # At least 1 share
        
    def open_position(
        self,
        symbol: str,
        date: pd.Timestamp,
        price: float,
        side: str,
        volatility: Optional[float] = None
    ) -> Optional[Position]:
        """Open a new position."""
        if symbol in self.positions:
            return None  # Already have position
            
        shares = self.position_size(price, volatility)
        position_value = shares * price
        commission = position_value * self.config.transaction_cost
        slippage = position_value * self.config.slippage
        total_cost = position_value + commission + slippage
        
        # Check if we have enough cash
        required_cash = total_cost
        if side == 'short' and self.config.short_selling_allowed:
            required_cash = total_cost * self.config.margin_requirement
            
        if self.cash < required_cash:
            return None
            
        # Create position
        position = Position(
            symbol=symbol,
            entry_date=date,
            entry_price=price * (1 + self.config.slippage),  # Adjust for slippage
            shares=shares,
            side=side,
            current_price=price,
            highest_price=price
        )
        
        # Set stop loss and take profit
        if self.config.stop_loss:
            if side == 'long':
                position.stop_loss = price * (1 - self.config.stop_loss)
            else:
                position.stop_loss = price * (1 + self.config.stop_loss)
                
        if self.config.take_profit:
            if side == 'long':
                position.take_profit = price * (1 + self.config.take_profit)
            else:
                position.take_profit = price * (1 - self.config.take_profit)
                
        # Update cash
        if side == 'long':
            self.cash -= total_cost
        else:  # short - only margin requirement
            self.cash -= required_cash
            
        self.positions[symbol] = position
        return position
        
    def close_position(
        self,
        symbol: str,
        date: pd.Timestamp,
        price: float,
        reason: str = 'signal'
    ) -> Optional[Trade]:
        """Close an existing position."""
        if symbol not in self.positions:
            return None
            
        position = self.positions[symbol]
        position.exit_date = date
        position.exit_price = price * (1 - self.config.slippage)  # Adjust for slippage
        position.update_price(position.exit_price)
        
        # Calculate commission
        exit_value = position.shares * position.exit_price
        commission = exit_value * self.config.transaction_cost
        
        # Update cash
        if position.side == 'long':
            self.cash += exit_value - commission
        else:  # short
            # Return margin and adjust for P&L
            margin_return = position.shares * position.entry_price * self.config.margin_requirement
            self.cash += margin_return + position.pnl - commission
            
        # Create trade record
        trade = Trade(
            symbol=symbol,
            entry_date=position.entry_date,
            exit_date=date,
            entry_price=position.entry_price,
            exit_price=position.exit_price,
            shares=position.shares,
            side=position.side,
            pnl=position.pnl - commission,
            return_pct=position.return_pct,
            holding_period=(date - position.entry_date).days,
            exit_reason=reason,
            commission=commission * 2  # Entry + exit
        )
        
        self.closed_trades.append(trade)
        del self.positions[symbol]
        
        return trade
        
    def update_trailing_stops(self):
        """Update trailing stop prices for all positions."""
        if not self.config.use_trailing_stop:
            return
            
        for position in self.positions.values():
            if position.side == 'long':
                new_stop = position.highest_price * (1 - self.config.trailing_stop_distance)
                if position.trailing_stop_price is None or new_stop > position.trailing_stop_price:
                    position.trailing_stop_price = new_stop
            else:  # short
                new_stop = position.highest_price * (1 + self.config.trailing_stop_distance)
                if position.trailing_stop_price is None or new_stop < position.trailing_stop_price:
                    position.trailing_stop_price = new_stop
                    
    def update_portfolio(self, date: pd.Timestamp):
        """Update portfolio state."""
        self.dates.append(date)
        self.equity_curve.append(self.equity)
        self.cash_curve.append(self.cash)


class Backtester:
    """Main backtesting engine."""
    
    def __init__(
        self,
        config: Optional[BacktestConfig] = None,
        verbose: bool = True
    ):
        self.config = config or BacktestConfig()
        self.verbose = verbose
        self.portfolio: Optional[Portfolio] = None
        self.market_data: Optional[pd.DataFrame] = None
        self.signals: Optional[pd.DataFrame] = None
        self.results: Optional[Dict] = None
        
    def prepare_signals(
        self,
        predictions: Union[np.ndarray, torch.Tensor, pd.DataFrame],
        dates: pd.DatetimeIndex,
        symbols: List[str],
        prediction_type: str = 'regression'
    ) -> pd.DataFrame:
        """
        Convert model predictions to trading signals.
        
        Args:
            predictions: Model predictions
            dates: Date index
            symbols: List of symbols
            prediction_type: 'regression' or 'classification'
            
        Returns:
            DataFrame with trading signals
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
            
        if len(predictions.shape) == 1:
            predictions = predictions.reshape(-1, 1)
            
        signals_df = pd.DataFrame(
            predictions,
            index=dates,
            columns=symbols if len(symbols) == predictions.shape[1] else [f'signal_{i}' for i in range(predictions.shape[1])]
        )
        
        if prediction_type == 'regression':
            # Convert continuous predictions to signals
            # Positive predictions -> long, negative -> short
            signals_df = signals_df.apply(lambda x: np.sign(x))
        else:
            # Classification: 0=short, 1=neutral, 2=long
            signals_df = signals_df.apply(lambda x: x - 1)  # Convert to -1, 0, 1
            
        return signals_df
        
    def prepare_market_data(
        self,
        price_data: Dict[str, pd.DataFrame],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Prepare market data for backtesting.
        
        Args:
            price_data: Dictionary of symbol -> price DataFrame
            start_date: Start date for backtest
            end_date: End date for backtest
            
        Returns:
            Combined market data DataFrame
        """
        combined_data = {}
        
        for symbol, df in price_data.items():
            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"Missing required column '{col}' for {symbol}")
                    
            # Add returns and volatility
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(20).std()
            
            # Filter date range
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df.index <= pd.to_datetime(end_date)]
                
            combined_data[symbol] = df
            
        # Combine into multi-index DataFrame
        market_df = pd.concat(combined_data, axis=1)
        return market_df
        
    def run(
        self,
        signals: pd.DataFrame,
        market_data: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run backtest simulation.
        
        Args:
            signals: Trading signals DataFrame
            market_data: Market data DataFrame
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary with backtest results
        """
        # Initialize portfolio
        self.portfolio = Portfolio(self.config.initial_capital, self.config)
        
        # Align signals and market data
        common_dates = signals.index.intersection(market_data.index)
        if start_date:
            common_dates = common_dates[common_dates >= pd.to_datetime(start_date)]
        if end_date:
            common_dates = common_dates[common_dates <= pd.to_datetime(end_date)]
            
        if len(common_dates) == 0:
            raise ValueError("No overlapping dates between signals and market data")
            
        # Store for later use
        self.signals = signals
        self.market_data = market_data
        
        # Run simulation
        symbols = signals.columns.tolist()
        rebalance_dates = self._get_rebalance_dates(common_dates)
        
        for date in common_dates:
            # Update current prices
            for symbol in symbols:
                if symbol in self.portfolio.positions:
                    try:
                        current_price = market_data.loc[date, (symbol, 'close')]
                        self.portfolio.positions[symbol].update_price(current_price)
                        self.portfolio.positions[symbol].holding_period += 1
                    except:
                        continue
                        
            # Check exit conditions
            positions_to_close = []
            for symbol, position in self.portfolio.positions.items():
                should_exit, reason = position.should_exit(self.config)
                if should_exit:
                    positions_to_close.append((symbol, reason))
                    
            # Close positions
            for symbol, reason in positions_to_close:
                try:
                    price = market_data.loc[date, (symbol, 'close')]
                    self.portfolio.close_position(symbol, date, price, reason)
                    if self.verbose:
                        logger.info(f"{date}: Closed {symbol} position - {reason}")
                except:
                    continue
                    
            # Rebalance portfolio
            if date in rebalance_dates:
                self._rebalance_portfolio(date, signals.loc[date], market_data.loc[date])
                
            # Update trailing stops
            self.portfolio.update_trailing_stops()
            
            # Update portfolio state
            self.portfolio.update_portfolio(date)
            
        # Close all remaining positions
        final_date = common_dates[-1]
        for symbol in list(self.portfolio.positions.keys()):
            try:
                price = market_data.loc[final_date, (symbol, 'close')]
                self.portfolio.close_position(symbol, final_date, price, 'end_of_backtest')
            except:
                continue
                
        # Calculate results
        self.results = self._calculate_results()
        
        return self.results
        
    def _get_rebalance_dates(self, dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
        """Get rebalancing dates based on frequency."""
        if self.config.rebalance_frequency == 'daily':
            return dates
        elif self.config.rebalance_frequency == 'weekly':
            return dates[dates.weekday == 0]  # Mondays
        elif self.config.rebalance_frequency == 'monthly':
            return dates[dates.is_month_start]
        else:
            return dates
            
    def _rebalance_portfolio(
        self,
        date: pd.Timestamp,
        signals: pd.Series,
        market_data: pd.DataFrame
    ):
        """Rebalance portfolio based on signals."""
        # Get current positions
        current_symbols = set(self.portfolio.positions.keys())
        
        # Get target positions based on signals
        target_symbols = set()
        for symbol in signals.index:
            signal = signals[symbol]
            if not pd.isna(signal) and signal != 0:
                target_symbols.add(symbol)
                
        # Close positions no longer in signals
        for symbol in current_symbols - target_symbols:
            try:
                price = market_data[(symbol, 'close')]
                self.portfolio.close_position(symbol, date, price, 'rebalance')
            except:
                continue
                
        # Open new positions
        for symbol in target_symbols - current_symbols:
            if not self.portfolio.can_open_position():
                break
                
            try:
                signal = signals[symbol]
                price = market_data[(symbol, 'close')]
                volatility = market_data.get((symbol, 'volatility'), None)
                
                side = 'long' if signal > 0 else 'short'
                if not self.config.short_selling_allowed and side == 'short':
                    continue
                    
                position = self.portfolio.open_position(symbol, date, price, side, volatility)
                if position and self.verbose:
                    logger.info(f"{date}: Opened {side} position in {symbol} at {price:.2f}")
            except:
                continue
                
    def _calculate_results(self) -> Dict[str, Any]:
        """Calculate comprehensive backtest results."""
        if not self.portfolio:
            return {}
            
        # Convert to arrays
        equity_curve = np.array(self.portfolio.equity_curve)
        dates = pd.to_datetime(self.portfolio.dates)
        
        # Calculate returns
        returns = np.diff(equity_curve) / equity_curve[:-1]
        
        # Initialize metrics calculator
        metrics_calc = FinancialMetrics(
            transaction_cost=0,  # Already accounted for
            risk_free_rate=0.02
        )
        
        # Calculate all metrics
        metrics = metrics_calc.calculate_all_metrics(
            returns, returns  # Using returns as both actual and predicted
        )
        
        # Trade statistics
        trades = self.portfolio.closed_trades
        if trades:
            winning_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl < 0]
            
            trade_stats = {
                'total_trades': len(trades),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': len(winning_trades) / len(trades) if trades else 0,
                'avg_win': np.mean([t.pnl for t in winning_trades]) if winning_trades else 0,
                'avg_loss': np.mean([t.pnl for t in losing_trades]) if losing_trades else 0,
                'avg_win_pct': np.mean([t.return_pct for t in winning_trades]) if winning_trades else 0,
                'avg_loss_pct': np.mean([t.return_pct for t in losing_trades]) if losing_trades else 0,
                'largest_win': max([t.pnl for t in trades], default=0),
                'largest_loss': min([t.pnl for t in trades], default=0),
                'avg_holding_period': np.mean([t.holding_period for t in trades]) if trades else 0,
                'profit_factor': abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) if losing_trades else np.inf,
            }
            
            # Exit reason breakdown
            exit_reasons = {}
            for trade in trades:
                exit_reasons[trade.exit_reason] = exit_reasons.get(trade.exit_reason, 0) + 1
            trade_stats['exit_reasons'] = exit_reasons
        else:
            trade_stats = {}
            
        # Portfolio statistics
        portfolio_stats = {
            'initial_capital': self.config.initial_capital,
            'final_equity': equity_curve[-1],
            'total_return': (equity_curve[-1] - equity_curve[0]) / equity_curve[0],
            'total_pnl': equity_curve[-1] - equity_curve[0],
            'max_equity': np.max(equity_curve),
            'min_equity': np.min(equity_curve),
            'total_commission': sum(t.commission for t in trades) if trades else 0,
        }
        
        # Combine all results
        results = {
            'metrics': metrics,
            'trade_statistics': trade_stats,
            'portfolio_statistics': portfolio_stats,
            'equity_curve': equity_curve.tolist(),
            'cash_curve': self.portfolio.cash_curve,
            'dates': [d.isoformat() for d in dates],
            'returns': returns.tolist(),
            'trades': [self._trade_to_dict(t) for t in trades] if trades else [],
            'config': self.config.to_dict()
        }
        
        return results
        
    def _trade_to_dict(self, trade: Trade) -> Dict:
        """Convert Trade object to dictionary."""
        return {
            'symbol': trade.symbol,
            'entry_date': trade.entry_date.isoformat(),
            'exit_date': trade.exit_date.isoformat(),
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'shares': trade.shares,
            'side': trade.side,
            'pnl': trade.pnl,
            'return_pct': trade.return_pct,
            'holding_period': trade.holding_period,
            'exit_reason': trade.exit_reason,
            'commission': trade.commission
        }
        
    def save_results(self, filepath: Union[str, Path]):
        """Save backtest results to file."""
        if not self.results:
            raise ValueError("No results to save. Run backtest first.")
            
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj
            
        results_serializable = convert_types(self.results)
        
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=2)
            
        logger.info(f"Results saved to {filepath}")
        
    def generate_report(self) -> pd.DataFrame:
        """Generate a summary report of backtest results."""
        if not self.results:
            return pd.DataFrame()
            
        # Create summary DataFrame
        summary_data = []
        
        # Portfolio metrics
        portfolio_stats = self.results['portfolio_statistics']
        summary_data.extend([
            ('Portfolio', 'Initial Capital', f"${portfolio_stats['initial_capital']:,.2f}"),
            ('Portfolio', 'Final Equity', f"${portfolio_stats['final_equity']:,.2f}"),
            ('Portfolio', 'Total Return', f"{portfolio_stats['total_return']:.2%}"),
            ('Portfolio', 'Total P&L', f"${portfolio_stats['total_pnl']:,.2f}"),
        ])
        
        # Performance metrics
        metrics = self.results['metrics']
        summary_data.extend([
            ('Performance', 'Annual Return', f"{metrics.get('annual_return', 0):.2%}"),
            ('Performance', 'Volatility', f"{metrics.get('volatility', 0):.2%}"),
            ('Performance', 'Sharpe Ratio', f"{metrics.get('sharpe_ratio', 0):.2f}"),
            ('Performance', 'Sortino Ratio', f"{metrics.get('sortino_ratio', 0):.2f}"),
            ('Performance', 'Calmar Ratio', f"{metrics.get('calmar_ratio', 0):.2f}"),
            ('Performance', 'Max Drawdown', f"{metrics.get('max_drawdown', 0):.2%}"),
        ])
        
        # Trade statistics
        trade_stats = self.results['trade_statistics']
        if trade_stats:
            summary_data.extend([
                ('Trading', 'Total Trades', trade_stats['total_trades']),
                ('Trading', 'Win Rate', f"{trade_stats['win_rate']:.2%}"),
                ('Trading', 'Profit Factor', f"{trade_stats['profit_factor']:.2f}"),
                ('Trading', 'Avg Win', f"${trade_stats['avg_win']:,.2f}"),
                ('Trading', 'Avg Loss', f"${trade_stats['avg_loss']:,.2f}"),
                ('Trading', 'Avg Holding Period', f"{trade_stats['avg_holding_period']:.1f} days"),
            ])
            
        # Create DataFrame
        report_df = pd.DataFrame(
            summary_data,
            columns=['Category', 'Metric', 'Value']
        )
        
        return report_df