"""
Visualization utilities for backtesting results and model performance.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.dates as mdates
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')


class BacktestVisualizer:
    """Visualize backtest results with various charts and plots."""
    
    def __init__(self, results: Dict, save_dir: Optional[Path] = None):
        self.results = results
        self.save_dir = Path(save_dir) if save_dir else Path('results/figures')
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_equity_curve(self, 
                         figsize: Tuple[int, int] = (12, 6),
                         save: bool = True) -> plt.Figure:
        """Plot portfolio equity curve over time."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
        
        # Convert dates
        dates = pd.to_datetime(self.results['dates'])
        equity = np.array(self.results['equity_curve'])
        cash = np.array(self.results['cash_curve'])
        
        # Equity curve
        ax1.plot(dates, equity, label='Total Equity', linewidth=2, color='blue')
        ax1.plot(dates, cash, label='Cash', linewidth=1, alpha=0.7, color='green')
        ax1.fill_between(dates, cash, equity, alpha=0.2, color='blue')
        
        # Add initial capital line
        initial_capital = self.results['portfolio_statistics']['initial_capital']
        ax1.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5, 
                   label='Initial Capital')
        
        # Mark peak equity
        peak_idx = np.argmax(equity)
        ax1.plot(dates[peak_idx], equity[peak_idx], 'go', markersize=8, 
                label=f'Peak: ${equity[peak_idx]:,.0f}')
        
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_title('Portfolio Equity Curve')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Drawdown plot
        drawdown = (equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity) * 100
        ax2.fill_between(dates, drawdown, 0, color='red', alpha=0.3)
        ax2.plot(dates, drawdown, color='red', linewidth=1)
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        
        # Mark max drawdown
        max_dd_idx = np.argmin(drawdown)
        ax2.plot(dates[max_dd_idx], drawdown[max_dd_idx], 'ro', markersize=8, 
                label=f'Max DD: {drawdown[max_dd_idx]:.1f}%')
        ax2.legend()
        
        plt.tight_layout()
        
        if save:
            filepath = self.save_dir / 'equity_curve.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_returns_distribution(self, 
                                figsize: Tuple[int, int] = (12, 5),
                                save: bool = True) -> plt.Figure:
        """Plot returns distribution and statistics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        returns = np.array(self.results['returns']) * 100  # Convert to percentage
        
        # Histogram
        n_bins = min(50, len(returns) // 10)
        ax1.hist(returns, bins=n_bins, alpha=0.7, density=True, color='blue', edgecolor='black')
        
        # Fit normal distribution
        from scipy import stats
        mu, sigma = stats.norm.fit(returns)
        x = np.linspace(returns.min(), returns.max(), 100)
        ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
                label=f'Normal(μ={mu:.2f}%, σ={sigma:.2f}%)')
        
        # Add statistics
        ax1.axvline(returns.mean(), color='green', linestyle='--', linewidth=2, 
                   label=f'Mean: {returns.mean():.2f}%')
        ax1.axvline(np.median(returns), color='orange', linestyle='--', linewidth=2, 
                   label=f'Median: {np.median(returns):.2f}%')
        
        ax1.set_xlabel('Daily Returns (%)')
        ax1.set_ylabel('Density')
        ax1.set_title('Returns Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(returns, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = self.save_dir / 'returns_distribution.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_rolling_metrics(self, 
                           window: int = 60,
                           figsize: Tuple[int, int] = (12, 10),
                           save: bool = True) -> plt.Figure:
        """Plot rolling performance metrics."""
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        
        dates = pd.to_datetime(self.results['dates'])
        returns = np.array(self.results['returns'])
        
        # Calculate rolling metrics
        rolling_returns = pd.Series(returns, index=dates[1:])
        
        # Rolling returns (annualized)
        rolling_mean = rolling_returns.rolling(window).mean() * 252
        rolling_std = rolling_returns.rolling(window).std() * np.sqrt(252)
        
        axes[0].plot(rolling_mean.index, rolling_mean, label='Annualized Return', linewidth=2)
        axes[0].fill_between(rolling_mean.index, 
                           rolling_mean - rolling_std, 
                           rolling_mean + rolling_std, 
                           alpha=0.2, label='±1 Std Dev')
        axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0].set_ylabel('Return (%)')
        axes[0].set_title(f'Rolling {window}-Day Metrics')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
        
        # Rolling Sharpe ratio
        rolling_sharpe = (rolling_mean - 0.02) / rolling_std  # Assuming 2% risk-free rate
        axes[1].plot(rolling_sharpe.index, rolling_sharpe, color='green', linewidth=2)
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Sharpe = 1')
        axes[1].set_ylabel('Sharpe Ratio')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Rolling win rate
        rolling_wins = (rolling_returns > 0).rolling(window).mean()
        axes[2].plot(rolling_wins.index, rolling_wins, color='orange', linewidth=2)
        axes[2].axhline(y=0.5, color='black', linestyle='-', alpha=0.3)
        axes[2].set_ylabel('Win Rate')
        axes[2].set_ylim(0, 1)
        axes[2].grid(True, alpha=0.3)
        axes[2].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
        
        # Rolling max drawdown
        equity = np.array(self.results['equity_curve'])[1:]  # Skip initial
        rolling_dd = pd.Series(index=dates[1:])
        for i in range(window, len(equity)):
            window_equity = equity[i-window:i]
            dd = (window_equity - np.maximum.accumulate(window_equity)) / np.maximum.accumulate(window_equity)
            rolling_dd.iloc[i] = dd.min()
            
        axes[3].fill_between(rolling_dd.index, rolling_dd * 100, 0, color='red', alpha=0.3)
        axes[3].plot(rolling_dd.index, rolling_dd * 100, color='red', linewidth=1)
        axes[3].set_ylabel('Max Drawdown (%)')
        axes[3].set_xlabel('Date')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = self.save_dir / 'rolling_metrics.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_trade_analysis(self, 
                          figsize: Tuple[int, int] = (14, 10),
                          save: bool = True) -> plt.Figure:
        """Analyze and visualize trade statistics."""
        trades = self.results.get('trades', [])
        if not trades:
            return None
            
        # Convert to DataFrame for easier analysis
        trades_df = pd.DataFrame(trades)
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
        trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
        
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # P&L distribution
        ax1 = fig.add_subplot(gs[0, 0])
        trades_df['pnl'].hist(bins=30, ax=ax1, alpha=0.7, color='blue')
        ax1.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax1.axvline(x=trades_df['pnl'].mean(), color='red', linestyle='--', 
                   label=f"Mean: ${trades_df['pnl'].mean():.2f}")
        ax1.set_xlabel('P&L ($)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('P&L Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Returns distribution
        ax2 = fig.add_subplot(gs[0, 1])
        trades_df['return_pct'].hist(bins=30, ax=ax2, alpha=0.7, color='green')
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_xlabel('Return (%)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Returns Distribution')
        ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        ax2.grid(True, alpha=0.3)
        
        # Holding period distribution
        ax3 = fig.add_subplot(gs[0, 2])
        trades_df['holding_period'].hist(bins=30, ax=ax3, alpha=0.7, color='orange')
        ax3.set_xlabel('Holding Period (days)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Holding Period Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Win/Loss by exit reason
        ax4 = fig.add_subplot(gs[1, 0])
        exit_reasons = trades_df.groupby('exit_reason')['pnl'].agg(['count', 'sum', 'mean'])
        exit_reasons['count'].plot(kind='bar', ax=ax4, color='skyblue')
        ax4.set_xlabel('Exit Reason')
        ax4.set_ylabel('Number of Trades')
        ax4.set_title('Trades by Exit Reason')
        ax4.grid(True, alpha=0.3)
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        # Cumulative P&L
        ax5 = fig.add_subplot(gs[1, 1:])
        trades_df = trades_df.sort_values('exit_date')
        trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
        ax5.plot(trades_df['exit_date'], trades_df['cumulative_pnl'], linewidth=2)
        ax5.fill_between(trades_df['exit_date'], 0, trades_df['cumulative_pnl'], 
                        where=trades_df['cumulative_pnl'] >= 0, alpha=0.3, color='green', label='Profit')
        ax5.fill_between(trades_df['exit_date'], 0, trades_df['cumulative_pnl'], 
                        where=trades_df['cumulative_pnl'] < 0, alpha=0.3, color='red', label='Loss')
        ax5.set_xlabel('Date')
        ax5.set_ylabel('Cumulative P&L ($)')
        ax5.set_title('Cumulative Trading P&L')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Trade scatter plot
        ax6 = fig.add_subplot(gs[2, :])
        colors = ['green' if pnl > 0 else 'red' for pnl in trades_df['pnl']]
        sizes = np.abs(trades_df['pnl']) / np.abs(trades_df['pnl']).max() * 200 + 20
        
        scatter = ax6.scatter(trades_df['exit_date'], trades_df['return_pct'], 
                            c=colors, s=sizes, alpha=0.6, edgecolors='black', linewidth=1)
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax6.set_xlabel('Exit Date')
        ax6.set_ylabel('Return (%)')
        ax6.set_title('Individual Trade Returns (size = P&L magnitude)')
        ax6.grid(True, alpha=0.3)
        ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
        
        plt.tight_layout()
        
        if save:
            filepath = self.save_dir / 'trade_analysis.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_monthly_returns_heatmap(self, 
                                   figsize: Tuple[int, int] = (12, 8),
                                   save: bool = True) -> plt.Figure:
        """Plot monthly returns heatmap."""
        dates = pd.to_datetime(self.results['dates'])
        equity = np.array(self.results['equity_curve'])
        
        # Create monthly returns
        monthly_equity = pd.Series(equity, index=dates).resample('M').last()
        monthly_returns = monthly_equity.pct_change().dropna()
        
        # Reshape for heatmap
        years = monthly_returns.index.year.unique()
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        heatmap_data = pd.DataFrame(index=years, columns=months)
        
        for date, ret in monthly_returns.items():
            year = date.year
            month = date.month - 1
            if year in heatmap_data.index:
                heatmap_data.iloc[heatmap_data.index.get_loc(year), month] = ret * 100
        
        # Convert to numeric
        heatmap_data = heatmap_data.astype(float)
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create mask for missing values
        mask = heatmap_data.isna()
        
        # Plot heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Monthly Return (%)'}, mask=mask, ax=ax,
                   linewidths=1, linecolor='gray')
        
        ax.set_title('Monthly Returns Heatmap (%)')
        ax.set_xlabel('Month')
        ax.set_ylabel('Year')
        
        plt.tight_layout()
        
        if save:
            filepath = self.save_dir / 'monthly_returns_heatmap.png'
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            
        return fig
    
    def create_interactive_dashboard(self, 
                                   save_html: bool = True) -> go.Figure:
        """Create interactive Plotly dashboard."""
        dates = pd.to_datetime(self.results['dates'])
        equity = np.array(self.results['equity_curve'])
        returns = np.array(self.results['returns'])
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Equity Curve', 'Drawdown', 
                          'Daily Returns', 'Rolling Sharpe Ratio',
                          'Trade P&L Distribution', 'Monthly Returns'),
            row_heights=[0.4, 0.3, 0.3],
            column_widths=[0.6, 0.4],
            specs=[[{"colspan": 2}, None],
                  [{"type": "scatter"}, {"type": "scatter"}],
                  [{"type": "histogram"}, {"type": "heatmap"}]]
        )
        
        # Equity curve
        fig.add_trace(
            go.Scatter(x=dates, y=equity, name='Equity', line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # Cash curve
        cash = np.array(self.results['cash_curve'])
        fig.add_trace(
            go.Scatter(x=dates, y=cash, name='Cash', line=dict(color='green', width=1)),
            row=1, col=1
        )
        
        # Drawdown
        drawdown = (equity - np.maximum.accumulate(equity)) / np.maximum.accumulate(equity) * 100
        fig.add_trace(
            go.Scatter(x=dates, y=drawdown, name='Drawdown', 
                      fill='tozeroy', line=dict(color='red', width=1)),
            row=2, col=1
        )
        
        # Rolling Sharpe
        window = 60
        rolling_returns = pd.Series(returns, index=dates[1:])
        rolling_sharpe = (rolling_returns.rolling(window).mean() * 252 - 0.02) / (rolling_returns.rolling(window).std() * np.sqrt(252))
        
        fig.add_trace(
            go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe, 
                      name='Rolling Sharpe', line=dict(color='purple', width=2)),
            row=2, col=2
        )
        
        # Trade P&L histogram
        if 'trades' in self.results and self.results['trades']:
            trades_pnl = [t['pnl'] for t in self.results['trades']]
            fig.add_trace(
                go.Histogram(x=trades_pnl, name='Trade P&L', nbinsx=30),
                row=3, col=1
            )
        
        # Monthly returns heatmap
        monthly_equity = pd.Series(equity, index=dates).resample('M').last()
        monthly_returns = monthly_equity.pct_change().dropna() * 100
        
        # Prepare data for heatmap
        z_data = []
        y_labels = []
        x_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for year in monthly_returns.index.year.unique():
            year_data = []
            for month in range(1, 13):
                try:
                    value = monthly_returns[(monthly_returns.index.year == year) & 
                                          (monthly_returns.index.month == month)].iloc[0]
                    year_data.append(value)
                except:
                    year_data.append(None)
            z_data.append(year_data)
            y_labels.append(str(year))
        
        fig.add_trace(
            go.Heatmap(z=z_data, x=x_labels, y=y_labels, 
                      colorscale='RdYlGn', zmid=0),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            showlegend=True,
            title_text="Backtest Results Dashboard",
            title_font_size=20
        )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_yaxes(title_text="Value ($)", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=2)
        fig.update_yaxes(title_text="Sharpe Ratio", row=2, col=2)
        fig.update_xaxes(title_text="P&L ($)", row=3, col=1)
        fig.update_yaxes(title_text="Frequency", row=3, col=1)
        fig.update_xaxes(title_text="Month", row=3, col=2)
        fig.update_yaxes(title_text="Year", row=3, col=2)
        
        if save_html:
            filepath = self.save_dir / 'interactive_dashboard.html'
            fig.write_html(filepath)
            
        return fig
    
    def generate_all_plots(self):
        """Generate all visualization plots."""
        print("Generating backtest visualizations...")
        
        # Generate each plot
        self.plot_equity_curve()
        print("✓ Equity curve plotted")
        
        self.plot_returns_distribution()
        print("✓ Returns distribution plotted")
        
        self.plot_rolling_metrics()
        print("✓ Rolling metrics plotted")
        
        self.plot_trade_analysis()
        print("✓ Trade analysis plotted")
        
        self.plot_monthly_returns_heatmap()
        print("✓ Monthly returns heatmap plotted")
        
        self.create_interactive_dashboard()
        print("✓ Interactive dashboard created")
        
        print(f"\nAll visualizations saved to: {self.save_dir}")


def plot_prediction_analysis(
    predictions: np.ndarray,
    actuals: np.ndarray,
    dates: pd.DatetimeIndex,
    symbols: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Analyze model predictions vs actual returns."""
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    # Scatter plot
    axes[0].scatter(actuals, predictions, alpha=0.5, s=20)
    axes[0].plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 
                'r--', label='Perfect Prediction')
    axes[0].set_xlabel('Actual Returns')
    axes[0].set_ylabel('Predicted Returns')
    axes[0].set_title('Predictions vs Actuals')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residuals
    residuals = predictions - actuals
    axes[1].scatter(actuals, residuals, alpha=0.5, s=20)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel('Actual Returns')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residual Plot')
    axes[1].grid(True, alpha=0.3)
    
    # Time series of predictions
    if len(predictions.shape) == 1:
        axes[2].plot(dates, actuals, label='Actual', alpha=0.7)
        axes[2].plot(dates, predictions, label='Predicted', alpha=0.7)
        axes[2].set_xlabel('Date')
        axes[2].set_ylabel('Returns')
        axes[2].set_title('Returns Over Time')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    else:
        # Multiple symbols
        for i in range(min(3, predictions.shape[1])):
            axes[2].plot(dates, predictions[:, i], 
                       label=symbols[i] if symbols else f'Symbol {i}', alpha=0.7)
        axes[2].set_xlabel('Date')
        axes[2].set_ylabel('Predicted Returns')
        axes[2].set_title('Multi-Asset Predictions')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    # Directional accuracy over time
    window = 20
    correct_direction = (np.sign(predictions) == np.sign(actuals)).astype(float)
    if len(correct_direction.shape) == 1:
        rolling_accuracy = pd.Series(correct_direction, index=dates).rolling(window).mean()
        axes[3].plot(dates, rolling_accuracy, linewidth=2)
        axes[3].axhline(y=0.5, color='r', linestyle='--', label='Random Guess')
        axes[3].set_xlabel('Date')
        axes[3].set_ylabel('Accuracy')
        axes[3].set_title(f'Rolling {window}-Day Directional Accuracy')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        axes[3].set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    return fig