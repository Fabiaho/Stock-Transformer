#!/usr/bin/env python3
"""
Script to run backtesting on trained models.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
import yaml
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data.datamodule import StockDataModule
from src.models.lightning_module import StockTransformerLightning
from src.evaluation.backtester import Backtester, BacktestConfig
from src.evaluation.visualization import BacktestVisualizer, plot_prediction_analysis
from src.utils.logging import setup_logging


def load_checkpoint(checkpoint_path: str) -> StockTransformerLightning:
    """Load model from checkpoint."""
    logging.info(f"Loading model from {checkpoint_path}")
    model = StockTransformerLightning.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model


def get_predictions(model: StockTransformerLightning, datamodule: StockDataModule) -> Dict:
    """Get model predictions on test set."""
    logging.info("Generating predictions...")
    
    test_loader = datamodule.test_dataloader()
    if test_loader is None:
        logging.warning("No test data available, using validation data")
        test_loader = datamodule.val_dataloader()
    
    predictions = []
    actuals = []
    dates = []
    symbols = []
    metadata_list = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Get predictions
            outputs = model(batch['sequence'])
            pred = outputs['output'].cpu().numpy()
            
            # Get targets
            target = batch['target'].cpu().numpy()
            
            # Get metadata
            metadata = batch.get('metadata', [])
            
            predictions.append(pred)
            actuals.append(target)
            
            # Extract dates and symbols from metadata
            for meta in metadata:
                dates.append(pd.to_datetime(meta['target_date']))
                symbols.append(meta['symbol'])
                metadata_list.append(meta)
    
    # Concatenate all predictions
    predictions = np.concatenate(predictions, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    
    # Create DataFrame
    results_df = pd.DataFrame({
        'date': dates,
        'symbol': symbols,
        'prediction': predictions.squeeze(),
        'actual': actuals.squeeze()
    })
    
    return {
        'predictions': predictions,
        'actuals': actuals,
        'dates': dates,
        'symbols': symbols,
        'results_df': results_df,
        'metadata': metadata_list
    }


def prepare_backtest_data(prediction_results: Dict, datamodule: StockDataModule) -> tuple:
    """Prepare data for backtesting."""
    results_df = prediction_results['results_df']
    
    # Pivot predictions by symbol
    pivot_predictions = results_df.pivot(index='date', columns='symbol', values='prediction')
    
    # Get market data
    market_data = {}
    for symbol in datamodule.symbols:
        # Get price data from the datamodule
        symbol_data = datamodule.price_collector.fetch_stock_data(
            symbol, 
            datamodule.start_date, 
            datamodule.end_date
        )
        market_data[symbol] = symbol_data
    
    # Prepare market data for backtester
    backtester = Backtester()
    market_df = backtester.prepare_market_data(market_data)
    
    return pivot_predictions, market_df


def main():
    """Main function for running backtest."""
    parser = argparse.ArgumentParser(description='Run backtest on trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config',
                       help='Path to config directory')
    parser.add_argument('--output-dir', type=str, default='results/backtests',
                       help='Directory to save backtest results')
    parser.add_argument('--initial-capital', type=float, default=100000,
                       help='Initial capital for backtesting')
    parser.add_argument('--position-size', type=str, default='fixed_fraction',
                       choices=['fixed_fraction', 'kelly', 'risk_parity', 'equal_weight'],
                       help='Position sizing method')
    parser.add_argument('--max-positions', type=int, default=10,
                       help='Maximum number of concurrent positions')
    parser.add_argument('--transaction-cost', type=float, default=0.001,
                       help='Transaction cost (as fraction)')
    parser.add_argument('--stop-loss', type=float, default=0.05,
                       help='Stop loss percentage')
    parser.add_argument('--take-profit', type=float, default=0.10,
                       help='Take profit percentage')
    parser.add_argument('--no-short', action='store_true',
                       help='Disable short selling')
    parser.add_argument('--use-trailing-stop', action='store_true',
                       help='Use trailing stop loss')
    parser.add_argument('--rebalance-freq', type=str, default='daily',
                       choices=['daily', 'weekly', 'monthly'],
                       help='Portfolio rebalancing frequency')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level='INFO' if args.verbose else 'WARNING')
    
    # Create output directory
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_dir / f'backtest_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"Starting backtest run: {run_dir}")
    
    try:
        # Load config
        config_path = Path(args.config)
        with open(config_path / 'data_config.yaml', 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Create data module
        logging.info("Setting up data module...")
        datamodule = StockDataModule(
            symbols=data_config['data']['symbols'],
            start_date=data_config['data']['start_date'],
            end_date=data_config['data']['end_date'],
            sequence_length=data_config['data']['sequence_length'],
            prediction_horizon=data_config['data']['prediction_horizon'],
            target_column=data_config['data']['target_column'],
            target_type=data_config['data']['target_type'],
            add_technical_indicators=data_config['data']['add_technical_indicators'],
            add_market_data=data_config['data']['add_market_data'],
            cache_dir=data_config['data']['cache_dir']
        )
        
        # Prepare data
        datamodule.prepare_data()
        datamodule.setup('test')
        
        # Load model
        model = load_checkpoint(args.checkpoint)
        
        # Get predictions
        prediction_results = get_predictions(model, datamodule)
        
        # Save prediction results
        prediction_results['results_df'].to_csv(run_dir / 'predictions.csv', index=False)
        logging.info(f"Predictions saved to {run_dir / 'predictions.csv'}")
        
        # Plot prediction analysis
        plot_prediction_analysis(
            prediction_results['predictions'],
            prediction_results['actuals'],
            pd.DatetimeIndex(prediction_results['dates']),
            prediction_results['symbols'][:5],  # First 5 symbols
            save_path=run_dir / 'prediction_analysis.png'
        )
        
        # Prepare backtest data
        pivot_predictions, market_df = prepare_backtest_data(prediction_results, datamodule)
        
        # Configure backtester
        backtest_config = BacktestConfig(
            initial_capital=args.initial_capital,
            position_size_method=args.position_size,
            max_positions=args.max_positions,
            transaction_cost=args.transaction_cost,
            stop_loss=args.stop_loss if args.stop_loss > 0 else None,
            take_profit=args.take_profit if args.take_profit > 0 else None,
            short_selling_allowed=not args.no_short,
            use_trailing_stop=args.use_trailing_stop,
            rebalance_frequency=args.rebalance_freq
        )
        
        # Run backtest
        logging.info("Running backtest...")
        backtester = Backtester(config=backtest_config, verbose=args.verbose)
        
        # Convert predictions to signals
        signals = backtester.prepare_signals(
            pivot_predictions.values,
            pivot_predictions.index,
            pivot_predictions.columns.tolist(),
            prediction_type=data_config['data']['target_type']
        )
        
        # Run backtest
        results = backtester.run(signals, market_df)
        
        # Save results
        backtester.save_results(run_dir / 'backtest_results.json')
        
        # Generate report
        report = backtester.generate_report()
        report.to_csv(run_dir / 'backtest_report.csv', index=False)
        print("\nBacktest Summary Report:")
        print(report.to_string(index=False))
        
        # Generate visualizations
        logging.info("Generating visualizations...")
        visualizer = BacktestVisualizer(results, save_dir=run_dir)
        visualizer.generate_all_plots()
        
        # Calculate additional metrics
        logging.info("\nCalculating additional metrics...")
        
        # Model accuracy metrics
        predictions = prediction_results['predictions']
        actuals = prediction_results['actuals']
        
        if data_config['data']['target_type'] == 'regression':
            mae = np.mean(np.abs(predictions - actuals))
            rmse = np.sqrt(np.mean((predictions - actuals)**2))
            direction_accuracy = np.mean(np.sign(predictions) == np.sign(actuals))
            
            print(f"\nModel Performance:")
            print(f"  MAE: {mae:.6f}")
            print(f"  RMSE: {rmse:.6f}")
            print(f"  Directional Accuracy: {direction_accuracy:.2%}")
        else:
            accuracy = np.mean(predictions.argmax(axis=-1) == actuals)
            print(f"\nModel Performance:")
            print(f"  Classification Accuracy: {accuracy:.2%}")
        
        # Save config used
        with open(run_dir / 'backtest_config.yaml', 'w') as f:
            yaml.dump(backtest_config.to_dict(), f)
        
        logging.info(f"\nBacktest complete! Results saved to: {run_dir}")
        
        # Print key metrics
        metrics = results['metrics']
        print(f"\nKey Metrics:")
        print(f"  Total Return: {results['portfolio_statistics']['total_return']:.2%}")
        print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        print(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
        
        if 'trade_statistics' in results and results['trade_statistics']:
            trade_stats = results['trade_statistics']
            print(f"\nTrade Statistics:")
            print(f"  Total Trades: {trade_stats['total_trades']}")
            print(f"  Winning Trades: {trade_stats['winning_trades']}")
            print(f"  Average Win: ${trade_stats['avg_win']:.2f}")
            print(f"  Average Loss: ${trade_stats['avg_loss']:.2f}")
            print(f"  Profit Factor: {trade_stats.get('profit_factor', 0):.2f}")
        
    except Exception as e:
        logging.error(f"Backtest failed: {e}")
        raise


if __name__ == '__main__':
    main()