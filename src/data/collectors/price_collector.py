"""
Stock price data collector using yfinance with proper error handling and caching.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta
import logging
from pathlib import Path
import time
from functools import wraps
import pickle

logger = logging.getLogger(__name__)


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying failed API calls."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(delay * (attempt + 1))
            return None
        return wrapper
    return decorator


class PriceCollector:
    """Collects and manages historical price data for stocks."""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/raw/price_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    @retry_on_failure(max_retries=3)
    def fetch_stock_data(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch stock data for a single symbol.
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date for data collection
            end_date: End date for data collection
            interval: Data frequency (1d, 1h, 5m, etc.)
            
        Returns:
            DataFrame with OHLCV data
        """
        ticker = yf.Ticker(symbol)
        
        # Convert dates to string format if datetime objects
        if isinstance(start_date, datetime):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')
            
        df = ticker.history(start=start_date, end=end_date, interval=interval)
        
        if df.empty:
            raise ValueError(f"No data found for {symbol}")
            
        # Clean and standardize the data
        df = self._clean_price_data(df, symbol)
        
        return df
    
    def fetch_multiple_stocks(
        self,
        symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1d",
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks with caching support.
        
        Args:
            symbols: List of stock ticker symbols
            start_date: Start date for data collection
            end_date: End date for data collection
            interval: Data frequency
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        results = {}
        
        for symbol in symbols:
            try:
                cache_key = f"{symbol}_{start_date}_{end_date}_{interval}"
                cache_path = self.cache_dir / f"{cache_key}.pkl"
                
                if use_cache and cache_path.exists():
                    logger.info(f"Loading cached data for {symbol}")
                    with open(cache_path, 'rb') as f:
                        results[symbol] = pickle.load(f)
                else:
                    logger.info(f"Fetching data for {symbol}")
                    df = self.fetch_stock_data(symbol, start_date, end_date, interval)
                    results[symbol] = df
                    
                    # Cache the data
                    with open(cache_path, 'wb') as f:
                        pickle.dump(df, f)
                        
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")
                continue
                
        return results
    
    def _clean_price_data(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Clean and standardize price data.
        
        Args:
            df: Raw price DataFrame
            symbol: Stock symbol
            
        Returns:
            Cleaned DataFrame
        """
        # Rename columns to lowercase
        df.columns = df.columns.str.lower()
        
        # Add symbol column
        df['symbol'] = symbol
        
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Remove any duplicate indices
        df = df[~df.index.duplicated(keep='first')]
        
        # Sort by date
        df = df.sort_index()
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Add price movement features
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        return df
    
    def get_adjusted_prices(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime]
    ) -> pd.DataFrame:
        """
        Get split and dividend adjusted prices.
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with adjusted prices
        """
        ticker = yf.Ticker(symbol)
        
        # Get regular price data
        df = self.fetch_stock_data(symbol, start_date, end_date)
        
        # Get splits and dividends
        actions = ticker.actions
        
        if not actions.empty:
            # Filter actions to date range
            actions = actions.loc[start_date:end_date]
            
            # Merge with price data
            df = df.join(actions, how='left')
            df['dividends'] = df['dividends'].fillna(0)
            df['stock splits'] = df['stock splits'].fillna(1)
            
        return df
    
    def get_market_indices(
        self,
        indices: List[str] = ['^GSPC', '^DJI', '^IXIC', '^VIX'],
        start_date: Union[str, datetime] = None,
        end_date: Union[str, datetime] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch major market indices data.
        
        Args:
            indices: List of index symbols
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary of index DataFrames
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        return self.fetch_multiple_stocks(indices, start_date, end_date)
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, any]:
        """
        Validate data quality and return statistics.
        
        Args:
            df: Price DataFrame
            
        Returns:
            Dictionary with quality metrics
        """
        metrics = {
            'total_rows': len(df),
            'date_range': (df.index.min(), df.index.max()),
            'missing_values': df.isnull().sum().to_dict(),
            'zero_volumes': (df['volume'] == 0).sum(),
            'price_anomalies': self._detect_price_anomalies(df),
            'trading_days': len(df),
            'data_gaps': self._find_data_gaps(df)
        }
        
        return metrics
    
    def _detect_price_anomalies(self, df: pd.DataFrame, threshold: float = 0.5) -> int:
        """Detect unusual price movements."""
        if 'returns' not in df.columns:
            return 0
            
        # Count returns greater than threshold (50% by default)
        anomalies = (df['returns'].abs() > threshold).sum()
        return int(anomalies)
    
    def _find_data_gaps(self, df: pd.DataFrame) -> List[tuple]:
        """Find gaps in trading days."""
        # Expected trading days (excluding weekends)
        expected_days = pd.bdate_range(start=df.index.min(), end=df.index.max())
        actual_days = pd.to_datetime(df.index)
        
        missing_days = expected_days.difference(actual_days)
        
        # Group consecutive missing days
        gaps = []
        if len(missing_days) > 0:
            gap_start = missing_days[0]
            gap_end = missing_days[0]
            
            for i in range(1, len(missing_days)):
                if missing_days[i] - missing_days[i-1] > timedelta(days=1):
                    gaps.append((gap_start, gap_end))
                    gap_start = missing_days[i]
                gap_end = missing_days[i]
                
            gaps.append((gap_start, gap_end))
            
        return gaps