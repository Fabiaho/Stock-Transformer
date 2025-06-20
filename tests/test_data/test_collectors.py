"""
Tests for data collectors.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

from src.data.collectors.price_collector import PriceCollector


class TestPriceCollector:
    """Test price data collection functionality."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def collector(self, temp_cache_dir):
        """Create price collector instance."""
        return PriceCollector(cache_dir=temp_cache_dir)
    
    def test_initialization(self, collector, temp_cache_dir):
        """Test collector initialization."""
        assert collector.cache_dir == temp_cache_dir
        assert temp_cache_dir.exists()
    
    def test_fetch_single_stock(self, collector):
        """Test fetching data for a single stock."""
        # Use a short recent period to ensure data availability
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        df = collector.fetch_stock_data(
            'AAPL',
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        # Basic assertions
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])
        assert 'returns' in df.columns
        assert 'log_returns' in df.columns
        assert df.index.is_monotonic_increasing
    
    def test_fetch_multiple_stocks(self, collector):
        """Test fetching data for multiple stocks."""
        symbols = ['AAPL', 'MSFT']
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        data = collector.fetch_multiple_stocks(
            symbols,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        assert isinstance(data, dict)
        assert len(data) == len(symbols)
        assert all(symbol in data for symbol in symbols)
        assert all(isinstance(df, pd.DataFrame) for df in data.values())
    
    def test_caching(self, collector):
        """Test that caching works correctly."""
        symbol = 'AAPL'
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # First fetch - should hit API
        data1 = collector.fetch_multiple_stocks(
            [symbol],
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            use_cache=True
        )
        
        # Check cache file exists
        cache_files = list(collector.cache_dir.glob('*.pkl'))
        assert len(cache_files) > 0
        
        # Second fetch - should use cache
        data2 = collector.fetch_multiple_stocks(
            [symbol],
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            use_cache=True
        )
        
        # Data should be identical
        pd.testing.assert_frame_equal(data1[symbol], data2[symbol])
    
    def test_data_cleaning(self, collector):
        """Test data cleaning functionality."""
        # Create sample data with issues
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        df = pd.DataFrame({
            'open': [100, 101, np.nan, 103, 104, 105, 106, 107, 108, 109],
            'high': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            'close': [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
            'volume': [1000000, 1100000, 1200000, 0, 1400000, 1500000, 1600000, 1700000, 1800000, 1900000]
        }, index=dates)
        
        # Clean data
        cleaned = collector._clean_price_data(df, 'TEST')
        
        # Check cleaning results
        assert cleaned['open'].isna().sum() == 0  # NaN should be filled
        assert 'symbol' in cleaned.columns
        assert cleaned['symbol'].iloc[0] == 'TEST'
        assert 'returns' in cleaned.columns
        assert 'high_low_ratio' in cleaned.columns
    
    def test_data_quality_validation(self, collector):
        """Test data quality validation."""
        # Fetch real data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=100)
        
        df = collector.fetch_stock_data(
            'AAPL',
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        # Validate quality
        metrics = collector.validate_data_quality(df)
        
        assert 'total_rows' in metrics
        assert 'date_range' in metrics
        assert 'missing_values' in metrics
        assert 'zero_volumes' in metrics
        assert 'price_anomalies' in metrics
        assert 'trading_days' in metrics
        assert 'data_gaps' in metrics
        
        # Basic sanity checks
        assert metrics['total_rows'] > 0
        assert metrics['trading_days'] > 0
        assert isinstance(metrics['data_gaps'], list)
    
    def test_error_handling(self, collector):
        """Test error handling for invalid inputs."""
        # Invalid symbol
        with pytest.raises(Exception):
            collector.fetch_stock_data(
                'INVALID_SYMBOL_XYZ',
                '2023-01-01',
                '2023-12-31'
            )
        
        # Invalid date range
        with pytest.raises(Exception):
            collector.fetch_stock_data(
                'AAPL',
                '2030-01-01',  # Future date
                '2030-12-31'
            )
    
    def test_market_indices(self, collector):
        """Test fetching market indices."""
        indices_data = collector.get_market_indices()
        
        assert isinstance(indices_data, dict)
        assert len(indices_data) > 0
        
        # Check that major indices are present
        expected_indices = ['^GSPC', '^VIX']
        for index in expected_indices:
            if index in indices_data:  # Some indices might not be available
                assert isinstance(indices_data[index], pd.DataFrame)
                assert len(indices_data[index]) > 0


class TestTechnicalIndicators:
    """Test technical indicators calculation."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample price data for testing."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # Generate realistic price data
        close_prices = 100 + np.cumsum(np.random.randn(100) * 2)
        
        df = pd.DataFrame({
            'open': close_prices + np.random.randn(100) * 0.5,
            'high': close_prices + np.abs(np.random.randn(100)) * 2,
            'low': close_prices - np.abs(np.random.randn(100)) * 2,
            'close': close_prices,
            'volume': np.random.randint(1000000, 5000000, 100)
        }, index=dates)
        
        # Ensure high >= close >= low
        df['high'] = df[['high', 'close']].max(axis=1)
        df['low'] = df[['low', 'close']].min(axis=1)
        
        return df
    
    def test_moving_averages(self, sample_data):
        """Test moving average calculations."""
        from src.data.processors.technical_indicators import TechnicalIndicators
        
        indicators = TechnicalIndicators()
        df = indicators.add_moving_averages(sample_data)
        
        # Check that MA columns exist
        for period in [5, 10, 20, 50]:
            assert f'sma_{period}' in df.columns
            assert f'ema_{period}' in df.columns
            
        # Check calculations
        assert df['sma_5'].iloc[4] == pytest.approx(df['close'].iloc[:5].mean(), rel=1e-5)
        
        # Check that MAs have appropriate NaN values at start
        assert df['sma_20'].iloc[:19].isna().all()
        assert df['sma_20'].iloc[19:].notna().all()
    
    def test_momentum_indicators(self, sample_data):
        """Test momentum indicators."""
        from src.data.processors.technical_indicators import TechnicalIndicators
        
        indicators = TechnicalIndicators()
        df = indicators.add_momentum_indicators(sample_data)
        
        # Check RSI
        assert 'rsi_14' in df.columns
        assert 'rsi_30' in df.columns
        
        # RSI should be between 0 and 100
        rsi_values = df['rsi_14'].dropna()
        assert (rsi_values >= 0).all()
        assert (rsi_values <= 100).all()
        
        # Check MACD
        assert 'macd' in df.columns
        assert 'macd_signal' in df.columns
        assert 'macd_hist' in df.columns
        
        # Check other momentum indicators
        expected_indicators = ['slowk', 'slowd', 'williams_r', 'roc_10', 'cci']
        for indicator in expected_indicators:
            assert indicator in df.columns
    
    def test_volatility_indicators(self, sample_data):
        """Test volatility indicators."""
        from src.data.processors.technical_indicators import TechnicalIndicators
        
        indicators = TechnicalIndicators()
        df = indicators.add_volatility_indicators(sample_data)
        
        # Check Bollinger Bands
        assert 'bb_lower' in df.columns
        assert 'bb_middle' in df.columns
        assert 'bb_upper' in df.columns
        assert 'bb_pct' in df.