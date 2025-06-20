"""
Tests for dataset classes.
"""

import pytest
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data.dataset import StockSequenceDataset, MultiStockDataset


class TestStockSequenceDataset:
    """Test StockSequenceDataset functionality."""
    
    @pytest.fixture
    def sample_stock_data(self):
        """Create sample stock data for testing."""
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        np.random.seed(42)
        
        data = {}
        for symbol in ['AAPL', 'MSFT']:
            close_prices = 100 + np.cumsum(np.random.randn(200) * 2)
            df = pd.DataFrame({
                'open': close_prices + np.random.randn(200) * 0.5,
                'high': close_prices + np.abs(np.random.randn(200)) * 2,
                'low': close_prices - np.abs(np.random.randn(200)) * 2,
                'close': close_prices,
                'volume': np.random.randint(1000000, 5000000, 200),
                'returns': np.random.randn(200) * 0.02,
                'rsi_14': np.random.uniform(20, 80, 200),
                'macd': np.random.randn(200) * 0.5,
                'symbol': symbol
            }, index=dates)
            data[symbol] = df
        
        return data
    
    def test_initialization(self, sample_stock_data):
        """Test dataset initialization."""
        dataset = StockSequenceDataset(
            data=sample_stock_data,
            sequence_length=60,
            prediction_horizon=1,
            target_column='returns'
        )
        
        assert dataset.sequence_length == 60
        assert dataset.prediction_horizon == 1
        assert dataset.target_column == 'returns'
        assert len(dataset) > 0
    
    def test_sequence_creation(self, sample_stock_data):
        """Test that sequences are created correctly."""
        sequence_length = 30
        prediction_horizon = 5
        
        dataset = StockSequenceDataset(
            data=sample_stock_data,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            target_column='returns'
        )
        
        # Check sequence dimensions
        sample = dataset[0]
        assert sample['sequence'].shape[0] == sequence_length
        assert sample['sequence'].shape[1] == len(dataset.feature_columns)
        assert sample['target'].shape == torch.Size([1])
        
        # Check metadata
        assert 'metadata' in sample
        assert 'symbol' in sample['metadata']
        assert 'start_date' in sample['metadata']
        assert 'end_date' in sample['metadata']
        assert 'target_date' in sample['metadata']
    
    def test_classification_mode(self, sample_stock_data):
        """Test classification target type."""
        dataset = StockSequenceDataset(
            data=sample_stock_data,
            sequence_length=30,
            prediction_horizon=1,
            target_column='returns',
            target_type='classification',
            classification_bins=[-0.01, 0, 0.01]
        )
        
        sample = dataset[0]
        target = sample['target'].item()
        
        # Target should be 0, 1, or 2 (3 classes)
        assert target in [0, 1, 2]
    
    def test_feature_scaling(self, sample_stock_data):
        """Test feature scaling."""
        dataset_scaled = StockSequenceDataset(
            data=sample_stock_data,
            sequence_length=30,
            prediction_horizon=1,
            target_column='returns',
            scale_features=True
        )
        
        dataset_unscaled = StockSequenceDataset(
            data=sample_stock_data,
            sequence_length=30,
            prediction_horizon=1,
            target_column='returns',
            scale_features=False
        )
        
        # Scaled features should have different values
        sample_scaled = dataset_scaled[0]['sequence']
        sample_unscaled = dataset_unscaled[0]['sequence']
        
        assert not torch.allclose(sample_scaled, sample_unscaled)
        
        # Scaled features should have reasonable range
        assert sample_scaled.abs().max() < 100  # Reasonable bound
    
    def test_data_split(self, sample_stock_data):
        """Test temporal data splitting."""
        dataset = StockSequenceDataset(
            data=sample_stock_data,
            sequence_length=30,
            prediction_horizon=1,
            target_column='returns'
        )
        
        train_end = '2023-04-01'
        val_end = '2023-05-01'
        
        train_dataset, val_dataset, test_dataset = dataset.split_by_date(
            train_end_date=train_end,
            val_end_date=val_end,
            gap_days=5
        )
        
        assert len(train_dataset) > 0
        assert len(val_dataset) > 0
        assert len(test_dataset) > 0
        
        # Check no data leakage
        train_dates = [pd.to_datetime(m['target_date']) for m in train_dataset.metadata]
        val_dates = [pd.to_datetime(m['target_date']) for m in val_dataset.metadata]
        test_dates = [pd.to_datetime(m['target_date']) for m in test_dataset.metadata]
        
        assert max(train_dates) < min(val_dates)
        assert max(val_dates) < min(test_dates)
    
    def test_getitem(self, sample_stock_data):
        """Test __getitem__ method."""
        dataset = StockSequenceDataset(
            data=sample_stock_data,
            sequence_length=30,
            prediction_horizon=1,
            target_column='returns'
        )
        
        # Test single item
        item = dataset[0]
        assert isinstance(item, dict)
        assert 'sequence' in item
        assert 'target' in item
        assert 'metadata' in item
        
        assert isinstance(item['sequence'], torch.Tensor)
        assert isinstance(item['target'], torch.Tensor)
        assert isinstance(item['metadata'], dict)
        
        # Test multiple items
        for i in range(min(10, len(dataset))):
            item = dataset[i]
            assert item['sequence'].shape[0] == 30
    
    def test_feature_selection(self, sample_stock_data):
        """Test custom feature selection."""
        feature_columns = ['close', 'volume', 'rsi_14']
        
        dataset = StockSequenceDataset(
            data=sample_stock_data,
            sequence_length=30,
            prediction_horizon=1,
            target_column='returns',
            feature_columns=feature_columns
        )
        
        assert dataset.feature_columns == feature_columns
        assert dataset[0]['sequence'].shape[1] == len(feature_columns)


class TestMultiStockDataset:
    """Test MultiStockDataset functionality."""
    
    @pytest.fixture
    def sample_multi_stock_data(self):
        """Create sample data for multi-stock dataset."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        stock_data = {}
        for symbol in ['AAPL', 'MSFT', 'GOOGL']:
            close_prices = 100 + np.cumsum(np.random.randn(100) * 2)
            df = pd.DataFrame({
                'open': close_prices + np.random.randn(100) * 0.5,
                'high': close_prices + np.abs(np.random.randn(100)) * 2,
                'low': close_prices - np.abs(np.random.randn(100)) * 2,
                'close': close_prices,
                'volume': np.random.randint(1000000, 5000000, 100),
                'returns': np.random.randn(100) * 0.02
            }, index=dates)
            stock_data[symbol] = df
        
        # Market data
        market_data = pd.DataFrame({
            'vix': np.random.uniform(15, 30, 100),
            'spy_returns': np.random.randn(100) * 0.015
        }, index=dates)
        
        return stock_data, market_data
    
    def test_multi_stock_initialization(self, sample_multi_stock_data):
        """Test multi-stock dataset initialization."""
        stock_data, market_data = sample_multi_stock_data
        
        dataset = MultiStockDataset(
            stock_data=stock_data,
            market_data=market_data,
            sequence_length=30,
            prediction_horizon=1,
            include_market_features=True
        )
        
        assert len(dataset) > 0
        assert dataset.sequence_length == 30
        assert dataset.prediction_horizon == 1
    
    def test_multi_stock_getitem(self, sample_multi_stock_data):
        """Test multi-stock dataset __getitem__."""
        stock_data, market_data = sample_multi_stock_data
        
        dataset = MultiStockDataset(
            stock_data=stock_data,
            market_data=market_data,
            sequence_length=30,
            prediction_horizon=1,
            include_market_features=True
        )
        
        item = dataset[0]
        
        # Check that we have features for each stock
        for symbol in stock_data.keys():
            assert f'{symbol}_features' in item
            assert f'{symbol}_target' in item
            assert item[f'{symbol}_features'].shape[0] == 30
        
        # Check market features
        assert 'market_features' in item
        assert item['market_features'].shape[0] == 30
        
        # Check metadata
        assert 'metadata' in item
        assert all(key in item['metadata'] for key in ['start_date', 'end_date', 'target_date'])
    
    def test_correlation_features(self, sample_multi_stock_data):
        """Test correlation feature calculation."""
        stock_data, _ = sample_multi_stock_data
        
        dataset = MultiStockDataset(
            stock_data=stock_data,
            sequence_length=30,
            prediction_horizon=1,
            include_market_features=True,
            correlation_lookback=20
        )
        
        item = dataset[0]
        
        # Should have correlation features
        assert 'correlation_features' in item
        
        # Correlation matrix should be flattened
        n_stocks = len(stock_data)
        expected_corr_features = n_stocks * n_stocks
        assert item['correlation_features'].shape[1] == expected_corr_features


class TestDataLoader:
    """Test DataLoader compatibility."""
    
    def test_dataloader_single_stock(self, sample_stock_data):
        """Test DataLoader with StockSequenceDataset."""
        from torch.utils.data import DataLoader
        
        dataset = StockSequenceDataset(
            data=sample_stock_data,
            sequence_length=30,
            prediction_horizon=1,
            target_column='returns'
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=True,
            num_workers=0  # Use 0 for testing
        )
        
        # Test iteration
        for batch in dataloader:
            assert 'sequence' in batch
            assert 'target' in batch
            assert 'metadata' in batch
            
            # Check batch dimensions
            assert batch['sequence'].dim() == 3  # (batch, seq_len, features)
            assert batch['target'].dim() == 2  # (batch, 1)
            
            # Check batch size
            assert batch['sequence'].shape[0] <= 16
            
            break  # Just test first batch
    
    def test_dataloader_multi_stock(self, sample_multi_stock_data):
        """Test DataLoader with MultiStockDataset."""
        from torch.utils.data import DataLoader
        from src.data.dataset import custom_collate_fn
        
        stock_data, market_data = sample_multi_stock_data
        
        dataset = MultiStockDataset(
            stock_data=stock_data,
            market_data=market_data,
            sequence_length=30,
            prediction_horizon=1
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=8,
            shuffle=False,
            num_workers=0,
            collate_fn=custom_collate_fn
        )
        
        # Test iteration
        for batch in dataloader:
            # Check that all expected keys are present
            for symbol in stock_data.keys():
                assert f'{symbol}_features' in batch
                assert f'{symbol}_target' in batch
            
            assert 'metadata' in batch
            
            # Metadata should be a list
            assert isinstance(batch['metadata'], list)
            assert len(batch['metadata']) <= 8
            
            break  # Just test first batch