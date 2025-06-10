"""
Fixed PyTorch Lightning DataModule for stock price prediction.
Addresses the issue with MultiStockDataset format and proper data splitting.
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Union, Tuple
from pathlib import Path
import logging
from datetime import datetime, timedelta

from src.data.collectors.price_collector import PriceCollector
from src.data.processors.technical_indicators import TechnicalIndicators
from src.data.dataset import StockSequenceDataset, MultiStockDataset, custom_collate_fn

logger = logging.getLogger(__name__)


class StockDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for stock price data.
    Handles data collection, preprocessing, and loading.
    """
    
    def __init__(
        self,
        symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        sequence_length: int = 60,
        prediction_horizon: int = 1,
        batch_size: int = 32,
        num_workers: int = 4,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        target_column: str = 'returns',
        target_type: str = 'regression',
        feature_columns: Optional[List[str]] = None,
        add_technical_indicators: bool = True,
        add_market_data: bool = False,  # Changed default to False for simplicity
        market_indices: List[str] = ['^GSPC', '^VIX'],
        scale_features: bool = True,
        cache_dir: Optional[Path] = None,
        gap_days: int = 5,
        use_weighted_sampling: bool = False,
        classification_bins: Optional[List[float]] = None
    ):
        """
        Initialize the data module.
        """
        super().__init__()
        
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split
        self.target_column = target_column
        self.target_type = target_type
        self.feature_columns = feature_columns
        self.add_technical_indicators = add_technical_indicators
        self.add_market_data = add_market_data
        self.market_indices = market_indices
        self.scale_features = scale_features
        self.cache_dir = cache_dir
        self.gap_days = gap_days
        self.use_weighted_sampling = use_weighted_sampling
        self.classification_bins = classification_bins
        
        # Initialize components
        self.price_collector = PriceCollector(cache_dir=cache_dir)
        self.technical_indicators = TechnicalIndicators()
        
        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def prepare_data(self):
        """Download data if needed. Called only on rank 0 in distributed training."""
        logger.info(f"Preparing data for {len(self.symbols)} symbols")
        
        # Download stock data
        _ = self.price_collector.fetch_multiple_stocks(
            self.symbols,
            self.start_date,
            self.end_date,
            use_cache=True
        )
        
        # Download market data if needed
        if self.add_market_data:
            _ = self.price_collector.fetch_multiple_stocks(
                self.market_indices,
                self.start_date,
                self.end_date,
                use_cache=True
            )
            
    def setup(self, stage: Optional[str] = None):
        """
        Set up datasets for training/validation/testing.
        """
        # Load all data
        stock_data = self.price_collector.fetch_multiple_stocks(
            self.symbols,
            self.start_date,
            self.end_date,
            use_cache=True
        )
        
        # Add technical indicators
        if self.add_technical_indicators:
            logger.info("Adding technical indicators")
            for symbol in stock_data:
                stock_data[symbol] = self.technical_indicators.add_all_indicators(
                    stock_data[symbol]
                )
        
        # For now, always use StockSequenceDataset for consistency
        # You can enable MultiStockDataset later after fixing the model compatibility
        full_dataset = StockSequenceDataset(
            data=stock_data,
            sequence_length=self.sequence_length,
            prediction_horizon=self.prediction_horizon,
            target_column=self.target_column,
            feature_columns=self.feature_columns,
            scale_features=self.scale_features,
            target_type=self.target_type,
            classification_bins=self.classification_bins
        )
        
        # Calculate split dates
        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        total_days = len(date_range)
        
        train_days = int(total_days * self.train_val_test_split[0])
        val_days = int(total_days * self.train_val_test_split[1])
        
        train_end_date = date_range[train_days]
        val_end_date = date_range[train_days + val_days]
        
        # Split dataset by date
        self.train_dataset, self.val_dataset, self.test_dataset = full_dataset.split_by_date(
            train_end_date=train_end_date,
            val_end_date=val_end_date,
            gap_days=self.gap_days
        )
            
        logger.info(f"Dataset sizes - Train: {len(self.train_dataset)}, "
                   f"Val: {len(self.val_dataset) if self.val_dataset else 0}, "
                   f"Test: {len(self.test_dataset) if self.test_dataset else 0}")
        
    def train_dataloader(self) -> DataLoader:
        """Create training DataLoader."""
        sampler = None
        
        if self.use_weighted_sampling and self.target_type == 'classification':
            sampler = self._create_weighted_sampler(self.train_dataset)
            
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
    def val_dataloader(self) -> DataLoader:
        """Create validation DataLoader."""
        if self.val_dataset is None:
            return None
            
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
    def test_dataloader(self) -> DataLoader:
        """Create test DataLoader."""
        if self.test_dataset is None:
            return None
            
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
    def _create_weighted_sampler(self, dataset) -> WeightedRandomSampler:
        """Create weighted sampler for imbalanced classes."""
        targets = []
        for i in range(len(dataset)):
            targets.append(dataset[i]['target'].item())
            
        class_counts = np.bincount(targets)
        class_weights = 1.0 / class_counts
        weights = [class_weights[t] for t in targets]
        
        return WeightedRandomSampler(
            weights=weights,
            num_samples=len(dataset),
            replacement=True
        )
        
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        if hasattr(self.train_dataset, 'get_feature_names'):
            return self.train_dataset.get_feature_names()
        return []
        
    def get_num_features(self) -> int:
        """Get number of features."""
        if self.train_dataset is not None:
            sample = self.train_dataset[0]
            if isinstance(sample, dict) and 'sequence' in sample:
                return sample['sequence'].shape[-1]
        return 0
        
    def get_num_classes(self) -> int:
        """Get number of classes for classification."""
        if self.target_type == 'classification' and hasattr(self.train_dataset, 'classification_bins'):
            return len(self.train_dataset.classification_bins) - 1
        return 1  # Regression