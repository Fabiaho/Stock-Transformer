"""
PyTorch Dataset for stock price time series with proper handling of temporal data.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class StockSequenceDataset(Dataset):
    """
    Dataset for stock price sequences with technical indicators.
    Handles proper temporal windowing and prevents data leakage.
    """
    
    def __init__(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        sequence_length: int = 60,
        prediction_horizon: int = 1,
        target_column: str = 'returns',
        feature_columns: Optional[List[str]] = None,
        scale_features: bool = True,
        target_type: str = 'regression',  # 'regression' or 'classification'
        classification_bins: Optional[List[float]] = None,
        min_sequence_length: Optional[int] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            data: DataFrame or dict of DataFrames with stock data
            sequence_length: Number of time steps to look back
            prediction_horizon: Number of steps ahead to predict
            target_column: Column to predict
            feature_columns: List of feature columns to use (None = all numeric)
            scale_features: Whether to scale features
            target_type: Type of prediction task
            classification_bins: Bins for classification targets
            min_sequence_length: Minimum valid sequence length
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.target_column = target_column
        self.scale_features = scale_features
        self.target_type = target_type
        self.classification_bins = classification_bins or [-0.01, 0, 0.01]  # down, neutral, up
        self.min_sequence_length = min_sequence_length or sequence_length
        
        # Process data
        self.data = self._process_data(data)
        self.feature_columns = feature_columns or self._get_feature_columns()
        
        # Create sequences
        self.sequences, self.targets, self.metadata = self._create_sequences()
        
        # Fit scalers if needed
        if self.scale_features:
            self._fit_scalers()
            
        logger.info(f"Created dataset with {len(self.sequences)} sequences")
        
    def _process_data(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
        """Process and combine data into a single DataFrame."""
        if isinstance(data, dict):
            # Combine multiple stocks
            dfs = []
            for symbol, df in data.items():
                df = df.copy()
                if 'symbol' not in df.columns:
                    df['symbol'] = symbol
                dfs.append(df)
            combined_df = pd.concat(dfs, axis=0)
            combined_df = combined_df.sort_index()
        else:
            combined_df = data.copy()
            
        # Ensure we have the target column
        if self.target_column not in combined_df.columns:
            if self.target_column == 'returns' and 'close' in combined_df.columns:
                combined_df['returns'] = combined_df.groupby('symbol')['close'].pct_change()
            else:
                raise ValueError(f"Target column {self.target_column} not found")
                
        # Drop rows with NaN in target column
        combined_df = combined_df.dropna(subset=[self.target_column])
        
        return combined_df
    
    def _get_feature_columns(self) -> List[str]:
        """Get list of feature columns to use."""
        # Exclude non-feature columns
        exclude_cols = [
            'symbol', 'dividends', 'stock splits', 
            self.target_column, 'open', 'high', 'low', 'close', 'volume'
        ]
        
        # Get numeric columns
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter out excluded columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Add back OHLCV as they're important features
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        feature_cols = ohlcv_cols + feature_cols
        
        return feature_cols
    
    def _create_sequences(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]:
        """Create sequences from the data."""
        sequences = []
        targets = []
        metadata = []
        
        # Group by symbol to create sequences per stock
        for symbol, group_df in self.data.groupby('symbol'):
            group_df = group_df.sort_index()
            
            # Skip if not enough data
            if len(group_df) < self.sequence_length + self.prediction_horizon:
                continue
                
            # Create sequences using sliding window
            for i in range(len(group_df) - self.sequence_length - self.prediction_horizon + 1):
                # Get sequence data
                seq_data = group_df.iloc[i:i + self.sequence_length]
                
                # Get target data
                target_idx = i + self.sequence_length + self.prediction_horizon - 1
                target_data = group_df.iloc[target_idx]
                
                # Extract features
                seq_features = seq_data[self.feature_columns].values
                
                # Skip if too many NaN values
                if np.isnan(seq_features).sum() > len(seq_features) * 0.1:
                    continue
                    
                # Forward fill NaN values
                seq_features = pd.DataFrame(seq_features).fillna(method='ffill').fillna(0).values
                
                # Get target value
                target_value = target_data[self.target_column]
                
                if self.target_type == 'classification':
                    # Convert to classification target
                    target_value = np.digitize(target_value, self.classification_bins) - 1
                    
                sequences.append(seq_features)
                targets.append(target_value)
                
                # Store metadata
                metadata.append({
                    'symbol': symbol,
                    'start_date': seq_data.index[0],
                    'end_date': seq_data.index[-1],
                    'target_date': target_data.name
                })
                
        return sequences, targets, metadata
    
    def _fit_scalers(self):
        """Fit scalers for feature normalization."""
        from sklearn.preprocessing import StandardScaler, RobustScaler
        
        # Concatenate all sequences for fitting
        all_data = np.vstack(self.sequences)
        
        # Fit scaler
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.scaler.fit(all_data)
        
        # Transform sequences
        self.sequences = [self.scaler.transform(seq) for seq in self.sequences]
        
    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sequence and target.
        
        Args:
            idx: Index of the sequence
            
        Returns:
            Dictionary with 'sequence', 'target', and 'metadata'
        """
        sequence = torch.FloatTensor(self.sequences[idx])
        
        if self.target_type == 'classification':
            target = torch.LongTensor([self.targets[idx]])
        else:
            target = torch.FloatTensor([self.targets[idx]])
            
        return {
            'sequence': sequence,
            'target': target,
            'metadata': self.metadata[idx]
        }
    
    def get_feature_names(self) -> List[str]:
        """Get the names of features used."""
        return self.feature_columns
    
    def get_sequence_dates(self, idx: int) -> Tuple[datetime, datetime]:
        """Get start and end dates for a sequence."""
        meta = self.metadata[idx]
        return meta['start_date'], meta['end_date']
    
    def split_by_date(
        self, 
        train_end_date: Union[str, datetime],
        val_end_date: Optional[Union[str, datetime]] = None,
        gap_days: int = 0
    ) -> Tuple['StockSequenceDataset', 'StockSequenceDataset', Optional['StockSequenceDataset']]:
        """
        Split dataset by date for proper temporal validation.
        
        Args:
            train_end_date: End date for training data
            val_end_date: End date for validation data (None = no validation set)
            gap_days: Gap days between train/val and val/test to prevent leakage
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        train_indices = []
        val_indices = []
        test_indices = []
        
        if isinstance(train_end_date, str):
            train_end_date = pd.to_datetime(train_end_date)
        if val_end_date and isinstance(val_end_date, str):
            val_end_date = pd.to_datetime(val_end_date)
            
        # Add gap
        train_cutoff = train_end_date - timedelta(days=gap_days)
        val_cutoff = val_end_date - timedelta(days=gap_days) if val_end_date else None
        
        for i, meta in enumerate(self.metadata):
            target_date = meta['target_date']
            
            if target_date <= train_cutoff:
                train_indices.append(i)
            elif val_cutoff and target_date <= val_cutoff:
                val_indices.append(i)
            else:
                test_indices.append(i)
                
        # Create subset datasets
        train_dataset = self._create_subset(train_indices)
        val_dataset = self._create_subset(val_indices) if val_indices else None
        test_dataset = self._create_subset(test_indices) if test_indices else None
        
        return train_dataset, val_dataset, test_dataset
    
    def _create_subset(self, indices: List[int]) -> 'StockSequenceDataset':
        """Create a subset of the dataset."""
        subset = StockSequenceDataset.__new__(StockSequenceDataset)
        
        # Copy attributes
        subset.sequence_length = self.sequence_length
        subset.prediction_horizon = self.prediction_horizon
        subset.target_column = self.target_column
        subset.scale_features = self.scale_features
        subset.target_type = self.target_type
        subset.classification_bins = self.classification_bins
        subset.feature_columns = self.feature_columns
        
        # Copy scaler if exists
        if hasattr(self, 'scaler'):
            subset.scaler = self.scaler
            
        # Subset data
        subset.sequences = [self.sequences[i] for i in indices]
        subset.targets = [self.targets[i] for i in indices]
        subset.metadata = [self.metadata[i] for i in indices]
        
        return subset


class MultiStockDataset(Dataset):
    """
    Dataset that handles multiple stocks with cross-asset features.
    Useful for models that need market context.
    """
    
    def __init__(
        self,
        stock_data: Dict[str, pd.DataFrame],
        market_data: Optional[pd.DataFrame] = None,
        sequence_length: int = 60,
        prediction_horizon: int = 1,
        target_symbols: Optional[List[str]] = None,
        include_market_features: bool = True,
        correlation_lookback: int = 20
    ):
        """
        Initialize multi-stock dataset.
        
        Args:
            stock_data: Dictionary mapping symbols to DataFrames
            market_data: DataFrame with market indices (SPY, VIX, etc.)
            sequence_length: Sequence length for each stock
            prediction_horizon: Prediction horizon
            target_symbols: Symbols to predict (None = all)
            include_market_features: Whether to include market context
            correlation_lookback: Lookback for correlation features
        """
        self.stock_data = stock_data
        self.market_data = market_data
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.target_symbols = target_symbols or list(stock_data.keys())
        self.include_market_features = include_market_features
        self.correlation_lookback = correlation_lookback
        
        # Align all data to same dates
        self._align_data()
        
        # Create sequences
        self.sequences = self._create_multi_stock_sequences()
        
    def _align_data(self):
        """Align all stock data to the same dates."""
        # Get common dates across all stocks
        all_dates = None
        
        for symbol, df in self.stock_data.items():
            if all_dates is None:
                all_dates = set(df.index)
            else:
                all_dates = all_dates.intersection(set(df.index))
                
        if self.market_data is not None:
            all_dates = all_dates.intersection(set(self.market_data.index))
            
        # Convert to sorted list
        self.common_dates = sorted(list(all_dates))
        
        # Reindex all data
        for symbol in self.stock_data:
            self.stock_data[symbol] = self.stock_data[symbol].loc[self.common_dates]
            
        if self.market_data is not None:
            self.market_data = self.market_data.loc[self.common_dates]
            
    def _create_multi_stock_sequences(self) -> List[Dict]:
        """Create sequences with cross-asset features."""
        sequences = []
        
        # Calculate rolling correlations if needed
        if self.include_market_features:
            correlations = self._calculate_rolling_correlations()
            
        # Create sequences
        for i in range(len(self.common_dates) - self.sequence_length - self.prediction_horizon + 1):
            seq_data = {}
            
            # Get data for each stock
            for symbol in self.target_symbols:
                stock_df = self.stock_data[symbol]
                seq_data[symbol] = {
                    'features': stock_df.iloc[i:i + self.sequence_length].values,
                    'target': stock_df.iloc[i + self.sequence_length + self.prediction_horizon - 1]['returns']
                }
                
            # Add market features
            if self.include_market_features and self.market_data is not None:
                seq_data['market'] = self.market_data.iloc[i:i + self.sequence_length].values
                
            # Add correlation features
            if self.include_market_features and correlations is not None:
                seq_data['correlations'] = correlations[i:i + self.sequence_length]
                
            # Add metadata
            seq_data['metadata'] = {
                'start_date': self.common_dates[i],
                'end_date': self.common_dates[i + self.sequence_length - 1],
                'target_date': self.common_dates[i + self.sequence_length + self.prediction_horizon - 1]
            }
            
            sequences.append(seq_data)
            
        return sequences
    
    def _calculate_rolling_correlations(self) -> Optional[np.ndarray]:
        """Calculate rolling correlations between stocks."""
        if len(self.target_symbols) < 2:
            return None
            
        # Get returns for all stocks
        returns_data = {}
        for symbol in self.target_symbols:
            returns_data[symbol] = self.stock_data[symbol]['returns']
            
        returns_df = pd.DataFrame(returns_data)
        
        # Calculate rolling correlations
        correlations = []
        for i in range(len(returns_df)):
            if i < self.correlation_lookback:
                # Not enough data, use NaN
                corr_matrix = np.full((len(self.target_symbols), len(self.target_symbols)), np.nan)
            else:
                window_data = returns_df.iloc[i-self.correlation_lookback:i]
                corr_matrix = window_data.corr().values
                
            correlations.append(corr_matrix.flatten())
            
        return np.array(correlations)
    
    def __len__(self) -> int:
        """Return number of sequences."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single multi-stock sequence."""
        seq_data = self.sequences[idx]
        
        # Convert to tensors
        output = {}
        for symbol in self.target_symbols:
            output[f'{symbol}_features'] = torch.FloatTensor(seq_data[symbol]['features'])
            output[f'{symbol}_target'] = torch.FloatTensor([seq_data[symbol]['target']])
            
        if 'market' in seq_data:
            output['market_features'] = torch.FloatTensor(seq_data['market'])
            
        if 'correlations' in seq_data:
            output['correlation_features'] = torch.FloatTensor(seq_data['correlations'])
            
        output['metadata'] = seq_data['metadata']
        
        return output