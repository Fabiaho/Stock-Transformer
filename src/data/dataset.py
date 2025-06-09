"""
Fixed Dataset classes with proper metadata handling for DataLoader compatibility.
"""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def custom_collate_fn(batch):
    """Custom collate function that handles metadata with timestamps."""
    if not batch:
        return {}
    
    # Separate tensors from metadata
    collated = {}
    
    for key in batch[0].keys():
        if key == 'metadata':
            # For metadata, just keep as list (don't try to stack)
            collated[key] = [item[key] for item in batch]
        else:
            # For tensors, use default stacking
            values = [item[key] for item in batch]
            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values)
            else:
                collated[key] = values
    
    return collated


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
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.target_column = target_column
        self.scale_features = scale_features
        self.target_type = target_type
        self.classification_bins = classification_bins or [-0.01, 0, 0.01]
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
        exclude_cols = [
            'symbol', 'dividends', 'stock splits', 
            self.target_column, 'open', 'high', 'low', 'close', 'volume'
        ]
        
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Add back OHLCV as important features
        ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        feature_cols = ohlcv_cols + feature_cols
        
        return feature_cols
    
    def _create_sequences(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]:
        """Create sequences from the data."""
        sequences = []
        targets = []
        metadata = []
        
        for symbol, group_df in self.data.groupby('symbol'):
            group_df = group_df.sort_index()
            
            if len(group_df) < self.sequence_length + self.prediction_horizon:
                continue
                
            for i in range(len(group_df) - self.sequence_length - self.prediction_horizon + 1):
                seq_data = group_df.iloc[i:i + self.sequence_length]
                target_idx = i + self.sequence_length + self.prediction_horizon - 1
                target_data = group_df.iloc[target_idx]
                
                seq_features = seq_data[self.feature_columns].values
                
                if np.isnan(seq_features).sum() > len(seq_features) * 0.1:
                    continue
                    
                seq_features = pd.DataFrame(seq_features).ffill().fillna(0).values
                target_value = target_data[self.target_column]
                
                if self.target_type == 'classification':
                    target_value = np.digitize(target_value, self.classification_bins) - 1
                    
                sequences.append(seq_features)
                targets.append(target_value)
                
                # Convert timestamps to strings for metadata
                metadata.append({
                    'symbol': symbol,
                    'start_date': seq_data.index[0].strftime('%Y-%m-%d'),
                    'end_date': seq_data.index[-1].strftime('%Y-%m-%d'),
                    'target_date': target_data.name.strftime('%Y-%m-%d')
                })
                
        return sequences, targets, metadata
    
    def _fit_scalers(self):
        """Fit scalers for feature normalization."""
        from sklearn.preprocessing import RobustScaler
        
        all_data = np.vstack(self.sequences)
        self.scaler = RobustScaler()
        self.scaler.fit(all_data)
        self.sequences = [self.scaler.transform(seq) for seq in self.sequences]
        
    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sequence and target."""
        sequence = torch.FloatTensor(self.sequences[idx])
        
        if self.target_type == 'classification':
            target = torch.LongTensor([self.targets[idx]])
        else:
            target = torch.FloatTensor([self.targets[idx]])
            
        return {
            'sequence': sequence,
            'target': target,
            'metadata': self.metadata[idx]  # Now contains strings, not Timestamps
        }
    
    def get_feature_names(self) -> List[str]:
        """Get the names of features used."""
        return self.feature_columns
    
    def split_by_date(
        self, 
        train_end_date: Union[str, datetime],
        val_end_date: Optional[Union[str, datetime]] = None,
        gap_days: int = 0
    ) -> Tuple['StockSequenceDataset', 'StockSequenceDataset', Optional['StockSequenceDataset']]:
        """Split dataset by date for proper temporal validation."""
        train_indices = []
        val_indices = []
        test_indices = []
        
        if isinstance(train_end_date, str):
            train_end_date = pd.to_datetime(train_end_date)
        if val_end_date and isinstance(val_end_date, str):
            val_end_date = pd.to_datetime(val_end_date)
            
        train_cutoff = train_end_date - timedelta(days=gap_days)
        val_cutoff = val_end_date - timedelta(days=gap_days) if val_end_date else None
        
        for i, meta in enumerate(self.metadata):
            # Convert string back to datetime for comparison
            target_date = pd.to_datetime(meta['target_date'])
            
            if target_date <= train_cutoff:
                train_indices.append(i)
            elif val_cutoff and target_date <= val_cutoff:
                val_indices.append(i)
            else:
                test_indices.append(i)
                
        train_dataset = self._create_subset(train_indices)
        val_dataset = self._create_subset(val_indices) if val_indices else None
        test_dataset = self._create_subset(test_indices) if test_indices else None
        
        return train_dataset, val_dataset, test_dataset
    
    def _create_subset(self, indices: List[int]) -> 'StockSequenceDataset':
        """Create a subset of the dataset."""
        subset = StockSequenceDataset.__new__(StockSequenceDataset)
        
        subset.sequence_length = self.sequence_length
        subset.prediction_horizon = self.prediction_horizon
        subset.target_column = self.target_column
        subset.scale_features = self.scale_features
        subset.target_type = self.target_type
        subset.classification_bins = self.classification_bins
        subset.feature_columns = self.feature_columns
        
        if hasattr(self, 'scaler'):
            subset.scaler = self.scaler
            
        subset.sequences = [self.sequences[i] for i in indices]
        subset.targets = [self.targets[i] for i in indices]
        subset.metadata = [self.metadata[i] for i in indices]
        
        return subset


class MultiStockDataset(Dataset):
    """Dataset that handles multiple stocks with cross-asset features."""
    
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
        """Initialize multi-stock dataset."""
        self.stock_data = stock_data
        self.market_data = market_data
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.target_symbols = target_symbols or list(stock_data.keys())
        self.include_market_features = include_market_features
        self.correlation_lookback = correlation_lookback
        
        self._clean_data()
        self.feature_columns = self._get_feature_columns()
        self._align_data()
        self.sequences = self._create_multi_stock_sequences()
        
    def _clean_data(self):
        """Clean data to ensure numeric types and handle NaN values."""
        for symbol in self.stock_data:
            df = self.stock_data[symbol].copy()
            
            for col in df.columns:
                if col != 'symbol':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            self.stock_data[symbol] = df
            
        if self.market_data is not None:
            for col in self.market_data.columns:
                self.market_data[col] = pd.to_numeric(self.market_data[col], errors='coerce')
            
            self.market_data = self.market_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    def _get_feature_columns(self) -> Dict[str, List[str]]:
        """Get feature columns for each stock."""
        feature_cols = {}
        exclude_cols = ['symbol', 'dividends', 'stock splits']
        
        for symbol in self.target_symbols:
            df = self.stock_data[symbol]
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            features = [col for col in numeric_cols if col not in exclude_cols]
            
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col in df.columns and col not in features:
                    features.append(col)
                    
            feature_cols[symbol] = features
            
        return feature_cols
        
    def _align_data(self):
        """Align all stock data to the same dates."""
        all_dates = None
        
        for symbol, df in self.stock_data.items():
            if all_dates is None:
                all_dates = set(df.index)
            else:
                all_dates = all_dates.intersection(set(df.index))
                
        if self.market_data is not None:
            all_dates = all_dates.intersection(set(self.market_data.index))
            
        self.common_dates = sorted(list(all_dates))
        
        for symbol in self.stock_data:
            self.stock_data[symbol] = self.stock_data[symbol].loc[self.common_dates]
            
        if self.market_data is not None:
            self.market_data = self.market_data.loc[self.common_dates]
            
    def _create_multi_stock_sequences(self) -> List[Dict]:
        """Create sequences with cross-asset features."""
        sequences = []
        
        correlations = None
        if self.include_market_features and len(self.target_symbols) > 1:
            correlations = self._calculate_rolling_correlations()
            
        total_length = len(self.common_dates)
        max_idx = total_length - self.sequence_length - self.prediction_horizon + 1
        
        for i in range(max_idx):
            seq_data = {}
            
            for symbol in self.target_symbols:
                stock_df = self.stock_data[symbol]
                
                feature_data = stock_df[self.feature_columns[symbol]].iloc[i:i + self.sequence_length]
                features_array = feature_data.values.astype(np.float64)
                
                target_idx = i + self.sequence_length + self.prediction_horizon - 1
                if 'returns' in stock_df.columns:
                    target_value = stock_df['returns'].iloc[target_idx]
                else:
                    current_close = stock_df['close'].iloc[target_idx]
                    prev_close = stock_df['close'].iloc[target_idx - 1]
                    target_value = (current_close - prev_close) / prev_close if prev_close != 0 else 0
                
                seq_data[symbol] = {
                    'features': features_array,
                    'target': float(target_value)
                }
                
            if self.include_market_features and self.market_data is not None:
                market_features = self.market_data.iloc[i:i + self.sequence_length].values.astype(np.float64)
                seq_data['market'] = market_features
                
            if correlations is not None:
                corr_features = correlations[i:i + self.sequence_length].astype(np.float64)
                seq_data['correlations'] = corr_features
                
            # Convert timestamps to strings for metadata
            seq_data['metadata'] = {
                'start_date': self.common_dates[i].strftime('%Y-%m-%d'),
                'end_date': self.common_dates[i + self.sequence_length - 1].strftime('%Y-%m-%d'),
                'target_date': self.common_dates[i + self.sequence_length + self.prediction_horizon - 1].strftime('%Y-%m-%d')
            }
            
            sequences.append(seq_data)
            
        return sequences
    
    def _calculate_rolling_correlations(self) -> Optional[np.ndarray]:
        """Calculate rolling correlations between stocks."""
        if len(self.target_symbols) < 2:
            return None
            
        returns_data = {}
        for symbol in self.target_symbols:
            if 'returns' in self.stock_data[symbol].columns:
                returns_data[symbol] = self.stock_data[symbol]['returns']
            else:
                close_prices = self.stock_data[symbol]['close']
                returns_data[symbol] = close_prices.pct_change()
                
        returns_df = pd.DataFrame(returns_data)
        
        correlations = []
        for i in range(len(returns_df)):
            if i < self.correlation_lookback:
                corr_matrix = np.zeros((len(self.target_symbols), len(self.target_symbols)))
            else:
                window_data = returns_df.iloc[i-self.correlation_lookback:i]
                corr_matrix = window_data.corr().fillna(0).values
                
            correlations.append(corr_matrix.flatten())
            
        return np.array(correlations, dtype=np.float64)
    
    def __len__(self) -> int:
        """Return number of sequences."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single multi-stock sequence."""
        seq_data = self.sequences[idx]
        
        output = {}
        for symbol in self.target_symbols:
            features_array = np.ascontiguousarray(seq_data[symbol]['features'], dtype=np.float64)
            output[f'{symbol}_features'] = torch.from_numpy(features_array).float()
            output[f'{symbol}_target'] = torch.tensor([seq_data[symbol]['target']], dtype=torch.float32)
            
        if 'market' in seq_data:
            market_array = np.ascontiguousarray(seq_data['market'], dtype=np.float64)
            output['market_features'] = torch.from_numpy(market_array).float()
            
        if 'correlations' in seq_data:
            corr_array = np.ascontiguousarray(seq_data['correlations'], dtype=np.float64)
            output['correlation_features'] = torch.from_numpy(corr_array).float()
            
        output['metadata'] = seq_data['metadata']  # Now contains strings, not Timestamps
        
        return output
    
    def get_feature_names(self) -> Dict[str, List[str]]:
        """Get feature names for each symbol."""
        return self.feature_columns
    
    def get_num_features(self) -> Dict[str, int]:
        """Get number of features for each symbol."""
        return {symbol: len(features) for symbol, features in self.feature_columns.items()}