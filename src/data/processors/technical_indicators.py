"""
Technical indicators calculation using pandas-ta (pure Python, no C dependencies).
Fixed version with proper error handling and dynamic column detection.
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Optional, Union, Tuple
from functools import wraps
import warnings


def validate_price_data(func):
    """Decorator to validate price data before calculating indicators."""
    @wraps(func)
    def wrapper(df: pd.DataFrame, *args, **kwargs):
        required_cols = {'high', 'low', 'close', 'volume'}
        df_cols = set(df.columns.str.lower())
        
        if not required_cols.issubset(df_cols):
            missing = required_cols - df_cols
            raise ValueError(f"Missing required columns: {missing}")
            
        return func(df, *args, **kwargs)
    return wrapper


class TechnicalIndicators:
    """Calculate various technical indicators for stock data using pandas-ta."""
    
    @staticmethod
    @validate_price_data
    def add_moving_averages(
        df: pd.DataFrame,
        periods: list = [5, 10, 20, 50, 200]
    ) -> pd.DataFrame:
        """
        Add simple and exponential moving averages.
        
        Args:
            df: Price DataFrame
            periods: List of periods for MA calculation
            
        Returns:
            DataFrame with MA columns added
        """
        df = df.copy()
        
        for period in periods:
            try:
                # Simple Moving Average
                df[f'sma_{period}'] = ta.sma(df['close'], length=period)
                
                # Exponential Moving Average
                df[f'ema_{period}'] = ta.ema(df['close'], length=period)
                
                # Volume Weighted Average Price (simplified)
                if period <= 50:  # Only for shorter periods to avoid computation issues
                    typical_price = (df['high'] + df['low'] + df['close']) / 3
                    vwap = (typical_price * df['volume']).rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
                    df[f'vwap_{period}'] = vwap
                    
            except Exception as e:
                warnings.warn(f"Failed to calculate MA for period {period}: {e}")
                continue
                
        return df
    
    @staticmethod
    @validate_price_data
    def add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum-based indicators.
        
        Args:
            df: Price DataFrame
            
        Returns:
            DataFrame with momentum indicators
        """
        df = df.copy()
        
        try:
            # RSI (Relative Strength Index)
            df['rsi_14'] = ta.rsi(df['close'], length=14)
            df['rsi_30'] = ta.rsi(df['close'], length=30)
        except Exception as e:
            warnings.warn(f"Failed to calculate RSI: {e}")
        
        try:
            # MACD
            macd_result = ta.macd(df['close'], fast=12, slow=26, signal=9)
            if macd_result is not None and not macd_result.empty:
                # Get the actual column names
                macd_cols = macd_result.columns.tolist()
                if len(macd_cols) >= 3:
                    df['macd'] = macd_result.iloc[:, 0]  # MACD line
                    df['macd_signal'] = macd_result.iloc[:, 1]  # Signal line
                    df['macd_hist'] = macd_result.iloc[:, 2]  # Histogram
        except Exception as e:
            warnings.warn(f"Failed to calculate MACD: {e}")
        
        try:
            # Stochastic Oscillator
            stoch_result = ta.stoch(df['high'], df['low'], df['close'])
            if stoch_result is not None and not stoch_result.empty:
                stoch_cols = stoch_result.columns.tolist()
                if len(stoch_cols) >= 2:
                    df['slowk'] = stoch_result.iloc[:, 0]
                    df['slowd'] = stoch_result.iloc[:, 1]
        except Exception as e:
            warnings.warn(f"Failed to calculate Stochastic: {e}")
        
        try:
            # Williams %R
            df['williams_r'] = ta.willr(df['high'], df['low'], df['close'], length=14)
        except Exception as e:
            warnings.warn(f"Failed to calculate Williams %R: {e}")
        
        try:
            # Rate of Change
            df['roc_10'] = ta.roc(df['close'], length=10)
            df['roc_20'] = ta.roc(df['close'], length=20)
        except Exception as e:
            warnings.warn(f"Failed to calculate ROC: {e}")
        
        try:
            # Commodity Channel Index
            df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=14)
        except Exception as e:
            warnings.warn(f"Failed to calculate CCI: {e}")
        
        return df
    
    @staticmethod
    @validate_price_data
    def add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility-based indicators.
        
        Args:
            df: Price DataFrame
            
        Returns:
            DataFrame with volatility indicators
        """
        df = df.copy()
        
        try:
            # Bollinger Bands
            bbands_result = ta.bbands(df['close'], length=20, std=2)
            if bbands_result is not None and not bbands_result.empty:
                # Get actual column names and map them
                bb_cols = bbands_result.columns.tolist()
                for col in bb_cols:
                    if 'BBL' in col:
                        df['bb_lower'] = bbands_result[col]
                    elif 'BBM' in col:
                        df['bb_middle'] = bbands_result[col]
                    elif 'BBU' in col:
                        df['bb_upper'] = bbands_result[col]
                    elif 'BBB' in col:
                        df['bb_width'] = bbands_result[col]
                    elif 'BBP' in col:
                        df['bb_pct'] = bbands_result[col]
        except Exception as e:
            warnings.warn(f"Failed to calculate Bollinger Bands: {e}")
        
        try:
            # Average True Range
            df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            df['atr_20'] = ta.atr(df['high'], df['low'], df['close'], length=20)
        except Exception as e:
            warnings.warn(f"Failed to calculate ATR: {e}")
        
        try:
            # Normalized ATR
            df['natr_14'] = ta.natr(df['high'], df['low'], df['close'], length=14)
        except Exception as e:
            warnings.warn(f"Failed to calculate NATR: {e}")
        
        try:
            # Standard deviation
            df['std_20'] = df['close'].rolling(window=20).std()
            df['std_50'] = df['close'].rolling(window=50).std()
        except Exception as e:
            warnings.warn(f"Failed to calculate STD: {e}")
        
        try:
            # Keltner Channels - with dynamic column detection
            kc_result = ta.kc(df['high'], df['low'], df['close'], length=20, scalar=2)
            if kc_result is not None and not kc_result.empty:
                kc_cols = kc_result.columns.tolist()
                for col in kc_cols:
                    if 'KCL' in col:
                        df['kc_lower'] = kc_result[col]
                    elif 'KCU' in col:
                        df['kc_upper'] = kc_result[col]
                        
                # Calculate percentage if both bounds exist
                if 'kc_lower' in df.columns and 'kc_upper' in df.columns:
                    df['kc_pct'] = (df['close'] - df['kc_lower']) / (df['kc_upper'] - df['kc_lower'])
        except Exception as e:
            warnings.warn(f"Failed to calculate Keltner Channels: {e}")
        
        return df
    
    @staticmethod
    @validate_price_data
    def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based indicators.
        
        Args:
            df: Price DataFrame
            
        Returns:
            DataFrame with volume indicators
        """
        df = df.copy()
        
        try:
            # On Balance Volume
            df['obv'] = ta.obv(df['close'], df['volume'])
        except Exception as e:
            warnings.warn(f"Failed to calculate OBV: {e}")
        
        try:
            # Accumulation/Distribution
            df['ad'] = ta.ad(df['high'], df['low'], df['close'], df['volume'])
        except Exception as e:
            warnings.warn(f"Failed to calculate A/D: {e}")
        
        try:
            # Chaikin Money Flow
            df['cmf'] = ta.cmf(df['high'], df['low'], df['close'], df['volume'])
        except Exception as e:
            warnings.warn(f"Failed to calculate CMF: {e}")
        
        try:
            # Money Flow Index
            df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
        except Exception as e:
            warnings.warn(f"Failed to calculate MFI: {e}")
        
        try:
            # Volume Rate of Change
            df['vroc'] = ta.roc(df['volume'], length=10)
        except Exception as e:
            warnings.warn(f"Failed to calculate VROC: {e}")
        
        try:
            # Force Index
            df['force_index'] = (df['close'] - df['close'].shift(1)) * df['volume']
            df['force_index_ema'] = ta.ema(df['force_index'], length=13)
        except Exception as e:
            warnings.warn(f"Failed to calculate Force Index: {e}")
        
        try:
            # Volume moving averages
            df['volume_sma_10'] = ta.sma(df['volume'], length=10)
            df['volume_sma_20'] = ta.sma(df['volume'], length=20)
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        except Exception as e:
            warnings.warn(f"Failed to calculate volume averages: {e}")
        
        return df
    
    @staticmethod
    @validate_price_data
    def add_pattern_recognition(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add candlestick pattern recognition using pandas-ta.
        
        Args:
            df: Price DataFrame
            
        Returns:
            DataFrame with pattern indicators
        """
        df = df.copy()
        
        try:
            # Get all candlestick patterns
            df_copy = df.copy()
            df_copy.ta.cdl_pattern(name="all", append=True)
            
            # Get pattern columns
            pattern_cols = [col for col in df_copy.columns if col.startswith('CDL_')]
            
            if pattern_cols:
                # Add pattern columns to main dataframe
                for col in pattern_cols:
                    new_name = 'pattern_' + col[4:].lower()
                    df[new_name] = df_copy[col]
                
                # Count total patterns
                pattern_data_cols = [col for col in df.columns if col.startswith('pattern_')]
                if pattern_data_cols:
                    df['pattern_count'] = df[pattern_data_cols].abs().sum(axis=1)
        except Exception as e:
            warnings.warn(f"Failed to calculate candlestick patterns: {e}")
        
        return df
    
    @staticmethod
    @validate_price_data
    def add_support_resistance(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Add support and resistance levels.
        
        Args:
            df: Price DataFrame
            window: Lookback window for S/R calculation
            
        Returns:
            DataFrame with support/resistance levels
        """
        df = df.copy()
        
        try:
            # Rolling max/min as resistance/support
            df['resistance_20'] = df['high'].rolling(window=window).max()
            df['support_20'] = df['low'].rolling(window=window).min()
            
            df['resistance_50'] = df['high'].rolling(window=50).max()
            df['support_50'] = df['low'].rolling(window=50).min()
            
            # Distance from support/resistance
            df['dist_from_resistance_20'] = (df['resistance_20'] - df['close']) / df['close']
            df['dist_from_support_20'] = (df['close'] - df['support_20']) / df['close']
            
            # Pivot points
            df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
            df['r1'] = 2 * df['pivot'] - df['low']
            df['s1'] = 2 * df['pivot'] - df['high']
            df['r2'] = df['pivot'] + (df['high'] - df['low'])
            df['s2'] = df['pivot'] - (df['high'] - df['low'])
        except Exception as e:
            warnings.warn(f"Failed to calculate support/resistance: {e}")
        
        return df
    
    @staticmethod
    def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add all technical indicators to the DataFrame.
        
        Args:
            df: Price DataFrame
            
        Returns:
            DataFrame with all indicators
        """
        indicators = TechnicalIndicators()
        
        # Add all indicator categories with error handling
        try:
            df = indicators.add_moving_averages(df)
        except Exception as e:
            warnings.warn(f"Failed to add moving averages: {e}")
            
        try:
            df = indicators.add_momentum_indicators(df)
        except Exception as e:
            warnings.warn(f"Failed to add momentum indicators: {e}")
            
        try:
            df = indicators.add_volatility_indicators(df)
        except Exception as e:
            warnings.warn(f"Failed to add volatility indicators: {e}")
            
        try:
            df = indicators.add_volume_indicators(df)
        except Exception as e:
            warnings.warn(f"Failed to add volume indicators: {e}")
            
        try:
            df = indicators.add_pattern_recognition(df)
        except Exception as e:
            warnings.warn(f"Failed to add pattern recognition: {e}")
            
        try:
            df = indicators.add_support_resistance(df)
        except Exception as e:
            warnings.warn(f"Failed to add support/resistance: {e}")
        
        # Add custom features
        try:
            df = indicators._add_custom_features(df)
        except Exception as e:
            warnings.warn(f"Failed to add custom features: {e}")
        
        return df
    
    @staticmethod
    def _add_custom_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add custom engineered features."""
        df = df.copy()
        
        try:
            # Price position within the day's range
            range_val = df['high'] - df['low']
            df['price_position'] = np.where(range_val != 0, (df['close'] - df['low']) / range_val, 0.5)
            
            # Gap indicators
            prev_close = df['close'].shift(1)
            df['gap_up'] = np.where(prev_close != 0, ((df['open'] - prev_close) / prev_close) * 100, 0)
            df['gap_down'] = np.where(df['gap_up'] < 0, df['gap_up'], 0)
            df['gap_up'] = np.where(df['gap_up'] > 0, df['gap_up'], 0)
            
            # Volatility ratios
            df['high_low_pct'] = np.where(df['close'] != 0, ((df['high'] - df['low']) / df['close']) * 100, 0)
            df['close_open_pct'] = np.where(df['open'] != 0, ((df['close'] - df['open']) / df['open']) * 100, 0)
            
            # Volume features
            df['volume_delta'] = df['volume'] - df['volume'].shift(1)
            df['volume_delta_pct'] = df['volume'].pct_change()
            
            # Multi-timeframe returns
            for days in [2, 5, 10, 20]:
                df[f'return_{days}d'] = df['close'].pct_change(days)
                
            # Trend strength indicator (if SMA exists)
            if 'sma_20' in df.columns:
                df['trend_strength'] = np.where(df['sma_20'] != 0, (df['close'] - df['sma_20']) / df['sma_20'], 0)
        except Exception as e:
            warnings.warn(f"Failed to add some custom features: {e}")
        
        return df
    
    @staticmethod
    def calculate_feature_importance(
        df: pd.DataFrame,
        target_col: str = 'returns',
        method: str = 'correlation'
    ) -> pd.Series:
        """
        Calculate feature importance scores.
        
        Args:
            df: DataFrame with features
            target_col: Target column name
            method: Method for importance calculation
            
        Returns:
            Series with feature importance scores
        """
        if method == 'correlation':
            # Simple correlation-based importance
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            importance = df[numeric_cols].corrwith(df[target_col]).abs()
            return importance.sort_values(ascending=False)
        else:
            raise ValueError(f"Unknown method: {method}")