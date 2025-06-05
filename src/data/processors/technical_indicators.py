"""
Technical indicators calculation using pandas-ta (pure Python, no C dependencies).
"""

import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Optional, Union, Tuple
from functools import wraps


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
            # Simple Moving Average
            df[f'sma_{period}'] = ta.sma(df['close'], length=period)
            
            # Exponential Moving Average
            df[f'ema_{period}'] = ta.ema(df['close'], length=period)
            
            # Volume Weighted Average Price
            df[f'vwap_{period}'] = ta.vwap(
                high=df['high'], 
                low=df['low'], 
                close=df['close'], 
                volume=df['volume']
            ).rolling(window=period).mean()
            
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
        
        # RSI (Relative Strength Index)
        df['rsi_14'] = ta.rsi(df['close'], length=14)
        df['rsi_30'] = ta.rsi(df['close'], length=30)
        
        # MACD
        macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
        df['macd'] = macd['MACD_12_26_9']
        df['macd_signal'] = macd['MACDs_12_26_9']
        df['macd_hist'] = macd['MACDh_12_26_9']
        
        # Stochastic Oscillator
        stoch = ta.stoch(df['high'], df['low'], df['close'])
        df['slowk'] = stoch['STOCHk_14_3_3']
        df['slowd'] = stoch['STOCHd_14_3_3']
        
        # Williams %R
        df['williams_r'] = ta.willr(df['high'], df['low'], df['close'], length=14)
        
        # Rate of Change
        df['roc_10'] = ta.roc(df['close'], length=10)
        df['roc_20'] = ta.roc(df['close'], length=20)
        
        # Commodity Channel Index
        df['cci'] = ta.cci(df['high'], df['low'], df['close'], length=14)
        
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
        
        # Bollinger Bands
        bbands = ta.bbands(df['close'], length=20, std=2)
        df['bb_lower'] = bbands['BBL_20_2.0']
        df['bb_middle'] = bbands['BBM_20_2.0']
        df['bb_upper'] = bbands['BBU_20_2.0']
        df['bb_width'] = bbands['BBB_20_2.0']
        df['bb_pct'] = bbands['BBP_20_2.0']
        
        # Average True Range
        df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['atr_20'] = ta.atr(df['high'], df['low'], df['close'], length=20)
        
        # Normalized ATR
        df['natr_14'] = ta.natr(df['high'], df['low'], df['close'], length=14)
        
        # Standard deviation
        df['std_20'] = df['close'].rolling(window=20).std()
        df['std_50'] = df['close'].rolling(window=50).std()
        
        # Keltner Channels
        kc = ta.kc(df['high'], df['low'], df['close'], length=20, scalar=2)
        df['kc_lower'] = kc['KCLe_20_2']
        df['kc_upper'] = kc['KCUe_20_2']
        df['kc_pct'] = (df['close'] - df['kc_lower']) / (df['kc_upper'] - df['kc_lower'])
        
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
        
        # On Balance Volume
        df['obv'] = ta.obv(df['close'], df['volume'])
        
        # Accumulation/Distribution
        df['ad'] = ta.ad(df['high'], df['low'], df['close'], df['volume'])
        
        # Chaikin Money Flow
        df['cmf'] = ta.cmf(df['high'], df['low'], df['close'], df['volume'])
        
        # Money Flow Index
        df['mfi'] = ta.mfi(df['high'], df['low'], df['close'], df['volume'], length=14)
        
        # Volume Rate of Change
        df['vroc'] = ta.roc(df['volume'], length=10)
        
        # Force Index
        df['force_index'] = (df['close'] - df['close'].shift(1)) * df['volume']
        df['force_index_ema'] = ta.ema(df['force_index'], length=13)
        
        # Volume moving averages
        df['volume_sma_10'] = ta.sma(df['volume'], length=10)
        df['volume_sma_20'] = ta.sma(df['volume'], length=20)
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
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
        
        # Get all candlestick patterns
        df.ta.cdl_pattern(name="all", append=True)
        
        # Count total patterns
        pattern_cols = [col for col in df.columns if col.startswith('CDL_')]
        df['pattern_count'] = df[pattern_cols].abs().sum(axis=1)
        
        # Rename for consistency
        rename_dict = {}
        for col in pattern_cols:
            new_name = 'pattern_' + col[4:].lower()
            rename_dict[col] = new_name
        df.rename(columns=rename_dict, inplace=True)
        
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
        
        # Add all indicator categories
        df = indicators.add_moving_averages(df)
        df = indicators.add_momentum_indicators(df)
        df = indicators.add_volatility_indicators(df)
        df = indicators.add_volume_indicators(df)
        df = indicators.add_pattern_recognition(df)
        df = indicators.add_support_resistance(df)
        
        # Add custom features
        df = indicators._add_custom_features(df)
        
        return df
    
    @staticmethod
    def _add_custom_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add custom engineered features."""
        df = df.copy()
        
        # Price position within the day's range
        df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        
        # Gap indicators
        df['gap_up'] = ((df['open'] - df['close'].shift(1)) / df['close'].shift(1)) * 100
        df['gap_down'] = df['gap_up'].apply(lambda x: x if x < 0 else 0)
        df['gap_up'] = df['gap_up'].apply(lambda x: x if x > 0 else 0)
        
        # Volatility ratios
        df['high_low_pct'] = ((df['high'] - df['low']) / df['close']) * 100
        df['close_open_pct'] = ((df['close'] - df['open']) / df['open']) * 100
        
        # Volume features
        df['volume_delta'] = df['volume'] - df['volume'].shift(1)
        df['volume_delta_pct'] = df['volume'].pct_change()
        
        # Multi-timeframe returns
        for days in [2, 5, 10, 20]:
            df[f'return_{days}d'] = df['close'].pct_change(days)
            
        # Trend strength indicator
        df['trend_strength'] = (df['close'] - df['sma_20']) / df['sma_20']
        
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