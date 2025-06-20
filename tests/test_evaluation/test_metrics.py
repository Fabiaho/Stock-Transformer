"""
Tests for financial metrics.
"""

import pytest
import numpy as np
import pandas as pd

from src.training.metrics import FinancialMetrics


class TestFinancialMetrics:
    """Test financial metrics calculations."""
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample return data."""
        np.random.seed(42)
        # Generate returns with positive drift
        returns = np.random.randn(252) * 0.02 + 0.0005  # Daily returns
        return returns
    
    @pytest.fixture
    def metrics_calculator(self):
        """Create metrics calculator."""
        return FinancialMetrics(
            transaction_cost=0.001,
            risk_free_rate=0.02,
            trading_days=252
        )
    
    def test_initialization(self):
        """Test metrics initialization."""
        metrics = FinancialMetrics(
            transaction_cost=0.002,
            risk_free_"""
Tests for financial metrics.
"""

import pytest
import numpy as np
import pandas as pd

from src.training.metrics import FinancialMetrics


class TestFinancialMetrics:
    """Test financial metrics calculations."""
    
    @pytest.fixture
    def sample_returns(self):
        """Create sample return data."""
        np.random.seed(42)
        # Generate returns with positive drift
        returns = np.random.randn(252) * 0.02 + 0.0005  # Daily returns
        return returns
    
    @pytest.fixture
    def metrics_calculator(self):
        """Create metrics calculator."""
        return FinancialMetrics(
            transaction_cost=0.001,
            risk_free_rate=0.02,
            trading_days=252
        )
    
    def test_initialization(self):
        """Test metrics initialization."""
        metrics = FinancialMetrics(
            transaction_cost=0.002,
            risk_free_rate=0.03,
            trading_days=252
        )
        
        assert metrics.transaction_cost == 0.002
        assert metrics.risk_free_rate == 0.03
        assert metrics.trading_days == 252
        assert metrics.daily_rf == 0.03 / 252
    
    def test_calculate_returns(self, metrics_calculator):
        """Test strategy returns calculation.