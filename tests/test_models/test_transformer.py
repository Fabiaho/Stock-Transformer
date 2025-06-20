"""
Tests for transformer model.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from src.models.transformer import (
    StockTransformer, PositionalEncoding, MultiHeadAttentionWithFinancialBias,
    TemporalConvolutionBlock, TransformerBlock
)


class TestPositionalEncoding:
    """Test positional encoding module."""
    
    def test_initialization(self):
        """Test initialization of positional encoding."""
        d_model = 512
        max_len = 1000
        
        pe = PositionalEncoding(d_model, max_len=max_len)
        
        assert hasattr(pe, 'pe')
        assert pe.pe.shape == torch.Size([max_len, 1, d_model])
    
    def test_forward(self):
        """Test forward pass of positional encoding."""
        batch_size = 16
        seq_len = 60
        d_model = 256
        
        pe = PositionalEncoding(d_model)
        x = torch.randn(seq_len, batch_size, d_model)
        
        output = pe(x)
        
        assert output.shape == x.shape
        assert not torch.allclose(output, x)  # Should be different due to PE
    
    def test_time_gaps(self):
        """Test positional encoding with time gaps."""
        batch_size = 8
        seq_len = 30
        d_model = 128
        
        pe = PositionalEncoding(d_model)
        x = torch.randn(seq_len, batch_size, d_model)
        
        # Create time gaps (e.g., weekends)
        time_gaps = torch.ones(seq_len, batch_size)
        time_gaps[5] = 3  # Weekend gap
        time_gaps[10] = 3  # Another weekend
        
        output = pe(x, time_gaps)
        
        assert output.shape == x.shape


class TestMultiHeadAttention:
    """Test multi-head attention with financial bias."""
    
    def test_initialization(self):
        """Test initialization."""
        d_model = 512
        n_heads = 8
        
        attention = MultiHeadAttentionWithFinancialBias(d_model, n_heads)
        
        assert attention.d_model == d_model
        assert attention.n_heads == n_heads
        assert attention.d_k == d_model // n_heads
    
    def test_forward(self):
        """Test forward pass."""
        batch_size = 16
        seq_len = 60
        d_model = 256
        n_heads = 8
        
        attention = MultiHeadAttentionWithFinancialBias(d_model, n_heads)
        
        query = torch.randn(batch_size, seq_len, d_model)
        key = torch.randn(batch_size, seq_len, d_model)
        value = torch.randn(batch_size, seq_len, d_model)
        
        output, attn_weights = attention(query, key, value)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert attn_weights.shape == (batch_size, n_heads, seq_len, seq_len)
    
    def test_regime_bias(self):
        """Test attention with regime bias."""
        batch_size = 8
        seq_len = 30
        d_model = 128
        n_heads = 4
        
        attention = MultiHeadAttentionWithFinancialBias(
            d_model, n_heads, use_regime_bias=True
        )
        
        x = torch.randn(batch_size, seq_len, d_model)
        market_regime = torch.randint(0, 3, (batch_size,))  # 3 regimes
        
        output, attn_weights = attention(x, x, x, market_regime=market_regime)
        
        assert output.shape == (batch_size, seq_len, d_model)


class TestTemporalConvolution:
    """Test temporal convolution block."""
    
    def test_initialization(self):
        """Test initialization."""
        in_channels = 64
        out_channels = 128
        
        conv_block = TemporalConvolutionBlock(in_channels, out_channels)
        
        assert hasattr(conv_block, 'conv1')
        assert hasattr(conv_block, 'conv2')
        assert hasattr(conv_block, 'downsample')
    
    def test_forward(self):
        """Test forward pass."""
        batch_size = 16
        seq_len = 60
        in_channels = 64
        out_channels = 128
        
        conv_block = TemporalConvolutionBlock(in_channels, out_channels)
        
        # Input shape: (batch, channels, seq_len)
        x = torch.randn(batch_size, in_channels, seq_len)
        output = conv_block(x)
        
        assert output.shape == (batch_size, out_channels, seq_len)
    
    def test_residual_connection(self):
        """Test residual connection."""
        batch_size = 8
        seq_len = 30
        channels = 64
        
        # Same input/output channels - should have identity residual
        conv_block = TemporalConvolutionBlock(channels, channels)
        
        x = torch.randn(batch_size, channels, seq_len)
        output = conv_block(x)
        
        assert output.shape == x.shape


class TestTransformerBlock:
    """Test transformer block."""
    
    def test_initialization(self):
        """Test initialization."""
        d_model = 256
        n_heads = 8
        d_ff = 1024
        
        block = TransformerBlock(d_model, n_heads, d_ff)
        
        assert hasattr(block, 'attention')
        assert hasattr(block, 'ff')
        assert hasattr(block, 'norm1')
        assert hasattr(block, 'norm2')
    
    def test_forward(self):
        """Test forward pass."""
        batch_size = 16
        seq_len = 60
        d_model = 256
        n_heads = 8
        d_ff = 1024
        
        block = TransformerBlock(d_model, n_heads, d_ff)
        
        x = torch.randn(batch_size, seq_len, d_model)
        output, attn_weights = block(x)
        
        assert output.shape == x.shape
        assert attn_weights.shape == (batch_size, n_heads, seq_len, seq_len)


class TestStockTransformer:
    """Test the complete stock transformer model."""
    
    def test_initialization(self):
        """Test model initialization."""
        input_dim = 50
        d_model = 256
        
        model = StockTransformer(
            input_dim=input_dim,
            d_model=d_model,
            n_heads=8,
            n_layers=4,
            d_ff=1024
        )
        
        assert hasattr(model, 'input_projection')
        assert hasattr(model, 'pos_encoding')
        assert hasattr(model, 'transformer_blocks')
        assert hasattr(model, 'output_head')
        assert len(model.transformer_blocks) == 4
    
    def test_forward_regression(self):
        """Test forward pass for regression."""
        batch_size = 16
        seq_len = 60
        input_dim = 50
        
        model = StockTransformer(
            input_dim=input_dim,
            d_model=256,
            n_heads=8,
            n_layers=4,
            output_type='regression'
        )
        
        x = torch.randn(batch_size, seq_len, input_dim)
        outputs = model(x)
        
        assert 'output' in outputs
        assert 'attention_weights' in outputs
        assert 'hidden_states' in outputs
        
        assert outputs['output'].shape == (batch_size, 1)
        assert len(outputs['attention_weights']) == 4
        assert outputs['hidden_states'].shape == (batch_size, seq_len, 256)
    
    def test_forward_classification(self):
        """Test forward pass for classification."""
        batch_size = 16
        seq_len = 60
        input_dim = 50
        n_classes = 3
        
        model = StockTransformer(
            input_dim=input_dim,
            d_model=256,
            n_heads=8,
            n_layers=4,
            output_type='classification',
            n_classes=n_classes
        )
        
        x = torch.randn(batch_size, seq_len, input_dim)
        outputs = model(x)
        
        assert outputs['output'].shape == (batch_size, n_classes)
    
    def test_temporal_convolution(self):
        """Test model with temporal convolution."""
        batch_size = 8
        seq_len = 60
        input_dim = 50
        
        model = StockTransformer(
            input_dim=input_dim,
            d_model=256,
            n_heads=8,
            n_layers=2,
            use_temporal_conv=True,
            conv_channels=[64, 128]
        )
        
        x = torch.randn(batch_size, seq_len, input_dim)
        outputs = model(x)
        
        assert outputs['output'].shape == (batch_size, 1)
    
    def test_multi_horizon_prediction(self):
        """Test multi-horizon prediction."""
        batch_size = 8
        seq_len = 60
        input_dim = 50
        horizons = [1, 5, 10]
        
        model = StockTransformer(
            input_dim=input_dim,
            d_model=256,
            n_heads=8,
            n_layers=2
        )
        
        x = torch.randn(batch_size, seq_len, input_dim)
        predictions = model.predict_multi_horizon(x, horizons=horizons)
        
        for h in horizons:
            assert f'horizon_{h}' in predictions
            assert predictions[f'horizon_{h}'].shape == (batch_size, 1)
    
    def test_gradient_flow(self):
        """Test that gradients flow properly."""
        batch_size = 4
        seq_len = 30
        input_dim = 20
        
        model = StockTransformer(
            input_dim=input_dim,
            d_model=128,
            n_heads=4,
            n_layers=2
        )
        
        x = torch.randn(batch_size, seq_len, input_dim, requires_grad=True)
        outputs = model(x)
        
        loss = outputs['output'].sum()
        loss.backward()
        
        # Check that gradients are computed
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        # Check model parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}""""
Tests for transformer model.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from src.models.transformer import (
    StockTransformer, PositionalEncoding, MultiHeadAttentionWithFinancialBias,
    TemporalConvolutionBlock, TransformerBlock
)


class TestPositionalEncoding:
    """Test positional encoding module."""
    
    def test_initialization(self):
        """Test initialization of positional encoding."""
        d_model = 512
        max_len = 1000
        
        pe = PositionalEncoding(d_model