"""
Transformer model for stock price prediction with financial-specific modifications.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict
import numpy as np


class PositionalEncoding(nn.Module):
    """Positional encoding for time series with irregular intervals (weekends, holidays)."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor, time_gaps: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch, d_model)
            time_gaps: Optional tensor indicating time gaps between observations
        """
        if time_gaps is not None:
            # Adjust positional encoding based on actual time gaps
            # This handles weekends and holidays in financial data
            adjusted_positions = torch.cumsum(time_gaps, dim=0).unsqueeze(-1)
            pe_adjusted = self.pe[:x.size(0)] * adjusted_positions.unsqueeze(-1)
            x = x + pe_adjusted
        else:
            x = x + self.pe[:x.size(0)]
            
        return self.dropout(x)


class MultiHeadAttentionWithFinancialBias(nn.Module):
    """Multi-head attention with optional financial market regime awareness."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_regime_bias: bool = True
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_regime_bias = use_regime_bias
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        if use_regime_bias:
            # Learnable regime embeddings for different market conditions
            self.regime_embeddings = nn.Parameter(torch.randn(3, n_heads, 1, 1))
            
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        market_regime: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # Linear transformations and reshape
        Q = self.W_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply regime bias if available
        if self.use_regime_bias and market_regime is not None:
            regime_bias = self.regime_embeddings[market_regime].squeeze()
            scores = scores + regime_bias
            
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)
        
        # Reshape and project
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self.W_o(context)
        
        return output, attn_weights


class TemporalConvolutionBlock(nn.Module):
    """Temporal convolution for capturing short-term patterns before transformer."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        
        if self.downsample is not None:
            residual = self.downsample(residual)
            
        out = self.relu(out + residual)
        return out


class TransformerBlock(nn.Module):
    """Single transformer block with financial-specific modifications."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = 'gelu',
        use_regime_bias: bool = True
    ):
        super().__init__()
        
        self.attention = MultiHeadAttentionWithFinancialBias(
            d_model, n_heads, dropout, use_regime_bias
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        market_regime: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self-attention with residual
        attn_out, attn_weights = self.attention(x, x, x, mask, market_regime)
        x = self.norm1(x + attn_out)
        
        # Feed-forward with residual
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x, attn_weights


class StockTransformer(nn.Module):
    """
    Transformer model for stock price prediction with financial domain adaptations.
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        output_dim: int = 1,
        use_temporal_conv: bool = True,
        conv_channels: List[int] = [64, 128, 256],
        use_regime_bias: bool = True,
        output_type: str = 'regression',  # 'regression' or 'classification'
        n_classes: int = 3
    ):
        super().__init__()
        
        self.d_model = d_model
        self.output_type = output_type
        self.use_temporal_conv = use_temporal_conv
        
        # Temporal convolution layers (optional)
        if use_temporal_conv:
            conv_layers = []
            in_channels = input_dim
            
            for out_channels in conv_channels:
                conv_layers.append(
                    TemporalConvolutionBlock(
                        in_channels, out_channels,
                        kernel_size=3, dilation=1, dropout=dropout
                    )
                )
                in_channels = out_channels
                
            self.temporal_conv = nn.Sequential(*conv_layers)
            self.input_projection = nn.Linear(conv_channels[-1], d_model)
        else:
            self.input_projection = nn.Linear(input_dim, d_model)
            
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads, d_ff, dropout,
                use_regime_bias=use_regime_bias
            )
            for _ in range(n_layers)
        ])
        
        # Output heads
        if output_type == 'regression':
            self.output_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, output_dim)
            )
        else:  # classification
            self.output_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, n_classes)
            )
            
        # Additional components for interpretability
        self.attention_weights = []
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        time_gaps: Optional[torch.Tensor] = None,
        market_regime: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            mask: Optional attention mask
            time_gaps: Optional tensor indicating time gaps between observations
            market_regime: Optional market regime indicators
            
        Returns:
            Dictionary containing:
                - 'output': Model predictions
                - 'attention_weights': List of attention weights from each layer
                - 'hidden_states': Final hidden states
        """
        batch_size, seq_len, _ = x.shape
        
        # Apply temporal convolution if enabled
        if self.use_temporal_conv:
            # Reshape for conv1d: (batch, channels, seq_len)
            x = x.transpose(1, 2)
            x = self.temporal_conv(x)
            x = x.transpose(1, 2)
            
        # Project to model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        x = self.pos_encoding(x, time_gaps)
        x = x.transpose(0, 1)  # (batch, seq_len, d_model)
        
        # Pass through transformer blocks
        attention_weights = []
        for transformer in self.transformer_blocks:
            x, attn_w = transformer(x, mask, market_regime)
            attention_weights.append(attn_w)
            
        # Store hidden states before output projection
        hidden_states = x
        
        # Apply output head
        # For sequence prediction, we typically use the last position
        output = self.output_head(x[:, -1, :])
        
        return {
            'output': output,
            'attention_weights': attention_weights,
            'hidden_states': hidden_states
        }
    
    def get_attention_weights(self) -> List[torch.Tensor]:
        """Return stored attention weights for visualization."""
        return self.attention_weights
    
    def predict_multi_horizon(
        self,
        x: torch.Tensor,
        horizons: List[int] = [1, 5, 20],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Predict multiple time horizons using different output positions.
        
        Args:
            x: Input tensor
            horizons: List of prediction horizons
            **kwargs: Additional arguments for forward pass
            
        Returns:
            Dictionary with predictions for each horizon
        """
        # Get hidden states from forward pass
        results = self.forward(x, **kwargs)
        hidden_states = results['hidden_states']
        
        predictions = {}
        for horizon in horizons:
            # Use hidden state at position (-horizon) for each horizon
            if horizon <= hidden_states.size(1):
                horizon_hidden = hidden_states[:, -horizon, :]
                predictions[f'horizon_{horizon}'] = self.output_head(horizon_hidden)
            else:
                predictions[f'horizon_{horizon}'] = None
                
        return predictions