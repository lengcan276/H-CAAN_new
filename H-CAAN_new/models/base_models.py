# models/base_models.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union, Any


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer-based models."""
    
    def __init__(self, d_model: int, max_length: int = 5000, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register buffer (persistent but not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LayerNorm(nn.Module):
    """Layer normalization with optional conditional scaling and shifting."""
    
    def __init__(self, features: int, eps: float = 1e-6, conditional: bool = False, cond_dim: int = None):
        super(LayerNorm, self).__init__()
        self.features = features
        self.eps = eps
        self.conditional = conditional
        
        # Standard layer norm parameters
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        
        # Conditional layer norm parameters (for adaptive processing)
        if conditional and cond_dim is not None:
            self.cond_scale = nn.Linear(cond_dim, features)
            self.cond_shift = nn.Linear(cond_dim, features)
    
    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, ..., features]
            cond: Optional conditioning tensor for adaptive processing
                  of shape [batch_size, cond_dim]
                  
        Returns:
            Normalized tensor with same shape as input
        """
        # Calculate mean and std
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        
        # Normalize
        x_norm = (x - mean) / (std + self.eps)
        
        # Apply scaling and shifting
        if self.conditional and cond is not None:
            # Generate conditional gamma and beta
            cond_gamma = self.cond_scale(cond).unsqueeze(1)  # [batch_size, 1, features]
            cond_beta = self.cond_shift(cond).unsqueeze(1)   # [batch_size, 1, features]
            
            # Apply conditional scaling and shifting
            return x_norm * (self.gamma + cond_gamma) + (self.beta + cond_beta)
        else:
            return x_norm * self.gamma + self.beta


class FeedForward(nn.Module):
    """Feed-forward network with residual connection and layer normalization."""
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = 'relu'):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(d_model)
        
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'gelu':
            self.activation = F.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, ..., d_model]
            
        Returns:
            Output tensor of shape [batch_size, ..., d_model]
        """
        # Apply feed-forward network with residual connection
        residual = x
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = residual + x
        x = self.norm(x)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism."""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(d_model)
    
    def forward(self, 
               query: torch.Tensor, 
               key: torch.Tensor, 
               value: torch.Tensor, 
               mask: Optional[torch.Tensor] = None, 
               return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            query: Query tensor of shape [batch_size, q_len, d_model]
            key: Key tensor of shape [batch_size, k_len, d_model]
            value: Value tensor of shape [batch_size, k_len, d_model]
            mask: Optional mask tensor of shape [batch_size, q_len, k_len]
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor of shape [batch_size, q_len, d_model]
            Attention weights of shape [batch_size, num_heads, q_len, k_len] (if return_attention is True)
        """
        batch_size = query.size(0)
        residual = query
        
        # Linear projections and reshape for multi-head attention
        q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for multi-head attention
            if mask.dim() == 3:  # [batch_size, q_len, k_len]
                mask = mask.unsqueeze(1)  # [batch_size, 1, q_len, k_len]
            # Apply mask (set -inf where mask is 0)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Calculate attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        output = torch.matmul(attn_weights, v)
        
        # Reshape and apply output projection
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.o_proj(output)
        output = self.dropout(output)
        
        # Apply residual connection and layer normalization
        output = residual + output
        output = self.norm(output)
        
        if return_attention:
            return output, attn_weights
        return output


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with multi-head attention and feed-forward network."""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1, activation: str = 'relu'):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout, activation)
    
    def forward(self, 
               src: torch.Tensor, 
               src_mask: Optional[torch.Tensor] = None, 
               return_attention: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            src: Input tensor of shape [batch_size, src_len, d_model]
            src_mask: Optional mask tensor of shape [batch_size, src_len, src_len]
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor of shape [batch_size, src_len, d_model]
            Attention weights of shape [batch_size, num_heads, src_len, src_len] (if return_attention is True)
        """
        if return_attention:
            src, attn_weights = self.self_attn(src, src, src, src_mask, return_attention=True)
            src = self.feed_forward(src)
            return src, attn_weights
        else:
            src = self.self_attn(src, src, src, src_mask)
            src = self.feed_forward(src)
            return src


class BiGRULayer(nn.Module):
    """Bidirectional GRU layer with attention mechanism."""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.1):
        super(BiGRULayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # BiGRU
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm(hidden_dim * 2)
    
    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            hidden: Optional initial hidden state
            
        Returns:
            Output tensor of shape [batch_size, hidden_dim * 2]
        """
        # BiGRU forward pass
        outputs, _ = self.gru(x, hidden)  # [batch_size, seq_len, hidden_dim * 2]
        
        # Apply attention
        attn_weights = F.softmax(self.attn(outputs).squeeze(-1), dim=1)  # [batch_size, seq_len]
        context = torch.bmm(attn_weights.unsqueeze(1), outputs).squeeze(1)  # [batch_size, hidden_dim * 2]
        
        # Apply dropout and normalization
        context = self.dropout(context)
        context = self.norm(context)
        
        return context


class GraphConvBlock(nn.Module):
    """Graph convolutional block with residual connection and layer normalization."""
    
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1, residual: bool = True):
        super(GraphConvBlock, self).__init__()
        self.conv = torch.nn.Linear(in_dim, out_dim)
        self.norm = LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.residual = residual and (in_dim == out_dim)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node feature tensor of shape [batch_size, num_nodes, in_dim]
            adj: Adjacency matrix of shape [batch_size, num_nodes, num_nodes]
            
        Returns:
            Updated node features of shape [batch_size, num_nodes, out_dim]
        """
        residual = x
        
        # Apply graph convolution
        x = self.conv(x)  # [batch_size, num_nodes, out_dim]
        
        # Propagate features through graph
        x = torch.bmm(adj, x)  # [batch_size, num_nodes, out_dim]
        
        # Apply activation, dropout, and normalization
        x = self.activation(x)
        x = self.dropout(x)
        
        # Apply residual connection if possible
        if self.residual:
            x = x + residual
        
        x = self.norm(x)
        
        return x


class GraphAttentionLayer(nn.Module):
    """Graph attention layer with multi-head attention."""
    
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 8, dropout: float = 0.1, residual: bool = True):
        super(GraphAttentionLayer, self).__init__()
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.residual = residual and (in_dim == out_dim)
        
        # Linear transformation for input features
        self.W = nn.Linear(in_dim, out_dim)
        
        # Attention parameters for each head
        self.a = nn.Parameter(torch.zeros(num_heads, 2 * self.head_dim))
        nn.init.xavier_uniform_(self.a)
        
        # Dropout and activation
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(0.2)
        self.norm = LayerNorm(out_dim)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node feature tensor of shape [batch_size, num_nodes, in_dim]
            adj: Adjacency matrix of shape [batch_size, num_nodes, num_nodes]
            
        Returns:
            Updated node features of shape [batch_size, num_nodes, out_dim]
        """
        batch_size, num_nodes, _ = x.size()
        residual = x
        
        # Linear transformation
        x = self.W(x)  # [batch_size, num_nodes, out_dim]
        
        # Reshape for multi-head attention
        x = x.view(batch_size, num_nodes, self.num_heads, self.head_dim)
        x = x.permute(0, 2, 1, 3)  # [batch_size, num_heads, num_nodes, head_dim]
        
        # Prepare for attention computation
        x_i = x.unsqueeze(-2).expand(-1, -1, -1, num_nodes, -1)  # [batch_size, num_heads, num_nodes, num_nodes, head_dim]
        x_j = x.unsqueeze(-3).expand(-1, -1, num_nodes, -1, -1)  # [batch_size, num_heads, num_nodes, num_nodes, head_dim]
        
        # Concatenate features for attention computation
        x_pair = torch.cat([x_i, x_j], dim=-1)  # [batch_size, num_heads, num_nodes, num_nodes, 2*head_dim]
        
        # Compute attention weights
        attn_coef = torch.einsum('bnijh,bh->bnij', x_pair, self.a)  # [batch_size, num_heads, num_nodes, num_nodes]
        attn_coef = self.activation(attn_coef)
        
        # Apply mask from adjacency matrix
        adj = adj.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  # [batch_size, num_heads, num_nodes, num_nodes]
        mask = (1 - adj) * -1e9
        attn_coef = attn_coef + mask
        
        # Normalize attention weights
        attn_coef = F.softmax(attn_coef, dim=-1)
        attn_coef = self.dropout(attn_coef)
        
        # Apply attention to node features
        out = torch.matmul(attn_coef, x)  # [batch_size, num_heads, num_nodes, head_dim]
        
        # Reshape output
        out = out.permute(0, 2, 1, 3).contiguous()  # [batch_size, num_nodes, num_heads, head_dim]
        out = out.view(batch_size, num_nodes, -1)  # [batch_size, num_nodes, out_dim]
        
        # Apply residual connection if possible
        if self.residual:
            out = out + residual
        
        # Apply normalization
        out = self.norm(out)
        
        return out


class MLP(nn.Module):
    """Multi-layer perceptron with dropout and layer normalization."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, 
                 dropout: float = 0.1, activation: str = 'relu', use_layer_norm: bool = True):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.use_layer_norm = use_layer_norm
        
        # Build network layers
        layers = []
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            
            if use_layer_norm:
                layers.append(LayerNorm(dims[i+1]))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'silu':
                layers.append(nn.SiLU())
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(dims[-1], output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, ..., input_dim]
            
        Returns:
            Output tensor of shape [batch_size, ..., output_dim]
        """
        return self.network(x)