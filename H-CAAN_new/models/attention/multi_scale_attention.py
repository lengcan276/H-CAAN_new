import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention mechanism as described in 'Attention Is All You Need'.
    
    This is the core attention mechanism used in the transformer architecture.
    """
    
    def __init__(self, temperature, attn_dropout=0.1):
        """
        Initialize the scaled dot-product attention.
        
        Args:
            temperature (float): Temperature parameter for scaling
            attn_dropout (float): Dropout probability for attention weights
        """
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
    
    def forward(self, q, k, v, mask=None):
        """
        Compute the attention.
        
        Args:
            q (torch.Tensor): Query tensor of shape [batch_size, n_heads, len_q, d_k]
            k (torch.Tensor): Key tensor of shape [batch_size, n_heads, len_k, d_k]
            v (torch.Tensor): Value tensor of shape [batch_size, n_heads, len_v, d_v]
            mask (torch.Tensor, optional): Mask tensor of shape [batch_size, n_heads, len_q, len_k]
            
        Returns:
            tuple: (output, attn_weights)
                - output: Attention output of shape [batch_size, n_heads, len_q, d_v]
                - attn_weights: Attention weights of shape [batch_size, n_heads, len_q, len_k]
        """
        # Compute attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.temperature
        
        # Apply mask if provided
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        # Apply softmax and dropout
        attn_weights = self.dropout(F.softmax(attn, dim=-1))
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as described in 'Attention Is All You Need'.
    
    This splits the input into multiple heads, applies scaled dot-product attention
    to each head independently, and then concatenates the results.
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        """
        Initialize multi-head attention.
        
        Args:
            d_model (int): Model dimension
            n_heads (int): Number of attention heads
            dropout (float): Dropout probability
        """
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        
        # Output projection
        self.fc = nn.Linear(d_model, d_model)
        
        # Scaled dot-product attention
        self.attention = ScaledDotProductAttention(temperature=self.d_k ** 0.5, attn_dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
    def forward(self, q, k, v, mask=None, return_attns=False):
        """
        Compute multi-head attention.
        
        Args:
            q (torch.Tensor): Query tensor of shape [batch_size, len_q, d_model]
            k (torch.Tensor): Key tensor of shape [batch_size, len_k, d_model]
            v (torch.Tensor): Value tensor of shape [batch_size, len_v, d_model]
            mask (torch.Tensor, optional): Mask tensor
            return_attns (bool): Whether to return attention weights
            
        Returns:
            tuple: (output, attn)
                - output: Attention output of shape [batch_size, len_q, d_model]
                - attn: Attention weights if return_attns=True
        """
        batch_size, len_q, _ = q.size()
        batch_size, len_k, _ = k.size()
        batch_size, len_v, _ = v.size()
        
        residual = q
        
        # Linear projections and reshape for multi-head attention
        q = self.w_q(q).view(batch_size, len_q, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, len_k, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, len_v, self.n_heads, self.d_k).transpose(1, 2)
        
        # Adjust mask for multi-head attention
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1)
        
        # Apply scaled dot-product attention
        output, attn_weights = self.attention(q, k, v, mask=mask)
        
        # Reshape back to original dimensions
        output = output.transpose(1, 2).contiguous().view(batch_size, len_q, self.d_model)
        
        # Final projection and residual connection
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        
        if return_attns:
            return output, attn_weights
        return output


class MultiScaleAttention(nn.Module):
    """
    Multi-Scale Attention for capturing information at different scales.
    
    This module applies attention at multiple scales by using different kernel sizes
    or dilation rates to capture both local and global dependencies.
    """
    
    def __init__(self, d_model, n_heads, scales=[1, 2, 4, 8], dropout=0.1):
        """
        Initialize multi-scale attention.
        
        Args:
            d_model (int): Model dimension
            n_heads (int): Number of attention heads
            scales (list): List of scale factors for multi-scale processing
            dropout (float): Dropout probability
        """
        super(MultiScaleAttention, self).__init__()
        
        self.d_model = d_model
        self.scales = scales
        self.n_scales = len(scales)
        
        # Create multi-head attention for each scale
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(d_model, n_heads, dropout)
            for _ in range(self.n_scales)
        ])
        
        # Scale-specific projections
        self.scale_projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=scale, padding=scale//2, dilation=scale),
                nn.LayerNorm(d_model)
            )
            for scale in scales
        ])
        
        # Integration layer
        self.integration = nn.Sequential(
            nn.Linear(d_model * self.n_scales, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        """
        Apply multi-scale attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model]
            mask (torch.Tensor, optional): Mask tensor
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.size()
        
        # Process each scale
        scale_outputs = []
        
        for i, scale in enumerate(self.scales):
            # Apply scale-specific projection
            x_scale = x.transpose(1, 2)  # [batch_size, d_model, seq_len]
            x_scale = self.scale_projections[i](x_scale)
            x_scale = x_scale.transpose(1, 2)  # [batch_size, seq_len, d_model]
            
            # Apply attention at this scale
            output = self.attention_layers[i](x_scale, x_scale, x_scale, mask)
            scale_outputs.append(output)
        
        # Concatenate outputs from all scales
        multi_scale_output = torch.cat(scale_outputs, dim=-1)
        
        # Integrate across scales
        output = self.integration(multi_scale_output)
        
        return output


class HierarchicalMultiScaleAttention(nn.Module):
    """
    Hierarchical Multi-Scale Attention for processing molecular data at different levels.
    
    This module applies multi-scale attention in a hierarchical manner, allowing the model
    to capture dependencies at different semantic levels.
    """
    
    def __init__(self, d_model, n_heads, n_levels=3, scales=[1, 2, 4, 8], dropout=0.1):
        """
        Initialize hierarchical multi-scale attention.
        
        Args:
            d_model (int): Model dimension
            n_heads (int): Number of attention heads
            n_levels (int): Number of hierarchical levels
            scales (list): List of scale factors for multi-scale processing
            dropout (float): Dropout probability
        """
        super(HierarchicalMultiScaleAttention, self).__init__()
        
        self.d_model = d_model
        self.n_levels = n_levels
        
        # Create multi-scale attention for each level
        self.level_attentions = nn.ModuleList([
            MultiScaleAttention(d_model, n_heads, scales, dropout)
            for _ in range(n_levels)
        ])
        
        # Pooling layers for hierarchy
        self.pooling_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=2, stride=2),
                nn.LayerNorm(d_model)
            )
            for _ in range(n_levels - 1)
        ])
        
        # Unpooling layers for hierarchy
        self.unpooling_layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose1d(d_model, d_model, kernel_size=2, stride=2),
                nn.LayerNorm(d_model)
            )
            for _ in range(n_levels - 1)
        ])
        
        # Final integration
        self.final_integration = nn.Sequential(
            nn.Linear(d_model * n_levels, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        """
        Apply hierarchical multi-scale attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model]
            mask (torch.Tensor, optional): Mask tensor
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.size()
        
        # Process each level in the hierarchy
        level_outputs = []
        level_features = x
        
        # Top-down pass (encoding)
        level_features_list = [level_features]
        for i in range(self.n_levels - 1):
            # Apply pooling
            level_features = level_features.transpose(1, 2)  # [batch_size, d_model, seq_len]
            level_features = self.pooling_layers[i](level_features)
            level_features = level_features.transpose(1, 2)  # [batch_size, seq_len/2, d_model]
            
            level_features_list.append(level_features)
        
        # Process each level with multi-scale attention
        processed_features = []
        for i in range(self.n_levels):
            level_input = level_features_list[i]
            
            # Create appropriate mask for this level
            level_mask = None
            if mask is not None:
                if i == 0:
                    level_mask = mask
                else:
                    # Downsample mask for this level
                    level_mask = mask[:, ::2**i][:, :level_input.size(1)]
            
            # Apply multi-scale attention
            level_output = self.level_attentions[i](level_input, level_mask)
            processed_features.append(level_output)
        
        # Bottom-up pass (decoding)
        final_features = processed_features[-1]
        for i in range(self.n_levels - 2, -1, -1):
            # Apply unpooling
            final_features = final_features.transpose(1, 2)  # [batch_size, d_model, seq_len]
            final_features = self.unpooling_layers[i](final_features)
            
            # Handle potential size mismatch after transposed convolution
            target_seq_len = processed_features[i].size(1)
            if final_features.size(2) != target_seq_len:
                final_features = F.interpolate(final_features, size=target_seq_len, mode='linear')
                
            final_features = final_features.transpose(1, 2)  # [batch_size, seq_len*2, d_model]
            
            # Add features from this level (residual connection)
            final_features = final_features + processed_features[i]
            
            level_outputs.append(final_features)
        
        # Interpolate all outputs to original sequence length if needed
        aligned_outputs = []
        for output in level_outputs:
            if output.size(1) != seq_len:
                output = output.transpose(1, 2)  # [batch_size, d_model, seq_len_i]
                output = F.interpolate(output, size=seq_len, mode='linear')
                output = output.transpose(1, 2)  # [batch_size, seq_len, d_model]
            aligned_outputs.append(output)
        
        # Concatenate all level outputs
        hierarchical_output = torch.cat(aligned_outputs, dim=-1)
        
        # Final integration
        output = self.final_integration(hierarchical_output)
        
        return output


class ChemicalMultiScaleAttention(nn.Module):
    """
    Chemistry-aware Multi-Scale Attention specifically designed for molecular data.
    
    This module extends the multi-scale attention with chemistry-aware mechanisms
    that capture molecular substructures at different scales.
    """
    
    def __init__(self, d_model, n_heads, substructure_scales=[1, 2, 4, 8], 
                chemical_context_window=3, use_functional_groups=True, dropout=0.1):
        """
        Initialize chemistry-aware multi-scale attention.
        
        Args:
            d_model (int): Model dimension
            n_heads (int): Number of attention heads
            substructure_scales (list): List of substructure scales
            chemical_context_window (int): Size of chemical context window
            use_functional_groups (bool): Whether to use functional group information
            dropout (float): Dropout probability
        """
        super(ChemicalMultiScaleAttention, self).__init__()
        
        self.d_model = d_model
        self.substructure_scales = substructure_scales
        self.n_scales = len(substructure_scales)
        self.chemical_context_window = chemical_context_window
        self.use_functional_groups = use_functional_groups
        
        # Create multi-head attention for each substructure scale
        self.substructure_attentions = nn.ModuleList([
            MultiHeadAttention(d_model, n_heads, dropout)
            for _ in range(self.n_scales)
        ])
        
        # Chemical context projections
        self.chemical_projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    d_model, d_model, 
                    kernel_size=scale * chemical_context_window,
                    padding=(scale * chemical_context_window) // 2,
                    groups=d_model // 4  # Grouped convolution for efficiency
                ),
                nn.LayerNorm(d_model),
                nn.GELU()
            )
            for scale in substructure_scales
        ])
        
        # Functional group awareness module (if enabled)
        if use_functional_groups:
            self.functional_group_embedding = nn.Embedding(100, d_model)  # Assuming up to 100 functional group types
            self.functional_group_attention = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Substructure importance weighting
        self.scale_importance = nn.Parameter(torch.ones(self.n_scales))
        
        # Integration layer
        self.integration = nn.Sequential(
            nn.Linear(d_model * self.n_scales, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, functional_groups=None, mask=None):
        """
        Apply chemistry-aware multi-scale attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model]
            functional_groups (torch.Tensor, optional): Functional group indices
                of shape [batch_size, seq_len]
            mask (torch.Tensor, optional): Mask tensor
            
        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.size()
        
        # Process each substructure scale
        scale_outputs = []
        scale_weights = F.softmax(self.scale_importance, dim=0)
        
        for i, scale in enumerate(self.substructure_scales):
            # Apply chemical context projection
            x_scale = x.transpose(1, 2)  # [batch_size, d_model, seq_len]
            x_scale = self.chemical_projections[i](x_scale)
            x_scale = x_scale.transpose(1, 2)  # [batch_size, seq_len, d_model]
            
            # Apply attention at this scale
            output = self.substructure_attentions[i](x_scale, x_scale, x_scale, mask)
            
            # Weight by scale importance
            output = output * scale_weights[i]
            scale_outputs.append(output)
        
        # Process functional groups if provided
        if self.use_functional_groups and functional_groups is not None:
            # Convert functional group indices to embeddings
            fg_embeddings = self.functional_group_embedding(functional_groups)
            
            # Apply functional group attention
            fg_output = self.functional_group_attention(fg_embeddings, fg_embeddings, fg_embeddings, mask)
            scale_outputs.append(fg_output)
        
        # Concatenate outputs from all scales
        multi_scale_output = torch.cat(scale_outputs, dim=-1)
        
        # Integrate across scales
        output = self.integration(multi_scale_output)
        
        return output