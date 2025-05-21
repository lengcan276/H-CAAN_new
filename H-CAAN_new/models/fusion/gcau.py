import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GatedCrossModalAttentionUnit(nn.Module):
    """
    Gated Cross-modal Attention Unit (GCAU) for enabling targeted information
    flow between different modalities.
    
    This module implements a novel attention mechanism specifically designed for 
    cross-modal information exchange, with a chemical-aware gating mechanism that
    controls how much information should flow between modalities.
    """
    
    def __init__(self, query_dim, key_dim, hidden_dim=None, num_heads=4, dropout=0.1):
        """
        Initialize the GCAU module.
        
        Args:
            query_dim (int): Dimension of the query modality features
            key_dim (int): Dimension of the key modality features
            hidden_dim (int, optional): Dimension of the hidden layer. If None, use min(query_dim, key_dim)
            num_heads (int): Number of attention heads
            dropout (float): Dropout probability
        """
        super(GatedCrossModalAttentionUnit, self).__init__()
        
        # Set dimensions
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else min(query_dim, key_dim)
        self.num_heads = num_heads
        
        # Ensure hidden_dim is divisible by num_heads
        assert self.hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.head_dim = self.hidden_dim // num_heads
        
        # Layers for query modality
        self.query_proj = nn.Linear(query_dim, self.hidden_dim)
        self.query_norm = nn.LayerNorm(self.hidden_dim)
        
        # Layers for key modality
        self.key_proj = nn.Linear(key_dim, self.hidden_dim)
        self.key_norm = nn.LayerNorm(self.hidden_dim)
        
        # Value projections
        self.value_q_proj = nn.Linear(query_dim, self.hidden_dim)
        self.value_k_proj = nn.Linear(key_dim, self.hidden_dim)
        
        # Output projections
        self.output_proj = nn.Linear(self.hidden_dim, query_dim)
        self.output_norm = nn.LayerNorm(query_dim)
        
        # Gating mechanism
        self.gate_proj = nn.Sequential(
            nn.Linear(query_dim + key_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Chemical relevance scoring
        self.chemical_relevance = nn.Sequential(
            nn.Linear(query_dim + key_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # For storing attention weights
        self.attention_weights = None
    
    def forward(self, query_features, key_features):
        """
        Forward pass through the GCAU.
        
        Args:
            query_features (torch.Tensor): Features from the query modality [batch_size, query_dim]
            key_features (torch.Tensor): Features from the key modality [batch_size, key_dim]
            
        Returns:
            torch.Tensor: Enhanced query features with information from key modality
        """
        batch_size = query_features.size(0)
        
        # Project inputs
        q = self.query_proj(query_features)  # [batch_size, hidden_dim]
        k = self.key_proj(key_features)      # [batch_size, hidden_dim]
        
        # Apply layer normalization
        q = self.query_norm(q)
        k = self.key_norm(k)
        
        # Project values
        v_q = self.value_q_proj(query_features)  # [batch_size, hidden_dim]
        v_k = self.value_k_proj(key_features)    # [batch_size, hidden_dim]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, self.num_heads, self.head_dim)  # [batch_size, num_heads, head_dim]
        k = k.view(batch_size, self.num_heads, self.head_dim)  # [batch_size, num_heads, head_dim]
        v_k = v_k.view(batch_size, self.num_heads, self.head_dim)  # [batch_size, num_heads, head_dim]
        
        # Compute attention scores
        attention_scores = torch.einsum('bhd,bjd->bhj', q, k)  # [batch_size, num_heads, 1]
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)  # [batch_size, num_heads, 1]
        attention_weights = self.dropout(attention_weights)
        
        # Store attention weights for interpretation
        self.attention_weights = attention_weights.detach()
        
        # Apply attention to values from key modality
        attended_values = torch.einsum('bhj,bjd->bhd', attention_weights, v_k)  # [batch_size, num_heads, head_dim]
        
        # Reshape and project
        attended_values = attended_values.contiguous().view(batch_size, self.hidden_dim)  # [batch_size, hidden_dim]
        output = self.output_proj(attended_values)  # [batch_size, query_dim]
        
        # Apply layer normalization
        output = self.output_norm(output)
        
        # Compute gating value
        combined_features = torch.cat([query_features, key_features], dim=1)
        gate_value = self.gate_proj(combined_features)  # [batch_size, 1]
        
        # Compute chemical relevance score
        chemical_score = self.chemical_relevance(combined_features)  # [batch_size, 1]
        
        # Apply gating with chemical relevance weighting
        effective_gate = gate_value * chemical_score
        
        # Apply gated residual connection
        enhanced_features = query_features + effective_gate * output
        
        return enhanced_features
    
    def get_attention_weights(self):
        """
        Get the attention weights for interpretation.
        
        Returns:
            torch.Tensor: Attention weights [batch_size, num_heads, 1]
        """
        return self.attention_weights

class ChemicallyInformedGCAU(GatedCrossModalAttentionUnit):
    """
    An extension of GCAU with more specific chemical knowledge integration.
    This variant leverages domain-specific chemical properties and functional
    group information to enhance the cross-modal attention.
    """
    
    def __init__(self, query_dim, key_dim, hidden_dim=None, num_heads=4, dropout=0.1, 
                 functional_group_vocab_size=100):
        """
        Initialize the ChemicallyInformedGCAU module.
        
        Args:
            query_dim (int): Dimension of the query modality features
            key_dim (int): Dimension of the key modality features
            hidden_dim (int, optional): Dimension of the hidden layer. If None, use min(query_dim, key_dim)
            num_heads (int): Number of attention heads
            dropout (float): Dropout probability
            functional_group_vocab_size (int): Size of the functional group vocabulary
        """
        super(ChemicallyInformedGCAU, self).__init__(
            query_dim, key_dim, hidden_dim, num_heads, dropout
        )
        
        # Functional group embedding
        self.func_group_embedding = nn.Embedding(functional_group_vocab_size, self.hidden_dim)
        
        # Chemical knowledge integration
        self.chemical_knowledge_gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, query_features, key_features, functional_groups=None):
        """
        Forward pass through the ChemicallyInformedGCAU.
        
        Args:
            query_features (torch.Tensor): Features from the query modality [batch_size, query_dim]
            key_features (torch.Tensor): Features from the key modality [batch_size, key_dim]
            functional_groups (torch.LongTensor, optional): Indices of functional groups [batch_size, num_groups]
            
        Returns:
            torch.Tensor: Enhanced query features with information from key modality
        """
        batch_size = query_features.size(0)
        
        # Get base GCAU output
        base_output = super().forward(query_features, key_features)
        
        # If functional groups are provided, enhance the output further
        if functional_groups is not None:
            # Get functional group embeddings
            func_group_embeds = self.func_group_embedding(functional_groups)  # [batch_size, num_groups, hidden_dim]
            
            # Average over groups
            func_group_embeds = func_group_embeds.mean(dim=1)  # [batch_size, hidden_dim]
            
            # Project query features to the same space
            query_projected = self.query_proj(query_features)  # [batch_size, hidden_dim]
            
            # Combine with functional group information
            combined = torch.cat([query_projected, func_group_embeds], dim=1)  # [batch_size, hidden_dim*2]
            
            # Compute chemical knowledge gate
            chem_gate = self.chemical_knowledge_gate(combined)  # [batch_size, 1]
            
            # Apply chemical knowledge weighting
            final_output = query_features + chem_gate * (base_output - query_features)
            
            return final_output
        
        return base_output