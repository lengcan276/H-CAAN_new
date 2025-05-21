import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class ChemicalAwareAttention(nn.Module):
    """
    Chemical-Aware Attention module for enhancing cross-modal understanding
    of molecular substructures.
    
    This attention mechanism is specifically designed to identify and focus on
    chemically relevant parts of molecules across different molecular representations.
    """
    
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1, use_chemical_priors=True):
        """
        Initialize the Chemical-Aware Attention module.
        
        Args:
            hidden_dim (int): Dimension of the feature vectors
            num_heads (int): Number of attention heads
            dropout (float): Dropout probability
            use_chemical_priors (bool): Whether to use chemical prior knowledge
        """
        super(ChemicalAwareAttention, self).__init__()
        
        # Set parameters
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_chemical_priors = use_chemical_priors
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.head_dim = hidden_dim // num_heads
        
        # Multi-head attention layers
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Chemical prior knowledge
        if use_chemical_priors:
            # Embedding for common functional groups
            self.func_group_embedding = nn.Parameter(torch.randn(64, hidden_dim))
            
            # Chemical relevance scoring
            self.chemical_relevance = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Store attention weights
        self.attention_weights = None
    
    def forward(self, feature_list):
        """
        Forward pass through the Chemical-Aware Attention module.
        
        Args:
            feature_list (list): List of feature tensors from different modalities
                                [batch_size, hidden_dim]
        
        Returns:
            list: List of enhanced feature tensors
        """
        batch_size = feature_list[0].shape[0]
        num_modalities = len(feature_list)
        
        # Concatenate all features
        # Shape: [batch_size, num_modalities, hidden_dim]
        all_features = torch.stack(feature_list, dim=1)
        
        # Apply self-attention across modalities
        enhanced_features = self.cross_modal_attention(all_features)
        
        # If using chemical priors, apply chemical relevance weighting
        if self.use_chemical_priors:
            enhanced_features = self.apply_chemical_priors(enhanced_features, all_features)
        
        # Split back into list of features
        enhanced_feature_list = [enhanced_features[:, i] for i in range(num_modalities)]
        
        return enhanced_feature_list
    
    def cross_modal_attention(self, features):
        """
        Apply cross-modal self-attention to the features.
        
        Args:
            features (torch.Tensor): Feature tensor [batch_size, num_modalities, hidden_dim]
        
        Returns:
            torch.Tensor: Enhanced features [batch_size, num_modalities, hidden_dim]
        """
        batch_size, num_modalities, hidden_dim = features.shape
        
        # Apply layer normalization
        normed_features = self.norm1(features)
        
        # Project to queries, keys, and values
        q = self.q_proj(normed_features)  # [batch_size, num_modalities, hidden_dim]
        k = self.k_proj(normed_features)  # [batch_size, num_modalities, hidden_dim]
        v = self.v_proj(normed_features)  # [batch_size, num_modalities, hidden_dim]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, num_modalities, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)  # [batch_size, num_heads, num_modalities, head_dim]
        
        k = k.view(batch_size, num_modalities, self.num_heads, self.head_dim)
        k = k.permute(0, 2, 1, 3)  # [batch_size, num_heads, num_modalities, head_dim]
        
        v = v.view(batch_size, num_modalities, self.num_heads, self.head_dim)
        v = v.permute(0, 2, 1, 3)  # [batch_size, num_heads, num_modalities, head_dim]
        
        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-1, -2))  # [batch_size, num_heads, num_modalities, num_modalities]
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Store attention weights for interpretation
        self.attention_weights = attention_weights.detach()
        
        # Apply attention to values
        attended_values = torch.matmul(attention_weights, v)  # [batch_size, num_heads, num_modalities, head_dim]
        
        # Reshape back
        attended_values = attended_values.permute(0, 2, 1, 3)  # [batch_size, num_modalities, num_heads, head_dim]
        attended_values = attended_values.contiguous().view(batch_size, num_modalities, hidden_dim)
        
        # Project output
        output = self.output_proj(attended_values)
        output = self.dropout(output)
        
        # Add residual connection
        output = output + features
        
        # Apply feed-forward network with residual connection
        output2 = self.norm2(output)
        output2 = self.ffn(output2)
        output = output + output2
        
        return output
    
    def apply_chemical_priors(self, enhanced_features, original_features):
        """
        Apply chemical prior knowledge to enhance features.
        
        Args:
            enhanced_features (torch.Tensor): Enhanced features from attention
                                             [batch_size, num_modalities, hidden_dim]
            original_features (torch.Tensor): Original features
                                             [batch_size, num_modalities, hidden_dim]
        
        Returns:
            torch.Tensor: Chemically enhanced features [batch_size, num_modalities, hidden_dim]
        """
        batch_size, num_modalities, hidden_dim = enhanced_features.shape
        
        # Expand functional group embeddings
        func_groups = self.func_group_embedding.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_groups, hidden_dim]
        
        # Compute chemical relevance for each modality
        chemical_relevance_scores = []
        
        for i in range(num_modalities):
            # Get features for current modality
            modal_feats = original_features[:, i].unsqueeze(1)  # [batch_size, 1, hidden_dim]
            
            # Compute similarity to functional groups
            # [batch_size, num_groups, hidden_dim] x [batch_size, hidden_dim, 1]
            # = [batch_size, num_groups, 1]
            similarity = torch.bmm(func_groups, modal_feats.transpose(1, 2))
            
            # Normalize similarity
            similarity = torch.sigmoid(similarity / math.sqrt(hidden_dim))
            
            # Get max similarity across functional groups
            relevance = similarity.max(dim=1)[0]  # [batch_size, 1]
            
            chemical_relevance_scores.append(relevance)
        
        # Stack relevance scores
        chemical_relevance_scores = torch.cat(chemical_relevance_scores, dim=1)  # [batch_size, num_modalities]
        chemical_relevance_scores = chemical_relevance_scores.unsqueeze(-1)  # [batch_size, num_modalities, 1]
        
        # Apply chemical relevance weighting
        chemically_enhanced = original_features + chemical_relevance_scores * (enhanced_features - original_features)
        
        return chemically_enhanced
    
    def get_attention_weights(self):
        """
        Get the attention weights for interpretation.
        
        Returns:
            torch.Tensor: Attention weights
        """
        return self.attention_weights


class FunctionalGroupAttention(nn.Module):
    """
    An extension of ChemicalAwareAttention that specifically focuses on
    identifying and highlighting functional groups across different molecular
    representations.
    """
    
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1, num_functional_groups=32):
        """
        Initialize the FunctionalGroupAttention module.
        
        Args:
            hidden_dim (int): Dimension of the feature vectors
            num_heads (int): Number of attention heads
            dropout (float): Dropout probability
            num_functional_groups (int): Number of functional groups to identify
        """
        super(FunctionalGroupAttention, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_functional_groups = num_functional_groups
        
        # Base chemical-aware attention
        self.chemical_attention = ChemicalAwareAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_chemical_priors=True
        )
        
        # Functional group detection network
        self.functional_group_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_functional_groups),
            nn.Sigmoid()
        )
        
        # Functional group embedding
        self.functional_group_embedding = nn.Parameter(
            torch.randn(num_functional_groups, hidden_dim)
        )
        
        # Functional group attention
        self.func_group_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, feature_list):
        """
        Forward pass through the FunctionalGroupAttention module.
        
        Args:
            feature_list (list): List of feature tensors from different modalities
                                [batch_size, hidden_dim]
        
        Returns:
            list: List of enhanced feature tensors
        """
        batch_size = feature_list[0].shape[0]
        num_modalities = len(feature_list)
        
        # Apply base chemical-aware attention
        base_enhanced = self.chemical_attention(feature_list)
        
        # Detect functional groups in each modality
        functional_group_scores = []
        for features in feature_list:
            scores = self.functional_group_detector(features)  # [batch_size, num_functional_groups]
            functional_group_scores.append(scores)
        
        # Average functional group scores across modalities
        avg_functional_group_scores = torch.stack(functional_group_scores, dim=1).mean(dim=1)  # [batch_size, num_functional_groups]
        
        # Weight functional group embeddings
        weighted_func_groups = avg_functional_group_scores.unsqueeze(-1) * self.functional_group_embedding.unsqueeze(0)
        weighted_func_groups = weighted_func_groups.sum(dim=1)  # [batch_size, hidden_dim]
        
        # Enhanced features with functional group attention
        enhanced_feature_list = []
        
        for i, features in enumerate(base_enhanced):
            # Apply functional group attention
            attended_features, _ = self.func_group_attention(
                query=features.unsqueeze(1),
                key=weighted_func_groups.unsqueeze(1),
                value=weighted_func_groups.unsqueeze(1)
            )
            
            # Concatenate original and attended features
            combined = torch.cat([features, attended_features.squeeze(1)], dim=1)
            
            # Project to original dimension
            enhanced = self.output_proj(combined)
            enhanced = self.dropout(enhanced)
            
            # Add residual connection
            enhanced = self.norm(enhanced + features)
            
            enhanced_feature_list.append(enhanced)
        
        return enhanced_feature_list
    
    def get_functional_group_scores(self, feature_list):
        """
        Get the functional group detection scores for interpretation.
        
        Args:
            feature_list (list): List of feature tensors from different modalities
        
        Returns:
            torch.Tensor: Functional group scores [batch_size, num_modalities, num_functional_groups]
        """
        scores = []
        for features in feature_list:
            score = self.functional_group_detector(features)
            scores.append(score)
        
        return torch.stack(scores, dim=1)