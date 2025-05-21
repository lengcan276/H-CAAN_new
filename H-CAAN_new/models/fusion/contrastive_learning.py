import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CrossModalContrastiveLoss(nn.Module):
    """
    Implementation of Cross-Modal Contrastive Learning for molecular representations.
    
    This module applies contrastive learning across different modalities by maximizing
    agreement between different representations of the same molecule while minimizing
    similarity between representations of different molecules.
    """
    
    def __init__(self, temperature=0.07, base_temperature=0.07, reduction='mean'):
        """
        Initialize the cross-modal contrastive loss.
        
        Args:
            temperature (float): The temperature parameter for softmax scaling
            base_temperature (float): The base temperature parameter for scaling
            reduction (str): Specifies the reduction to apply to the output ('none', 'mean', 'sum')
        """
        super(CrossModalContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.reduction = reduction
    
    def forward(self, features_list, batch_size=None):
        """
        Calculate the cross-modal contrastive loss across multiple modality features.
        
        Args:
            features_list (list): List of feature tensors from different modalities
                                 [tensor1, tensor2, ...], each with shape [batch_size, feature_dim]
            batch_size (int, optional): Batch size. If None, inferred from features
            
        Returns:
            torch.Tensor: Contrastive loss value
        """
        if batch_size is None:
            batch_size = features_list[0].shape[0]
        
        device = features_list[0].device
        
        # Normalize all features to unit vectors
        normalized_features = []
        for features in features_list:
            norm_features = F.normalize(features, dim=1)
            normalized_features.append(norm_features)
        
        # Calculate contrastive loss across all modality pairs
        loss = torch.tensor(0.0, device=device)
        num_modalities = len(normalized_features)
        
        for i in range(num_modalities):
            for j in range(i+1, num_modalities):
                anchor = normalized_features[i]
                contrast = normalized_features[j]
                
                # Calculate similarity matrix
                similarity = torch.matmul(anchor, contrast.T) / self.temperature
                
                # For numerical stability
                sim_max, _ = torch.max(similarity, dim=1, keepdim=True)
                similarity = similarity - sim_max.detach()
                
                # Create labels: positives are diagonal elements (same molecule in different modalities)
                labels = torch.arange(batch_size, device=device)
                
                # Calculate cross-entropy loss
                exp_sim = torch.exp(similarity)
                pos_sim = torch.exp(torch.gather(similarity, 1, labels.view(-1, 1)))
                
                loss_i_to_j = -torch.log(pos_sim / exp_sim.sum(dim=1, keepdim=True))
                
                # Calculate loss from j to i (symmetric)
                similarity_T = similarity.T
                sim_max_T, _ = torch.max(similarity_T, dim=1, keepdim=True)
                similarity_T = similarity_T - sim_max_T.detach()
                
                exp_sim_T = torch.exp(similarity_T)
                pos_sim_T = torch.exp(torch.gather(similarity_T, 1, labels.view(-1, 1)))
                
                loss_j_to_i = -torch.log(pos_sim_T / exp_sim_T.sum(dim=1, keepdim=True))
                
                # Combine losses
                if self.reduction == 'mean':
                    loss_pair = (loss_i_to_j.mean() + loss_j_to_i.mean()) / 2
                elif self.reduction == 'sum':
                    loss_pair = (loss_i_to_j.sum() + loss_j_to_i.sum()) / 2
                else:  # 'none'
                    loss_pair = (loss_i_to_j + loss_j_to_i) / 2
                
                loss += loss_pair
        
        # Average over all modality pairs
        num_pairs = num_modalities * (num_modalities - 1) // 2
        loss = loss / num_pairs
        
        # Apply temperature scaling
        loss = loss * (self.temperature / self.base_temperature)
        
        return loss


class ModalAlignmentModule(nn.Module):
    """
    Module for aligning and contrasting features across different modalities.
    
    This module helps in learning cross-modal representations by aligning 
    features from different modalities and applying contrastive learning.
    """
    
    def __init__(self, input_dims, projection_dim=128, temperature=0.07):
        """
        Initialize the modal alignment module.
        
        Args:
            input_dims (list): List of input dimensions for each modality
            projection_dim (int): Dimension of the projection space
            temperature (float): Temperature parameter for contrastive loss
        """
        super(ModalAlignmentModule, self).__init__()
        
        # Create projection heads for each modality
        self.projection_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, projection_dim * 2),
                nn.BatchNorm1d(projection_dim * 2),
                nn.ReLU(inplace=True),
                nn.Linear(projection_dim * 2, projection_dim)
            ) for dim in input_dims
        ])
        
        # Contrastive loss
        self.contrastive_loss = CrossModalContrastiveLoss(temperature=temperature)
    
    def forward(self, features_list):
        """
        Project features from different modalities and compute contrastive loss.
        
        Args:
            features_list (list): List of feature tensors from different modalities
            
        Returns:
            tuple: (projected_features, contrastive_loss)
                - projected_features: List of projected feature tensors
                - contrastive_loss: Contrastive loss value
        """
        # Project each modality's features
        projected_features = [
            head(features) for head, features in zip(self.projection_heads, features_list)
        ]
        
        # Compute contrastive loss
        loss = self.contrastive_loss(projected_features)
        
        return projected_features, loss


class HierarchicalContrastiveLearning(nn.Module):
    """
    Hierarchical Contrastive Learning module for molecular representations.
    
    This module performs contrastive learning at multiple levels of the feature hierarchy,
    helping to learn better aligned representations at different semantic levels.
    """
    
    def __init__(self, feature_dims, hierarchy_levels=3, temperature=0.07):
        """
        Initialize the hierarchical contrastive learning module.
        
        Args:
            feature_dims (list): List of feature dimensions for each modality at input level
            hierarchy_levels (int): Number of hierarchy levels for contrastive learning
            temperature (float): Temperature parameter for contrastive loss
        """
        super(HierarchicalContrastiveLearning, self).__init__()
        
        self.hierarchy_levels = hierarchy_levels
        
        # Create alignment modules for each level in the hierarchy
        self.alignment_modules = nn.ModuleList()
        
        current_dims = feature_dims
        for level in range(hierarchy_levels):
            # Define projection dimension based on level
            proj_dim = 128 // (2 ** level)
            
            # Create alignment module for current level
            self.alignment_modules.append(
                ModalAlignmentModule(
                    input_dims=current_dims,
                    projection_dim=proj_dim,
                    temperature=temperature
                )
            )
            
            # Update dimensions for next level (assuming aligned dimensions at each level)
            current_dims = [proj_dim] * len(feature_dims)
    
    def forward(self, features_hierarchy):
        """
        Apply hierarchical contrastive learning to feature hierarchy.
        
        Args:
            features_hierarchy (list): List of feature lists at each hierarchy level
                                      [level1_features, level2_features, ...]
                                      where each level_features is a list of tensors
            
        Returns:
            tuple: (aligned_features_hierarchy, total_loss)
                - aligned_features_hierarchy: List of aligned feature lists at each level
                - total_loss: Total contrastive loss across all levels
        """
        total_loss = 0
        aligned_features = []
        
        # Process only available levels in hierarchy
        levels_to_process = min(self.hierarchy_levels, len(features_hierarchy))
        
        for level in range(levels_to_process):
            level_features = features_hierarchy[level]
            
            # Apply alignment module
            projected_features, level_loss = self.alignment_modules[level](level_features)
            
            # Add to aligned features
            aligned_features.append(projected_features)
            
            # Accumulate loss (with optional level weighting)
            level_weight = 1.0 / (2 ** level)  # Higher weight for earlier levels
            total_loss += level_weight * level_loss
        
        return aligned_features, total_loss


def create_molecular_feature_projector(input_dim, output_dim):
    """
    Create a projection network for molecular feature transformation.
    
    Args:
        input_dim (int): Input feature dimension
        output_dim (int): Output feature dimension
        
    Returns:
        nn.Module: Feature projection network
    """
    return nn.Sequential(
        nn.Linear(input_dim, input_dim // 2),
        nn.LayerNorm(input_dim // 2),
        nn.ReLU(inplace=True),
        nn.Linear(input_dim // 2, output_dim),
        nn.LayerNorm(output_dim)
    )