import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TaskSpecificWeightGenerator(nn.Module):
    """
    Generates task-specific weights for different modalities based on their relevance
    to the specific prediction task.
    
    This module learns to dynamically weight the importance of each modality based on
    input features and task requirements.
    """
    
    def __init__(self, n_modalities, feature_dim, hidden_dim=64, temperature=1.0, 
                 gumbel_softmax=False, hard=False, use_gate=True):
        """
        Initialize the task-specific weight generator.
        
        Args:
            n_modalities (int): Number of modalities to weight
            feature_dim (int): Dimension of input features
            hidden_dim (int): Hidden layer dimension
            temperature (float): Temperature parameter for softmax/gumbel-softmax
            gumbel_softmax (bool): Whether to use Gumbel-Softmax instead of regular softmax
            hard (bool): Whether to use hard Gumbel-Softmax (only applicable if gumbel_softmax=True)
            use_gate (bool): Whether to use gating mechanism
        """
        super(TaskSpecificWeightGenerator, self).__init__()
        
        self.n_modalities = n_modalities
        self.temperature = temperature
        self.gumbel_softmax = gumbel_softmax
        self.hard = hard
        self.use_gate = use_gate
        
        # Feature compression network
        self.feature_net = nn.Sequential(
            nn.Linear(feature_dim * n_modalities, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Weight prediction network
        self.weight_net = nn.Linear(hidden_dim, n_modalities)
        
        # Optional gating mechanism
        if use_gate:
            self.gate_net = nn.Sequential(
                nn.Linear(feature_dim * n_modalities, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, n_modalities),
                nn.Sigmoid()
            )
    
    def forward(self, features_list, task_embedding=None):
        """
        Generate weights for each modality based on input features.
        
        Args:
            features_list (list): List of feature tensors from different modalities,
                                 each with shape [batch_size, feature_dim]
            task_embedding (torch.Tensor, optional): Optional task embedding for
                                                    task-specific weighting
            
        Returns:
            tuple: (modality_weights, raw_weights)
                - modality_weights: Normalized weights for each modality
                - raw_weights: Raw weights before normalization
        """
        batch_size = features_list[0].size(0)
        
        # Concatenate all features
        concat_features = torch.cat(features_list, dim=1)
        
        # Add task embedding if provided
        if task_embedding is not None:
            concat_features = torch.cat([concat_features, task_embedding], dim=1)
        
        # Process features
        hidden = self.feature_net(concat_features)
        
        # Predict raw weights
        raw_weights = self.weight_net(hidden)
        
        # Apply temperature scaling
        scaled_weights = raw_weights / self.temperature
        
        # Apply softmax or gumbel-softmax
        if self.gumbel_softmax:
            modality_weights = F.gumbel_softmax(scaled_weights, tau=self.temperature, hard=self.hard, dim=1)
        else:
            modality_weights = F.softmax(scaled_weights, dim=1)
        
        # Apply gating if enabled
        if self.use_gate:
            gates = self.gate_net(concat_features)
            modality_weights = modality_weights * gates
            
            # Re-normalize if needed
            if not self.gumbel_softmax:
                weight_sum = modality_weights.sum(dim=1, keepdim=True)
                weight_sum = torch.clamp(weight_sum, min=1e-6)  # Avoid division by zero
                modality_weights = modality_weights / weight_sum
        
        return modality_weights, raw_weights


class AdaptiveModalityWeighting(nn.Module):
    """
    Adaptive modality weighting module that adjusts weights based on both
    task requirements and input features.
    
    This module provides a more flexible weighting mechanism compared to
    the basic TaskSpecificWeightGenerator.
    """
    
    def __init__(self, n_modalities, feature_dims, hidden_dim=64, 
                 use_attention=True, use_feature_correlations=True):
        """
        Initialize the adaptive modality weighting module.
        
        Args:
            n_modalities (int): Number of modalities to weight
            feature_dims (list): List of feature dimensions for each modality
            hidden_dim (int): Hidden layer dimension
            use_attention (bool): Whether to use attention mechanism
            use_feature_correlations (bool): Whether to use feature correlations
        """
        super(AdaptiveModalityWeighting, self).__init__()
        
        self.n_modalities = n_modalities
        self.feature_dims = feature_dims
        self.use_attention = use_attention
        self.use_feature_correlations = use_feature_correlations
        
        # Feature projections to common dimension
        self.feature_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True)
            )
            for dim in feature_dims
        ])
        
        # Attention-based weighting (if enabled)
        if use_attention:
            self.query_proj = nn.Linear(hidden_dim, hidden_dim)
            self.key_projs = nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim)
                for _ in range(n_modalities)
            ])
            self.attention_scale = hidden_dim ** -0.5
        
        # Feature correlation module (if enabled)
        if use_feature_correlations:
            self.correlation_projections = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.LayerNorm(hidden_dim // 2),
                    nn.ReLU(inplace=True)
                )
                for _ in range(n_modalities)
            ])
            
            self.correlation_importance = nn.Sequential(
                nn.Linear(n_modalities * (n_modalities - 1) // 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, n_modalities),
                nn.Softmax(dim=1)
            )
        
        # Final weight generator
        self.weight_generator = nn.Sequential(
            nn.Linear(n_modalities * hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, n_modalities),
            nn.Softmax(dim=1)
        )
    
    def forward(self, features_list, task_embedding=None):
        """
        Generate adaptive weights for each modality.
        
        Args:
            features_list (list): List of feature tensors from different modalities
            task_embedding (torch.Tensor, optional): Optional task embedding
            
        Returns:
            torch.Tensor: Weights for each modality with shape [batch_size, n_modalities]
        """
        batch_size = features_list[0].size(0)
        
        # Project features to common dimension
        projected_features = [
            projection(features)
            for features, projection in zip(features_list, self.feature_projections)
        ]
        
        # Calculate attention-based weights (if enabled)
        if self.use_attention:
            # Create global query from average of all features
            global_feature = torch.stack(projected_features, dim=1).mean(dim=1)
            query = self.query_proj(global_feature)
            
            # Calculate attention scores
            attention_scores = []
            for i in range(self.n_modalities):
                key = self.key_projs[i](projected_features[i])
                # Calculate attention score (dot product)
                score = torch.sum(query * key, dim=1) * self.attention_scale
                attention_scores.append(score)
            
            # Stack and normalize scores
            attention_scores = torch.stack(attention_scores, dim=1)
            attention_weights = F.softmax(attention_scores, dim=1)
        else:
            attention_weights = torch.ones(batch_size, self.n_modalities, device=features_list[0].device)
            attention_weights = attention_weights / self.n_modalities
        
        # Calculate feature correlation importance (if enabled)
        if self.use_feature_correlations:
            # Project features for correlation calculation
            corr_features = [
                projection(features)
                for features, projection in zip(projected_features, self.correlation_projections)
            ]
            
            # Calculate pairwise correlations
            correlations = []
            for i in range(self.n_modalities):
                for j in range(i+1, self.n_modalities):
                    # Normalized correlation
                    f_i = F.normalize(corr_features[i], p=2, dim=1)
                    f_j = F.normalize(corr_features[j], p=2, dim=1)
                    corr = torch.sum(f_i * f_j, dim=1)
                    correlations.append(corr)
            
            # Stack correlations
            correlations = torch.stack(correlations, dim=1)
            
            # Calculate correlation-based importance
            correlation_weights = self.correlation_importance(correlations)
        else:
            correlation_weights = torch.ones(batch_size, self.n_modalities, device=features_list[0].device)
            correlation_weights = correlation_weights / self.n_modalities
        
        # Combine attention and correlation weights
        if self.use_attention and self.use_feature_correlations:
            combined_weights = (attention_weights + correlation_weights) / 2
        elif self.use_attention:
            combined_weights = attention_weights
        elif self.use_feature_correlations:
            combined_weights = correlation_weights
        else:
            combined_weights = torch.ones(batch_size, self.n_modalities, device=features_list[0].device)
            combined_weights = combined_weights / self.n_modalities
        
        # Concatenate all projected features
        concat_features = torch.cat(projected_features, dim=1)
        
        # Generate final weights
        final_weights = self.weight_generator(concat_features)
        
        # Blend with combined weights
        weights = (final_weights + combined_weights) / 2
        
        return weights


class MultiTaskWeightGenerator(nn.Module):
    """
    Generates task-specific weights for different modalities in a multi-task setting.
    
    This module extends the TaskSpecificWeightGenerator to handle multiple tasks
    simultaneously, allowing different weightings for each task.
    """
    
    def __init__(self, n_modalities, n_tasks, feature_dim, hidden_dim=64, 
                 share_feature_net=True, temperature=1.0):
        """
        Initialize the multi-task weight generator.
        
        Args:
            n_modalities (int): Number of modalities to weight
            n_tasks (int): Number of tasks
            feature_dim (int): Dimension of input features
            hidden_dim (int): Hidden layer dimension
            share_feature_net (bool): Whether to share feature network across tasks
            temperature (float): Temperature parameter for softmax
        """
        super(MultiTaskWeightGenerator, self).__init__()
        
        self.n_modalities = n_modalities
        self.n_tasks = n_tasks
        self.share_feature_net = share_feature_net
        self.temperature = temperature
        
        # Shared feature network (if enabled)
        if share_feature_net:
            self.feature_net = nn.Sequential(
                nn.Linear(feature_dim * n_modalities, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True)
            )
        else:
            self.feature_nets = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(feature_dim * n_modalities, hidden_dim * 2),
                    nn.LayerNorm(hidden_dim * 2),
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(inplace=True)
                )
                for _ in range(n_tasks)
            ])
        
        # Task-specific embedding
        self.task_embeddings = nn.Parameter(torch.randn(n_tasks, hidden_dim))
        
        # Weight prediction networks (one per task)
        self.weight_nets = nn.ModuleList([
            nn.Linear(hidden_dim * 2, n_modalities)
            for _ in range(n_tasks)
        ])
    
    def forward(self, features_list, task_ids=None):
        """
        Generate weights for each modality for specified tasks.
        
        Args:
            features_list (list): List of feature tensors from different modalities,
                                 each with shape [batch_size, feature_dim]
            task_ids (torch.Tensor, optional): Task indices of shape [batch_size]
                                            If None, weights for all tasks are generated
            
        Returns:
            torch.Tensor: Weights for each modality and task with shape:
                         - If task_ids is provided: [batch_size, n_modalities]
                         - If task_ids is None: [batch_size, n_tasks, n_modalities]
        """
        batch_size = features_list[0].size(0)
        
        # Concatenate all features
        concat_features = torch.cat(features_list, dim=1)
        
        # Process features (shared or task-specific)
        if self.share_feature_net:
            hidden = self.feature_net(concat_features)
            
            # Repeat for all tasks if task_ids is None
            if task_ids is None:
                hidden = hidden.unsqueeze(1).repeat(1, self.n_tasks, 1)
        else:
            if task_ids is None:
                # Process features for all tasks
                hidden = torch.stack([
                    feature_net(concat_features)
                    for feature_net in self.feature_nets
                ], dim=1)
            else:
                # Process features only for specified tasks
                hidden = torch.stack([
                    self.feature_nets[task_id](concat_features[i:i+1])
                    for i, task_id in enumerate(task_ids)
                ], dim=0).squeeze(1)
        
        # Get task embeddings
        if task_ids is None:
            # Use all task embeddings
            task_emb = self.task_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            # Use specified task embeddings
            task_emb = self.task_embeddings[task_ids]
        
        # Combine hidden and task embeddings
        if task_ids is None:
            combined = torch.cat([hidden, task_emb], dim=2)
        else:
            combined = torch.cat([hidden, task_emb], dim=1)
        
        # Predict weights
        if task_ids is None:
            # Predict weights for all tasks
            raw_weights = torch.stack([
                self.weight_nets[i](combined[:, i])
                for i in range(self.n_tasks)
            ], dim=1)
        else:
            # Predict weights only for specified tasks
            raw_weights = torch.stack([
                self.weight_nets[task_id](combined[i:i+1])
                for i, task_id in enumerate(task_ids)
            ], dim=0).squeeze(1)
        
        # Apply temperature scaling and softmax
        if task_ids is None:
            # For all tasks
            weights = F.softmax(raw_weights / self.temperature, dim=2)
        else:
            # For specified tasks
            weights = F.softmax(raw_weights / self.temperature, dim=1)
        
        return weights


class TaskSpecificGatedFusion(nn.Module):
    """
    Task-specific gated fusion module that combines features from different modalities
    using task-specific gates.
    
    This module uses both task-specific weights and gating mechanisms to fuse
    multimodal features effectively.
    """
    
    def __init__(self, n_modalities, feature_dims, output_dim, hidden_dim=128):
        """
        Initialize the task-specific gated fusion module.
        
        Args:
            n_modalities (int): Number of modalities to fuse
            feature_dims (list): List of feature dimensions for each modality
            output_dim (int): Output feature dimension
            hidden_dim (int): Hidden layer dimension
        """
        super(TaskSpecificGatedFusion, self).__init__()
        
        self.n_modalities = n_modalities
        self.feature_dims = feature_dims
        self.output_dim = output_dim
        
        # Feature projections to common dimension
        self.feature_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True)
            )
            for dim in feature_dims
        ])
        
        # Gate generators
        self.gate_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(sum(feature_dims), hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid()
            )
            for _ in range(n_modalities)
        ])
        
        # Fusion network
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_dim * n_modalities, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, output_dim)
        )
        
        # Task-specific weight generator
        self.weight_generator = TaskSpecificWeightGenerator(
            n_modalities=n_modalities,
            feature_dim=sum(feature_dims) // n_modalities,
            hidden_dim=hidden_dim // 2
        )
    
    def forward(self, features_list, task_embedding=None):
        """
        Fuse features from different modalities using task-specific gates.
        
        Args:
            features_list (list): List of feature tensors from different modalities
            task_embedding (torch.Tensor, optional): Optional task embedding
            
        Returns:
            torch.Tensor: Fused features of shape [batch_size, output_dim]
        """
        batch_size = features_list[0].size(0)
        
        # Project features to common dimension
        projected_features = [
            projection(features)
            for features, projection in zip(features_list, self.feature_projections)
        ]
        
        # Concatenate all raw features for gate generation
        concat_features = torch.cat(features_list, dim=1)
        
        # Generate gates for each modality
        gates = [
            gate_gen(concat_features)
            for gate_gen in self.gate_generators
        ]
        
        # Apply gates to projected features
        gated_features = [
            features * gate
            for features, gate in zip(projected_features, gates)
        ]
        
        # Generate task-specific weights
        weights, _ = self.weight_generator(features_list, task_embedding)
        
        # Apply weights and stack features
        weighted_gated_features = [
            gated * weights[:, i:i+1]
            for i, gated in enumerate(gated_features)
        ]
        
        # Concatenate weighted and gated features
        fused_features = torch.cat(weighted_gated_features, dim=1)
        
        # Apply fusion network
        output = self.fusion_net(fused_features)
        
        return output, weights