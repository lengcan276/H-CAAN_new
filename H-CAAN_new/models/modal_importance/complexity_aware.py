import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors


class MolecularComplexityAssessment:
    """
    Utility class for assessing the complexity of molecular structures.
    
    This class provides methods to calculate various complexity metrics
    for molecules, which can be used to guide modality selection.
    """
    
    @staticmethod
    def calculate_complexity_score(mol, normalize=True):
        """
        Calculate a comprehensive complexity score for a molecule.
        
        Args:
            mol (rdkit.Chem.Mol): RDKit molecule object
            normalize (bool): Whether to normalize the score to [0, 1]
            
        Returns:
            float: Complexity score
        """
        if mol is None:
            return 0.0
        
        # Calculate basic descriptors
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()
        num_rings = rdMolDescriptors.CalcNumRings(mol)
        num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        num_heterocycles = rdMolDescriptors.CalcNumHeterocycles(mol)
        num_rotatable_bonds = Lipinski.NumRotatableBonds(mol)
        num_h_donors = Lipinski.NumHDonors(mol)
        num_h_acceptors = Lipinski.NumHAcceptors(mol)
        
        # Calculate additional complexity metrics
        bertz_index = Descriptors.BertzCT(mol)
        tpsa = Descriptors.TPSA(mol)
        complexity = Descriptors.MolWt(mol) * (0.5 + num_rings * 0.5) * (0.5 + num_rotatable_bonds * 0.1)
        
        # Combine metrics into a single score
        score = (
            0.1 * num_atoms +
            0.1 * num_bonds +
            0.2 * num_rings * (1 + 0.5 * num_aromatic_rings) * (1 + 0.3 * num_heterocycles) +
            0.15 * num_rotatable_bonds +
            0.05 * (num_h_donors + num_h_acceptors) +
            0.2 * bertz_index / 1000 +  # Normalize Bertz index
            0.1 * tpsa / 150 +  # Normalize TPSA
            0.1 * complexity / 1000  # Normalize complexity
        )
        
        # Normalize score if requested
        if normalize:
            # Apply sigmoid-like normalization
            score = 2.0 / (1.0 + np.exp(-0.05 * score)) - 1.0
        
        return float(score)
    
    @staticmethod
    def calculate_specific_complexity_metrics(mol):
        """
        Calculate specific complexity metrics for a molecule.
        
        Args:
            mol (rdkit.Chem.Mol): RDKit molecule object
            
        Returns:
            dict: Dictionary of complexity metrics
        """
        if mol is None:
            return {
                'structural_complexity': 0.0,
                'functional_complexity': 0.0,
                'topological_complexity': 0.0,
                'stereochemical_complexity': 0.0
            }
        
        # Structural complexity (based on atom and bond counts)
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()
        structural_complexity = (num_atoms + num_bonds) / 30.0  # Normalize
        
        # Functional complexity (based on functional groups)
        num_h_donors = Lipinski.NumHDonors(mol)
        num_h_acceptors = Lipinski.NumHAcceptors(mol)
        num_heteroatoms = len([atom for atom in mol.GetAtoms() if atom.GetAtomicNum() not in [1, 6]])
        functional_complexity = (num_h_donors + num_h_acceptors + num_heteroatoms) / 15.0  # Normalize
        
        # Topological complexity (based on rings and connectivity)
        num_rings = rdMolDescriptors.CalcNumRings(mol)
        num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        num_rotatable_bonds = Lipinski.NumRotatableBonds(mol)
        bertz_index = Descriptors.BertzCT(mol)
        topological_complexity = (
            (num_rings * 2.0 + num_aromatic_rings + num_rotatable_bonds * 0.5 + bertz_index / 1000.0) / 10.0
        )
        
        # Stereochemical complexity (based on stereocenters)
        num_stereocenters = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
        stereochemical_complexity = num_stereocenters / 5.0  # Normalize
        
        # Return all metrics as a dictionary
        return {
            'structural_complexity': min(1.0, structural_complexity),
            'functional_complexity': min(1.0, functional_complexity),
            'topological_complexity': min(1.0, topological_complexity),
            'stereochemical_complexity': min(1.0, stereochemical_complexity)
        }


class ComplexityAwareModalitySelection(nn.Module):
    """
    Module for selecting and weighting modalities based on molecular complexity.
    
    This module assesses molecular complexity and adjusts modality weights accordingly,
    as different modalities may be more effective for molecules of different complexities.
    """
    
    def __init__(self, n_modalities=4, complexity_dims=4, hidden_dim=64):
        """
        Initialize the complexity-aware modality selection module.
        
        Args:
            n_modalities (int): Number of modalities to weight
            complexity_dims (int): Number of complexity dimensions
            hidden_dim (int): Hidden layer dimension
        """
        super(ComplexityAwareModalitySelection, self).__init__()
        
        self.n_modalities = n_modalities
        self.complexity_dims = complexity_dims
        
        # Complexity to modality importance mapping
        self.complexity_to_weight = nn.Sequential(
            nn.Linear(complexity_dims, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_modalities)
        )
        
        # Optional learnable parameters for fixed modality preferences
        self.modality_preferences = nn.Parameter(torch.ones(n_modalities))
    
    def forward(self, complexity_vectors, feature_vectors=None):
        """
        Calculate modality weights based on molecular complexity.
        
        Args:
            complexity_vectors (torch.Tensor): Tensor of complexity metrics
                                              with shape [batch_size, complexity_dims]
            feature_vectors (list, optional): List of feature tensors from different modalities
            
        Returns:
            torch.Tensor: Weights for each modality with shape [batch_size, n_modalities]
        """
        batch_size = complexity_vectors.size(0)
        
        # Calculate raw weights based on complexity
        raw_weights = self.complexity_to_weight(complexity_vectors)
        
        # Apply modality preferences
        raw_weights = raw_weights + self.modality_preferences.unsqueeze(0)
        
        # Apply softmax to get final weights
        weights = F.softmax(raw_weights, dim=1)
        
        # Adjust weights based on feature vectors if provided
        if feature_vectors is not None:
            # Calculate feature quality (e.g., norm of feature vectors)
            feature_quality = torch.stack([
                torch.norm(features, dim=1)
                for features in feature_vectors
            ], dim=1)
            
            # Normalize feature quality
            feature_quality = feature_quality / (feature_quality.sum(dim=1, keepdim=True) + 1e-8)
            
            # Combine complexity-based weights with feature quality
            weights = weights * feature_quality
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        
        return weights


class LearningCurveBasedModalityImportance(nn.Module):
    """
    Module for dynamically adjusting modality importance based on learning curves.
    
    This module tracks the learning progress of each modality and adjusts their
    importance weights accordingly during training.
    """
    
    def __init__(self, n_modalities=4, hidden_dim=64, ema_alpha=0.9):
        """
        Initialize the learning curve based modality importance module.
        
        Args:
            n_modalities (int): Number of modalities to track
            hidden_dim (int): Hidden layer dimension
            ema_alpha (float): Exponential moving average decay factor
        """
        super(LearningCurveBasedModalityImportance, self).__init__()
        
        self.n_modalities = n_modalities
        self.ema_alpha = ema_alpha
        
        # Track loss history for each modality
        self.register_buffer('loss_history', torch.ones(n_modalities))
        self.register_buffer('loss_improvement', torch.zeros(n_modalities))
        
        # Learning rate for each modality
        self.learning_rates = nn.Parameter(torch.ones(n_modalities))
        
        # Mapping from learning curve to importance
        self.importance_net = nn.Sequential(
            nn.Linear(n_modalities * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_modalities)
        )
    
    def update_learning_curves(self, modality_losses):
        """
        Update learning curves based on current losses.
        
        Args:
            modality_losses (torch.Tensor): Current losses for each modality
                                           with shape [n_modalities]
        """
        # Calculate loss improvement (negative change in loss)
        improvement = self.loss_history - modality_losses
        
        # Update loss history with EMA
        self.loss_history = self.ema_alpha * self.loss_history + (1 - self.ema_alpha) * modality_losses
        
        # Update loss improvement with EMA
        self.loss_improvement = self.ema_alpha * self.loss_improvement + (1 - self.ema_alpha) * improvement
    
    def forward(self, modality_losses=None, complexity=None):
        """
        Calculate modality importance based on learning curves.
        
        Args:
            modality_losses (torch.Tensor, optional): Current losses for each modality
                                                    with shape [batch_size, n_modalities]
            complexity (torch.Tensor, optional): Complexity metrics
                                               with shape [batch_size, complexity_dims]
                                               
        Returns:
            torch.Tensor: Importance weights for each modality with shape [batch_size, n_modalities]
        """
        batch_size = modality_losses.size(0) if modality_losses is not None else 1
        
        # If modality losses are provided, update learning curves
        if modality_losses is not None:
            mean_losses = modality_losses.mean(dim=0)
            self.update_learning_curves(mean_losses)
        
        # Combine loss history and improvement for importance calculation
        learning_curve_features = torch.cat([
            self.loss_history.expand(batch_size, -1),
            self.loss_improvement.expand(batch_size, -1)
        ], dim=1)
        
        # Calculate importance based on learning curves
        raw_importance = self.importance_net(learning_curve_features)
        
        # Scale by learning rates
        raw_importance = raw_importance * self.learning_rates.expand(batch_size, -1)
        
        # Apply softmax to get final weights
        importance_weights = F.softmax(raw_importance, dim=1)
        
        return importance_weights


class TaskAdaptiveModalitySelection(nn.Module):
    """
    Module for adaptively selecting and weighting modalities based on the task.
    
    This module learns to assign different weights to modalities for different
    prediction tasks, recognizing that some modalities may be more informative
    for certain properties than others.
    """
    
    def __init__(self, n_modalities=4, n_tasks=1, task_embedding_dim=32, hidden_dim=64):
        """
        Initialize the task-adaptive modality selection module.
        
        Args:
            n_modalities (int): Number of modalities to weight
            n_tasks (int): Number of prediction tasks
            task_embedding_dim (int): Dimension of task embeddings
            hidden_dim (int): Hidden layer dimension
        """
        super(TaskAdaptiveModalitySelection, self).__init__()
        
        self.n_modalities = n_modalities
        self.n_tasks = n_tasks
        
        # Task embeddings (learnable)
        if n_tasks > 1:
            self.task_embeddings = nn.Parameter(torch.randn(n_tasks, task_embedding_dim))
        else:
            self.task_embeddings = nn.Parameter(torch.randn(1, task_embedding_dim))
        
        # Mapping from task embedding to modality weights
        self.task_to_weights = nn.Sequential(
            nn.Linear(task_embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_modalities)
        )
        
        # Base importance for each modality
        self.base_importance = nn.Parameter(torch.ones(n_modalities))
    
    def forward(self, task_indices=None):
        """
        Calculate modality weights based on the task.
        
        Args:
            task_indices (torch.Tensor, optional): Indices of tasks
                                                  with shape [batch_size]
                                                  
        Returns:
            torch.Tensor: Weights for each modality with shape [batch_size, n_modalities]
        """
        batch_size = task_indices.size(0) if task_indices is not None else 1
        
        # Get task embeddings
        if self.n_tasks > 1 and task_indices is not None:
            task_embs = self.task_embeddings[task_indices]
        else:
            task_embs = self.task_embeddings.expand(batch_size, -1)
        
        # Calculate raw weights based on task
        raw_weights = self.task_to_weights(task_embs)
        
        # Add base importance
        raw_weights = raw_weights + self.base_importance.unsqueeze(0)
        
        # Apply softmax to get final weights
        weights = F.softmax(raw_weights, dim=1)
        
        return weights


class DomainKnowledgeBasedWeights(nn.Module):
    """
    Module for incorporating domain knowledge into modality weighting.
    
    This module uses chemical domain knowledge to prioritize different modalities
    for different types of molecules or prediction tasks.
    """
    
    def __init__(self, n_modalities=4, n_property_types=5, hidden_dim=64):
        """
        Initialize the domain knowledge based weights module.
        
        Args:
            n_modalities (int): Number of modalities to weight
            n_property_types (int): Number of property types
            hidden_dim (int): Hidden layer dimension
        """
        super(DomainKnowledgeBasedWeights, self).__init__()
        
        self.n_modalities = n_modalities
        self.n_property_types = n_property_types
        
        # Domain knowledge lookup table
        # Maps property types to initial modality weights
        self.property_modality_weights = nn.Parameter(
            torch.ones(n_property_types, n_modalities)
        )
        
        # Learnable transformation for fine-tuning weights
        self.weight_transform = nn.Sequential(
            nn.Linear(n_modalities, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_modalities)
        )
    
    def forward(self, property_indices=None, mol_fingerprints=None):
        """
        Calculate modality weights based on domain knowledge.
        
        Args:
            property_indices (torch.Tensor, optional): Indices of property types
                                                      with shape [batch_size]
            mol_fingerprints (torch.Tensor, optional): Molecular fingerprints
                                                      with shape [batch_size, fingerprint_dim]
                                                      
        Returns:
            torch.Tensor: Weights for each modality with shape [batch_size, n_modalities]
        """
        batch_size = property_indices.size(0) if property_indices is not None else 1
        
        # Get initial weights from domain knowledge
        if property_indices is not None:
            initial_weights = self.property_modality_weights[property_indices]
        else:
            # Default to uniform weighting
            initial_weights = torch.ones(batch_size, self.n_modalities, device=self.property_modality_weights.device)
            initial_weights = initial_weights / self.n_modalities
        
        # Transform weights based on molecular fingerprints if available
        if mol_fingerprints is not None:
            # Project fingerprints to modality weights
            fp_weights = self.weight_transform(initial_weights)
            
            # Combine initial weights with fingerprint-based weights
            combined_weights = initial_weights + fp_weights
        else:
            combined_weights = initial_weights
        
        # Apply softmax to get final weights
        weights = F.softmax(combined_weights, dim=1)
        
        return weights


class AdaptiveModalityFusion(nn.Module):
    """
    A comprehensive module that combines multiple modality weighting approaches.
    
    This module integrates complexity-aware, learning-curve-based, task-adaptive,
    and domain-knowledge-based weighting to determine the optimal contribution 
    of each modality for molecular property prediction.
    """
    
    def __init__(self, n_modalities=4, n_tasks=1, n_property_types=5, 
                 complexity_dims=4, hidden_dim=64, fusion_method='weighted_sum'):
        """
        Initialize the adaptive modality fusion module.
        
        Args:
            n_modalities (int): Number of modalities to weight
            n_tasks (int): Number of prediction tasks
            n_property_types (int): Number of property types
            complexity_dims (int): Number of complexity dimensions
            hidden_dim (int): Hidden layer dimension
            fusion_method (str): Method for fusing modality outputs
                                ('weighted_sum', 'attention', 'gating')
        """
        super(AdaptiveModalityFusion, self).__init__()
        
        self.n_modalities = n_modalities
        self.fusion_method = fusion_method
        
        # Complexity-aware modality selection
        self.complexity_selector = ComplexityAwareModalitySelection(
            n_modalities=n_modalities,
            complexity_dims=complexity_dims,
            hidden_dim=hidden_dim
        )
        
        # Learning curve based modality importance
        self.learning_curve_weighter = LearningCurveBasedModalityImportance(
            n_modalities=n_modalities,
            hidden_dim=hidden_dim
        )
        
        # Task-adaptive modality selection
        self.task_selector = TaskAdaptiveModalitySelection(
            n_modalities=n_modalities,
            n_tasks=n_tasks,
            hidden_dim=hidden_dim
        )
        
        # Domain knowledge based weights
        self.domain_weighter = DomainKnowledgeBasedWeights(
            n_modalities=n_modalities,
            n_property_types=n_property_types,
            hidden_dim=hidden_dim
        )
        
        # Meta-weights for combining different weighting strategies
        self.meta_weights = nn.Parameter(torch.ones(4))
        
        # For attention-based fusion
        if fusion_method == 'attention':
            self.modality_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                batch_first=True
            )
            self.modality_projections = nn.ModuleList([
                nn.Linear(hidden_dim, hidden_dim) for _ in range(n_modalities)
            ])
        
        # For gating-based fusion
        elif fusion_method == 'gating':
            self.modality_gates = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Sigmoid()
                ) for _ in range(n_modalities)
            ])
    
    def compute_weights(self, complexity_vectors=None, modality_losses=None, 
                       task_indices=None, property_indices=None, mol_fingerprints=None,
                       feature_vectors=None):
        """
        Compute comprehensive modality weights using multiple strategies.
        
        Args:
            complexity_vectors (torch.Tensor, optional): Complexity metrics
            modality_losses (torch.Tensor, optional): Current losses for each modality
            task_indices (torch.Tensor, optional): Indices of tasks
            property_indices (torch.Tensor, optional): Indices of property types
            mol_fingerprints (torch.Tensor, optional): Molecular fingerprints
            feature_vectors (list, optional): List of feature tensors from different modalities
            
        Returns:
            torch.Tensor: Combined weights for each modality
        """
        weights_list = []
        
        # Get weights from each strategy if inputs are available
        if complexity_vectors is not None:
            complexity_weights = self.complexity_selector(complexity_vectors, feature_vectors)
            weights_list.append(complexity_weights)
        else:
            weights_list.append(None)
        
        if modality_losses is not None:
            learning_curve_weights = self.learning_curve_weighter(modality_losses)
            weights_list.append(learning_curve_weights)
        else:
            weights_list.append(None)
        
        if task_indices is not None:
            task_weights = self.task_selector(task_indices)
            weights_list.append(task_weights)
        else:
            weights_list.append(None)
        
        if property_indices is not None:
            domain_weights = self.domain_weighter(property_indices, mol_fingerprints)
            weights_list.append(domain_weights)
        else:
            weights_list.append(None)
        
        # Filter out None values
        valid_weights = [w for w in weights_list if w is not None]
        
        if not valid_weights:
            # If no valid weights, use uniform weighting
            batch_size = 1
            if complexity_vectors is not None:
                batch_size = complexity_vectors.size(0)
            elif modality_losses is not None:
                batch_size = modality_losses.size(0)
            
            return torch.ones(batch_size, self.n_modalities, device=self.meta_weights.device) / self.n_modalities
        
        # Get valid meta weights
        valid_meta_weights = torch.softmax(
            self.meta_weights[[i for i, w in enumerate(weights_list) if w is not None]],
            dim=0
        )
        
        # Combine weights using meta weights
        combined_weights = sum(w * m for w, m in zip(valid_weights, valid_meta_weights))
        
        # Ensure proper normalization
        combined_weights = combined_weights / (combined_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        return combined_weights
    
    def forward(self, modality_features, weights=None, **kwargs):
        """
        Fuse modality features based on computed weights.
        
        Args:
            modality_features (list): List of feature tensors from different modalities
                                     each with shape [batch_size, hidden_dim]
            weights (torch.Tensor, optional): Pre-computed modality weights
                                            with shape [batch_size, n_modalities]
            **kwargs: Additional arguments for weight computation
            
        Returns:
            torch.Tensor: Fused features with shape [batch_size, hidden_dim]
        """
        batch_size = modality_features[0].size(0)
        hidden_dim = modality_features[0].size(1)
        
        # Compute weights if not provided
        if weights is None:
            # Include feature vectors for quality-based weighting
            kwargs['feature_vectors'] = modality_features
            weights = self.compute_weights(**kwargs)
        
        if self.fusion_method == 'weighted_sum':
            # Simple weighted sum
            fused_features = sum(feat * weights[:, i:i+1] for i, feat in enumerate(modality_features))
            
        elif self.fusion_method == 'attention':
            # Project each modality's features
            projected_features = [
                proj(feat) for feat, proj in zip(modality_features, self.modality_projections)
            ]
            
            # Stack features into a sequence
            sequence = torch.stack(projected_features, dim=1)  # [batch_size, n_modalities, hidden_dim]
            
            # Apply attention with modality weights as attention bias
            attention_mask = torch.ones(batch_size, self.n_modalities, self.n_modalities, device=weights.device)
            attention_mask = attention_mask * weights.unsqueeze(1)  # Broadcast weights across rows
            
            attn_output, _ = self.modality_attention(sequence, sequence, sequence, 
                                                   attn_mask=attention_mask)
            
            # Fuse attended features
            fused_features = attn_output.mean(dim=1)  # [batch_size, hidden_dim]
            
        elif self.fusion_method == 'gating':
            # Apply gates to each modality's features
            gated_features = [
                gate(feat) * feat for feat, gate in zip(modality_features, self.modality_gates)
            ]
            
            # Weighted sum of gated features
            fused_features = sum(feat * weights[:, i:i+1] for i, feat in enumerate(gated_features))
        
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        
        return fused_features