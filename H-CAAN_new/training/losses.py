import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification tasks
    
    FL(p_t) = -alpha * (1 - p_t) ^ gamma * log(p_t)
    
    Args:
        alpha: Weighting factor for the rare class
        gamma: Focusing parameter
        reduction: Reduction method ('mean', 'sum', or 'none')
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Forward pass
        
        Args:
            inputs: Predicted probabilities (N,)
            targets: Target values (N,)
            
        Returns:
            Loss value
        """
        # Convert targets to float for calculations
        targets = targets.float()
        
        # Apply sigmoid to raw logits if needed
        if not (inputs.min() >= 0 and inputs.max() <= 1):
            inputs = torch.sigmoid(inputs)
        
        # Calculate binary cross entropy
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        
        # Calculate focal term
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        focal_term = (1 - pt) ** self.gamma
        
        # Calculate alpha term
        alpha_term = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Calculate focal loss
        focal_loss = alpha_term * focal_term * BCE_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for regression and classification tasks
    
    Supports task-specific weights and uncertainty-based weighting
    
    Args:
        task_types: List of task types ('regression' or 'classification')
        task_weights: Initial weights for each task
        learn_weights: Whether to learn task weights automatically
        reduction: Reduction method ('mean', 'sum', or 'none')
    """
    def __init__(self, task_types, task_weights=None, learn_weights=True, reduction='mean'):
        super(MultiTaskLoss, self).__init__()
        self.task_types = task_types
        self.num_tasks = len(task_types)
        self.reduction = reduction
        self.learn_weights = learn_weights
        
        # Initialize task weights
        if task_weights is None:
            task_weights = torch.ones(self.num_tasks)
        
        # Initialize log task variances for uncertainty weighting
        if learn_weights:
            self.log_vars = nn.Parameter(torch.zeros(self.num_tasks))
        else:
            self.register_buffer('task_weights', torch.tensor(task_weights, dtype=torch.float))
    
    def forward(self, outputs, targets):
        """
        Forward pass
        
        Args:
            outputs: List of task-specific outputs
            targets: List of task-specific targets
            
        Returns:
            Loss value
        """
        # Calculate losses for each task
        task_losses = []
        for i, (task_type, output, target) in enumerate(zip(self.task_types, outputs, targets)):
            if task_type == 'regression':
                # Mean squared error for regression tasks
                loss = F.mse_loss(output, target, reduction='none')
            else:
                # Binary cross entropy for binary classification tasks
                loss = F.binary_cross_entropy_with_logits(output, target, reduction='none')
            
            task_losses.append(loss.mean(dim=0, keepdim=True))
        
        # Stack task losses
        stacked_task_losses = torch.cat(task_losses)
        
        # Apply task weights
        if self.learn_weights:
            # Uncertainty weighting (Kendall et al., 2018)
            weighted_losses = 0.5 * torch.exp(-self.log_vars) * stacked_task_losses + 0.5 * self.log_vars
        else:
            weighted_losses = self.task_weights * stacked_task_losses
        
        # Apply reduction
        if self.reduction == 'mean':
            return weighted_losses.mean()
        elif self.reduction == 'sum':
            return weighted_losses.sum()
        else:
            return weighted_losses


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for multimodal representation learning
    
    Encourages representations from different modalities of the same molecule to be similar,
    while representations of different molecules should be dissimilar.
    
    Args:
        temperature: Temperature parameter for softmax
        margin: Margin for negative pairs
        reduction: Reduction method ('mean', 'sum', or 'none')
    """
    def __init__(self, temperature=0.5, margin=1.0, reduction='mean'):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, modal_embeddings, batch_indices):
        """
        Forward pass
        
        Args:
            modal_embeddings: List of embeddings from different modalities [N, D]
            batch_indices: Batch indices to identify the same molecules
            
        Returns:
            Loss value
        """
        losses = []
        
        # Process each modality pair
        num_modalities = len(modal_embeddings)
        for i in range(num_modalities):
            for j in range(i + 1, num_modalities):
                # Get embeddings for the pair of modalities
                z_i = F.normalize(modal_embeddings[i], dim=1)
                z_j = F.normalize(modal_embeddings[j], dim=1)
                
                # Compute similarity matrix
                sim_matrix = torch.matmul(z_i, z_j.t()) / self.temperature
                
                # Create positive and negative masks
                batch_size = z_i.size(0)
                pos_mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
                
                # Set positive pairs based on batch indices
                for k in range(batch_size):
                    pos_indices = (batch_indices == batch_indices[k]).nonzero(as_tuple=True)[0]
                    pos_mask[k, pos_indices] = True
                
                # Exclude self-pairs
                pos_mask.fill_diagonal_(False)
                neg_mask = ~pos_mask
                
                # Compute positive and negative scores
                pos_scores = sim_matrix[pos_mask]
                neg_scores = sim_matrix[neg_mask]
                
                # Compute InfoNCE loss
                pos_loss = -torch.mean(pos_scores)
                
                # Compute hinge loss for negative pairs
                neg_loss = torch.mean(torch.clamp(self.margin + neg_scores, min=0))
                
                # Combine positive and negative losses
                loss = pos_loss + neg_loss
                losses.append(loss)
        
        # Combine losses from all modality pairs
        combined_loss = torch.stack(losses)
        
        # Apply reduction
        if self.reduction == 'mean':
            return combined_loss.mean()
        elif self.reduction == 'sum':
            return combined_loss.sum()
        else:
            return combined_loss


class HierarchicalModularLoss(nn.Module):
    """
    Hierarchical loss for H-CAAN
    
    Combines losses from different modality pairs at each fusion level
    
    Args:
        task_type: Type of primary task ('regression' or 'classification')
        aux_weight: Weight for auxiliary losses
        contrastive_weight: Weight for contrastive loss
        reduction: Reduction method ('mean', 'sum', or 'none')
    """
    def __init__(self, task_type='regression', aux_weight=0.1, contrastive_weight=0.05, reduction='mean'):
        super(HierarchicalModularLoss, self).__init__()
        self.task_type = task_type
        self.aux_weight = aux_weight
        self.contrastive_weight = contrastive_weight
        self.reduction = reduction
        
        # Primary loss function
        if task_type == 'regression':
            self.primary_loss_fn = nn.MSELoss(reduction='none')
        else:
            self.primary_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        
        # Auxiliary loss function for modality alignment
        self.aux_loss_fn = nn.MSELoss(reduction='none')
        
        # Contrastive loss for multimodal representation learning
        self.contrastive_loss_fn = ContrastiveLoss(reduction='none')
    
    def forward(self, outputs, targets, modal_outputs=None, modal_embeddings=None, batch_indices=None):
        """
        Forward pass
        
        Args:
            outputs: Final outputs for the primary task
            targets: Targets for the primary task
            modal_outputs: Outputs from individual modalities (optional)
            modal_embeddings: Embeddings from different modalities (optional)
            batch_indices: Batch indices for contrastive loss (optional)
            
        Returns:
            Loss value and loss breakdown
        """
        # Calculate primary loss
        primary_loss = self.primary_loss_fn(outputs, targets)
        
        # Apply reduction for primary loss
        if self.reduction == 'mean':
            primary_loss = primary_loss.mean()
        elif self.reduction == 'sum':
            primary_loss = primary_loss.sum()
        
        loss_breakdown = {'primary': primary_loss.item()}
        total_loss = primary_loss
        
        # Calculate auxiliary alignment loss if modal outputs are provided
        if modal_outputs is not None and len(modal_outputs) > 1:
            aux_losses = []
            
            for i, modal_output in enumerate(modal_outputs):
                # Calculate alignment loss between modal output and final output
                aux_loss = self.aux_loss_fn(modal_output, outputs)
                
                # Apply reduction
                if self.reduction == 'mean':
                    aux_loss = aux_loss.mean()
                elif self.reduction == 'sum':
                    aux_loss = aux_loss.sum()
                
                aux_losses.append(aux_loss)
                loss_breakdown[f'aux_{i}'] = aux_loss.item()
            
            # Combine auxiliary losses
            aux_loss_combined = torch.stack(aux_losses).mean()
            total_loss = total_loss + self.aux_weight * aux_loss_combined
            loss_breakdown['aux_combined'] = aux_loss_combined.item()
        
        # Calculate contrastive loss if modal embeddings are provided
        if modal_embeddings is not None and batch_indices is not None:
            contrastive_loss = self.contrastive_loss_fn(modal_embeddings, batch_indices)
            
            # Apply reduction if needed
            if isinstance(contrastive_loss, torch.Tensor) and contrastive_loss.dim() > 0:
                if self.reduction == 'mean':
                    contrastive_loss = contrastive_loss.mean()
                elif self.reduction == 'sum':
                    contrastive_loss = contrastive_loss.sum()
            
            total_loss = total_loss + self.contrastive_weight * contrastive_loss
            loss_breakdown['contrastive'] = contrastive_loss.item()
        
        loss_breakdown['total'] = total_loss.item()
        return total_loss, loss_breakdown


class ModalityAlignmentLoss(nn.Module):
    """
    Modality alignment loss for H-CAAN
    
    Enforces alignment between modality representations
    
    Args:
        alignment_type: Type of alignment ('cosine', 'l2', 'mse')
        temperature: Temperature parameter for cosine similarity
        reduction: Reduction method ('mean', 'sum', or 'none')
    """
    def __init__(self, alignment_type='cosine', temperature=0.5, reduction='mean'):
        super(ModalityAlignmentLoss, self).__init__()
        self.alignment_type = alignment_type
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self, modal_embeddings):
        """
        Forward pass
        
        Args:
            modal_embeddings: List of embeddings from different modalities
            
        Returns:
            Loss value
        """
        losses = []
        
        # Process each modality pair
        num_modalities = len(modal_embeddings)
        for i in range(num_modalities):
            for j in range(i + 1, num_modalities):
                # Get embeddings for the pair of modalities
                z_i = modal_embeddings[i]
                z_j = modal_embeddings[j]
                
                if self.alignment_type == 'cosine':
                    # Normalize embeddings
                    z_i_norm = F.normalize(z_i, dim=1)
                    z_j_norm = F.normalize(z_j, dim=1)
                    
                    # Compute cosine similarity
                    similarity = torch.bmm(
                        z_i_norm.unsqueeze(1),
                        z_j_norm.unsqueeze(2)
                    ).squeeze() / self.temperature
                    
                    # Cosine similarity loss (1 - cos_sim)
                    loss = 1 - similarity
                
                elif self.alignment_type == 'l2':
                    # L2 distance between normalized embeddings
                    z_i_norm = F.normalize(z_i, dim=1)
                    z_j_norm = F.normalize(z_j, dim=1)
                    loss = torch.norm(z_i_norm - z_j_norm, p=2, dim=1)
                
                else:  # 'mse'
                    # Mean squared error between embeddings
                    loss = F.mse_loss(z_i, z_j, reduction='none').mean(dim=1)
                
                losses.append(loss)
        
        # Stack losses
        stacked_losses = torch.stack([loss.mean() for loss in losses])
        
        # Apply reduction
        if self.reduction == 'mean':
            return stacked_losses.mean()
        elif self.reduction == 'sum':
            return stacked_losses.sum()
        else:
            return stacked_losses


def get_loss_function(loss_name, **kwargs):
    """
    Factory function to get loss function by name
    
    Args:
        loss_name: Name of the loss function
        **kwargs: Additional arguments for the loss function
        
    Returns:
        Loss function
    """
    if loss_name == 'MSE' or loss_name == 'mse':
        return nn.MSELoss()
    elif loss_name == 'MAE' or loss_name == 'mae' or loss_name == 'L1':
        return nn.L1Loss()
    elif loss_name == 'Huber':
        delta = kwargs.get('delta', 1.0)
        return nn.SmoothL1Loss(beta=delta)
    elif loss_name == 'BCE' or loss_name == 'bce':
        return nn.BCEWithLogitsLoss()
    elif loss_name == 'CrossEntropy' or loss_name == 'ce':
        return nn.CrossEntropyLoss()
    elif loss_name == 'FocalLoss' or loss_name == 'focal':
        alpha = kwargs.get('alpha', 0.25)
        gamma = kwargs.get('gamma', 2.0)
        return FocalLoss(alpha=alpha, gamma=gamma)
    elif loss_name == 'MultiTask' or loss_name == 'multitask':
        task_types = kwargs.get('task_types', ['regression'])
        task_weights = kwargs.get('task_weights', None)
        learn_weights = kwargs.get('learn_weights', True)
        return MultiTaskLoss(task_types, task_weights, learn_weights)
    elif loss_name == 'Contrastive' or loss_name == 'contrastive':
        temperature = kwargs.get('temperature', 0.5)
        margin = kwargs.get('margin', 1.0)
        return ContrastiveLoss(temperature=temperature, margin=margin)
    elif loss_name == 'HierarchicalModular' or loss_name == 'hierarchical':
        task_type = kwargs.get('task_type', 'regression')
        aux_weight = kwargs.get('aux_weight', 0.1)
        contrastive_weight = kwargs.get('contrastive_weight', 0.05)
        return HierarchicalModularLoss(task_type, aux_weight, contrastive_weight)
    elif loss_name == 'ModalityAlignment' or loss_name == 'alignment':
        alignment_type = kwargs.get('alignment_type', 'cosine')
        temperature = kwargs.get('temperature', 0.5)
        return ModalityAlignmentLoss(alignment_type, temperature)
    elif loss_name == 'Custom Multi-objective':
        # Custom loss for H-CAAN combining primary task loss with alignment and contrastive losses
        task_type = kwargs.get('task_type', 'regression')
        aux_weight = kwargs.get('aux_weight', 0.1)
        contrastive_weight = kwargs.get('contrastive_weight', 0.05)
        return HierarchicalModularLoss(task_type, aux_weight, contrastive_weight)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")