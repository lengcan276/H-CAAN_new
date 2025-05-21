import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import entropy


class MCDropout(nn.Module):
    """
    Implementation of Monte Carlo Dropout for Bayesian uncertainty estimation.
    
    This module wraps a model and applies dropout at test time to generate
    multiple predictions for uncertainty estimation.
    """
    
    def __init__(self, model, num_samples=10, dropout_rate=0.2):
        """
        Initialize MC Dropout.
        
        Args:
            model (nn.Module): Base model
            num_samples (int): Number of Monte Carlo samples
            dropout_rate (float): Dropout rate
        """
        super(MCDropout, self).__init__()
        
        self.model = model
        self.num_samples = num_samples
        self.dropout_rate = dropout_rate
        
        # Add dropout to the model if not already present
        self._add_dropout_layers(self.model)
    
    def _add_dropout_layers(self, module):
        """
        Recursively add dropout layers to the model.
        
        Args:
            module (nn.Module): Module to add dropout to
        """
        for name, child in module.named_children():
            if isinstance(child, nn.Dropout):
                # Replace with dropout of specified rate
                setattr(module, name, nn.Dropout(p=self.dropout_rate))
            else:
                # Recursively add dropout to children
                self._add_dropout_layers(child)
    
    def forward(self, *args, **kwargs):
        """
        Generate multiple predictions with MC Dropout.
        
        Args:
            *args: Arguments to pass to the model
            **kwargs: Keyword arguments to pass to the model
            
        Returns:
            tuple: (mean_prediction, variance)
                - mean_prediction: Mean of sampled predictions
                - variance: Variance of sampled predictions
        """
        # Enable dropout during inference
        self.model.train()
        
        # Generate multiple predictions
        predictions = []
        for _ in range(self.num_samples):
            output = self.model(*args, **kwargs)
            predictions.append(output)
        
        # Calculate mean and variance of predictions
        pred_stack = torch.stack(predictions, dim=0)
        mean_prediction = torch.mean(pred_stack, dim=0)
        variance = torch.var(pred_stack, dim=0)
        
        return mean_prediction, variance


class EnsembleUncertainty(nn.Module):
    """
    Ensemble-based uncertainty estimation.
    
    This module combines predictions from an ensemble of models to
    estimate prediction uncertainty.
    """
    
    def __init__(self, models, ensemble_type='average'):
        """
        Initialize the ensemble uncertainty module.
        
        Args:
            models (list): List of models in the ensemble
            ensemble_type (str): Type of ensemble ('average', 'weighted', or 'boosting')
        """
        super(EnsembleUncertainty, self).__init__()
        
        self.models = nn.ModuleList(models)
        self.ensemble_type = ensemble_type
        
        # If using weighted ensemble, create weights
        if ensemble_type == 'weighted':
            self.model_weights = nn.Parameter(torch.ones(len(models)))
    
    def forward(self, *args, **kwargs):
        """
        Generate ensemble predictions with uncertainty estimates.
        
        Args:
            *args: Arguments to pass to the models
            **kwargs: Keyword arguments to pass to the models
            
        Returns:
            tuple: (ensemble_prediction, uncertainty)
                - ensemble_prediction: Combined prediction from ensemble
                - uncertainty: Uncertainty estimate
        """
        # Set all models to eval mode
        for model in self.models:
            model.eval()
        
        # Generate predictions from each model
        predictions = []
        for model in self.models:
            with torch.no_grad():
                output = model(*args, **kwargs)
            predictions.append(output)
        
        # Combine predictions based on ensemble type
        if self.ensemble_type == 'average':
            # Simple averaging
            pred_stack = torch.stack(predictions, dim=0)
            ensemble_prediction = torch.mean(pred_stack, dim=0)
            
            # Calculate uncertainty as variance of predictions
            uncertainty = torch.var(pred_stack, dim=0)
            
        elif self.ensemble_type == 'weighted':
            # Weighted averaging
            weights = F.softmax(self.model_weights, dim=0)
            
            pred_stack = torch.stack(predictions, dim=0)
            weighted_preds = pred_stack * weights.view(-1, 1, 1)
            ensemble_prediction = torch.sum(weighted_preds, dim=0)
            
            # Calculate weighted variance
            squared_diff = (pred_stack - ensemble_prediction.unsqueeze(0)) ** 2
            uncertainty = torch.sum(squared_diff * weights.view(-1, 1, 1), dim=0)
            
        elif self.ensemble_type == 'boosting':
            # Simple boosting (each model focuses on errors of previous ones)
            ensemble_prediction = predictions[0]
            uncertainty = torch.zeros_like(ensemble_prediction)
            
            for i in range(1, len(predictions)):
                # Calculate error of current ensemble
                error = torch.abs(ensemble_prediction - predictions[i])
                
                # Update ensemble prediction
                ensemble_prediction = (ensemble_prediction * i + predictions[i]) / (i + 1)
                
                # Update uncertainty estimate
                uncertainty = uncertainty + error
            
            # Normalize uncertainty
            uncertainty = uncertainty / len(predictions)
        
        else:
            raise ValueError(f"Unknown ensemble type: {self.ensemble_type}")
        
        return ensemble_prediction, uncertainty


class ModalUncertaintyEstimator(nn.Module):
    """
    Estimates uncertainty for each modality to guide modality weighting.
    
    This module uses Bayesian techniques to estimate the uncertainty of
    predictions from different modalities, which can be used to weight
    their contributions accordingly.
    """
    
    def __init__(self, n_modalities, feature_dims, hidden_dim=64, dropout_rate=0.2, num_samples=10):
        """
        Initialize the modal uncertainty estimator.
        
        Args:
            n_modalities (int): Number of modalities
            feature_dims (list): List of feature dimensions for each modality
            hidden_dim (int): Hidden layer dimension
            dropout_rate (float): Dropout rate for MC Dropout
            num_samples (int): Number of Monte Carlo samples
        """
        super(ModalUncertaintyEstimator, self).__init__()
        
        self.n_modalities = n_modalities
        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.num_samples = num_samples
        
        # Create uncertainty estimation networks for each modality
        self.uncertainty_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_rate),
                nn.Linear(hidden_dim, 2)  # Mean and log variance
            )
            for dim in feature_dims
        ])
    
    def forward(self, features_list, enable_dropout=True):
        """
        Estimate uncertainty for each modality.
        
        Args:
            features_list (list): List of feature tensors from different modalities
            enable_dropout (bool): Whether to enable dropout for uncertainty estimation
            
        Returns:
            tuple: (means, uncertainties)
                - means: Mean prediction for each modality
                - uncertainties: Uncertainty for each modality
        """
        batch_size = features_list[0].size(0)
        
        # Set model to training mode if dropout is enabled
        if enable_dropout:
            self.train()  # Enable dropout
        else:
            self.eval()  # Disable dropout
        
        means = []
        uncertainties = []
        
        # Process each modality
        for i, (features, net) in enumerate(zip(features_list, self.uncertainty_nets)):
            if enable_dropout:
                # Use Monte Carlo Dropout for uncertainty estimation
                outputs = []
                for _ in range(self.num_samples):
                    output = net(features)
                    mean, log_var = output.chunk(2, dim=1)
                    outputs.append(mean)
                
                # Calculate mean and uncertainty
                outputs_stack = torch.stack(outputs, dim=0)
                mean = torch.mean(outputs_stack, dim=0)
                uncertainty = torch.var(outputs_stack, dim=0)
            else:
                # Single forward pass
                output = net(features)
                mean, log_var = output.chunk(2, dim=1)
                
                # Convert log variance to uncertainty
                uncertainty = torch.exp(log_var)
            
            means.append(mean)
            uncertainties.append(uncertainty)
        
        return means, uncertainties


class UncertaintyGuidedFusion(nn.Module):
    """
    Fuses modality features based on their estimated uncertainties.
    
    This module weights the contribution of each modality inversely
    proportional to its uncertainty, giving more weight to more
    confident modalities.
    """
    
    def __init__(self, n_modalities, feature_dims, output_dim, hidden_dim=128):
        """
        Initialize uncertainty-guided fusion.
        
        Args:
            n_modalities (int): Number of modalities
            feature_dims (list): List of feature dimensions for each modality
            output_dim (int): Output feature dimension
            hidden_dim (int): Hidden layer dimension
        """
        super(UncertaintyGuidedFusion, self).__init__()
        
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
        
        # Uncertainty estimator
        self.uncertainty_estimator = ModalUncertaintyEstimator(
            n_modalities=n_modalities,
            feature_dims=feature_dims,
            hidden_dim=hidden_dim // 2
        )
        
        # Fusion network
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_dim * n_modalities, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, output_dim)
        )
        
        # Calibration parameters
        self.uncertainty_scaling = nn.Parameter(torch.ones(n_modalities))
        self.uncertainty_shift = nn.Parameter(torch.zeros(n_modalities))
    
    def forward(self, features_list, return_weights=False):
        """
        Fuse features based on uncertainty estimates.
        
        Args:
            features_list (list): List of feature tensors from different modalities
            return_weights (bool): Whether to return modality weights
            
        Returns:
            torch.Tensor or tuple: Fused features or (fused_features, weights)
        """
        batch_size = features_list[0].size(0)
        
        # Project features to common dimension
        projected_features = [
            projection(features)
            for features, projection in zip(features_list, self.feature_projections)
        ]
        
        # Estimate uncertainty for each modality
        _, uncertainties = self.uncertainty_estimator(features_list, enable_dropout=self.training)
        
        # Calibrate uncertainties
        calibrated_uncertainties = [
            uncertainty * self.uncertainty_scaling[i] + self.uncertainty_shift[i]
            for i, uncertainty in enumerate(uncertainties)
        ]
        
        # Calculate modality weights based on uncertainties (inverse relationship)
        # Higher uncertainty means lower weight
        inverse_uncertainties = [
            1.0 / (uncertainty + 1e-6)  # Add small epsilon to avoid division by zero
            for uncertainty in calibrated_uncertainties
        ]
        
        # Normalize weights
        inverse_sum = sum(inverse_uncertainties)
        weights = [
            inverse / inverse_sum
            for inverse in inverse_uncertainties
        ]
        
        # Apply weights to features
        weighted_features = [
            features * weight
            for features, weight in zip(projected_features, weights)
        ]
        
        # Concatenate weighted features
        concat_features = torch.cat(weighted_features, dim=1)
        
        # Apply fusion network
        fused_features = self.fusion_net(concat_features)
        
        if return_weights:
            return fused_features, torch.stack(weights, dim=1)
        else:
            return fused_features


class BayesianUncertaintyAttention(nn.Module):
    """
    Attention mechanism that incorporates Bayesian uncertainty estimation.
    
    This module extends standard attention mechanisms with Bayesian uncertainty
    estimation, allowing it to focus more on certain inputs and less on
    uncertain ones.
    """
    
    def __init__(self, d_model, n_heads=8, dropout_rate=0.1, num_samples=10):
        """
        Initialize Bayesian uncertainty attention.
        
        Args:
            d_model (int): Model dimension
            n_heads (int): Number of attention heads
            dropout_rate (float): Dropout rate
            num_samples (int): Number of Monte Carlo samples
        """
        super(BayesianUncertaintyAttention, self).__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_rate = dropout_rate
        self.num_samples = num_samples
        
        assert self.head_dim * n_heads == d_model, "d_model must be divisible by n_heads"
        
        # Query, key, value projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Dropout layers
        self.dropout_q = nn.Dropout(p=dropout_rate)
        self.dropout_k = nn.Dropout(p=dropout_rate)
        self.dropout_v = nn.Dropout(p=dropout_rate)
        self.dropout_attn = nn.Dropout(p=dropout_rate)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Uncertainty estimation network
        self.uncertainty_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(d_model // 2, 1)
        )
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, q, k, v, mask=None, return_uncertainty=False):
        """
        Apply Bayesian uncertainty attention.
        
        Args:
            q (torch.Tensor): Query tensor of shape [batch_size, seq_len_q, d_model]
            k (torch.Tensor): Key tensor of shape [batch_size, seq_len_k, d_model]
            v (torch.Tensor): Value tensor of shape [batch_size, seq_len_v, d_model]
            mask (torch.Tensor, optional): Mask tensor
            return_uncertainty (bool): Whether to return uncertainty estimates
            
        Returns:
            torch.Tensor or tuple: Output tensor or (output, uncertainty)
        """
        batch_size, seq_len_q, _ = q.size()
        seq_len_k = k.size(1)
        seq_len_v = v.size(1)
        
        if self.training or return_uncertainty:
            # Use Monte Carlo Dropout for uncertainty estimation
            outputs = []
            uncertainties = []
            
            for _ in range(self.num_samples):
                # Apply projections with dropout
                q_mc = self.dropout_q(self.q_proj(q))
                k_mc = self.dropout_k(self.k_proj(k))
                v_mc = self.dropout_v(self.v_proj(v))
                
                # Reshape for multi-head attention
                q_mc = q_mc.view(batch_size, seq_len_q, self.n_heads, self.head_dim).transpose(1, 2)
                k_mc = k_mc.view(batch_size, seq_len_k, self.n_heads, self.head_dim).transpose(1, 2)
                v_mc = v_mc.view(batch_size, seq_len_v, self.n_heads, self.head_dim).transpose(1, 2)
                
                # Compute attention scores
                scores = torch.matmul(q_mc, k_mc.transpose(-2, -1)) * self.scale
                
                # Apply mask if provided
                if mask is not None:
                    scores = scores.masked_fill(mask == 0, -1e9)
                
                # Apply softmax and dropout
                attn_weights = F.softmax(scores, dim=-1)
                attn_weights = self.dropout_attn(attn_weights)
                
                # Apply attention to values
                context = torch.matmul(attn_weights, v_mc)
                
                # Reshape and apply output projection
                context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
                output = self.out_proj(context)
                
                # Estimate uncertainty
                uncertainty = self.uncertainty_net(output)
                
                outputs.append(output)
                uncertainties.append(uncertainty)
            
            # Calculate mean and uncertainty
            outputs_stack = torch.stack(outputs, dim=0)
            mean_output = torch.mean(outputs_stack, dim=0)
            
            # Calculate predictive variance
            pred_variance = torch.var(outputs_stack, dim=0)
            
            # Combine with uncertainty estimates
            uncertainties_stack = torch.stack(uncertainties, dim=0)
            mean_uncertainty = torch.mean(uncertainties_stack, dim=0)
            
            # Total uncertainty = predictive variance + mean uncertainty
            total_uncertainty = pred_variance + mean_uncertainty
            
            if return_uncertainty:
                return mean_output, total_uncertainty
            else:
                return mean_output
            
        else:
            # Standard forward pass for inference (without MC Dropout)
            # Apply projections
            q_proj = self.q_proj(q)
            k_proj = self.k_proj(k)
            v_proj = self.v_proj(v)
            
            # Reshape for multi-head attention
            q_proj = q_proj.view(batch_size, seq_len_q, self.n_heads, self.head_dim).transpose(1, 2)
            k_proj = k_proj.view(batch_size, seq_len_k, self.n_heads, self.head_dim).transpose(1, 2)
            v_proj = v_proj.view(batch_size, seq_len_v, self.n_heads, self.head_dim).transpose(1, 2)
            
            # Compute attention scores
            scores = torch.matmul(q_proj, k_proj.transpose(-2, -1)) * self.scale
            
            # Apply mask if provided
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            # Apply softmax
            attn_weights = F.softmax(scores, dim=-1)
            
            # Apply attention to values
            context = torch.matmul(attn_weights, v_proj)
            
            # Reshape and apply output projection
            context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
            output = self.out_proj(context)
            
            return output