import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from models.encoders.smiles_encoder import SmilesEncoder
from models.encoders.ecfp_encoder import ECFPEncoder
from models.encoders.gcn_encoder import GCNEncoder
from models.encoders.mfbert_encoder import MFBERTEncoder

from models.fusion.gcau import GatedCrossModalAttentionUnit
from models.fusion.hierarchical_fusion import HierarchicalFusion
from models.fusion.contrastive_learning import CrossModalContrastiveLoss

from models.attention.chemical_aware_attention import ChemicalAwareAttention
from models.attention.adaptive_gating import AdaptiveGating
from models.attention.multi_scale_attention import MultiScaleAttention

from models.modal_importance.task_specific_weights import TaskSpecificWeightGenerator
from models.modal_importance.complexity_aware import MolecularComplexityAssessor
from models.modal_importance.uncertainty_estimation import BayesianUncertaintyEstimator

class HCAAN(nn.Module):
    """
    Hierarchical Cross-modal Adaptive Attention Network (H-CAAN)
    for enhanced drug property prediction.
    
    This model integrates multiple molecular representations through
    a hierarchical fusion approach with adaptive attention mechanisms.
    """
    
    def __init__(self, config):
        """
        Initialize the H-CAAN model.
        
        Args:
            config: Dictionary containing model configuration
        """
        super(HCAAN, self).__init__()
        
        self.config = config
        
        # Set up encoders
        self.setup_encoders()
        
        # Set up fusion modules
        self.setup_fusion_modules()
        
        # Set up attention mechanisms
        self.setup_attention_mechanisms()
        
        # Set up modal importance modules
        self.setup_modal_importance_modules()
        
        # Set up output layers
        self.setup_output_layers()
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def setup_encoders(self):
        """Set up the encoder modules for each modality"""
        # SMILES encoder
        self.smiles_encoder = SmilesEncoder(
            vocab_size=self.config['smiles_encoder']['vocab_size'],
            hidden_dim=self.config['smiles_encoder']['hidden_dim'],
            num_layers=self.config['smiles_encoder']['num_layers'],
            num_heads=self.config['smiles_encoder']['num_heads'],
            dropout=self.config['general']['dropout']
        )
        
        # ECFP encoder
        self.ecfp_encoder = ECFPEncoder(
            input_dim=self.config['ecfp_encoder']['input_dim'],
            hidden_dim=self.config['ecfp_encoder']['hidden_dim'],
            num_layers=self.config['ecfp_encoder']['num_layers'],
            dropout=self.config['general']['dropout']
        )
        
        # Graph (GCN) encoder
        self.gcn_encoder = GCNEncoder(
            input_dim=self.config['gcn_encoder']['input_dim'],
            hidden_dim=self.config['gcn_encoder']['hidden_dim'],
            num_layers=self.config['gcn_encoder']['num_layers'],
            dropout=self.config['general']['dropout']
        )
        
        # MFBERT encoder (optional)
        if self.config['mfbert_encoder']['use_mfbert']:
            self.mfbert_encoder = MFBERTEncoder(
                hidden_dim=self.config['mfbert_encoder']['hidden_dim'],
                dropout=self.config['general']['dropout'],
                pretrained_model_path=self.config['mfbert_encoder'].get('pretrained_model_path', None)
            )
            self.use_mfbert = True
        else:
            self.use_mfbert = False
        
        # Define output dimensions for each encoder
        self.smiles_dim = self.config['smiles_encoder']['hidden_dim']
        self.ecfp_dim = self.config['ecfp_encoder']['hidden_dim']
        self.gcn_dim = self.config['gcn_encoder']['hidden_dim']
        self.mfbert_dim = self.config['mfbert_encoder']['hidden_dim'] if self.use_mfbert else 0
        
        # Common projection size for all modalities
        self.common_dim = self.config['general']['output_dim']
        
        # Projection layers to common dimension
        self.smiles_projection = nn.Linear(self.smiles_dim, self.common_dim)
        self.ecfp_projection = nn.Linear(self.ecfp_dim, self.common_dim)
        self.gcn_projection = nn.Linear(self.gcn_dim, self.common_dim)
        
        if self.use_mfbert:
            self.mfbert_projection = nn.Linear(self.mfbert_dim, self.common_dim)
    
    def setup_fusion_modules(self):
        """Set up the fusion modules for cross-modal integration"""
        fusion_config = self.config['fusion']
        
        # Determine which fusion levels to use
        self.use_low_level_fusion = "Low-level (Feature)" in fusion_config['levels']
        self.use_mid_level_fusion = "Mid-level (Semantic)" in fusion_config['levels']
        self.use_high_level_fusion = "High-level (Decision)" in fusion_config['levels']
        
        # Gated Cross-modal Attention Units (GCAUs)
        if self.use_low_level_fusion:
            # Create GCAUs for each modality pair
            self.gcau_smiles_ecfp = GatedCrossModalAttentionUnit(self.common_dim, self.common_dim)
            self.gcau_smiles_gcn = GatedCrossModalAttentionUnit(self.common_dim, self.common_dim)
            self.gcau_ecfp_gcn = GatedCrossModalAttentionUnit(self.common_dim, self.common_dim)
            
            if self.use_mfbert:
                self.gcau_smiles_mfbert = GatedCrossModalAttentionUnit(self.common_dim, self.common_dim)
                self.gcau_ecfp_mfbert = GatedCrossModalAttentionUnit(self.common_dim, self.common_dim)
                self.gcau_gcn_mfbert = GatedCrossModalAttentionUnit(self.common_dim, self.common_dim)
        
        # Hierarchical Fusion Module
        self.hierarchical_fusion = HierarchicalFusion(
            common_dim=self.common_dim,
            num_modalities=4 if self.use_mfbert else 3,
            use_low_level=self.use_low_level_fusion,
            use_mid_level=self.use_mid_level_fusion,
            use_high_level=self.use_high_level_fusion,
            dropout=self.config['general']['dropout']
        )
        
        # Contrastive Learning (for mid-level fusion)
        if self.use_mid_level_fusion:
            self.contrastive_loss = CrossModalContrastiveLoss(
                temperature=0.07,
                use_chemical_equivalence=True
            )
    
    def setup_attention_mechanisms(self):
        """Set up the attention mechanisms for enhanced feature learning"""
        attention_config = self.config['fusion']
        
        # Chemical-Aware Attention
        if attention_config['use_chemical_aware']:
            self.chemical_aware_attention = ChemicalAwareAttention(
                hidden_dim=self.common_dim,
                num_heads=4,
                dropout=self.config['general']['dropout']
            )
        else:
            self.chemical_aware_attention = None
        
        # Adaptive Gating
        if attention_config['use_adaptive_gating']:
            self.adaptive_gating = AdaptiveGating(
                hidden_dim=self.common_dim,
                dropout=self.config['general']['dropout']
            )
        else:
            self.adaptive_gating = None
        
        # Multi-Scale Attention
        if attention_config['use_multi_scale']:
            self.multi_scale_attention = MultiScaleAttention(
                hidden_dim=self.common_dim,
                num_scales=3,
                dropout=self.config['general']['dropout']
            )
        else:
            self.multi_scale_attention = None
    
    def setup_modal_importance_modules(self):
        """Set up the modules for dynamic modal importance assessment"""
        modal_config = self.config['modal_importance']
        
        # Task-Specific Weight Generator
        if modal_config['use_task_specific']:
            self.task_weight_generator = TaskSpecificWeightGenerator(
                hidden_dim=self.common_dim,
                num_modalities=4 if self.use_mfbert else 3,
                dropout=self.config['general']['dropout']
            )
        else:
            self.task_weight_generator = None
        
        # Molecular Complexity Assessor
        if modal_config['use_complexity_aware']:
            self.complexity_assessor = MolecularComplexityAssessor(
                hidden_dim=self.common_dim,
                dropout=self.config['general']['dropout']
            )
        else:
            self.complexity_assessor = None
        
        # Bayesian Uncertainty Estimator
        if modal_config['use_uncertainty']:
            self.uncertainty_estimator = BayesianUncertaintyEstimator(
                hidden_dim=self.common_dim,
                num_modalities=4 if self.use_mfbert else 3,
                dropout=self.config['general']['dropout']
            )
        else:
            self.uncertainty_estimator = None
    
    def setup_output_layers(self):
        """Set up the output layers for prediction"""
        # Output dimension (1 for regression, more for classification)
        self.output_dim = self.config['task']['output_dim']
        
        # Expert mixture system
        num_experts = 3  # Number of specialized expert heads
        self.expert_classifiers = nn.ModuleList([
            nn.Linear(self.common_dim, self.common_dim // 2) for _ in range(num_experts)
        ])
        
        self.expert_gates = nn.Linear(self.common_dim, num_experts)
        
        # Expert output layers
        self.expert_outputs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.common_dim // 2, self.common_dim // 4),
                nn.ReLU(),
                nn.Linear(self.common_dim // 4, self.output_dim)
            ) for _ in range(num_experts)
        ])
        
        # Ensemble prediction layer
        self.ensemble_predictor = nn.Linear(self.common_dim, self.output_dim)
        
        # Calibration layer
        self.calibration_layer = nn.Linear(self.output_dim + self.common_dim, self.output_dim)
    
    def encode_modalities(self, smiles_ids, smiles_mask, ecfp, graph_data, mfbert_inputs=None):
        """
        Encode all molecular modalities into separate representations
        
        Args:
            smiles_ids: Tokenized SMILES sequences
            smiles_mask: Mask for SMILES sequences
            ecfp: ECFP fingerprints
            graph_data: Molecular graph data
            mfbert_inputs: Inputs for MFBERT model (optional)
            
        Returns:
            Dictionary of encoded representations for each modality
        """
        # Encode SMILES
        smiles_encoded = self.smiles_encoder(smiles_ids, smiles_mask)
        smiles_pooled = smiles_encoded.mean(dim=1)  # Global average pooling
        smiles_features = self.smiles_projection(smiles_pooled)
        
        # Encode ECFP
        ecfp_features = self.ecfp_encoder(ecfp)
        ecfp_features = self.ecfp_projection(ecfp_features)
        
        # Encode Graph
        graph_features = self.gcn_encoder(graph_data)
        graph_features = self.gcn_projection(graph_features)
        
        # Encode with MFBERT if available
        if self.use_mfbert and mfbert_inputs is not None:
            mfbert_features = self.mfbert_encoder(mfbert_inputs)
            mfbert_features = self.mfbert_projection(mfbert_features)
        else:
            mfbert_features = None
        
        # Return dictionary of encoded features
        encodings = {
            'smiles': smiles_features,
            'ecfp': ecfp_features,
            'graph': graph_features
        }
        
        if mfbert_features is not None:
            encodings['mfbert'] = mfbert_features
        
        return encodings
    
    def apply_low_level_fusion(self, encodings):
        """
        Apply low-level cross-modal fusion using GCAUs
        
        Args:
            encodings: Dictionary of encoded representations
            
        Returns:
            Dictionary of enhanced representations after cross-modal fusion
        """
        if not self.use_low_level_fusion:
            return encodings
        
        # Extract features
        smiles_features = encodings['smiles']
        ecfp_features = encodings['ecfp']
        graph_features = encodings['graph']
        mfbert_features = encodings.get('mfbert', None)
        
        # Apply GCAUs between pairs of modalities
        smiles_ecfp = self.gcau_smiles_ecfp(smiles_features, ecfp_features)
        smiles_graph = self.gcau_smiles_gcn(smiles_features, graph_features)
        ecfp_graph = self.gcau_ecfp_gcn(ecfp_features, graph_features)
        
        # Update features with cross-modal information
        smiles_features = smiles_features + smiles_ecfp + smiles_graph
        ecfp_features = ecfp_features + smiles_ecfp + ecfp_graph
        graph_features = graph_features + smiles_graph + ecfp_graph
        
        # If MFBERT is used, apply additional GCAUs
        if self.use_mfbert and mfbert_features is not None:
            smiles_mfbert = self.gcau_smiles_mfbert(smiles_features, mfbert_features)
            ecfp_mfbert = self.gcau_ecfp_mfbert(ecfp_features, mfbert_features)
            graph_mfbert = self.gcau_gcn_mfbert(graph_features, mfbert_features)
            
            smiles_features = smiles_features + smiles_mfbert
            ecfp_features = ecfp_features + ecfp_mfbert
            graph_features = graph_features + graph_mfbert
            mfbert_features = mfbert_features + smiles_mfbert + ecfp_mfbert + graph_mfbert
            
            # Update MFBERT features
            encodings['mfbert'] = mfbert_features
        
        # Update encodings
        encodings['smiles'] = smiles_features
        encodings['ecfp'] = ecfp_features
        encodings['graph'] = graph_features
        
        # Apply chemical-aware attention if enabled
        if self.chemical_aware_attention is not None:
            # Convert dictionary to list of features
            feature_list = [encodings[key] for key in encodings]
            
            # Apply chemical-aware attention
            enhanced_features = self.chemical_aware_attention(feature_list)
            
            # Update encodings
            for i, key in enumerate(encodings):
                encodings[key] = enhanced_features[i]
        
        return encodings
    
    def apply_mid_level_fusion(self, encodings):
        """
        Apply mid-level semantic fusion with contrastive learning
        
        Args:
            encodings: Dictionary of encoded representations
            
        Returns:
            Dictionary of enhanced representations after semantic fusion
            and contrastive loss value
        """
        if not self.use_mid_level_fusion:
            return encodings, 0.0
        
        # Extract features
        feature_list = [encodings[key] for key in encodings]
        
        # Apply adaptive gating if enabled
        if self.adaptive_gating is not None:
            gated_features = self.adaptive_gating(feature_list)
            
            # Update encodings
            for i, key in enumerate(encodings):
                encodings[key] = gated_features[i]
        
        # Calculate contrastive loss
        contrastive_loss = self.contrastive_loss(feature_list)
        
        return encodings, contrastive_loss
    
    def apply_high_level_fusion(self, encodings):
        """
        Apply high-level decision fusion
        
        Args:
            encodings: Dictionary of encoded representations
            
        Returns:
            Fused representation for final prediction
        """
        # Extract features
        feature_list = [encodings[key] for key in encodings]
        
        # Determine modality weights based on different strategies
        weights = torch.ones(len(feature_list), device=feature_list[0].device)
        weights = weights / weights.sum()  # Default equal weights
        
        # Apply task-specific weights if enabled
        if self.task_weight_generator is not None:
            task_weights = self.task_weight_generator(feature_list)
            weights = weights * task_weights
        
        # Apply complexity-aware weights if enabled
        if self.complexity_assessor is not None:
            complexity_weights = self.complexity_assessor(feature_list)
            weights = weights * complexity_weights
        
        # Apply uncertainty-based weights if enabled
        if self.uncertainty_estimator is not None:
            uncertainty_weights = self.uncertainty_estimator(feature_list)
            weights = weights * uncertainty_weights
        
        # Normalize weights
        weights = weights / weights.sum()
        
        # Apply multi-scale attention if enabled
        if self.multi_scale_attention is not None:
            feature_list = self.multi_scale_attention(feature_list)
        
        # Apply hierarchical fusion with determined weights
        fused_representation = self.hierarchical_fusion(feature_list, weights)
        
        return fused_representation, weights
    
    def forward(self, smiles_ids, smiles_mask, ecfp, graph_data, mfbert_inputs=None):
        """
        Forward pass through the H-CAAN model
        
        Args:
            smiles_ids: Tokenized SMILES sequences
            smiles_mask: Mask for SMILES sequences
            ecfp: ECFP fingerprints
            graph_data: Molecular graph data
            mfbert_inputs: Inputs for MFBERT model (optional)
            
        Returns:
            Dictionary containing:
                - predictions: Final property predictions
                - auxiliary_outputs: Predictions from individual modalities
                - contrastive_loss: Contrastive loss value
                - modality_weights: Weights assigned to each modality
        """
        # Encode all modalities
        encodings = self.encode_modalities(smiles_ids, smiles_mask, ecfp, graph_data, mfbert_inputs)
        
        # Store original encodings for auxiliary predictions
        original_encodings = {k: v.clone() for k, v in encodings.items()}
        
        # Apply low-level fusion
        encodings = self.apply_low_level_fusion(encodings)
        
        # Apply mid-level fusion
        encodings, contrastive_loss = self.apply_mid_level_fusion(encodings)
        
        # Apply high-level fusion
        fused_representation, modality_weights = self.apply_high_level_fusion(encodings)
        
        # Expert mixture system
        # Calculate expert gates (soft assignment to experts)
        gate_logits = self.expert_gates(fused_representation)
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # Apply each expert
        expert_outputs = []
        for i, expert in enumerate(self.expert_classifiers):
            expert_representation = expert(fused_representation)
            expert_output = self.expert_outputs[i](expert_representation)
            expert_outputs.append(expert_output)
        
        # Combine expert outputs using gate probabilities
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [batch_size, num_experts, output_dim]
        gate_probs = gate_probs.unsqueeze(-1)  # [batch_size, num_experts, 1]
        mixture_output = (expert_outputs * gate_probs).sum(dim=1)  # [batch_size, output_dim]
        
        # Ensemble prediction
        ensemble_output = self.ensemble_predictor(fused_representation)
        
        # Combine mixture and ensemble predictions
        combined_output = (mixture_output + ensemble_output) / 2.0
        
        # Calibration with uncertainty information
        if self.uncertainty_estimator is not None:
            uncertainty_info = self.uncertainty_estimator.get_uncertainty(fused_representation)
            calibration_input = torch.cat([combined_output, uncertainty_info], dim=1)
            final_prediction = self.calibration_layer(calibration_input)
        else:
            final_prediction = combined_output
        
        # Make auxiliary predictions from individual modalities
        auxiliary_outputs = {}
        for modality, features in original_encodings.items():
            aux_output = nn.Linear(features.size(-1), self.output_dim).to(features.device)(features)
            auxiliary_outputs[modality] = aux_output
        
        return {
            'predictions': final_prediction,
            'auxiliary_outputs': auxiliary_outputs,
            'contrastive_loss': contrastive_loss,
            'modality_weights': modality_weights,
            'fused_representation': fused_representation,
            'uncertainty': uncertainty_info if self.uncertainty_estimator is not None else None
        }
    
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Use Xavier initialization for linear layers
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Use normal distribution for embeddings
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            # Use ones for normalization layers
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        
    def get_attention_weights(self):
        """
        Get attention weights for interpretation
        
        Returns:
            Dictionary of attention weights from different components
        """
        attention_weights = {}
        
        # Get chemical-aware attention weights if available
        if self.chemical_aware_attention is not None:
            attention_weights['chemical_aware'] = self.chemical_aware_attention.get_attention_weights()
        
        # Get multi-scale attention weights if available
        if self.multi_scale_attention is not None:
            attention_weights['multi_scale'] = self.multi_scale_attention.get_attention_weights()
        
        # Get GCAU attention weights if available and low-level fusion is used
        if self.use_low_level_fusion:
            gcau_weights = {
                'smiles_ecfp': self.gcau_smiles_ecfp.get_attention_weights(),
                'smiles_gcn': self.gcau_smiles_gcn.get_attention_weights(),
                'ecfp_gcn': self.gcau_ecfp_gcn.get_attention_weights()
            }
            
            if self.use_mfbert:
                gcau_weights.update({
                    'smiles_mfbert': self.gcau_smiles_mfbert.get_attention_weights(),
                    'ecfp_mfbert': self.gcau_ecfp_mfbert.get_attention_weights(),
                    'gcn_mfbert': self.gcau_gcn_mfbert.get_attention_weights()
                })
            
            attention_weights['gcau'] = gcau_weights
        
        return attention_weights
    
    def get_feature_importance(self, smiles_ids, smiles_mask, ecfp, graph_data, mfbert_inputs=None):
        """
        Get feature importance scores for interpretation
        
        Args:
            smiles_ids: Tokenized SMILES sequences
            smiles_mask: Mask for SMILES sequences
            ecfp: ECFP fingerprints
            graph_data: Molecular graph data
            mfbert_inputs: Inputs for MFBERT model (optional)
            
        Returns:
            Dictionary of feature importance scores
        """
        # Register hooks to get gradients
        gradients = {}
        activations = {}
        
        def get_activation(name):
            def hook(model, input, output):
                activations[name] = output.detach()
            return hook
        
        def get_gradient(name):
            def hook(grad):
                gradients[name] = grad.detach()
            return hook
        
        # Register hooks
        hooks = []
        
        # SMILES encoder
        handle = self.smiles_projection.register_forward_hook(get_activation('smiles'))
        hooks.append(handle)
        
        # ECFP encoder
        handle = self.ecfp_projection.register_forward_hook(get_activation('ecfp'))
        hooks.append(handle)
        
        # GCN encoder
        handle = self.gcn_projection.register_forward_hook(get_activation('graph'))
        hooks.append(handle)
        
        # MFBERT encoder (if used)
        if self.use_mfbert and mfbert_inputs is not None:
            handle = self.mfbert_projection.register_forward_hook(get_activation('mfbert'))
            hooks.append(handle)
        
        # Forward pass
        self.zero_grad()
        outputs = self.forward(smiles_ids, smiles_mask, ecfp, graph_data, mfbert_inputs)
        predictions = outputs['predictions']
        
        # Register backward hooks
        for name, activation in activations.items():
            activation.register_hook(get_gradient(name))
        
        # Backward pass
        if self.output_dim > 1:
            # For classification, use the predicted class
            predictions.max(dim=1)[0].sum().backward()
        else:
            # For regression, use the prediction directly
            predictions.sum().backward()
        
        # Calculate feature importance
        feature_importance = {}
        for name in activations:
            # Gradient * Activation (Grad-CAM like approach)
            importance = (activations[name] * gradients[name]).sum(dim=1)
            feature_importance[name] = importance
        
        # Remove hooks
        for handle in hooks:
            handle.remove()
        
        return feature_importance
