# config/model_config.py

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any

@dataclass
class EncoderConfig:
    """Configuration for different encoder models."""
    # Shared parameters
    hidden_dim: int = 512
    dropout: float = 0.2
    
    # SMILES Transformer encoder
    smiles_embedding_dim: int = 128
    smiles_num_layers: int = 4
    smiles_num_heads: int = 8
    smiles_max_length: int = 512
    smiles_ff_dim: int = 2048
    
    # ECFP BiGRU encoder
    ecfp_input_dim: int = 1024  # ECFP fingerprint size
    ecfp_num_layers: int = 2
    ecfp_bidirectional: bool = True
    ecfp_attention_heads: int = 4
    
    # GCN encoder
    gcn_node_feat_dim: int = 78  # Node feature dimension
    gcn_edge_feat_dim: int = 10  # Edge feature dimension
    gcn_num_layers: int = 3
    gcn_residual: bool = True
    gcn_graph_pooling: str = "attention"  # [mean, sum, max, attention]
    gcn_virtual_node: bool = True
    
    # MFBERT encoder
    mfbert_pretrained_path: str = "Model/pre-trained"
    mfbert_freeze_layers: int = 8  # Number of layers to freeze
    mfbert_output_dim: int = 768
    mfbert_pooling: str = "mean"  # [cls, mean]


@dataclass
class FusionConfig:
    """Configuration for fusion modules."""
    # GCAU parameters
    gcau_hidden_dim: int = 256
    gcau_num_heads: int = 4
    gcau_dropout: float = 0.1
    
    # Hierarchical fusion parameters
    low_level_fusion: str = "attention"  # [concat, attention, gcau]
    mid_level_fusion: str = "gcau"      # [concat, attention, gcau]
    high_level_fusion: str = "adaptive" # [concat, attention, gcau, adaptive]
    
    # Contrastive learning parameters
    temperature: float = 0.07
    contrastive_weight: float = 0.1
    use_projection_head: bool = True
    projection_dim: int = 128


@dataclass
class AttentionConfig:
    """Configuration for attention mechanisms."""
    # Chemical-aware attention
    use_chemical_prior: bool = True
    chemical_embedding_dim: int = 64
    
    # Adaptive gating
    gating_hidden_dim: int = 128
    gating_activation: str = "sigmoid"  # [sigmoid, tanh, relu]
    
    # Multi-scale attention
    num_scales: int = 3
    scale_factors: List[int] = field(default_factory=lambda: [1, 2, 4])
    shared_parameters: bool = False
    scale_attention_dropout: float = 0.1


@dataclass
class ModalImportanceConfig:
    """Configuration for modal importance assessment."""
    # Task-specific weights
    learn_task_weights: bool = True
    weight_network_hidden_dim: int = 64
    
    # Complexity-aware mechanisms
    use_complexity_assessment: bool = True
    complexity_features: List[str] = field(default_factory=lambda: ["rings", "branches", "stereo"])
    
    # Uncertainty estimation
    monte_carlo_samples: int = 5
    use_temperature_scaling: bool = True
    

@dataclass
class TrainingConfig:
    """Configuration for training process."""
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    batch_size: int = 16
    num_epochs: int = 100
    gradient_accumulation_steps: int = 2
    max_grad_norm: float = 1.0
    
    # Optimizer settings
    optimizer: str = "adamw"  # [adam, adamw, sgd]
    scheduler: str = "cosine_with_warmup"  # [linear, cosine, cosine_with_warmup, none]
    warmup_ratio: float = 0.1
    
    # EMA settings
    use_ema: bool = True
    ema_decay: float = 0.999
    
    # Early stopping
    patience: int = 15
    
    # Checkpointing
    save_steps: int = 500
    checkpoint_dir: str = "checkpoints"


@dataclass
class DataConfig:
    """Configuration for datasets."""
    data_dir: str = "data"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42
    
    # Data preprocessing
    normalize_features: bool = True
    normalize_targets: bool = True
    augmentation: bool = False
    fold_validation: bool = True
    num_folds: int = 5
    
    # Dataset options
    datasets: List[str] = field(default_factory=lambda: ["Delaney", "Lipophilicity", "BACE", "SAMPL"])
    target_tasks: List[str] = field(default_factory=lambda: ["solubility", "lipophilicity", "inhibition", "partition"])


@dataclass
class HCANConfig:
    """Main configuration for H-CAAN model."""
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    modal_importance: ModalImportanceConfig = field(default_factory=ModalImportanceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # General model parameters
    output_dim: int = 1  # Number of prediction targets
    use_layer_norm: bool = True
    use_residual: bool = True
    dropout: float = 0.2
    
    # Multi-task learning
    multi_task: bool = False
    task_specific_heads: bool = True
    task_weights: Optional[Dict[str, float]] = None
    
    # Experiment settings
    experiment_name: str = "h_caan_default"
    log_dir: str = "logs"
    
    def save(self, filepath: str):
        """Save config to file."""
        import json
        
        # Convert dataclass to dict
        config_dict = {
            "encoder": self.encoder.__dict__,
            "fusion": self.fusion.__dict__,
            "attention": self.attention.__dict__,
            "modal_importance": self.modal_importance.__dict__,
            "training": self.training.__dict__,
            "data": self.data.__dict__,
            "output_dim": self.output_dim,
            "use_layer_norm": self.use_layer_norm,
            "use_residual": self.use_residual,
            "dropout": self.dropout,
            "multi_task": self.multi_task,
            "task_specific_heads": self.task_specific_heads,
            "task_weights": self.task_weights,
            "experiment_name": self.experiment_name,
            "log_dir": self.log_dir
        }
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save to JSON
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'HCANConfig':
        """Load config from file."""
        import json
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Create config objects from dict
        encoder_config = EncoderConfig(**config_dict.pop("encoder"))
        fusion_config = FusionConfig(**config_dict.pop("fusion"))
        attention_config = AttentionConfig(**config_dict.pop("attention"))
        modal_importance_config = ModalImportanceConfig(**config_dict.pop("modal_importance"))
        training_config = TrainingConfig(**config_dict.pop("training"))
        data_config = DataConfig(**config_dict.pop("data"))
        
        # Create main config
        return cls(
            encoder=encoder_config,
            fusion=fusion_config,
            attention=attention_config,
            modal_importance=modal_importance_config,
            training=training_config,
            data=data_config,
            **config_dict
        )


def get_default_config() -> HCANConfig:
    """Return the default configuration."""
    return HCANConfig()


def get_config_for_dataset(dataset_name: str) -> HCANConfig:
    """Get optimized configuration for a specific dataset."""
    config = get_default_config()
    
    if dataset_name == "Delaney":
        # Optimized for solubility prediction
        config.encoder.smiles_num_layers = 3
        config.encoder.gcn_num_layers = 4
        config.fusion.contrastive_weight = 0.15
        config.training.learning_rate = 3e-5
        
    elif dataset_name == "Lipophilicity":
        # Optimized for lipophilicity prediction
        config.encoder.smiles_num_layers = 4
        config.encoder.gcn_num_layers = 3
        config.fusion.contrastive_weight = 0.1
        config.attention.use_chemical_prior = True
        config.training.learning_rate = 4e-5
        
    elif dataset_name == "BACE":
        # Optimized for enzyme inhibition prediction
        config.encoder.smiles_num_layers = 5
        config.encoder.gcn_num_layers = 4
        config.fusion.contrastive_weight = 0.05
        config.modal_importance.use_complexity_assessment = True
        config.training.learning_rate = 2e-5
        
    elif dataset_name == "SAMPL":
        # Optimized for partition coefficient prediction
        config.encoder.smiles_num_layers = 4
        config.encoder.gcn_num_layers = 3
        config.fusion.contrastive_weight = 0.08
        config.modal_importance.use_complexity_assessment = True
        config.training.learning_rate = 3e-5
        
    elif dataset_name == "PDBbind":
        # Optimized for protein-ligand binding prediction
        config.encoder.smiles_num_layers = 6
        config.encoder.gcn_num_layers = 5
        config.fusion.contrastive_weight = 0.2
        config.modal_importance.use_complexity_assessment = True
        config.training.learning_rate = 1e-5
        config.training.batch_size = 8
        config.attention.num_scales = 4
    
    # Update experiment name
    config.experiment_name = f"h_caan_{dataset_name.lower()}"
    
    return config