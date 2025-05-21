import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import json
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
import torch.nn.functional as F
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tqdm import tqdm
import time
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GraphDataLoader

from models.hierarchical_model import HCAAN

class MolecularDataset(Dataset):
    """
    Dataset class for molecular data with multiple modalities
    """
    
    def __init__(self, smiles_data, ecfp_data, graph_data, mfbert_data=None, labels=None):
        """
        Initialize the dataset.
        
        Args:
            smiles_data (dict): Dict with 'ids' and 'mask' for SMILES
            ecfp_data (torch.Tensor): ECFP fingerprints
            graph_data (list): List of torch_geometric.data.Data objects
            mfbert_data (dict, optional): Dict with 'input_ids' and 'attention_mask' for MFBERT
            labels (torch.Tensor, optional): Target labels
        """
        self.smiles_ids = smiles_data['ids']
        self.smiles_mask = smiles_data['mask']
        self.ecfp = ecfp_data
        self.graph_data = graph_data
        self.mfbert_data = mfbert_data
        self.labels = labels
    
    def __len__(self):
        return len(self.smiles_ids)
    
    def __getitem__(self, idx):
        item = {
            'smiles_ids': self.smiles_ids[idx],
            'smiles_mask': self.smiles_mask[idx],
            'ecfp': self.ecfp[idx],
            'graph_data': self.graph_data[idx]
        }
        
        if self.mfbert_data is not None:
            item['mfbert_input_ids'] = self.mfbert_data['input_ids'][idx]
            item['mfbert_attention_mask'] = self.mfbert_data['attention_mask'][idx]
        
        if self.labels is not None:
            item['labels'] = self.labels[idx]
        
        return item

class ModelAgent:
    """
    Agent responsible for H-CAAN model configuration, training,
    and optimization.
    """
    
    def __init__(self, knowledge_base=None, openai_api_key=None, verbose=True):
        """
        Initialize the Model Agent.
        
        Args:
            knowledge_base (dict, optional): Shared knowledge base
            openai_api_key (str, optional): OpenAI API key for LLM integration
            verbose (bool): Whether to output detailed logs
        """
        self.knowledge_base = knowledge_base or {}
        self.openai_api_key = openai_api_key
        self.verbose = verbose
        
        # Set up logging
        self.logger = self._setup_logging()
        
        # Initialize required directories
        self.model_dir = os.path.join(os.getcwd(), 'models')
        self.output_dir = os.path.join(os.getcwd(), 'outputs')
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model configuration
        self.model_config = None
        self.model = None
        
        self.logger.info(f"Model Agent initialized successfully (Device: {self.device})")
    
    def _setup_logging(self):
        """Set up logging configuration"""
        logger = logging.getLogger("ModelAgent")
        
        if self.verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        
        # Create console handler
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(ch)
        
        return logger
    
    def configure_model(self, config):
        """
        Configure the H-CAAN model.
        
        Args:
            config (dict): Model configuration dictionary
            
        Returns:
            dict: Model summary
        """
        self.logger.info("Configuring H-CAAN model...")
        
        # Save model configuration
        self.model_config = self._process_model_config(config)
        
        # Save configuration to file
        config_path = os.path.join(self.model_dir, 'model_config.json')
        with open(config_path, 'w') as f:
            json.dump(self.model_config, f, indent=2)
        
        # Initialize the model
        self.model = self._initialize_model()
        
        # Generate model summary
        model_summary = self._generate_model_summary()
        
        # Update knowledge base
        if self.knowledge_base is not None:
            self.knowledge_base['model_config'] = self.model_config
            self.knowledge_base['model_summary'] = model_summary
        
        self.logger.info("Model configuration completed successfully")
        
        return model_summary
    
    def _process_model_config(self, config):
        """
        Process and validate the model configuration.
        
        Args:
            config (dict): Model configuration dictionary
            
        Returns:
            dict: Processed configuration dictionary
        """
        # Default configuration values
        default_config = {
            "smiles_encoder": {
                "vocab_size": 100,  # Will be updated based on tokenizer
                "hidden_dim": 256,
                "num_layers": 3,
                "num_heads": 8,
                "dropout": 0.1
            },
            "ecfp_encoder": {
                "input_dim": 1024,  # Default for ECFP4 with 1024 bits
                "hidden_dim": 256,
                "num_layers": 2,
                "dropout": 0.1
            },
            "gcn_encoder": {
                "input_dim": 78,  # Default atom feature dimension
                "hidden_dim": 256,
                "num_layers": 3,
                "dropout": 0.1
            },
            "mfbert_encoder": {
                "use_mfbert": True,
                "hidden_dim": 512,
                "pretrained_model_path": None,
                "dropout": 0.1
            },
            "fusion": {
                "levels": ["Low-level (Feature)", "Mid-level (Semantic)", "High-level (Decision)"],
                "use_chemical_aware": True,
                "use_adaptive_gating": True,
                "use_multi_scale": True
            },
            "modal_importance": {
                "use_task_specific": True,
                "use_complexity_aware": True,
                "use_uncertainty": True
            },
            "general": {
                "dropout": 0.1,
                "output_dim": 128
            },
            "task": {
                "task_type": "regression",  # or "classification"
                "output_dim": 1,  # 1 for regression, more for multi-class classification
                "num_classes": 1  # For classification tasks
            }
        }
        
        # Update default config with provided values
        processed_config = default_config.copy()
        
        # Update SMILES encoder config
        if 'smiles_encoder' in config:
            processed_config['smiles_encoder'].update(config['smiles_encoder'])
        
        # Update ECFP encoder config
        if 'ecfp_encoder' in config:
            processed_config['ecfp_encoder'].update(config['ecfp_encoder'])
        
        # Update GCN encoder config
        if 'gcn_encoder' in config:
            processed_config['gcn_encoder'].update(config['gcn_encoder'])
        
        # Update MFBERT encoder config
        if 'mfbert_encoder' in config:
            processed_config['mfbert_encoder'].update(config['mfbert_encoder'])
        
        # Update fusion config
        if 'fusion' in config:
            processed_config['fusion'].update(config['fusion'])
        
        # Update modal importance config
        if 'modal_importance' in config:
            processed_config['modal_importance'].update(config['modal_importance'])
        
        # Update general config
        if 'general' in config:
            processed_config['general'].update(config['general'])
        
        # Infer task type from knowledge base (if available)
        if self.knowledge_base and 'dataset' in self.knowledge_base:
            dataset_stats = self.knowledge_base['dataset'].get('stats', {})
            property_stats = dataset_stats.get('property_stats', {})
            
            if property_stats:
                # Check if it's likely a classification task
                if 'unique_values' in property_stats:
                    unique_values = property_stats['unique_values']
                    if len(unique_values) <= 10 and all(isinstance(v, int) for v in unique_values):
                        processed_config['task']['task_type'] = 'classification'
                        processed_config['task']['num_classes'] = len(unique_values)
                        processed_config['task']['output_dim'] = len(unique_values)
        
        # Override with specific task config if provided
        if 'task' in config:
            processed_config['task'].update(config['task'])
        
        # Ensure general dropout is consistent
        dropout = processed_config['general']['dropout']
        processed_config['smiles_encoder']['dropout'] = dropout
        processed_config['ecfp_encoder']['dropout'] = dropout
        processed_config['gcn_encoder']['dropout'] = dropout
        processed_config['mfbert_encoder']['dropout'] = dropout
        
        return processed_config
    
    def _initialize_model(self):
        """
        Initialize the H-CAAN model with the configured settings.
        
        Returns:
            HCAAN: Initialized model
        """
        try:
            model = HCAAN(self.model_config)
            model = model.to(self.device)
            
            self.logger.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
            
            return model
        except Exception as e:
            self.logger.error(f"Error initializing model: {str(e)}")
            raise
    
    def _generate_model_summary(self):
        """
        Generate a summary of the model architecture.
        
        Returns:
            dict: Model summary
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        summary = {
            "architecture": "H-CAAN",
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "modal_encoders": {
                "smiles_encoder": {
                    "type": "Transformer",
                    "hidden_dim": self.model_config['smiles_encoder']['hidden_dim'],
                    "num_layers": self.model_config['smiles_encoder']['num_layers'],
                    "num_heads": self.model_config['smiles_encoder']['num_heads']
                },
                "ecfp_encoder": {
                    "type": "BiGRU",
                    "hidden_dim": self.model_config['ecfp_encoder']['hidden_dim'],
                    "num_layers": self.model_config['ecfp_encoder']['num_layers']
                },
                "gcn_encoder": {
                    "type": "GCN",
                    "hidden_dim": self.model_config['gcn_encoder']['hidden_dim'],
                    "num_layers": self.model_config['gcn_encoder']['num_layers']
                }
            },
            "fusion_mechanisms": {
                "use_low_level": "Low-level (Feature)" in self.model_config['fusion']['levels'],
                "use_mid_level": "Mid-level (Semantic)" in self.model_config['fusion']['levels'],
                "use_high_level": "High-level (Decision)" in self.model_config['fusion']['levels'],
                "chemical_aware_attention": self.model_config['fusion']['use_chemical_aware'],
                "adaptive_gating": self.model_config['fusion']['use_adaptive_gating'],
                "multi_scale_attention": self.model_config['fusion']['use_multi_scale']
            },
            "modal_importance": {
                "task_specific_weights": self.model_config['modal_importance']['use_task_specific'],
                "complexity_aware_selection": self.model_config['modal_importance']['use_complexity_aware'],
                "uncertainty_estimation": self.model_config['modal_importance']['use_uncertainty']
            },
            "task": {
                "type": self.model_config['task']['task_type'],
                "output_dim": self.model_config['task']['output_dim']
            }
        }
        
        if self.model_config['mfbert_encoder']['use_mfbert']:
            summary["modal_encoders"]["mfbert_encoder"] = {
                "type": "MFBERT",
                "hidden_dim": self.model_config['mfbert_encoder']['hidden_dim']
            }
        
        return summary
    
    def train_model(self, training_config):
        """
        Train the H-CAAN model.
        
        Args:
            training_config (dict): Training configuration dictionary
            
        Returns:
            dict: Training results
        """
        self.logger.info("Starting model training...")
        
        # Check if model is configured
        if self.model is None:
            self.logger.error("Model not configured. Please call configure_model() first.")
            raise ValueError("Model not configured")
        
        # Process training configuration
        training_config = self._process_training_config(training_config)
        
        # Prepare data
        train_loader, valid_loader, test_loader = self._prepare_data(training_config)
        
        # Set up training components
        optimizer = self._setup_optimizer(training_config)
        lr_scheduler = self._setup_lr_scheduler(optimizer, training_config)
        loss_fn = self._setup_loss_function(training_config)
        
        # Train the model
        training_results = self._train_model(
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss_fn=loss_fn,
            training_config=training_config
        )
        
        # Run ablation study if enabled
        if training_config['ablation_study']['run']:
            ablation_results = self._run_ablation_study(
                train_loader=train_loader,
                valid_loader=valid_loader,
                test_loader=test_loader,
                training_config=training_config,
                components=training_config['ablation_study']['components']
            )
            training_results['ablation_results'] = ablation_results
        
        # Update knowledge base
        if self.knowledge_base is not None:
            self.knowledge_base['training_results'] = training_results
        
        self.logger.info("Model training completed successfully")
        
        return training_results
    
    def _process_training_config(self, config):
        """
        Process and validate the training configuration.
        
        Args:
            config (dict): Training configuration dictionary
            
        Returns:
            dict: Processed configuration dictionary
        """
        # Default configuration values
        default_config = {
            "batch_size": 64,
            "epochs": 100,
            "learning_rate": 0.001,
            "optimizer": "AdamW",
            "loss_function": "MSE",
            "early_stopping": {
                "use": True,
                "patience": 20
            },
            "regularization": {
                "weight_decay": 0.0001
            },
            "lr_scheduler": "ReduceLROnPlateau",
            "augmentation": False,
            "ablation_study": {
                "run": False,
                "components": []
            }
        }
        
        # Update default config with provided values
        processed_config = default_config.copy()
        
        # Update with provided values
        for key, value in config.items():
            if key in processed_config:
                if isinstance(value, dict) and isinstance(processed_config[key], dict):
                    processed_config[key].update(value)
                else:
                    processed_config[key] = value
        
        return processed_config
    
    def _prepare_data(self, training_config):
        """
        Prepare data loaders for training, validation, and testing.
        
        Args:
            training_config (dict): Training configuration dictionary
            
        Returns:
            tuple: (train_loader, valid_loader, test_loader)
        """
        self.logger.info("Preparing data loaders...")
        
        # Check if dataset is available in knowledge base
        if self.knowledge_base is None or 'dataset' not in self.knowledge_base:
            self.logger.error("Dataset not available in knowledge base")
            raise ValueError("Dataset not available")
        
        # Load data from the data directory
        try:
            train_df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'train.csv'))
            valid_df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'valid.csv'))
            test_df = pd.read_csv(os.path.join(os.getcwd(), 'data', 'test.csv'))
            
            self.logger.info(f"Loaded datasets: Train={len(train_df)}, Validation={len(valid_df)}, Test={len(test_df)}")
        except Exception as e:
            self.logger.error(f"Error loading datasets: {str(e)}")
            raise
        
        # Convert to PyTorch datasets
        train_dataset = self._create_dataset(train_df, augment=training_config['augmentation'])
        valid_dataset = self._create_dataset(valid_df, augment=False)
        test_dataset = self._create_dataset(test_df, augment=False)
        
        # Create data loaders
        batch_size = training_config['batch_size']
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
        
        valid_loader = DataLoader(
            valid_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
        
        return train_loader, valid_loader, test_loader
    
    def _create_dataset(self, df, augment=False):
        """
        Create a PyTorch dataset from a DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame containing molecular data
            augment (bool): Whether to apply data augmentation
            
        Returns:
            MolecularDataset: Dataset for the model
        """
        # Extract SMILES
        smiles_list = df['smiles'].tolist()
        
        # TODO: Implement actual tokenization with a proper tokenizer
        # For now, just use dummy tokenization
        smiles_data = {
            'ids': torch.zeros((len(smiles_list), 100), dtype=torch.long),
            'mask': torch.ones((len(smiles_list), 100), dtype=torch.long)
        }
        
        # Extract ECFP fingerprints
        ecfp_data = torch.FloatTensor(df['ECFP'].tolist())
        
        # Create graph data (placeholder implementation)
        graph_data = []
        for i in range(len(df)):
            # TODO: Replace with actual graph construction
            x = torch.randn(5, 78)
            edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4], 
                                       [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
            graph_data.append(Data(x=x, edge_index=edge_index))
        
        # Extract labels
        if 'Property' in df.columns:
            labels = torch.FloatTensor(df['Property'].values.reshape(-1, 1))
        else:
            labels = None
        
        # Create MFBERT data (placeholder implementation)
        # TODO: Replace with actual MFBERT tokenization
        mfbert_data = {
            'input_ids': torch.zeros((len(smiles_list), 512), dtype=torch.long),
            'attention_mask': torch.ones((len(smiles_list), 512), dtype=torch.long)
        }
        
        # Apply data augmentation if enabled
        if augment:
            # TODO: Implement data augmentation
            pass
        
        # Create dataset
        dataset = MolecularDataset(
            smiles_data=smiles_data,
            ecfp_data=ecfp_data,
            graph_data=graph_data,
            mfbert_data=mfbert_data,
            labels=labels
        )
        
        return dataset
    
    def _collate_fn(self, batch):
        """
        Custom collate function for DataLoader.
        
        Args:
            batch (list): List of items from the dataset
            
        Returns:
            dict: Batched data
        """
        batch_data = {
            'smiles_ids': torch.stack([item['smiles_ids'] for item in batch]),
            'smiles_mask': torch.stack([item['smiles_mask'] for item in batch]),
            'ecfp': torch.stack([item['ecfp'] for item in batch]),
        }
        
        # Handle graph data (use Batch.from_data_list from PyG)
        batch_data['graph_data'] = torch.stack([item['graph_data'] for item in batch])
        
        # Handle MFBERT data
        if 'mfbert_input_ids' in batch[0]:
            batch_data['mfbert_input_ids'] = torch.stack([item['mfbert_input_ids'] for item in batch])
            batch_data['mfbert_attention_mask'] = torch.stack([item['mfbert_attention_mask'] for item in batch])
        
        # Handle labels
        if 'labels' in batch[0]:
            batch_data['labels'] = torch.stack([item['labels'] for item in batch])
        
        return batch_data
    
    def _setup_optimizer(self, training_config):
        """
        Set up the optimizer.
        
        Args:
            training_config (dict): Training configuration dictionary
            
        Returns:
            torch.optim.Optimizer: Configured optimizer
        """
        optimizer_name = training_config['optimizer']
        lr = training_config['learning_rate']
        weight_decay = training_config['regularization']['weight_decay']
        
        if optimizer_name == 'Adam':
            optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == 'AdamW':
            optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            self.logger.warning(f"Optimizer {optimizer_name} not recognized. Using AdamW.")
            optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        return optimizer
    
    def _setup_lr_scheduler(self, optimizer, training_config):
        """
        Set up the learning rate scheduler.
        
        Args:
            optimizer (torch.optim.Optimizer): The optimizer
            training_config (dict): Training configuration dictionary
            
        Returns:
            torch.optim.lr_scheduler._LRScheduler: Configured scheduler
        """
        scheduler_name = training_config['lr_scheduler']
        
        if scheduler_name == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        elif scheduler_name == 'CosineAnnealing':
            scheduler = CosineAnnealingLR(optimizer, T_max=training_config['epochs'])
        elif scheduler_name == 'OneCycleLR':
            scheduler = OneCycleLR(
                optimizer, 
                max_lr=training_config['learning_rate'], 
                epochs=training_config['epochs'],
                steps_per_epoch=100  # This will be updated in the training loop
            )
        else:
            self.logger.warning(f"Scheduler {scheduler_name} not recognized. Using ReduceLROnPlateau.")
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        return scheduler
    
    def _setup_loss_function(self, training_config):
        """
        Set up the loss function.
        
        Args:
            training_config (dict): Training configuration dictionary
            
        Returns:
            function: Loss function
        """
        loss_name = training_config['loss_function']
        task_type = self.model_config['task']['task_type']
        
        if task_type == 'regression':
            if loss_name == 'MSE':
                return F.mse_loss
            elif loss_name == 'MAE':
                return F.l1_loss
            elif loss_name == 'Huber':
                return F.smooth_l1_loss
            elif loss_name == 'Custom Multi-objective':
                return self._custom_multiobjective_loss
            else:
                self.logger.warning(f"Loss function {loss_name} not recognized for regression. Using MSE.")
                return F.mse_loss
        elif task_type == 'classification':
            if loss_name == 'CrossEntropy':
                return F.cross_entropy
            elif loss_name == 'BCE':
                return F.binary_cross_entropy_with_logits
            else:
                self.logger.warning(f"Loss function {loss_name} not recognized for classification. Using CrossEntropy.")
                return F.cross_entropy
        else:
            self.logger.warning(f"Task type {task_type} not recognized. Using MSE loss.")
            return F.mse_loss
    
    def _custom_multiobjective_loss(self, pred, target, aux_outputs=None, contrastive_loss=None):
        """
        Custom multi-objective loss function combining multiple components.
        
        Args:
            pred (torch.Tensor): Main predictions
            target (torch.Tensor): Target values
            aux_outputs (dict, optional): Auxiliary outputs from different modalities
            contrastive_loss (float, optional): Contrastive loss value
            
        Returns:
            torch.Tensor: Combined loss value
        """
        # Main prediction loss (MSE)
        main_loss = F.mse_loss(pred, target)
        
        total_loss = main_loss
        
        # Add auxiliary losses if available
        if aux_outputs is not None:
            aux_loss = 0
            for modality, aux_pred in aux_outputs.items():
                aux_loss += F.mse_loss(aux_pred, target)
            
            # Weight auxiliary losses
            aux_weight = 0.1
            total_loss += aux_weight * aux_loss
        
        # Add contrastive loss if available
        if contrastive_loss is not None:
            contrastive_weight = 0.05
            total_loss += contrastive_weight * contrastive_loss
        
        return total_loss
    
    def _train_model(self, train_loader, valid_loader, test_loader, optimizer, lr_scheduler, loss_fn, training_config):
        """
        Train the model.
        
        Args:
            train_loader (DataLoader): Training data loader
            valid_loader (DataLoader): Validation data loader
            test_loader (DataLoader): Test data loader
            optimizer (torch.optim.Optimizer): Optimizer
            lr_scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler
            loss_fn (function): Loss function
            training_config (dict): Training configuration dictionary
            
        Returns:
            dict: Training results
        """
        self.logger.info("Training model...")
        
        # Set up early stopping
        early_stopping = training_config['early_stopping']['use']
        patience = training_config['early_stopping']['patience']
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Set up results tracking
        epochs = training_config['epochs']
        train_losses = []
        val_losses = []
        learning_rates = []
        
        # Update OneCycleLR steps_per_epoch if used
        if training_config['lr_scheduler'] == 'OneCycleLR':
            lr_scheduler.steps_per_epoch = len(train_loader)
        
        # Save best model
        best_model_path = os.path.join(self.model_dir, 'best_model.pth')
        
        # Start training
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training step
            train_loss = self._train_epoch(train_loader, optimizer, loss_fn, training_config)
            train_losses.append(train_loss)
            
            # Validation step
            val_loss = self._evaluate(valid_loader, loss_fn)
            val_losses.append(val_loss)
            
            # Update learning rate
            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)
            
            if training_config['lr_scheduler'] == 'ReduceLROnPlateau':
                lr_scheduler.step(val_loss)
            elif training_config['lr_scheduler'] == 'CosineAnnealing':
                lr_scheduler.step()
            
            # Log progress
            self.logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.6f}")
            
            # Check for early stopping
            if early_stopping:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model
                    torch.save(self.model.state_dict(), best_model_path)
                    self.logger.info(f"Saved best model with val_loss: {best_val_loss:.4f}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                        break
        
        # Training time
        training_time = time.time() - start_time
        
        # Load best model for evaluation
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
            self.logger.info("Loaded best model for evaluation")
        
        # Evaluate on test set
        test_metrics = self._evaluate_metrics(test_loader)
        
        # Create training curves
        loss_fig = self._plot_training_curves(train_losses, val_losses, learning_rates)
        
        # Save final model
        final_model_path = os.path.join(self.model_dir, 'final_model.pth')
        torch.save(self.model.state_dict(), final_model_path)
        
        # Compile results
        results = {
            "training_time": training_time,
            "epochs_completed": len(train_losses),
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "best_val_loss": best_val_loss,
            "test_metrics": test_metrics,
            "loss_history": {
                "train": train_losses,
                "val": val_losses
            },
            "learning_rates": learning_rates
        }
        
        return results
    
    def _train_epoch(self, train_loader, optimizer, loss_fn, training_config):
        """
        Train the model for one epoch.
        
        Args:
            train_loader (DataLoader): Training data loader
            optimizer (torch.optim.Optimizer): Optimizer
            loss_fn (function): Loss function
            training_config (dict): Training configuration dictionary
            
        Returns:
            float: Average training loss
        """
        self.model.train()
        total_loss = 0
        
        for i, batch in enumerate(tqdm(train_loader, desc="Training")):
            # Move data to device
            smiles_ids = batch['smiles_ids'].to(self.device)
            smiles_mask = batch['smiles_mask'].to(self.device)
            ecfp = batch['ecfp'].to(self.device)
            graph_data = batch['graph_data'].to(self.device)
            
            mfbert_input_ids = None
            mfbert_attention_mask = None
            if 'mfbert_input_ids' in batch:
                mfbert_input_ids = batch['mfbert_input_ids'].to(self.device)
                mfbert_attention_mask = batch['mfbert_attention_mask'].to(self.device)
            
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            
            mfbert_inputs = {
                'input_ids': mfbert_input_ids,
                'attention_mask': mfbert_attention_mask
            } if mfbert_input_ids is not None else None
            
            outputs = self.model(smiles_ids, smiles_mask, ecfp, graph_data, mfbert_inputs)
            
            # Calculate loss
            if training_config['loss_function'] == 'Custom Multi-objective':
                loss = self._custom_multiobjective_loss(
                    outputs['predictions'], 
                    labels,
                    aux_outputs=outputs['auxiliary_outputs'],
                    contrastive_loss=outputs['contrastive_loss']
                )
            else:
                loss = loss_fn(outputs['predictions'], labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Update OneCycleLR if used
            if training_config['lr_scheduler'] == 'OneCycleLR':
                lr_scheduler.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def _evaluate(self, data_loader, loss_fn):
        """
        Evaluate the model.
        
        Args:
            data_loader (DataLoader): Data loader
            loss_fn (function): Loss function
            
        Returns:
            float: Average validation loss
        """
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in data_loader:
                # Move data to device
                smiles_ids = batch['smiles_ids'].to(self.device)
                smiles_mask = batch['smiles_mask'].to(self.device)
                ecfp = batch['ecfp'].to(self.device)
                graph_data = batch['graph_data'].to(self.device)
                
                mfbert_input_ids = None
                mfbert_attention_mask = None
                if 'mfbert_input_ids' in batch:
                    mfbert_input_ids = batch['mfbert_input_ids'].to(self.device)
                    mfbert_attention_mask = batch['mfbert_attention_mask'].to(self.device)
                
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                mfbert_inputs = {
                    'input_ids': mfbert_input_ids,
                    'attention_mask': mfbert_attention_mask
                } if mfbert_input_ids is not None else None
                
                outputs = self.model(smiles_ids, smiles_mask, ecfp, graph_data, mfbert_inputs)
                
                # Calculate loss
                loss = loss_fn(outputs['predictions'], labels)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        return avg_loss
    
    def _evaluate_metrics(self, data_loader):
        """
        Evaluate the model and calculate various metrics.
        
        Args:
            data_loader (DataLoader): Data loader
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                # Move data to device
                smiles_ids = batch['smiles_ids'].to(self.device)
                smiles_mask = batch['smiles_mask'].to(self.device)
                ecfp = batch['ecfp'].to(self.device)
                graph_data = batch['graph_data'].to(self.device)
                
                mfbert_input_ids = None
                mfbert_attention_mask = None
                if 'mfbert_input_ids' in batch:
                    mfbert_input_ids = batch['mfbert_input_ids'].to(self.device)
                    mfbert_attention_mask = batch['mfbert_attention_mask'].to(self.device)
                
                labels = batch['labels']
                
                # Forward pass
                mfbert_inputs = {
                    'input_ids': mfbert_input_ids,
                    'attention_mask': mfbert_attention_mask
                } if mfbert_input_ids is not None else None
                
                outputs = self.model(smiles_ids, smiles_mask, ecfp, graph_data, mfbert_inputs)
                
                predictions = outputs['predictions'].cpu().numpy()
                
                all_preds.extend(predictions)
                all_labels.extend(labels.numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        metrics = {}
        
        if self.model_config['task']['task_type'] == 'regression':
            metrics['r2'] = r2_score(all_labels, all_preds)
            metrics['rmse'] = np.sqrt(mean_squared_error(all_labels, all_preds))
            metrics['mae'] = mean_absolute_error(all_labels, all_preds)
        else:
            # For classification tasks
            # TODO: Implement classification metrics
            pass
        
        # Get modality weights for interpretation
        modality_weights = outputs['modality_weights'].cpu().numpy().tolist()
        modalities = ['smiles', 'ecfp', 'graph']
        if self.model_config['mfbert_encoder']['use_mfbert']:
            modalities.append('mfbert')
        
        metrics['modality_weights'] = {
            modality: weight for modality, weight in zip(modalities, modality_weights)
        }
        
        return metrics
    
    def _plot_training_curves(self, train_losses, val_losses, learning_rates):
        """
        Plot training and validation loss curves.
        
        Args:
            train_losses (list): Training losses
            val_losses (list): Validation losses
            learning_rates (list): Learning rates
            
        Returns:
            matplotlib.figure.Figure: Figure with the curves
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        epochs = range(1, len(train_losses) + 1)
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot learning rate
        ax2.plot(epochs, learning_rates, 'g-')
        ax2.set_title('Learning Rate')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Learning Rate')
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(self.output_dir, 'training_curves.png')
        plt.savefig(fig_path)
        
        return fig
    
    def _run_ablation_study(self, train_loader, valid_loader, test_loader, training_config, components):
        """
        Run ablation study to analyze the contribution of different components.
        
        Args:
            train_loader (DataLoader): Training data loader
            valid_loader (DataLoader): Validation data loader
            test_loader (DataLoader): Test data loader
            training_config (dict): Training configuration dictionary
            components (list): List of components to ablate
            
        Returns:
            dict: Ablation study results
        """
        self.logger.info(f"Running ablation study for components: {components}")
        
        ablation_results = {
            "baseline": self._evaluate_metrics(test_loader)
        }
        
        # Create ablated model configurations
        for component in components:
            ablated_config = self._create_ablated_config(component)
            
            if ablated_config:
                # Initialize ablated model
                ablated_model = HCAAN(ablated_config).to(self.device)
                
                # Train ablated model
                self.logger.info(f"Training ablated model without {component}...")
                
                # Use fewer epochs for ablation study
                ablation_training_config = training_config.copy()
                ablation_training_config['epochs'] = min(20, training_config['epochs'])
                
                # Set up training components
                optimizer = self._setup_optimizer(ablation_training_config)
                lr_scheduler = self._setup_lr_scheduler(optimizer, ablation_training_config)
                loss_fn = self._setup_loss_function(ablation_training_config)
                
                # Train the ablated model
                # Note: This is a simplified training loop for ablation study
                ablated_model.train()
                for epoch in range(ablation_training_config['epochs']):
                    for batch in train_loader:
                        # Move data to device
                        smiles_ids = batch['smiles_ids'].to(self.device)
                        smiles_mask = batch['smiles_mask'].to(self.device)
                        ecfp = batch['ecfp'].to(self.device)
                        graph_data = batch['graph_data'].to(self.device)
                        
                        mfbert_input_ids = None
                        mfbert_attention_mask = None
                        if 'mfbert_input_ids' in batch:
                            mfbert_input_ids = batch['mfbert_input_ids'].to(self.device)
                            mfbert_attention_mask = batch['mfbert_attention_mask'].to(self.device)
                        
                        labels = batch['labels'].to(self.device)
                        
                        # Forward pass
                        optimizer.zero_grad()
                        
                        mfbert_inputs = {
                            'input_ids': mfbert_input_ids,
                            'attention_mask': mfbert_attention_mask
                        } if mfbert_input_ids is not None else None
                        
                        outputs = ablated_model(smiles_ids, smiles_mask, ecfp, graph_data, mfbert_inputs)
                        
                        # Calculate loss
                        loss = loss_fn(outputs['predictions'], labels)
                        
                        # Backward pass
                        loss.backward()
                        optimizer.step()
                
                # Evaluate ablated model
                ablated_metrics = self._evaluate_ablated_model(ablated_model, test_loader)
                
                # Save results
                component_name = component.replace(' ', '_').replace('-', '_')
                ablation_results[f"no_{component_name}"] = ablated_metrics
        
        return ablation_results
    
    def _create_ablated_config(self, component):
        """
        Create a configuration with a specific component ablated.
        
        Args:
            component (str): Component to ablate
            
        Returns:
            dict: Ablated configuration
        """
        ablated_config = self.model_config.copy()
        
        if component == "SMILES Encoder":
            # Replace SMILES encoder with a dummy version
            ablated_config['smiles_encoder']['num_layers'] = 1
            ablated_config['smiles_encoder']['hidden_dim'] = 64
            ablated_config['smiles_encoder']['num_heads'] = 2
        elif component == "ECFP Encoder":
            # Replace ECFP encoder with a dummy version
            ablated_config['ecfp_encoder']['num_layers'] = 1
            ablated_config['ecfp_encoder']['hidden_dim'] = 64
        elif component == "GCN Encoder":
            # Replace GCN encoder with a dummy version
            ablated_config['gcn_encoder']['num_layers'] = 1
            ablated_config['gcn_encoder']['hidden_dim'] = 64
        elif component == "MFBERT Encoder":
            # Disable MFBERT encoder
            ablated_config['mfbert_encoder']['use_mfbert'] = False
        elif component == "Chemical-Aware Attention":
            # Disable chemical-aware attention
            ablated_config['fusion']['use_chemical_aware'] = False
        elif component == "Adaptive Gating":
            # Disable adaptive gating
            ablated_config['fusion']['use_adaptive_gating'] = False
        elif component == "Multi-Scale Attention":
            # Disable multi-scale attention
            ablated_config['fusion']['use_multi_scale'] = False
        elif component == "Task-Specific Weights":
            # Disable task-specific weights
            ablated_config['modal_importance']['use_task_specific'] = False
        elif component == "Complexity-Aware Selection":
            # Disable complexity-aware selection
            ablated_config['modal_importance']['use_complexity_aware'] = False
        elif component == "Uncertainty Estimation":
            # Disable uncertainty estimation
            ablated_config['modal_importance']['use_uncertainty'] = False
        else:
            self.logger.warning(f"Component {component} not recognized for ablation study")
            return None
        
        return ablated_config
    
    def _evaluate_ablated_model(self, model, data_loader):
        """
        Evaluate an ablated model.
        
        Args:
            model (HCAAN): Ablated model
            data_loader (DataLoader): Data loader
            
        Returns:
            dict: Evaluation metrics
        """
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Move data to device
                smiles_ids = batch['smiles_ids'].to(self.device)
                smiles_mask = batch['smiles_mask'].to(self.device)
                ecfp = batch['ecfp'].to(self.device)
                graph_data = batch['graph_data'].to(self.device)
                
                mfbert_input_ids = None
                mfbert_attention_mask = None
                if 'mfbert_input_ids' in batch:
                    mfbert_input_ids = batch['mfbert_input_ids'].to(self.device)
                    mfbert_attention_mask = batch['mfbert_attention_mask'].to(self.device)
                
                labels = batch['labels']
                
                # Forward pass
                mfbert_inputs = {
                    'input_ids': mfbert_input_ids,
                    'attention_mask': mfbert_attention_mask
                } if mfbert_input_ids is not None else None
                
                outputs = model(smiles_ids, smiles_mask, ecfp, graph_data, mfbert_inputs)
                
                predictions = outputs['predictions'].cpu().numpy()
                
                all_preds.extend(predictions)
                all_labels.extend(labels.numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        metrics = {}
        
        if self.model_config['task']['task_type'] == 'regression':
            metrics['r2'] = r2_score(all_labels, all_preds)
            metrics['rmse'] = np.sqrt(mean_squared_error(all_labels, all_preds))
            metrics['mae'] = mean_absolute_error(all_labels, all_preds)
        else:
            # For classification tasks
            # TODO: Implement classification metrics
            pass
        
        return metrics