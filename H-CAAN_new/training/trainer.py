import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import json
from datetime import datetime

# Local imports
from .losses import get_loss_function
from .optimizers import get_optimizer, get_scheduler
from evaluation.metrics import calculate_metrics, print_metrics

class Trainer:
    """
    Trainer class for H-CAAN model
    
    Handles model training, validation, testing, and model checkpointing
    """
    
    def __init__(self, model, config, device='cuda', checkpoint_dir='checkpoints', 
                 experiment_name=None, wandb_logging=False):
        """
        Initialize the trainer
        
        Args:
            model: The H-CAAN model
            config: Training configuration dictionary
            device: Device to use for training ('cuda' or 'cpu')
            checkpoint_dir: Directory to save model checkpoints
            experiment_name: Name of the experiment for logging
            wandb_logging: Whether to log metrics to Weights & Biases
        """
        self.model = model
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        # Create experiment name
        if experiment_name is None:
            self.experiment_name = f"H-CAAN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            self.experiment_name = experiment_name
        
        # Setup logging
        self.setup_logging()
        
        # Setup wandb logging
        self.wandb_logging = wandb_logging
        if wandb_logging:
            import wandb
            wandb.init(project="H-CAAN", name=self.experiment_name, config=config)
            self.wandb = wandb
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.join(checkpoint_dir, self.experiment_name), exist_ok=True)
        
        # Save config
        with open(os.path.join(checkpoint_dir, self.experiment_name, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)
        
        # Initialize training components
        self.optimizer = get_optimizer(model.parameters(), config['optimizer'], 
                                     config['learning_rate'], config['weight_decay'])
        
        self.loss_fn = get_loss_function(config['loss_function'])
        
        self.scheduler = get_scheduler(self.optimizer, config['lr_scheduler'], config)
        
        # Initialize tracking variables
        self.best_val_loss = float('inf')
        self.best_val_metric = 0.0  # For metrics like R2, higher is better
        self.epochs_without_improvement = 0
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': []
        }
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_file = os.path.join(self.checkpoint_dir, f"{self.experiment_name}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Starting experiment: {self.experiment_name}")
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_samples = 0
        all_targets = []
        all_predictions = []
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Training")
        
        for batch in pbar:
            # Get inputs and targets
            inputs, targets = self._prepare_batch(batch)
            batch_size = targets.size(0)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(*inputs)
            
            # Calculate loss
            loss = self.loss_fn(outputs, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping if configured
            if 'gradient_clip' in self.config and self.config['gradient_clip'] > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip'])
            
            # Update weights
            self.optimizer.step()
            
            # Track loss
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Track predictions and targets for metrics calculation
            all_targets.append(targets.detach().cpu().numpy())
            all_predictions.append(outputs.detach().cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({"loss": loss.item()})
        
        # Calculate epoch loss
        epoch_loss = total_loss / total_samples
        
        # Concatenate predictions and targets
        all_targets = np.concatenate(all_targets)
        all_predictions = np.concatenate(all_predictions)
        
        # Calculate metrics
        metrics = calculate_metrics(all_targets, all_predictions, task_type=self.config.get('task_type', 'regression'))
        
        return epoch_loss, metrics
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Get inputs and targets
                inputs, targets = self._prepare_batch(batch)
                batch_size = targets.size(0)
                
                # Forward pass
                outputs = self.model(*inputs)
                
                # Calculate loss
                loss = self.loss_fn(outputs, targets)
                
                # Track loss
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                # Track predictions and targets for metrics calculation
                all_targets.append(targets.detach().cpu().numpy())
                all_predictions.append(outputs.detach().cpu().numpy())
        
        # Calculate epoch loss
        epoch_loss = total_loss / total_samples
        
        # Concatenate predictions and targets
        all_targets = np.concatenate(all_targets)
        all_predictions = np.concatenate(all_predictions)
        
        # Calculate metrics
        metrics = calculate_metrics(all_targets, all_predictions, task_type=self.config.get('task_type', 'regression'))
        
        return epoch_loss, metrics, all_targets, all_predictions
    
    def test(self, test_loader):
        """Test the model"""
        self.model.eval()
        
        all_targets = []
        all_predictions = []
        modal_weights = []  # For storing modal weights if applicable
        attentions = []  # For storing attention matrices if applicable
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Testing"):
                # Get inputs and targets
                inputs, targets = self._prepare_batch(batch)
                
                # Forward pass
                if self.config.get('return_attention', False):
                    outputs, weights, attention = self.model(*inputs, return_attention=True)
                    modal_weights.append(weights.detach().cpu().numpy())
                    attentions.append(attention.detach().cpu().numpy())
                else:
                    outputs = self.model(*inputs)
                
                # Track predictions and targets
                all_targets.append(targets.detach().cpu().numpy())
                all_predictions.append(outputs.detach().cpu().numpy())
        
        # Concatenate predictions and targets
        all_targets = np.concatenate(all_targets)
        all_predictions = np.concatenate(all_predictions)
        
        # Calculate metrics
        metrics = calculate_metrics(all_targets, all_predictions, task_type=self.config.get('task_type', 'regression'))
        
        # Process model weights and attentions if available
        if modal_weights:
            modal_weights = np.concatenate(modal_weights)
            attentions = np.concatenate(attentions)
            
            results = {
                'targets': all_targets,
                'predictions': all_predictions,
                'metrics': metrics,
                'modal_weights': modal_weights,
                'attentions': attentions
            }
        else:
            results = {
                'targets': all_targets,
                'predictions': all_predictions,
                'metrics': metrics
            }
        
        return results
    
    def _prepare_batch(self, batch):
        """Prepare batch data for training/validation/testing"""
        # Extract batch data based on your data format
        # This is a placeholder - modify based on your actual data structure
        
        # Example for a batch with multiple modal inputs:
        smiles_encodings = batch['smiles_encodings'].to(self.device)
        smiles_masks = batch['smiles_masks'].to(self.device)
        ecfp = batch['ecfp'].to(self.device)
        graph_x = batch['graph_x'].to(self.device)
        graph_edge_index = batch['graph_edge_index'].to(self.device)
        graph_batch = batch['graph_batch'].to(self.device)
        mfbert_input_ids = batch['mfbert_input_ids'].to(self.device)
        mfbert_attention_mask = batch['mfbert_attention_mask'].to(self.device)
        
        targets = batch['targets'].to(self.device)
        
        inputs = (smiles_encodings, smiles_masks, ecfp, graph_x, 
                 graph_edge_index, graph_batch, mfbert_input_ids, mfbert_attention_mask)
        
        return inputs, targets
    
    def train(self, train_loader, val_loader, test_loader=None):
        """
        Train the model for the specified number of epochs
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            test_loader: DataLoader for test data (optional)
            
        Returns:
            Dictionary with training history and final test results
        """
        epochs = self.config['epochs']
        use_early_stopping = self.config['early_stopping']['use']
        patience = self.config['early_stopping']['patience']
        
        self.logger.info(f"Starting training for {epochs} epochs")
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            epoch_start_time = time.time()
            
            # Train one epoch
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_metrics, _, _ = self.validate(val_loader)
            
            # Update learning rate scheduler if using ReduceLROnPlateau
            if isinstance(self.scheduler, ReduceLROnPlateau):
                # Use validation loss for ReduceLROnPlateau scheduler
                self.scheduler.step(val_loss)
            elif self.scheduler is not None:
                # For other schedulers, step without validation loss
                self.scheduler.step()
            
            # Log current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['learning_rates'].append(current_lr)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_metrics)
            self.history['val_metrics'].append(val_metrics)
            
            # Log epoch results
            epoch_time = time.time() - epoch_start_time
            self.logger.info(
                f"Epoch {epoch}/{epochs} - {epoch_time:.2f}s - "
                f"Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - "
                f"Train R2: {train_metrics.get('r2', 0):.4f} - Val R2: {val_metrics.get('r2', 0):.4f} - "
                f"LR: {current_lr:.6f}"
            )
            
            # Print metrics
            self.logger.info("Train Metrics:")
            print_metrics(train_metrics)
            self.logger.info("Validation Metrics:")
            print_metrics(val_metrics)
            
            # Log to wandb
            if self.wandb_logging:
                log_dict = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": current_lr
                }
                # Add metrics
                for metric, value in train_metrics.items():
                    log_dict[f"train_{metric}"] = value
                for metric, value in val_metrics.items():
                    log_dict[f"val_{metric}"] = value
                
                self.wandb.log(log_dict)
            
            # Check for improvement
            primary_metric = self.config.get('primary_metric', 'loss')
            
            if primary_metric == 'loss':
                # For loss, lower is better
                current_metric = val_loss
                is_better = current_metric < self.best_val_loss
                
                if is_better:
                    improvement = self.best_val_loss - current_metric
                    self.best_val_loss = current_metric
                    self.logger.info(f"Validation loss improved by {improvement:.6f}")
                    self.epochs_without_improvement = 0
                    
                    # Save best model
                    self._save_checkpoint(epoch, is_best=True)
                else:
                    self.epochs_without_improvement += 1
            else:
                # For other metrics (e.g., R2), higher is better
                current_metric = val_metrics.get(primary_metric, 0)
                is_better = current_metric > self.best_val_metric
                
                if is_better:
                    improvement = current_metric - self.best_val_metric
                    self.best_val_metric = current_metric
                    self.logger.info(f"Validation {primary_metric} improved by {improvement:.6f}")
                    self.epochs_without_improvement = 0
                    
                    # Save best model
                    self._save_checkpoint(epoch, is_best=True)
                else:
                    self.epochs_without_improvement += 1
            
            # Save regular checkpoint
            if epoch % self.config.get('checkpoint_frequency', 10) == 0:
                self._save_checkpoint(epoch)
            
            # Early stopping
            if use_early_stopping and self.epochs_without_improvement >= patience:
                self.logger.info(f"Early stopping triggered after {epoch} epochs")
                break
        
        # Calculate training time
        training_time = time.time() - start_time
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Plot training history
        self._plot_training_history()
        
        # Test the model if test_loader is provided
        if test_loader:
            self.logger.info("Testing the best model...")
            
            # Load the best model
            best_model_path = os.path.join(self.checkpoint_dir, self.experiment_name, 'model_best.pth')
            self._load_checkpoint(best_model_path)
            
            # Run testing
            test_results = self.test(test_loader)
            
            # Log test results
            self.logger.info("Test Metrics:")
            print_metrics(test_results['metrics'])
            
            # Log to wandb
            if self.wandb_logging:
                log_dict = {}
                for metric, value in test_results['metrics'].items():
                    log_dict[f"test_{metric}"] = value
                self.wandb.log(log_dict)
            
            return {
                'history': self.history,
                'test_results': test_results
            }
        
        return {
            'history': self.history
        }
    
    def _save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_metric': self.best_val_metric,
            'config': self.config,
            'history': self.history
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, self.experiment_name, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if this is the best one
        if is_best:
            best_model_path = os.path.join(self.checkpoint_dir, self.experiment_name, 'model_best.pth')
            torch.save(checkpoint, best_model_path)
            self.logger.info(f"Saved best model checkpoint to {best_model_path}")
    
    def _load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        if not os.path.exists(checkpoint_path):
            self.logger.error(f"Checkpoint {checkpoint_path} does not exist")
            return False
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_metric = checkpoint['best_val_metric']
        self.history = checkpoint['history']
        
        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return True
    
    def _plot_training_history(self):
        """Plot training history"""
        # Create figures directory
        figures_dir = os.path.join(self.checkpoint_dir, self.experiment_name, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        
        # Plot training and validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['train_loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(figures_dir, 'loss_curve.png'))
        plt.close()
        
        # Plot learning rate
        plt.figure(figsize=(10, 6))
        plt.plot(self.history['learning_rates'])
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True)
        plt.savefig(os.path.join(figures_dir, 'learning_rate.png'))
        plt.close()
        
        # Plot metrics if available
        if self.history['train_metrics'] and 'r2' in self.history['train_metrics'][0]:
            train_r2 = [metrics['r2'] for metrics in self.history['train_metrics']]
            val_r2 = [metrics['r2'] for metrics in self.history['val_metrics']]
            
            plt.figure(figsize=(10, 6))
            plt.plot(train_r2, label='Training R²')
            plt.plot(val_r2, label='Validation R²')
            plt.xlabel('Epoch')
            plt.ylabel('R²')
            plt.title('Training and Validation R²')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(figures_dir, 'r2_curve.png'))
            plt.close()
        
        # Plot RMSE if available
        if self.history['train_metrics'] and 'rmse' in self.history['train_metrics'][0]:
            train_rmse = [metrics['rmse'] for metrics in self.history['train_metrics']]
            val_rmse = [metrics['rmse'] for metrics in self.history['val_metrics']]
            
            plt.figure(figsize=(10, 6))
            plt.plot(train_rmse, label='Training RMSE')
            plt.plot(val_rmse, label='Validation RMSE')
            plt.xlabel('Epoch')
            plt.ylabel('RMSE')
            plt.title('Training and Validation RMSE')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(figures_dir, 'rmse_curve.png'))
            plt.close()


class ModelTrainer(Trainer):
    """
    ModelTrainer class extending the base Trainer with additional functionality
    
    Provides a simplified interface for training H-CAAN models with automatic
    configuration management and performance tracking
    """
    
    def __init__(self, model, config, device='cuda', checkpoint_dir='checkpoints', 
                 experiment_name=None, wandb_logging=False):
        """
        Initialize the model trainer
        
        Args:
            model: The H-CAAN model
            config: Training configuration dictionary
            device: Device to use for training ('cuda' or 'cpu')
            checkpoint_dir: Directory to save model checkpoints
            experiment_name: Name of the experiment for logging
            wandb_logging: Whether to log metrics to Weights & Biases
        """
        super().__init__(model, config, device, checkpoint_dir, experiment_name, wandb_logging)
        
        # Additional attributes for ModelTrainer
        self.initial_lr = config['learning_rate']
        self.task_type = config.get('task_type', 'regression')
        self.task_name = config.get('task_name', 'Property Prediction')
        self.feature_importance = {}
        
    def train_model(self, train_loader, val_loader, test_loader=None, verbose=True):
        """
        Train the model with simplified interface
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            test_loader: DataLoader for test data (optional)
            verbose: Whether to print verbose output
            
        Returns:
            Dictionary with training results
        """
        if verbose:
            self.logger.info(f"Starting training for {self.task_name} task")
            self.logger.info(f"Task type: {self.task_type}")
            self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
            self.logger.info(f"Learning rate: {self.initial_lr}")
            self.logger.info(f"Batch size: {train_loader.batch_size}")
            self.logger.info(f"Training samples: {len(train_loader.dataset)}")
            self.logger.info(f"Validation samples: {len(val_loader.dataset)}")
            if test_loader:
                self.logger.info(f"Test samples: {len(test_loader.dataset)}")
        
        # Run training with the parent class
        results = self.train(train_loader, val_loader, test_loader)
        
        # Calculate feature importance if the model supports it
        if hasattr(self.model, 'get_feature_importance'):
            self.feature_importance = self.model.get_feature_importance()
            
        return results
    
    def run_cross_validation(self, dataset, n_folds=5, batch_size=32, random_state=42):
        """
        Run cross-validation
        
        Args:
            dataset: The dataset to use for cross-validation
            n_folds: Number of folds
            batch_size: Batch size for training
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary with cross-validation results
        """
        from sklearn.model_selection import KFold
        import copy
        
        self.logger.info(f"Running {n_folds}-fold cross-validation")
        
        # Initialize KFold
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        
        # Initialize results
        cv_results = {
            'fold_metrics': [],
            'mean_metrics': {},
            'std_metrics': {}
        }
        
        # Get dataset indices
        indices = list(range(len(dataset)))
        
        # Run for each fold
        for fold, (train_idx, val_idx) in enumerate(kf.split(indices)):
            self.logger.info(f"Training fold {fold+1}/{n_folds}")
            
            # Create data loaders for this fold
            train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
            
            train_loader = DataLoader(
                dataset, batch_size=batch_size, sampler=train_sampler,
                num_workers=self.config.get('num_workers', 0)
            )
            
            val_loader = DataLoader(
                dataset, batch_size=batch_size, sampler=val_sampler,
                num_workers=self.config.get('num_workers', 0)
            )
            
            # Reset model for this fold
            model_copy = copy.deepcopy(self.model)
            optimizer = get_optimizer(model_copy.parameters(), self.config['optimizer'],
                                     self.initial_lr, self.config['weight_decay'])
            scheduler = get_scheduler(optimizer, self.config['lr_scheduler'], self.config)
            
            # Create a new trainer for this fold
            fold_trainer = ModelTrainer(
                model_copy, self.config, self.device,
                checkpoint_dir=os.path.join(self.checkpoint_dir, f"fold_{fold+1}"),
                experiment_name=f"{self.experiment_name}_fold_{fold+1}",
                wandb_logging=False  # Disable wandb for folds
            )
            
            # Train the fold
            fold_results = fold_trainer.train_model(train_loader, val_loader, verbose=False)
            
            # Get the metrics
            fold_metrics = fold_results.get('test_results', {}).get('metrics', {})
            if not fold_metrics:
                # If test_results not available, use the last validation metrics
                val_metrics = fold_results['history']['val_metrics'][-1]
                fold_metrics = val_metrics
            
            # Store fold metrics
            cv_results['fold_metrics'].append(fold_metrics)
            
            # Log fold metrics
            self.logger.info(f"Fold {fold+1} metrics:")
            print_metrics(fold_metrics)
        
        # Calculate mean and std metrics
        # Get all metric keys from first fold
        metric_keys = cv_results['fold_metrics'][0].keys()
        
        for metric in metric_keys:
            # Extract metric values from all folds
            metric_values = [fold_metrics[metric] for fold_metrics in cv_results['fold_metrics']]
            
            # Calculate mean and std
            cv_results['mean_metrics'][metric] = np.mean(metric_values)
            cv_results['std_metrics'][metric] = np.std(metric_values)
        
        # Log cross-validation summary
        self.logger.info("Cross-validation summary:")
        for metric in metric_keys:
            mean = cv_results['mean_metrics'][metric]
            std = cv_results['std_metrics'][metric]
            self.logger.info(f"{metric}: {mean:.4f} ± {std:.4f}")
        
        return cv_results
    
    def evaluate_model(self, test_loader, return_predictions=False):
        """
        Evaluate the model on test data
        
        Args:
            test_loader: DataLoader for test data
            return_predictions: Whether to return predictions
            
        Returns:
            Dictionary with evaluation results
        """
        self.logger.info("Evaluating model on test data")
        
        # Run test
        test_results = self.test(test_loader)
        
        # Log test metrics
        self.logger.info("Test metrics:")
        print_metrics(test_results['metrics'])
        
        # Create evaluation results
        evaluation_results = {
            'metrics': test_results['metrics']
        }
        
        # Add predictions if requested
        if return_predictions:
            evaluation_results['predictions'] = test_results['predictions']
            evaluation_results['targets'] = test_results['targets']
            
            # If modal weights and attentions are available, include them
            if 'modal_weights' in test_results:
                evaluation_results['modal_weights'] = test_results['modal_weights']
            if 'attentions' in test_results:
                evaluation_results['attentions'] = test_results['attentions']
        
        return evaluation_results
    
    def plot_feature_importance(self):
        """Plot feature importance if available"""
        if not self.feature_importance:
            self.logger.warning("Feature importance not available")
            return
        
        # Create figures directory
        figures_dir = os.path.join(self.checkpoint_dir, self.experiment_name, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        
        # Sort features by importance
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        feature_names = [f[0] for f in sorted_features]
        importance_values = [f[1] for f in sorted_features]
        
        # Create bar plot
        plt.barh(range(len(feature_names)), importance_values, align='center')
        plt.yticks(range(len(feature_names)), feature_names)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance')
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(os.path.join(figures_dir, 'feature_importance.png'))
        plt.close()
    
    def save_predictions(self, predictions, targets, output_file):
        """
        Save predictions to a CSV file
        
        Args:
            predictions: Predicted values
            targets: True values
            output_file: Output file path
        """
        import pandas as pd
        
        # Create DataFrame
        df = pd.DataFrame({
            'true': targets.flatten(),
            'predicted': predictions.flatten()
        })
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        self.logger.info(f"Predictions saved to {output_file}")
    
    def visualize_attention(self, attention_matrices, molecule_ids=None, num_samples=5):
        """
        Visualize attention matrices
        
        Args:
            attention_matrices: Attention matrices
            molecule_ids: Molecule IDs (optional)
            num_samples: Number of samples to visualize
        """
        if attention_matrices is None or len(attention_matrices) == 0:
            self.logger.warning("No attention matrices available")
            return
        
        # Create figures directory
        figures_dir = os.path.join(self.checkpoint_dir, self.experiment_name, 'figures', 'attention')
        os.makedirs(figures_dir, exist_ok=True)
        
        # Select a subset of samples
        if len(attention_matrices) > num_samples:
            indices = np.random.choice(len(attention_matrices), num_samples, replace=False)
            attention_matrices = [attention_matrices[i] for i in indices]
            if molecule_ids is not None:
                molecule_ids = [molecule_ids[i] for i in indices]
        
        # Visualize attention matrices
        for i, attention in enumerate(attention_matrices):
            plt.figure(figsize=(10, 8))
            plt.imshow(attention, cmap='viridis')
            plt.colorbar()
            
            title = f"Attention Matrix {i+1}"
            if molecule_ids is not None:
                title = f"Attention Matrix for Molecule {molecule_ids[i]}"
            
            plt.title(title)
            plt.tight_layout()
            
            # Save the plot
            plt.savefig(os.path.join(figures_dir, f'attention_matrix_{i+1}.png'))
            plt.close()


class AblationTrainer(Trainer):
    """Extends the Trainer class to support ablation studies"""
    
    def run_ablation_study(self, train_loader, val_loader, test_loader, ablation_components):
        """
        Run ablation study by training multiple models with different components removed
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            test_loader: DataLoader for test data
            ablation_components: List of components to ablate
            
        Returns:
            Dictionary with ablation study results
        """
        self.logger.info(f"Running ablation study for components: {ablation_components}")
        
        # Store original model state
        original_state = self._get_model_state()
        
        # Dictionary to store results
        ablation_results = {
            'baseline': {}
        }
        
        # First, train and evaluate the baseline model (no ablation)
        self.logger.info("Training baseline model...")
        baseline_results = self.train(train_loader, val_loader, test_loader)
        ablation_results['baseline'] = {
            'test_metrics': baseline_results['test_results']['metrics'],
            'history': baseline_results['history']
        }
        
        # For each component, ablate it and train a new model
        for component in ablation_components:
            component_key = f"no_{component.replace(' ', '_')}"
            self.logger.info(f"Ablating component: {component}")
            
            # Reset model to original state
            self._reset_model_state(original_state)
            
            # Ablate the component
            self._ablate_component(component)
            
            # Train the ablated model
            component_results = self.train(train_loader, val_loader, test_loader)
            
            # Store results
            ablation_results[component_key] = {
                'test_metrics': component_results['test_results']['metrics'],
                'history': component_results['history']
            }
        
        # Reset model to original state
        self._reset_model_state(original_state)
        
        # Log ablation results
        self._log_ablation_results(ablation_results)
        
        # Plot ablation results
        self._plot_ablation_results(ablation_results)
        
        return ablation_results
    
    def _get_model_state(self):
        """Get the current model state"""
        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'best_val_metric': self.best_val_metric,
            'epochs_without_improvement': self.epochs_without_improvement,
            'history': self.history.copy()
        }
    
    def _reset_model_state(self, state):
        """Reset the model to a previous state"""
        self.model.load_state_dict(state['model'])
        self.optimizer.load_state_dict(state['optimizer'])
        if state['scheduler'] and self.scheduler:
            self.scheduler.load_state_dict(state['scheduler'])
        
        self.best_val_loss = state['best_val_loss']
        self.best_val_metric = state['best_val_metric']
        self.epochs_without_improvement = state['epochs_without_improvement']
        self.history = state['history'].copy()
    
    def _ablate_component(self, component):
        """
        Ablate a specific component of the model
        
        This is a placeholder - modify based on your model architecture
        """
        if component == "SMILES Encoder":
            # Disable SMILES encoder by setting its weights to zeros
            for name, param in self.model.named_parameters():
                if "smiles_encoder" in name:
                    param.data.zero_()
                    param.requires_grad = False
        
        elif component == "ECFP Encoder":
            # Disable ECFP encoder
            for name, param in self.model.named_parameters():
                if "ecfp_encoder" in name:
                    param.data.zero_()
                    param.requires_grad = False
        
        elif component == "GCN Encoder":
            # Disable GCN encoder
            for name, param in self.model.named_parameters():
                if "gcn_encoder" in name:
                    param.data.zero_()
                    param.requires_grad = False
        
        elif component == "MFBERT Encoder":
            # Disable MFBERT encoder
            for name, param in self.model.named_parameters():
                if "mfbert_encoder" in name:
                    param.data.zero_()
                    param.requires_grad = False
        
        elif component == "Chemical-Aware Attention":
            # Disable chemical-aware attention
            for name, param in self.model.named_parameters():
                if "chemical_aware_attention" in name:
                    param.data.zero_()
                    param.requires_grad = False
        
        elif component == "Adaptive Gating":
            # Disable adaptive gating
            for name, param in self.model.named_parameters():
                if "adaptive_gating" in name:
                    param.data.zero_()
                    param.requires_grad = False
        
        elif component == "Multi-Scale Attention":
            # Disable multi-scale attention
            for name, param in self.model.named_parameters():
                if "multi_scale_attention" in name:
                    param.data.zero_()
                    param.requires_grad = False
                    
        elif component == "Task-Specific Weights":
            # Disable task-specific weights
            for name, param in self.model.named_parameters():
                if "task_specific_weights" in name:
                    param.data.zero_()
                    param.requires_grad = False
        
        elif component == "Complexity-Aware Selection":
            # Disable complexity-aware selection
            for name, param in self.model.named_parameters():
                if "complexity_aware" in name:
                    param.data.zero_()
                    param.requires_grad = False
        
        elif component == "Uncertainty Estimation":
            # Disable uncertainty estimation
            for name, param in self.model.named_parameters():
                if "uncertainty_estimation" in name:
                    param.data.zero_()
                    param.requires_grad = False
        
        else:
            self.logger.warning(f"Unknown component: {component}. Skipping ablation.")
    
    def _log_ablation_results(self, ablation_results):
        """Log ablation study results"""
        self.logger.info("Ablation Study Results:")
        
        # Get the primary metric
        primary_metric = self.config.get('primary_metric', 'r2')
        
        # Print results for each configuration
        for config_name, results in ablation_results.items():
            metric_value = results['test_metrics'].get(primary_metric, 0)
            self.logger.info(f"{config_name}: {primary_metric}={metric_value:.4f}")
        
        # Calculate impact of each ablation
        baseline_metric = ablation_results['baseline']['test_metrics'].get(primary_metric, 0)
        
        for config_name, results in ablation_results.items():
            if config_name != 'baseline':
                metric_value = results['test_metrics'].get(primary_metric, 0)
                impact = baseline_metric - metric_value
                
                # For metrics where higher is better
                if primary_metric in ['r2', 'accuracy', 'auc']:
                    self.logger.info(f"Impact of {config_name}: {impact:.4f} decrease in {primary_metric}")
                # For metrics where lower is better
                else:
                    impact = -impact  # Reverse the sign for metrics where lower is better
                    self.logger.info(f"Impact of {config_name}: {impact:.4f} increase in {primary_metric}")
    
    def _plot_ablation_results(self, ablation_results):
        """Plot ablation study results"""
        # Create figures directory
        figures_dir = os.path.join(self.checkpoint_dir, self.experiment_name, 'figures')
        os.makedirs(figures_dir, exist_ok=True)
        
        # Get the primary metric
        primary_metric = self.config.get('primary_metric', 'r2')
        
        # Prepare data for plotting
        config_names = list(ablation_results.keys())
        metric_values = [results['test_metrics'].get(primary_metric, 0) for results in ablation_results.values()]
        
        # Create a bar plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(config_names, metric_values)
        
        # Highlight the baseline
        bars[0].set_color('gold')
        
        plt.xlabel('Model Configuration')
        plt.ylabel(primary_metric)
        plt.title(f'Ablation Study Results ({primary_metric})')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save the plot
        plt.savefig(os.path.join(figures_dir, f'ablation_results_{primary_metric}.png'))
        plt.close()
        
        # Plot multiple metrics if available
        common_metrics = set()
        for results in ablation_results.values():
            common_metrics.update(results['test_metrics'].keys())
        
        for metric in common_metrics:
            # Skip the primary metric as it's already plotted
            if metric == primary_metric:
                continue
            
            # Check if all configurations have this metric
            if all(metric in results['test_metrics'] for results in ablation_results.values()):
                metric_values = [results['test_metrics'][metric] for results in ablation_results.values()]
                
                plt.figure(figsize=(12, 6))
                bars = plt.bar(config_names, metric_values)
                
                # Highlight the baseline
                bars[0].set_color('gold')
                
                plt.xlabel('Model Configuration')
                plt.ylabel(metric)
                plt.title(f'Ablation Study Results ({metric})')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Save the plot
                plt.savefig(os.path.join(figures_dir, f'ablation_results_{metric}.png'))
                plt.close()