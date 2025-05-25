import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from pathlib import Path
import torch
from collections import OrderedDict
import logging

# Local imports
from .metrics import calculate_metrics, print_metrics, bootstrap_confidence_interval

# Set up logger
logger = logging.getLogger(__name__)

class AblationStudy:
    """
    Class for conducting ablation studies on H-CAAN components
    
    This class facilitates the systematic evaluation of the contribution
    of different components to the model's performance.
    """
    
    def __init__(self, config, output_dir='results/ablation'):
        """
        Initialize the ablation study
        
        Args:
            config: Configuration dictionary
            output_dir: Directory to save results
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize results dictionary
        self.results = {
            'config': config,
            'baseline': {},
            'ablations': {}
        }
    
    def add_baseline_results(self, predictions, targets, metrics=None, modality_weights=None):
        """
        Add baseline model results
        
        Args:
            predictions: Predictions from the baseline model
            targets: True targets
            metrics: Pre-computed metrics (optional)
            modality_weights: Modality weights used by the model (optional)
        """
        if metrics is None:
            # Calculate metrics based on task type
            task_type = self.config.get('task_type', 'regression')
            metrics = calculate_metrics(targets, predictions, task_type=task_type)
        
        self.results['baseline'] = {
            'metrics': metrics,
            'predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
            'modality_weights': modality_weights
        }
        
        # Log baseline results
        logger.info("Baseline model results:")
        print_metrics(metrics, task_type=self.config.get('task_type', 'regression'))
    
    def add_ablation_results(self, ablation_name, predictions, targets, metrics=None, modality_weights=None):
        """
        Add results from an ablated model
        
        Args:
            ablation_name: Name of the ablation
            predictions: Predictions from the ablated model
            targets: True targets
            metrics: Pre-computed metrics (optional)
            modality_weights: Modality weights used by the model (optional)
        """
        if metrics is None:
            # Calculate metrics based on task type
            task_type = self.config.get('task_type', 'regression')
            metrics = calculate_metrics(targets, predictions, task_type=task_type)
        
        self.results['ablations'][ablation_name] = {
            'metrics': metrics,
            'predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
            'modality_weights': modality_weights
        }
        
        # Log ablation results
        logger.info(f"Ablation '{ablation_name}' results:")
        print_metrics(metrics, task_type=self.config.get('task_type', 'regression'))
    
    def calculate_ablation_impact(self):
        """
        Calculate the impact of each ablation on model performance
        
        Returns:
            Dictionary of impact metrics
        """
        task_type = self.config.get('task_type', 'regression')
        
        # Get baseline metrics
        baseline_metrics = self.results['baseline']['metrics']
        
        # Determine the primary metric based on task type
        if task_type == 'regression':
            primary_metric = 'r2'
            # For R², higher is better
            higher_better = True
            secondary_metrics = ['rmse', 'mae', 'pearson_r']
        else:  # Classification
            primary_metric = 'accuracy'
            # For accuracy, higher is better
            higher_better = True
            if task_type == 'binary':
                secondary_metrics = ['f1', 'precision', 'recall', 'roc_auc']
            else:  # Multiclass
                secondary_metrics = ['f1_macro', 'precision_macro', 'recall_macro']
        
        # Calculate impact for each ablation
        impact = {}
        for ablation_name, ablation_results in self.results['ablations'].items():
            ablation_metrics = ablation_results['metrics']
            
            # Calculate impact on primary metric
            baseline_value = baseline_metrics[primary_metric]
            ablation_value = ablation_metrics[primary_metric]
            
            if higher_better:
                # For metrics where higher is better (R², accuracy)
                raw_impact = baseline_value - ablation_value
                relative_impact = raw_impact / baseline_value if baseline_value != 0 else 0
            else:
                # For metrics where lower is better (RMSE, MAE)
                raw_impact = ablation_value - baseline_value
                relative_impact = raw_impact / baseline_value if baseline_value != 0 else 0
            
            # Calculate impact on secondary metrics
            secondary_impact = {}
            for metric in secondary_metrics:
                if metric in baseline_metrics and metric in ablation_metrics:
                    if higher_better:
                        sec_raw_impact = baseline_metrics[metric] - ablation_metrics[metric]
                        sec_relative_impact = sec_raw_impact / baseline_metrics[metric] if baseline_metrics[metric] != 0 else 0
                    else:
                        sec_raw_impact = ablation_metrics[metric] - baseline_metrics[metric]
                        sec_relative_impact = sec_raw_impact / baseline_metrics[metric] if baseline_metrics[metric] != 0 else 0
                    
                    secondary_impact[metric] = {
                        'raw_impact': sec_raw_impact,
                        'relative_impact': sec_relative_impact
                    }
            
            impact[ablation_name] = {
                'primary_metric': primary_metric,
                'baseline_value': baseline_value,
                'ablation_value': ablation_value,
                'raw_impact': raw_impact,
                'relative_impact': relative_impact,
                'secondary_impact': secondary_impact
            }
        
        # Add impact to results
        self.results['impact'] = impact
        
        return impact
    
    def rank_components_by_importance(self):
        """
        Rank components by their importance to model performance
        
        Returns:
            Ordered dictionary of components ranked by importance
        """
        # Ensure impact has been calculated
        if 'impact' not in self.results:
            self.calculate_ablation_impact()
        
        impact = self.results['impact']
        
        # Get primary metric
        first_impact = next(iter(impact.values()))
        primary_metric = first_impact['primary_metric']
        
        # Rank components by absolute impact
        ranked_components = sorted(
            impact.items(),
            key=lambda x: abs(x[1]['relative_impact']),
            reverse=True
        )
        
        # Create ordered dictionary
        ranked_dict = OrderedDict()
        for component, impact_data in ranked_components:
            ranked_dict[component] = {
                'importance_score': abs(impact_data['relative_impact']),
                'raw_impact': impact_data['raw_impact'],
                'relative_impact': impact_data['relative_impact'],
                'baseline_value': impact_data['baseline_value'],
                'ablation_value': impact_data['ablation_value']
            }
        
        # Add ranking to results
        self.results['component_ranking'] = ranked_dict
        
        return ranked_dict
    
    def calculate_confidence_intervals(self, n_bootstraps=1000, confidence=0.95):
        """
        Calculate confidence intervals for ablation impacts using bootstrapping
        
        Args:
            n_bootstraps: Number of bootstrap samples
            confidence: Confidence level
            
        Returns:
            Dictionary of confidence intervals
        """
        task_type = self.config.get('task_type', 'regression')
        
        # Determine the primary metric function based on task type
        if task_type == 'regression':
            from sklearn.metrics import r2_score
            metric_func = r2_score
            primary_metric = 'r2'
        else:  # Classification
            from sklearn.metrics import accuracy_score
            metric_func = accuracy_score
            primary_metric = 'accuracy'
        
        # Calculate confidence intervals for baseline
        baseline_targets = np.array(self.results['baseline']['targets']) if 'targets' in self.results['baseline'] else None
        baseline_predictions = np.array(self.results['baseline']['predictions'])
        
        if baseline_targets is not None:
            baseline_ci = bootstrap_confidence_interval(
                baseline_targets, baseline_predictions, metric_func,
                n_bootstraps=n_bootstraps, confidence=confidence
            )
            self.results['baseline']['confidence_interval'] = baseline_ci
        
        # Calculate confidence intervals for each ablation
        confidence_intervals = {}
        for ablation_name, ablation_results in self.results['ablations'].items():
            ablation_targets = np.array(ablation_results['targets']) if 'targets' in ablation_results else None
            ablation_predictions = np.array(ablation_results['predictions'])
            
            if ablation_targets is not None:
                ablation_ci = bootstrap_confidence_interval(
                    ablation_targets, ablation_predictions, metric_func,
                    n_bootstraps=n_bootstraps, confidence=confidence
                )
                confidence_intervals[ablation_name] = ablation_ci
                self.results['ablations'][ablation_name]['confidence_interval'] = ablation_ci
        
        # Add confidence intervals to results
        self.results['confidence_intervals'] = {
            'baseline': baseline_ci if baseline_targets is not None else None,
            'ablations': confidence_intervals
        }
        
        return confidence_intervals
    
    def save_results(self, filename='ablation_results.json'):
        """
        Save ablation results to a JSON file
        
        Args:
            filename: Name of the output file
        """
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Ablation results saved to {output_path}")
    
    def plot_ablation_results(self, metric=None, save_fig=True, figsize=(12, 8)):
        """
        Plot ablation study results
        
        Args:
            metric: Metric to plot (default: primary metric)
            save_fig: Whether to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Ensure impact has been calculated
        if 'impact' not in self.results:
            self.calculate_ablation_impact()
        
        # Get primary metric if not specified
        if metric is None:
            first_impact = next(iter(self.results['impact'].values()))
            metric = first_impact['primary_metric']
        
        # Get task type
        task_type = self.config.get('task_type', 'regression')
        
        # Determine whether higher is better
        if metric in ['r2', 'accuracy', 'f1', 'precision', 'recall', 'roc_auc']:
            higher_better = True
            y_label = f"{metric.upper()}"
        else:
            higher_better = False
            y_label = f"{metric.upper()}"
        
        # Extract metric values
        baseline_value = self.results['baseline']['metrics'][metric]
        ablation_values = {
            ablation_name: results['metrics'][metric]
            for ablation_name, results in self.results['ablations'].items()
        }
        
        # Sort by impact
        if higher_better:
            sorted_ablations = sorted(
                ablation_values.items(),
                key=lambda x: x[1],
                reverse=False  # Lower values first (worse) for better visualization
            )
        else:
            sorted_ablations = sorted(
                ablation_values.items(),
                key=lambda x: x[1],
                reverse=True  # Higher values first (worse) for better visualization
            )
        
        # Prepare data for plotting
        names = ['Baseline'] + [name for name, _ in sorted_ablations]
        values = [baseline_value] + [value for _, value in sorted_ablations]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Set colors based on improvement/degradation
        if higher_better:
            # For metrics where higher is better (R², accuracy)
            colors = ['green'] + ['red' if v < baseline_value else 'green' for v in values[1:]]
        else:
            # For metrics where lower is better (RMSE, MAE)
            colors = ['green'] + ['red' if v > baseline_value else 'green' for v in values[1:]]
        
        # Create bar plot
        bars = ax.bar(range(len(names)), values, color=colors)
        
        # Set labels and title
        ax.set_xlabel('Model Configuration', fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)
        ax.set_title(f'Ablation Study Results ({metric.upper()})', fontsize=16)
        
        # Set x-axis ticks
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right', fontsize=12)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f'{height:.4f}',
                ha='center',
                va='bottom',
                fontsize=10
            )
        
        # Add grid
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if requested
        if save_fig:
            output_path = self.output_dir / f'ablation_results_{metric}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Ablation plot saved to {output_path}")
        
        return fig
    
    def plot_component_importance(self, save_fig=True, figsize=(12, 8)):
        """
        Plot component importance based on ablation results
        
        Args:
            save_fig: Whether to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Ensure components have been ranked
        if 'component_ranking' not in self.results:
            self.rank_components_by_importance()
        
        # Get component ranking
        ranking = self.results['component_ranking']
        
        # Prepare data for plotting
        components = list(ranking.keys())
        importance_scores = [data['importance_score'] for data in ranking.values()]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create bar plot
        bars = ax.barh(
            range(len(components)),
            importance_scores,
            color='skyblue'
        )
        
        # Set labels and title
        ax.set_xlabel('Importance Score (Impact on Performance)', fontsize=14)
        ax.set_ylabel('Component', fontsize=14)
        ax.set_title('Component Importance Ranking', fontsize=16)
        
        # Set y-axis ticks
        ax.set_yticks(range(len(components)))
        ax.set_yticklabels(components, fontsize=12)
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(
                width + 0.01,
                i,
                f'{width:.4f}',
                ha='left',
                va='center',
                fontsize=10
            )
        
        # Add grid
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if requested
        if save_fig:
            output_path = self.output_dir / 'component_importance.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Component importance plot saved to {output_path}")
        
        return fig
    
    def plot_correlation_matrix(self, metric='r2', save_fig=True, figsize=(10, 8)):
        """
        Plot correlation matrix of ablation results
        
        This shows how similar different ablations are in terms of their predictions
        
        Args:
            metric: Metric to use for color intensity
            save_fig: Whether to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        # Get all model configurations
        all_configs = ['baseline'] + list(self.results['ablations'].keys())
        
        # Create correlation matrix for predictions
        n_configs = len(all_configs)
        prediction_corr_matrix = np.zeros((n_configs, n_configs))
        metric_matrix = np.zeros((n_configs, n_configs))
        
        baseline_predictions = np.array(self.results['baseline']['predictions'])
        
        # Fill matrices
        for i, config_i in enumerate(all_configs):
            for j, config_j in enumerate(all_configs):
                # Get predictions
                if config_i == 'baseline':
                    pred_i = baseline_predictions
                else:
                    pred_i = np.array(self.results['ablations'][config_i]['predictions'])
                
                if config_j == 'baseline':
                    pred_j = baseline_predictions
                else:
                    pred_j = np.array(self.results['ablations'][config_j]['predictions'])
                
                # Calculate correlation
                prediction_corr_matrix[i, j] = np.corrcoef(pred_i, pred_j)[0, 1]
                
                # Get metric values
                if config_i == 'baseline':
                    metric_i = self.results['baseline']['metrics'][metric]
                else:
                    metric_i = self.results['ablations'][config_i]['metrics'][metric]
                
                if config_j == 'baseline':
                    metric_j = self.results['baseline']['metrics'][metric]
                else:
                    metric_j = self.results['ablations'][config_j]['metrics'][metric]
                
                # Calculate relative metric similarity (0-1)
                max_val = max(metric_i, metric_j)
                min_val = min(metric_i, metric_j)
                if max_val != 0:
                    metric_matrix[i, j] = min_val / max_val
                else:
                    metric_matrix[i, j] = 0
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot prediction correlation matrix
        sns.heatmap(
            prediction_corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='viridis',
            xticklabels=all_configs,
            yticklabels=all_configs,
            ax=ax1
        )
        ax1.set_title('Prediction Correlation Matrix', fontsize=14)
        
        # Plot metric similarity matrix
        sns.heatmap(
            metric_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            xticklabels=all_configs,
            yticklabels=all_configs,
            ax=ax2
        )
        ax2.set_title(f'{metric.upper()} Similarity Matrix', fontsize=14)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if requested
        if save_fig:
            output_path = self.output_dir / 'ablation_correlation_matrix.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Correlation matrix plot saved to {output_path}")
        
        return fig
    
    def plot_modality_weights(self, save_fig=True, figsize=(12, 8)):
        """
        Plot modality weights for different ablation configurations
        
        Args:
            save_fig: Whether to save the figure
            figsize: Figure size
            
        Returns:
            Matplotlib figure or None if no modality weights are available
        """
        # Check if modality weights are available
        baseline_weights = self.results['baseline'].get('modality_weights')
        if baseline_weights is None:
            logger.warning("No modality weights available for plotting")
            return None
        
        # Get modality names
        modality_names = list(baseline_weights.keys())
        
        # Get all configurations with modality weights
        configs_with_weights = ['baseline']
        for ablation_name, ablation_results in self.results['ablations'].items():
            if ablation_results.get('modality_weights') is not None:
                configs_with_weights.append(ablation_name)
        
        # Prepare data for plotting
        weights_data = []
        
        for config_name in configs_with_weights:
            if config_name == 'baseline':
                weights = baseline_weights
            else:
                weights = self.results['ablations'][config_name]['modality_weights']
            
            for modality_name, weight in weights.items():
                weights_data.append({
                    'Configuration': config_name,
                    'Modality': modality_name,
                    'Weight': weight
                })
        
        # Convert to DataFrame
        weights_df = pd.DataFrame(weights_data)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create grouped bar plot
        sns.barplot(
            x='Configuration',
            y='Weight',
            hue='Modality',
            data=weights_df,
            ax=ax
        )
        
        # Set labels and title
        ax.set_xlabel('Model Configuration', fontsize=14)
        ax.set_ylabel('Modality Weight', fontsize=14)
        ax.set_title('Modality Weights Across Ablations', fontsize=16)
        
        # Add legend
        ax.legend(title='Modality', fontsize=12, title_fontsize=12)
        
        # Add grid
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if requested
        if save_fig:
            output_path = self.output_dir / 'modality_weights.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Modality weights plot saved to {output_path}")
        
        return fig


def run_ablation_study(model, data_loaders, components_to_ablate, device, output_dir='results/ablation'):
    """
    Run an ablation study on a H-CAAN model
    
    Args:
        model: H-CAAN model
        data_loaders: Dictionary of data loaders ('train', 'val', 'test')
        components_to_ablate: List of components to ablate
        device: Device to run the model on
        output_dir: Directory to save results
        
    Returns:
        AblationStudy object with results
    """
    # Get config from model
    config = {
        'task_type': getattr(model, 'task_type', 'regression'),
        'modalities': getattr(model, 'modalities', ['smiles', 'ecfp', 'graph', 'mfbert']),
        'components': components_to_ablate
    }
    
    # Initialize ablation study
    ablation_study = AblationStudy(config, output_dir)
    
    # Run baseline (no ablation)
    logger.info("Running baseline model (no ablation)")
    
    # Set model to evaluation mode
    model.eval()
    
    # Get test loader
    test_loader = data_loaders['test']
    
    # Collect predictions and targets
    baseline_predictions = []
    baseline_targets = []
    baseline_modality_weights = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Forward pass with option to return modality weights
            if hasattr(model, 'return_modality_weights') and callable(getattr(model, 'return_modality_weights')):
                predictions, modality_weights = model.forward(batch, return_modality_weights=True)
                baseline_modality_weights.append(modality_weights)
            else:
                predictions = model.forward(batch)
            
            # Get targets
            targets = batch['targets']
            
            # Store predictions and targets
            baseline_predictions.append(predictions.cpu().numpy())
            baseline_targets.append(targets.cpu().numpy())
    
    # Concatenate predictions and targets
    baseline_predictions = np.concatenate(baseline_predictions)
    baseline_targets = np.concatenate(baseline_targets)
    
    # Process modality weights if available
    baseline_weights = None
    if baseline_modality_weights:
        # Average modality weights across batches
        baseline_weights = {}
        for modality in config['modalities']:
            baseline_weights[modality] = np.mean([weights[modality] for weights in baseline_modality_weights])
    
    # Add baseline results
    ablation_study.add_baseline_results(baseline_predictions, baseline_targets, modality_weights=baseline_weights)
    
    # Run ablations
    for component in components_to_ablate:
        logger.info(f"Running ablation for component: {component}")
        
        # Create a copy of the model
        ablated_model = type(model)(**model.get_config())
        ablated_model.load_state_dict(model.state_dict())
        ablated_model.to(device)
        
        # Ablate the component
        if hasattr(ablated_model, f'ablate_{component}') and callable(getattr(ablated_model, f'ablate_{component}')):
            getattr(ablated_model, f'ablate_{component}')()
        else:
            logger.warning(f"No ablation method found for component: {component}")
            continue
        
        # Set model to evaluation mode
        ablated_model.eval()
        
        # Collect predictions and targets
        ablated_predictions = []
        ablated_targets = []
        ablated_modality_weights = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Forward pass with option to return modality weights
                if hasattr(ablated_model, 'return_modality_weights') and callable(getattr(ablated_model, 'return_modality_weights')):
                    predictions, modality_weights = ablated_model.forward(batch, return_modality_weights=True)
                    ablated_modality_weights.append(modality_weights)
                else:
                    predictions = ablated_model.forward(batch)
                
                # Get targets
                targets = batch['targets']
                
                # Store predictions and targets
                ablated_predictions.append(predictions.cpu().numpy())
                ablated_targets.append(targets.cpu().numpy())
        
        # Concatenate predictions and targets
        ablated_predictions = np.concatenate(ablated_predictions)
        ablated_targets = np.concatenate(ablated_targets)
        
        # Process modality weights if available
        ablated_weights = None
        if ablated_modality_weights:
            # Average modality weights across batches
            ablated_weights = {}
            for modality in config['modalities']:
                ablated_weights[modality] = np.mean([weights[modality] for weights in ablated_modality_weights])
        
        # Add ablation results
        ablation_study.add_ablation_results(
            f"no_{component.lower()}",
            ablated_predictions,
            ablated_targets,
            modality_weights=ablated_weights
        )
    
    # Calculate impact and rank components
    ablation_study.calculate_ablation_impact()
    ablation_study.rank_components_by_importance()
    
    # Calculate confidence intervals
    ablation_study.calculate_confidence_intervals()
    
    # Save results
    ablation_study.save_results()
    
    # Generate plots
    ablation_study.plot_ablation_results()
    ablation_study.plot_component_importance()
    ablation_study.plot_correlation_matrix()
    ablation_study.plot_modality_weights()
    
    return ablation_study

def analyze_ablation_results(results_path, output_dir=None):
    """
    Analyze ablation study results from a saved JSON file
    
    This function loads previously saved ablation study results
    and generates visualizations and analysis.
    
    Args:
        results_path: Path to the JSON file with ablation results
        output_dir: Directory to save analysis results (default: same as results_path)
        
    Returns:
        Dictionary with analysis results
    """
    # Set output directory
    if output_dir is None:
        output_dir = Path(results_path).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Initialize analysis dictionary
    analysis = {
        'component_importance': {},
        'ablation_summary': {},
        'modality_weight_changes': {}
    }
    
    # Extract component importance if available
    if 'component_ranking' in results:
        analysis['component_importance'] = results['component_ranking']
    elif 'impact' in results:
        # Calculate component importance from impact
        component_importance = {}
        for component, impact_data in results['impact'].items():
            component_importance[component] = {
                'importance_score': abs(impact_data['relative_impact']),
                'raw_impact': impact_data['raw_impact'],
                'relative_impact': impact_data['relative_impact']
            }
        analysis['component_importance'] = component_importance
    
    # Create ablation summary
    if 'baseline' in results and 'ablations' in results:
        # Get primary metric
        task_type = results.get('config', {}).get('task_type', 'regression')
        if task_type == 'regression':
            primary_metric = 'r2'
        else:
            primary_metric = 'accuracy'
        
        # Get baseline metric value
        baseline_value = results['baseline']['metrics'][primary_metric]
        
        # Summarize ablations
        for ablation_name, ablation_data in results['ablations'].items():
            ablation_value = ablation_data['metrics'][primary_metric]
            
            # Calculate relative change
            if baseline_value != 0:
                relative_change = (ablation_value - baseline_value) / baseline_value
            else:
                relative_change = 0
            
            analysis['ablation_summary'][ablation_name] = {
                'metric': primary_metric,
                'baseline_value': baseline_value,
                'ablation_value': ablation_value,
                'absolute_change': ablation_value - baseline_value,
                'relative_change': relative_change
            }
    
    # Analyze modality weight changes if available
    if ('baseline' in results and 
        'modality_weights' in results['baseline'] and 
        results['baseline']['modality_weights'] is not None):
        
        baseline_weights = results['baseline']['modality_weights']
        
        for ablation_name, ablation_data in results['ablations'].items():
            if 'modality_weights' in ablation_data and ablation_data['modality_weights'] is not None:
                ablation_weights = ablation_data['modality_weights']
                
                # Calculate weight changes
                weight_changes = {}
                for modality, baseline_weight in baseline_weights.items():
                    if modality in ablation_weights:
                        ablation_weight = ablation_weights[modality]
                        
                        # Calculate absolute and relative change
                        absolute_change = ablation_weight - baseline_weight
                        if baseline_weight != 0:
                            relative_change = absolute_change / baseline_weight
                        else:
                            relative_change = 0
                        
                        weight_changes[modality] = {
                            'baseline_weight': baseline_weight,
                            'ablation_weight': ablation_weight,
                            'absolute_change': absolute_change,
                            'relative_change': relative_change
                        }
                
                analysis['modality_weight_changes'][ablation_name] = weight_changes
    
    # Generate visualizations
    # Create AblationStudy object to reuse visualization methods
    config = results.get('config', {})
    ablation_study = AblationStudy(config, output_dir)
    ablation_study.results = results
    
    # Generate plots
    try:
        ablation_study.plot_ablation_results(save_fig=True)
        ablation_study.plot_component_importance(save_fig=True)
        ablation_study.plot_correlation_matrix(save_fig=True)
        ablation_study.plot_modality_weights(save_fig=True)
    except Exception as e:
        logger.error(f"Error generating plots: {str(e)}")
    
    # Save analysis results
    analysis_path = output_dir / 'ablation_analysis.json'
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    logger.info(f"Ablation analysis saved to {analysis_path}")
    
    return analysis