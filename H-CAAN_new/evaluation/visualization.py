import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import os
import json
import torch
import networkx as nx
import logging
from pathlib import Path
import io
import base64

# Set up logger
logger = logging.getLogger(__name__)
def generate_molecule_visualizations(molecules, property_values=None, attention_weights=None, 
                                     n_molecules=5, output_dir='results/figures', 
                                     save_fig=True, figsize=(10, 10)):
    """
    Generate visualizations for molecules with optional property values and attention weights
    
    Args:
        molecules: List of RDKit molecules or SMILES strings
        property_values: Optional dictionary with molecule index as key and property value as value
        attention_weights: Optional dictionary with molecule index as key and atom attention weights as value
        n_molecules: Number of molecules to visualize (default: 5)
        output_dir: Directory to save figures
        save_fig: Whether to save the figure
        figsize: Figure size
        
    Returns:
        Dictionary of molecule visualizations
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw, rdMolDraw2D
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    import io
    import logging
    
    logger = logging.getLogger(__name__)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Limit number of molecules
    if n_molecules > 0:
        molecules = molecules[:n_molecules]
    
    # Convert molecules to RDKit molecules if they are SMILES strings
    rdkit_mols = []
    for i, mol in enumerate(molecules):
        if isinstance(mol, str):
            rdmol = Chem.MolFromSmiles(mol)
            if rdmol is not None:
                rdkit_mols.append(rdmol)
            else:
                logger.warning(f"Could not parse SMILES at index {i}: {mol}")
        else:
            rdkit_mols.append(mol)
    
    # Generate visualizations
    visualizations = {}
    
    for i, mol in enumerate(rdkit_mols):
        # Compute 2D coordinates
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        mol = Chem.RemoveHs(mol)
        
        # Get molecule title
        if property_values is not None and i in property_values:
            title = f"Molecule {i+1}: Property = {property_values[i]:.3f}"
        else:
            title = f"Molecule {i+1}"
        
        # If attention weights are provided, use them to color atoms
        if attention_weights is not None and i in attention_weights:
            # Normalize weights to [0, 1]
            weights = attention_weights[i]
            
            if weights:
                min_weight = min(weights.values())
                max_weight = max(weights.values())
                weight_range = max_weight - min_weight
                
                normalized_weights = {
                    idx: (weight - min_weight) / weight_range if weight_range > 0 else 0.5
                    for idx, weight in weights.items()
                }
            else:
                normalized_weights = {}
            
            # Convert to atom highlights and colors
            atom_highlights = []
            atom_colors = {}
            
            for atom_idx, weight in normalized_weights.items():
                if atom_idx < mol.GetNumAtoms():
                    atom_highlights.append(atom_idx)
                    # Convert weight to color (blue -> red gradient)
                    r = int(255 * weight)
                    b = int(255 * (1 - weight))
                    atom_colors[atom_idx] = (r, 0, b)
            
            # Draw molecule with highlights
            drawer = rdMolDraw2D.MolDraw2DCairo(int(figsize[0] * 100), int(figsize[1] * 100))
            drawer.DrawMolecule(
                mol,
                highlightAtoms=atom_highlights,
                highlightAtomColors=atom_colors
            )
            drawer.FinishDrawing()
            
            png_data = drawer.GetDrawingText()
            
            # Save figure
            if save_fig:
                with open(output_path / f"molecule_{i+1}_attention.png", 'wb') as f:
                    f.write(png_data)
            
            # Convert to matplotlib figure
            import io
            from PIL import Image
            
            img = Image.open(io.BytesIO(png_data))
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(title, fontsize=14)
            
            plt.tight_layout()
            
            visualizations[f"molecule_{i+1}"] = fig
        else:
            # Just draw the molecule without highlights
            img = Draw.MolToImage(mol, size=(int(figsize[0] * 100), int(figsize[1] * 100)))
            
            # Save figure
            if save_fig:
                img.save(output_path / f"molecule_{i+1}.png")
            
            # Convert to matplotlib figure
            fig, ax = plt.subplots(figsize=figsize)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(title, fontsize=14)
            
            plt.tight_layout()
            
            visualizations[f"molecule_{i+1}"] = fig
    
    return visualizations
def plot_training_curves(history, output_dir='results/figures', save_fig=True, figsize=(12, 8)):
    """
    Plot training and validation curves
    
    Args:
        history: Dictionary containing training history
        output_dir: Directory to save figures
        save_fig: Whether to save the figure
        figsize: Figure size
        
    Returns:
        Dictionary of figures
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    figures = {}
    
    # Plot loss curves
    if 'train_loss' in history and 'val_loss' in history:
        fig, ax = plt.subplots(figsize=figsize)
        
        epochs = range(1, len(history['train_loss']) + 1)
        ax.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
        ax.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        
        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel('Loss', fontsize=14)
        ax.set_title('Training and Validation Loss', fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(output_path / 'loss_curves.png', dpi=300, bbox_inches='tight')
        
        figures['loss_curves'] = fig
    
    # Plot learning rate curve
    if 'learning_rates' in history:
        fig, ax = plt.subplots(figsize=figsize)
        
        epochs = range(1, len(history['learning_rates']) + 1)
        ax.plot(epochs, history['learning_rates'], 'g-')
        
        ax.set_xlabel('Epoch', fontsize=14)
        ax.set_ylabel('Learning Rate', fontsize=14)
        ax.set_title('Learning Rate Schedule', fontsize=16)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(output_path / 'learning_rate.png', dpi=300, bbox_inches='tight')
        
        figures['learning_rate'] = fig
    
    # Plot metrics curves
    if 'train_metrics' in history and 'val_metrics' in history:
        train_metrics = history['train_metrics']
        val_metrics = history['val_metrics']
        
        # Get common metrics
        if train_metrics and val_metrics:
            common_metrics = set(train_metrics[0].keys()).intersection(set(val_metrics[0].keys()))
            
            for metric in common_metrics:
                if metric in ['confusion_matrix', 'roc_curve', 'pr_curve']:
                    continue
                
                fig, ax = plt.subplots(figsize=figsize)
                
                epochs = range(1, len(train_metrics) + 1)
                train_values = [metrics[metric] for metrics in train_metrics]
                val_values = [metrics[metric] for metrics in val_metrics]
                
                ax.plot(epochs, train_values, 'b-', label=f'Training {metric}')
                ax.plot(epochs, val_values, 'r-', label=f'Validation {metric}')
                
                ax.set_xlabel('Epoch', fontsize=14)
                ax.set_ylabel(metric, fontsize=14)
                ax.set_title(f'Training and Validation {metric}', fontsize=16)
                ax.legend(fontsize=12)
                ax.grid(True, linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                
                if save_fig:
                    plt.savefig(output_path / f'{metric}_curves.png', dpi=300, bbox_inches='tight')
                
                figures[f'{metric}_curves'] = fig
    
    return figures


def plot_regression_results(y_true, y_pred, output_dir='results/figures', save_fig=True, figsize=(12, 8)):
    """
    Plot regression results
    
    Args:
        y_true: True values
        y_pred: Predicted values
        output_dir: Directory to save figures
        save_fig: Whether to save the figure
        figsize: Figure size
        
    Returns:
        Dictionary of figures
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    figures = {}
    
    # Calculate metrics
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # Create actual vs predicted plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Scatter plot
    scatter = ax.scatter(y_true, y_pred, alpha=0.6, s=50)
    
    # Perfect prediction line
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    margin = (max_val - min_val) * 0.05
    ax.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin], 'r--')
    
    # Set limits
    ax.set_xlim(min_val - margin, max_val + margin)
    ax.set_ylim(min_val - margin, max_val + margin)
    
    # Set labels and title
    ax.set_xlabel('True Values', fontsize=14)
    ax.set_ylabel('Predicted Values', fontsize=14)
    ax.set_title('Actual vs Predicted Values', fontsize=16)
    
    # Add metrics as text
    metrics_text = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}'
    ax.text(
        0.05, 0.95, metrics_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(output_path / 'actual_vs_predicted.png', dpi=300, bbox_inches='tight')
    
    figures['actual_vs_predicted'] = fig
    
    # Create residuals plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate residuals
    residuals = y_true - y_pred
    
    # Scatter plot
    scatter = ax.scatter(y_pred, residuals, alpha=0.6, s=50)
    
    # Zero line
    ax.axhline(y=0, color='r', linestyle='--')
    
    # Set labels and title
    ax.set_xlabel('Predicted Values', fontsize=14)
    ax.set_ylabel('Residuals', fontsize=14)
    ax.set_title('Residuals vs Predicted Values', fontsize=16)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(output_path / 'residuals.png', dpi=300, bbox_inches='tight')
    
    figures['residuals'] = fig
    
    # Create residuals distribution plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Histogram with KDE
    sns.histplot(residuals, kde=True, ax=ax)
    
    # Zero line
    ax.axvline(x=0, color='r', linestyle='--')
    
    # Set labels and title
    ax.set_xlabel('Residuals', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_title('Residuals Distribution', fontsize=16)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(output_path / 'residuals_distribution.png', dpi=300, bbox_inches='tight')
    
    figures['residuals_distribution'] = fig
    
    # Create error by feature plot (interactive)
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'residuals': residuals,
        'abs_error': np.abs(residuals)
    })
    
    fig = px.scatter(
        df, x='y_true', y='abs_error',
        labels={'y_true': 'True Values', 'abs_error': 'Absolute Error'},
        title='Absolute Error vs True Values',
    )
    
    fig.update_layout(
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14),
        title_font=dict(size=16)
    )
    
    if save_fig:
        fig.write_html(str(output_path / 'error_by_feature.html'))
    
    figures['error_by_feature'] = fig
    
    return figures


def plot_classification_results(y_true, y_pred, y_prob=None, output_dir='results/figures', save_fig=True, figsize=(12, 8)):
    """
    Plot classification results
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for binary classification)
        output_dir: Directory to save figures
        save_fig: Whether to save the figure
        figsize: Figure size
        
    Returns:
        Dictionary of figures
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    figures = {}
    
    # Ensure arrays are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Check if binary or multiclass
    is_binary = len(np.unique(y_true)) == 2
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    classes = np.unique(np.concatenate((y_true, y_pred)))
    n_classes = len(classes)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title='Confusion Matrix',
        ylabel='True Label',
        xlabel='Predicted Label'
    )
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )
    
    fig.tight_layout()
    
    if save_fig:
        plt.savefig(output_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    
    figures['confusion_matrix'] = fig
    
    # Create ROC curve for binary classification
    if is_binary and y_prob is not None:
        fig, ax = plt.subplots(figsize=figsize)
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=14)
        ax.set_ylabel('True Positive Rate', fontsize=14)
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
        ax.legend(loc="lower right", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(output_path / 'roc_curve.png', dpi=300, bbox_inches='tight')
        
        figures['roc_curve'] = fig
        
        # Create Precision-Recall curve
        fig, ax = plt.subplots(figsize=figsize)
        
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        
        ax.plot(recall, precision, lw=2, label=f'PR curve (area = {pr_auc:.2f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=14)
        ax.set_ylabel('Precision', fontsize=14)
        ax.set_title('Precision-Recall Curve', fontsize=16)
        ax.legend(loc="lower left", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(output_path / 'pr_curve.png', dpi=300, bbox_inches='tight')
        
        figures['pr_curve'] = fig
    
    return figures


def plot_molecular_visualization(smiles, attention_weights=None, output_dir='results/figures', save_fig=True, figsize=(10, 10)):
    """
    Plot molecular visualization with optional attention weights
    
    Args:
        smiles: SMILES string of the molecule
        attention_weights: Optional attention weights for atoms (dict of atom_idx -> weight)
        output_dir: Directory to save figures
        save_fig: Whether to save the figure
        figsize: Figure size
        
    Returns:
        Molecular visualization figure
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.error(f"Could not parse SMILES: {smiles}")
        return None
    
    # Compute 2D coordinates
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    mol = Chem.RemoveHs(mol)
    
    # If attention weights are provided, use them to color atoms
    if attention_weights is not None:
        # Normalize weights to [0, 1]
        if attention_weights:
            min_weight = min(attention_weights.values())
            max_weight = max(attention_weights.values())
            weight_range = max_weight - min_weight
            
            normalized_weights = {
                idx: (weight - min_weight) / weight_range if weight_range > 0 else 0.5
                for idx, weight in attention_weights.items()
            }
        else:
            normalized_weights = {}
        
        # Convert to atom highlights and colors
        atom_highlights = []
        atom_colors = {}
        
        for atom_idx, weight in normalized_weights.items():
            if atom_idx < mol.GetNumAtoms():
                atom_highlights.append(atom_idx)
                # Convert weight to color (blue -> red gradient)
                r = int(255 * weight)
                b = int(255 * (1 - weight))
                atom_colors[atom_idx] = (r, 0, b)
        
        # Draw molecule with highlights
        drawer = rdMolDraw2D.MolDraw2DCairo(int(figsize[0] * 100), int(figsize[1] * 100))
        drawer.DrawMolecule(
            mol,
            highlightAtoms=atom_highlights,
            highlightAtomColors=atom_colors
        )
        drawer.FinishDrawing()
        
        png_data = drawer.GetDrawingText()
        
        # Save figure
        if save_fig:
            with open(output_path / f"molecule_attention.png", 'wb') as f:
                f.write(png_data)
        
        # Convert to matplotlib figure
        import io
        from PIL import Image
        
        img = Image.open(io.BytesIO(png_data))
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img)
        ax.axis('off')
        
        plt.tight_layout()
        
        return fig
    else:
        # Just draw the molecule without highlights
        img = Draw.MolToImage(mol, size=(int(figsize[0] * 100), int(figsize[1] * 100)))
        
        # Save figure
        if save_fig:
            img.save(output_path / "molecule.png")
        
        # Convert to matplotlib figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(img)
        ax.axis('off')
        
        plt.tight_layout()
        
        return fig


def plot_ablation_results(ablation_results, metric_name='R²', output_dir='results/figures', save_fig=True, figsize=(14, 8)):
    """
    Plot ablation study results
    
    Args:
        ablation_results: Dictionary with configuration as key and result dictionary as value
        metric_name: Name of the metric to plot
        output_dir: Directory to save figures
        save_fig: Whether to save the figure
        figsize: Figure size
        
    Returns:
        Ablation results figure
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract configurations and corresponding metric values
    configs = list(ablation_results.keys())
    
    # Check if the metric exists in results
    if metric_name.lower() not in ablation_results[configs[0]]:
        available_metrics = list(ablation_results[configs[0]].keys())
        logger.warning(f"Metric '{metric_name}' not found. Available metrics: {available_metrics}")
        if available_metrics:
            metric_name = available_metrics[0]
        else:
            return None
    
    metric_values = [result[metric_name.lower()] for result in ablation_results.values()]
    
    # Sort results by metric value (ascending or descending based on metric type)
    # For metrics like R², higher is better, for metrics like RMSE, lower is better
    higher_is_better = metric_name.lower() in ['r2', 'r²', 'accuracy', 'auc', 'f1', 'precision', 'recall']
    
    sorted_indices = np.argsort(metric_values)
    if higher_is_better:
        sorted_indices = sorted_indices[::-1]  # Reverse for higher is better
    
    sorted_configs = [configs[i] for i in sorted_indices]
    sorted_values = [metric_values[i] for i in sorted_indices]
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create colormap where best model has a distinct color
    colors = ['blue'] * len(sorted_configs)
    colors[0] = 'gold'  # Best model
    
    y_pos = np.arange(len(sorted_configs))
    bars = ax.barh(y_pos, sorted_values, align='center', color=colors)
    
    # Add value labels to the bars
    for i, v in enumerate(sorted_values):
        ax.text(
            v + (0.01 if higher_is_better else -0.05),
            i,
            f"{v:.4f}",
            va='center',
            fontsize=10
        )
    
    # Set labels and title
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_configs)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel(metric_name, fontsize=14)
    ax.set_title(f'Ablation Study Results ({metric_name})', fontsize=16)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7, axis='x')
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(output_path / f'ablation_results_{metric_name}.png', dpi=300, bbox_inches='tight')
    
    return fig


def plot_modal_importance(modal_weights, output_dir='results/figures', save_fig=True, figsize=(10, 6)):
    """
    Plot modal importance weights
    
    Args:
        modal_weights: Dictionary with modality as key and weight as value
        output_dir: Directory to save figures
        save_fig: Whether to save the figure
        figsize: Figure size
        
    Returns:
        Modal importance figure
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract modalities and weights
    modalities = list(modal_weights.keys())
    weights = list(modal_weights.values())
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.bar(modalities, weights, color=['blue', 'orange', 'green', 'red'])
    
    # Add value labels
    for i, v in enumerate(weights):
        ax.text(
            i,
            v + 0.01,
            f"{v:.2f}",
            ha='center',
            fontsize=12
        )
    
    # Set labels and title
    ax.set_ylabel('Importance Weight', fontsize=14)
    ax.set_title('Modal Importance Weights', fontsize=16)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(output_path / 'modal_importance.png', dpi=300, bbox_inches='tight')
    
    return fig


def plot_attention_heatmap(attention_matrix, row_labels, col_labels, output_dir='results/figures', 
                          save_fig=True, figsize=(12, 10), title="Attention Weights"):
    """
    Plot attention weights as a heatmap
    
    Args:
        attention_matrix: 2D numpy array of attention weights
        row_labels: Labels for rows
        col_labels: Labels for columns
        output_dir: Directory to save figures
        save_fig: Whether to save the figure
        figsize: Figure size
        title: Title for the heatmap
        
    Returns:
        Attention heatmap figure
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(attention_matrix, cmap="YlGnBu")
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Attention Weight", rotation=-90, va="bottom", fontsize=12)
    
    # Show all ticks and label them
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=10)
    ax.set_yticklabels(row_labels, fontsize=10)
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            text = ax.text(j, i, f"{attention_matrix[i, j]:.2f}",
                          ha="center", va="center", 
                          color="black" if attention_matrix[i, j] < 0.5 else "white")
    
    # Set title and labels
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Target Modality", fontsize=14)
    ax.set_ylabel("Source Modality", fontsize=14)
    
    fig.tight_layout()
    
    if save_fig:
        plt.savefig(output_path / 'attention_heatmap.png', dpi=300, bbox_inches='tight')
    
    return fig


def plot_embedding_space(embeddings, labels=None, method='PCA', n_components=2, 
                         output_dir='results/figures', save_fig=True, figsize=(12, 10)):
    """
    Plot embedding space using dimensionality reduction
    
    Args:
        embeddings: 2D numpy array of embeddings (n_samples, n_features)
        labels: Optional labels or values for coloring points
        method: Dimensionality reduction method ('PCA', 'TSNE', or 'UMAP')
        n_components: Number of components (2 or 3)
        output_dir: Directory to save figures
        save_fig: Whether to save the figure
        figsize: Figure size
        
    Returns:
        Embedding space figure
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Apply dimensionality reduction
    if method.upper() == 'PCA':
        reducer = PCA(n_components=n_components, random_state=42)
    elif method.upper() == 'TSNE':
        reducer = TSNE(n_components=n_components, random_state=42)
    elif method.upper() == 'UMAP':
        reducer = umap.UMAP(n_components=n_components, random_state=42)
    else:
        logger.error(f"Unknown dimensionality reduction method: {method}")
        return None
    
    reduced_data = reducer.fit_transform(embeddings)
    
    # Create dataframe for plotting
    df = pd.DataFrame(reduced_data, columns=[f'Component {i+1}' for i in range(n_components)])
    
    if labels is not None:
        df['Label'] = labels
    
    # Create plot
    if n_components == 2:
        if labels is not None:
            fig = px.scatter(
                df, x='Component 1', y='Component 2', color='Label',
                labels={'Component 1': f'{method} Component 1', 'Component 2': f'{method} Component 2'},
                title=f'{method} Visualization of Embedding Space',
            )
        else:
            fig = px.scatter(
                df, x='Component 1', y='Component 2',
                labels={'Component 1': f'{method} Component 1', 'Component 2': f'{method} Component 2'},
                title=f'{method} Visualization of Embedding Space',
            )
    elif n_components == 3:
        if labels is not None:
            fig = px.scatter_3d(
                df, x='Component 1', y='Component 2', z='Component 3', color='Label',
                labels={
                    'Component 1': f'{method} Component 1',
                    'Component 2': f'{method} Component 2',
                    'Component 3': f'{method} Component 3'
                },
                title=f'{method} Visualization of Embedding Space',
            )
        else:
            fig = px.scatter_3d(
                df, x='Component 1', y='Component 2', z='Component 3',
                labels={
                    'Component 1': f'{method} Component 1',
                    'Component 2': f'{method} Component 2',
                    'Component 3': f'{method} Component 3'
                },
                title=f'{method} Visualization of Embedding Space',
            )
    else:
        logger.error(f"Invalid number of components: {n_components}. Must be 2 or 3.")
        return None
    
    fig.update_layout(
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14),
        title_font=dict(size=16)
    )
    
    # Save figure
    if save_fig:
        html_path = output_path / f'{method.lower()}_embedding_space.html'
        png_path = output_path / f'{method.lower()}_embedding_space.png'
        
        fig.write_html(str(html_path))
        fig.write_image(str(png_path))
    
    return fig


def compare_model_performance(models_results, metric_name='R²', output_dir='results/figures', 
                             save_fig=True, figsize=(12, 8)):
    """
    Compare performance of multiple models
    
    Args:
        models_results: Dictionary with model name as key and result dictionary as value
        metric_name: Name of the metric to plot
        output_dir: Directory to save figures
        save_fig: Whether to save the figure
        figsize: Figure size
        
    Returns:
        Model comparison figure
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract model names and corresponding metric values
    model_names = list(models_results.keys())
    
    # Check if the metric exists in results
    if metric_name.lower() not in models_results[model_names[0]]:
        available_metrics = list(models_results[model_names[0]].keys())
        logger.warning(f"Metric '{metric_name}' not found. Available metrics: {available_metrics}")
        if available_metrics:
            metric_name = available_metrics[0]
        else:
            return None
    
    metric_values = [result[metric_name.lower()] for result in models_results.values()]
    
    # Sort results by metric value (ascending or descending based on metric type)
    # For metrics like R², higher is better, for metrics like RMSE, lower is better
    higher_is_better = metric_name.lower() in ['r2', 'r²', 'accuracy', 'auc', 'f1', 'precision', 'recall']
    
    sorted_indices = np.argsort(metric_values)
    if higher_is_better:
        sorted_indices = sorted_indices[::-1]  # Reverse for higher is better
    
    sorted_names = [model_names[i] for i in sorted_indices]
    sorted_values = [metric_values[i] for i in sorted_indices]
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create colormap where best model has a distinct color
    colors = ['blue'] * len(sorted_names)
    colors[0] = 'gold'  # Best model
    
    y_pos = np.arange(len(sorted_names))
    bars = ax.barh(y_pos, sorted_values, align='center', color=colors)
    
    # Add value labels to the bars
    for i, v in enumerate(sorted_values):
        ax.text(
            v + (0.01 if higher_is_better else -0.05),
            i,
            f"{v:.4f}",
            va='center',
            fontsize=10
        )
    
    # Set labels and title
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_names)
    ax.invert_yaxis()  # Labels read top-to-bottom
    ax.set_xlabel(metric_name, fontsize=14)
    ax.set_title(f'Model Performance Comparison ({metric_name})', fontsize=16)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7, axis='x')
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(output_path / f'model_comparison_{metric_name}.png', dpi=300, bbox_inches='tight')
    
    return fig


def plot_chemical_space(molecules, properties=None, fingerprints=None, method='UMAP', 
                       output_dir='results/figures', save_fig=True, figsize=(12, 10)):
    """
    Plot chemical space using dimensionality reduction on molecular fingerprints
    
    Args:
        molecules: List of RDKit molecules or SMILES strings
        properties: Optional dictionary with molecule index as key and property value as value
        fingerprints: Optional precomputed fingerprints (if None, will compute ECFP4)
        method: Dimensionality reduction method ('PCA', 'TSNE', or 'UMAP')
        output_dir: Directory to save figures
        save_fig: Whether to save the figure
        figsize: Figure size
        
    Returns:
        Chemical space figure
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Convert molecules to RDKit molecules if they are SMILES strings
    rdkit_mols = []
    for i, mol in enumerate(molecules):
        if isinstance(mol, str):
            rdmol = Chem.MolFromSmiles(mol)
            if rdmol is not None:
                rdkit_mols.append(rdmol)
            else:
                logger.warning(f"Could not parse SMILES at index {i}: {mol}")
        else:
            rdkit_mols.append(mol)
    
    # Compute fingerprints if not provided
    if fingerprints is None:
        fingerprints = []
        for mol in rdkit_mols:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            fingerprints.append(np.array(fp))
        
        fingerprints = np.array(fingerprints)
    
    # Apply dimensionality reduction
    if method.upper() == 'PCA':
        reducer = PCA(n_components=2, random_state=42)
    elif method.upper() == 'TSNE':
        reducer = TSNE(n_components=2, random_state=42)
    elif method.upper() == 'UMAP':
        reducer = umap.UMAP(n_components=2, random_state=42)
    else:
        logger.error(f"Unknown dimensionality reduction method: {method}")
        return None
    
    reduced_data = reducer.fit_transform(fingerprints)
    
    # Create dataframe for plotting
    df = pd.DataFrame(reduced_data, columns=['Component 1', 'Component 2'])
    
    # Add properties if provided
    if properties is not None:
        property_values = [properties.get(i, np.nan) for i in range(len(rdkit_mols))]
        df['Property'] = property_values
    
    # Create plot
    if properties is not None:
        fig = px.scatter(
            df, x='Component 1', y='Component 2', color='Property',
            labels={'Component 1': f'{method} Component 1', 'Component 2': f'{method} Component 2'},
            title=f'Chemical Space Visualization ({method})',
        )
    else:
        fig = px.scatter(
            df, x='Component 1', y='Component 2',
            labels={'Component 1': f'{method} Component 1', 'Component 2': f'{method} Component 2'},
            title=f'Chemical Space Visualization ({method})',
        )
    
    fig.update_layout(
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14),
        title_font=dict(size=16)
    )
    
    # Save figure
    if save_fig:
        html_path = output_path / f'chemical_space_{method.lower()}.html'
        png_path = output_path / f'chemical_space_{method.lower()}.png'
        
        fig.write_html(str(html_path))
        fig.write_image(str(png_path))
    
    return fig


def visualize_model_architecture(model_config, output_dir='results/figures', save_fig=True, figsize=(15, 10)):
    """
    Visualize the H-CAAN model architecture
    
    Args:
        model_config: Dictionary with model configuration
        output_dir: Directory to save figures
        save_fig: Whether to save the figure
        figsize: Figure size
        
    Returns:
        Model architecture figure
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes for different components
    # Input nodes
    G.add_node("SMILES", pos=(0, 0), node_type="input")
    G.add_node("ECFP", pos=(0, 2), node_type="input")
    G.add_node("Graph", pos=(0, 4), node_type="input")
    G.add_node("MFBERT", pos=(0, 6), node_type="input")
    
    # Encoder nodes
    G.add_node("TransformerEncoder", pos=(3, 0), node_type="encoder")
    G.add_node("BiGRU", pos=(3, 2), node_type="encoder")
    G.add_node("GCN", pos=(3, 4), node_type="encoder")
    G.add_node("MFBERTEncoder", pos=(3, 6), node_type="encoder")
    
    # Fusion nodes
    G.add_node("GCAU_1", pos=(6, 1), node_type="fusion")
    G.add_node("GCAU_2", pos=(6, 5), node_type="fusion")
    G.add_node("HierarchicalFusion", pos=(9, 3), node_type="fusion")
    
    # Modal importance node
    G.add_node("ModalImportance", pos=(12, 3), node_type="importance")
    
    # Output node
    G.add_node("Output", pos=(15, 3), node_type="output")
    
    # Add edges
    # Input to encoder
    G.add_edge("SMILES", "TransformerEncoder")
    G.add_edge("ECFP", "BiGRU")
    G.add_edge("Graph", "GCN")
    G.add_edge("MFBERT", "MFBERTEncoder")
    
    # Encoder to fusion
    G.add_edge("TransformerEncoder", "GCAU_1")
    G.add_edge("BiGRU", "GCAU_1")
    G.add_edge("GCN", "GCAU_2")
    G.add_edge("MFBERTEncoder", "GCAU_2")
    
    # Fusion to hierarchical fusion
    G.add_edge("GCAU_1", "HierarchicalFusion")
    G.add_edge("GCAU_2", "HierarchicalFusion")
    
    # Hierarchical fusion to modal importance
    G.add_edge("HierarchicalFusion", "ModalImportance")
    
    # Modal importance to output
    G.add_edge("ModalImportance", "Output")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Get node types
    node_types = nx.get_node_attributes(G, 'node_type')
    
    # Define colors for different node types
    color_map = {
        "input": "lightblue",
        "encoder": "lightgreen",
        "fusion": "salmon",
        "importance": "purple",
        "output": "gold"
    }
    
    # Draw nodes
    for node_type, color in color_map.items():
        nodes = [node for node, type in node_types.items() if type == node_type]
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=nodes,
            node_color=color,
            node_size=2000,
            alpha=0.8,
            ax=ax
        )
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        width=2,
        alpha=0.5,
        edge_color='gray',
        arrows=True,
        arrowsize=20,
        ax=ax
    )
    
    # Draw labels
    nx.draw_networkx_labels(
        G, pos,
        font_size=10,
        font_weight='bold',
        ax=ax
    )
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=15, label=node_type.capitalize())
        for node_type, color in color_map.items()
    ]
    
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5)
    
    # Set title
    ax.set_title("H-CAAN Model Architecture", fontsize=16)
    
    # Remove axis
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_fig:
        plt.savefig(output_path / 'model_architecture.png', dpi=300, bbox_inches='tight')
    
    return fig


def plot_error_analysis_by_complexity(y_true, y_pred, complexity_scores, output_dir='results/figures', 
                                     save_fig=True, figsize=(12, 8)):
    """
    Plot error analysis by molecular complexity
    
    Args:
        y_true: True values
        y_pred: Predicted values
        complexity_scores: Dictionary with molecule index as key and complexity score as value
        output_dir: Directory to save figures
        save_fig: Whether to save the figure
        figsize: Figure size
        
    Returns:
        Error analysis figure
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Calculate absolute errors
    abs_errors = np.abs(np.array(y_true) - np.array(y_pred))
    
    # Create dataframe with errors and complexity scores
    df = pd.DataFrame({
        'true': y_true,
        'pred': y_pred,
        'abs_error': abs_errors,
        'complexity': [complexity_scores.get(i, np.nan) for i in range(len(y_true))]
    })
    
    # Remove rows with missing complexity scores
    df = df.dropna(subset=['complexity'])
    
    # Create scatter plot of errors vs complexity
    fig = px.scatter(
        df, x='complexity', y='abs_error',
        labels={'complexity': 'Molecular Complexity', 'abs_error': 'Absolute Error'},
        title='Error Analysis by Molecular Complexity',
        trendline='lowess'
    )
    
    fig.update_layout(
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14),
        title_font=dict(size=16)
    )
    
    # Save figure
    if save_fig:
        html_path = output_path / 'error_by_complexity.html'
        png_path = output_path / 'error_by_complexity.png'
        
        fig.write_html(str(html_path))
        fig.write_image(str(png_path))
    
    # Create box plot of errors by complexity bins
    # Bin complexity scores
    n_bins = 5
    df['complexity_bin'] = pd.qcut(df['complexity'], n_bins, labels=False)
    
    # Calculate bin edges for labels
    bin_edges = pd.qcut(df['complexity'], n_bins).cat.categories
    bin_labels = [f"{bin_edges[i].left:.2f}-{bin_edges[i].right:.2f}" for i in range(n_bins)]
    
    # Group by bin and calculate statistics
    bin_stats = df.groupby('complexity_bin')['abs_error'].agg(['mean', 'median', 'std', 'count']).reset_index()
    bin_stats['bin_label'] = bin_labels
    
    # Create box plot
    fig_box = px.box(
        df, x='complexity_bin', y='abs_error',
        labels={'complexity_bin': 'Complexity Bin', 'abs_error': 'Absolute Error'},
        title='Error Distribution by Molecular Complexity',
        category_orders={'complexity_bin': range(n_bins)}
    )
    
    # Update x-axis labels
    fig_box.update_xaxes(
        tickvals=list(range(n_bins)),
        ticktext=bin_labels
    )
    
    fig_box.update_layout(
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14),
        title_font=dict(size=16)
    )
    
    # Save figure
    if save_fig:
        html_path = output_path / 'error_distribution_by_complexity.html'
        png_path = output_path / 'error_distribution_by_complexity.png'
        
        fig_box.write_html(str(html_path))
        fig_box.write_image(str(png_path))
    
    return fig, fig_box


def visualize_cross_modal_attention(attention_weights, modalities=["SMILES", "ECFP", "Graph", "MFBERT"], 
                                   output_dir='results/figures', save_fig=True, figsize=(10, 8)):
    """
    Visualize cross-modal attention weights
    
    Args:
        attention_weights: 2D numpy array of attention weights (n_modalities, n_modalities)
        modalities: List of modality names
        output_dir: Directory to save figures
        save_fig: Whether to save the figure
        figsize: Figure size
        
    Returns:
        Cross-modal attention figure
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=attention_weights,
        x=modalities,
        y=modalities,
        colorscale='Viridis',
        text=[[f"{val:.2f}" for val in row] for row in attention_weights],
        texttemplate="%{text}",
        textfont={"size": 12},
    ))
    
    fig.update_layout(
        title="Cross-Modal Attention Weights",
        xaxis_title="Target Modality",
        yaxis_title="Source Modality",
        xaxis_title_font=dict(size=14),
        yaxis_title_font=dict(size=14),
        title_font=dict(size=16),
        width=figsize[0] * 100,
        height=figsize[1] * 100
    )
    
    # Save figure
    if save_fig:
        html_path = output_path / 'cross_modal_attention.html'
        png_path = output_path / 'cross_modal_attention.png'
        
        fig.write_html(str(html_path))
        fig.write_image(str(png_path))
    
    return fig


def create_interactive_dashboard(results, model_config, output_dir='results/dashboard'):
    """
    Create an interactive dashboard for model results
    
    Args:
        results: Dictionary with model results
        model_config: Dictionary with model configuration
        output_dir: Directory to save dashboard
        
    Returns:
        Dashboard HTML file path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create dashboard HTML
    dashboard_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>H-CAAN Results Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
        <script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
        <script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            .header {
                background-color: #4527a0;
                color: white;
                padding: 20px;
                text-align: center;
                margin-bottom: 20px;
            }
            .metric-card {
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                padding: 20px;
                margin-bottom: 20px;
                text-align: center;
            }
            .metric-value {
                font-size: 2em;
                font-weight: bold;
                color: #4527a0;
            }
            .metric-label {
                color: #666;
                margin-top: 5px;
            }
            .chart-container {
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
                padding: 20px;
                margin-bottom: 20px;
            }
            .chart-title {
                font-size: 1.2em;
                font-weight: bold;
                margin-bottom: 15px;
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 20px;
            }
            .full-width {
                grid-column: 1 / -1;
            }
            .half-width {
                grid-column: span 2;
            }
            @media (max-width: 768px) {
                .grid {
                    grid-template-columns: 1fr;
                }
                .half-width {
                    grid-column: 1;
                }
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>H-CAAN Results Dashboard</h1>
            <p>Hierarchical Cross-modal Adaptive Attention Network for Molecular Property Prediction</p>
        </div>
        
        <div class="container">
            <h2>Performance Metrics</h2>
            <div class="grid">
                <div class="metric-card">
                    <div class="metric-value">{{r2_value}}</div>
                    <div class="metric-label">R²</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{rmse_value}}</div>
                    <div class="metric-label">RMSE</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{mae_value}}</div>
                    <div class="metric-label">MAE</div>
                </div>
            </div>
            
            <h2>Results Visualization</h2>
            <div class="grid">
                <div class="chart-container full-width">
                    <div class="chart-title">Training and Validation Loss</div>
                    <div id="loss-chart"></div>
                </div>
                
                <div class="chart-container half-width">
                    <div class="chart-title">Actual vs Predicted Values</div>
                    <div id="actual-vs-predicted"></div>
                </div>
                
                <div class="chart-container">
                    <div class="chart-title">Modal Importance</div>
                    <div id="modal-importance"></div>
                </div>
                
                <div class="chart-container full-width">
                    <div class="chart-title">Ablation Study Results</div>
                    <div id="ablation-results"></div>
                </div>
                
                <div class="chart-container full-width">
                    <div class="chart-title">Cross-Modal Attention Visualization</div>
                    <div id="cross-modal-attention"></div>
                </div>
                
                <div class="chart-container full-width">
                    <div class="chart-title">Error Analysis by Molecular Complexity</div>
                    <div id="error-analysis"></div>
                </div>
            </div>
            
            <h2>Model Configuration</h2>
            <div class="chart-container">
                <pre id="model-config"></pre>
            </div>
        </div>
        
        <script>
            // Load JSON data
            const modelConfig = {{model_config_json}};
            const results = {{results_json}};
            
            // Display model configuration
            document.getElementById('model-config').textContent = JSON.stringify(modelConfig, null, 2);
            
            // Create loss chart
            const lossData = [
                {
                    x: Array.from({length: results.training.train_loss.length}, (_, i) => i + 1),
                    y: results.training.train_loss,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Training Loss'
                },
                {
                    x: Array.from({length: results.training.val_loss.length}, (_, i) => i + 1),
                    y: results.training.val_loss,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Validation Loss'
                }
            ];
            
            Plotly.newPlot('loss-chart', lossData, {
                xaxis: { title: 'Epoch' },
                yaxis: { title: 'Loss' }
            });
            
            // Create actual vs predicted chart
            const actualVsPredicted = {
                x: results.evaluation.y_true,
                y: results.evaluation.y_pred,
                mode: 'markers',
                type: 'scatter',
                marker: { color: 'blue', opacity: 0.7 }
            };
            
            const perfectLine = {
                x: [Math.min(...results.evaluation.y_true), Math.max(...results.evaluation.y_true)],
                y: [Math.min(...results.evaluation.y_true), Math.max(...results.evaluation.y_true)],
                mode: 'lines',
                type: 'scatter',
                line: { color: 'red', dash: 'dash' }
            };
            
            Plotly.newPlot('actual-vs-predicted', [actualVsPredicted, perfectLine], {
                xaxis: { title: 'True Values' },
                yaxis: { title: 'Predicted Values' }
            });
            
            // Create modal importance chart
            const modalImportance = {
                x: Object.keys(results.modal_importance),
                y: Object.values(results.modal_importance),
                type: 'bar'
            };
            
            Plotly.newPlot('modal-importance', [modalImportance], {
                xaxis: { title: 'Modality' },
                yaxis: { title: 'Importance Weight' }
            });
            
            // Create ablation results chart
            const ablationResults = {
                x: Object.values(results.ablation_results).map(r => r.r2),
                y: Object.keys(results.ablation_results),
                type: 'bar',
                orientation: 'h',
                marker: {
                    color: 'blue'
                }
            };
            
            Plotly.newPlot('ablation-results', [ablationResults], {
                xaxis: { title: 'R²' },
                yaxis: { title: 'Configuration' }
            });
            
            // Create cross-modal attention visualization
            const crossModalAttention = {
                z: results.cross_modal_attention,
                x: ['SMILES', 'ECFP', 'Graph', 'MFBERT'],
                y: ['SMILES', 'ECFP', 'Graph', 'MFBERT'],
                type: 'heatmap',
                colorscale: 'Viridis'
            };
            
            Plotly.newPlot('cross-modal-attention', [crossModalAttention], {
                xaxis: { title: 'Target Modality' },
                yaxis: { title: 'Source Modality' }
            });
            
            // Create error analysis chart
            const errorAnalysis = {
                x: results.evaluation.complexity_scores,
                y: results.evaluation.abs_errors,
                mode: 'markers',
                type: 'scatter',
                marker: { color: 'blue', opacity: 0.7 }
            };
            
            Plotly.newPlot('error-analysis', [errorAnalysis], {
                xaxis: { title: 'Molecular Complexity' },
                yaxis: { title: 'Absolute Error' }
            });
        </script>
    </body>
    </html>
    """
    
    # Replace placeholders with actual values
    dashboard_html = dashboard_html.replace("{{model_config_json}}", json.dumps(model_config))
    dashboard_html = dashboard_html.replace("{{results_json}}", json.dumps(results))
    
    # Replace metric values
    if 'evaluation' in results and 'metrics' in results['evaluation']:
        metrics = results['evaluation']['metrics']
        dashboard_html = dashboard_html.replace("{{r2_value}}", f"{metrics.get('r2', 0.0):.3f}")
        dashboard_html = dashboard_html.replace("{{rmse_value}}", f"{metrics.get('rmse', 0.0):.3f}")
        dashboard_html = dashboard_html.replace("{{mae_value}}", f"{metrics.get('mae', 0.0):.3f}")
    else:
        dashboard_html = dashboard_html.replace("{{r2_value}}", "N/A")
        dashboard_html = dashboard_html.replace("{{rmse_value}}", "N/A")
        dashboard_html = dashboard_html.replace("{{mae_value}}", "N/A")
    
    # Save dashboard HTML
    dashboard_path = output_path / 'dashboard.html'
    with open(dashboard_path, 'w') as f:
        f.write(dashboard_html)
    
    return dashboard_path


def fig_to_base64(fig):
    """
    Convert matplotlib figure to base64 string for HTML embedding
    
    Args:
        fig: Matplotlib figure
        
    Returns:
        Base64 encoded string
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str


def save_figure_to_file(fig, filename, output_dir='results/figures', format='png', dpi=300):
    """
    Save figure to file
    
    Args:
        fig: Matplotlib or Plotly figure
        filename: Filename without extension
        output_dir: Directory to save figures
        format: Format to save figure (png, pdf, svg, etc.)
        dpi: DPI for raster formats
        
    Returns:
        Path to saved figure
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    filepath = output_path / f"{filename}.{format}"
    
    # Check if fig is a matplotlib figure
    if hasattr(fig, 'savefig'):
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    # Check if fig is a plotly figure
    elif hasattr(fig, 'write_image'):
        fig.write_image(str(filepath))
    else:
        logger.error(f"Unknown figure type: {type(fig)}")
        return None
    
    return filepath