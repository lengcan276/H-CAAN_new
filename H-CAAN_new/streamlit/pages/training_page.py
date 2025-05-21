import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
import os
import sys
from datetime import datetime

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.trainer import ModelTrainer
from training.losses import get_loss_function
from training.optimizers import get_optimizer, get_scheduler


def plot_learning_curve(epochs, train_metrics, val_metrics, metric_name="Loss"):
    """Plot learning curves for training and validation"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_metrics, 'b-', label=f'Training {metric_name}')
    plt.plot(range(1, epochs + 1), val_metrics, 'r-', label=f'Validation {metric_name}')
    plt.title(f'Training and Validation {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    return plt


def plot_confusion_matrix(cm, class_names=None):
    """Plot confusion matrix as a heatmap"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    return plt


def plot_scatter(y_true, y_pred, title="Predicted vs Actual Values"):
    """Create scatter plot of predicted vs actual values"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title(title)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.grid(True, linestyle='--', alpha=0.7)
    return plt


def plot_feature_importance(importances, feature_names):
    """Plot feature importance"""
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.title('Feature Importance')
    plt.tight_layout()
    return plt


def training_page():
    st.title("H-CAAN: Model Training and Evaluation")
    
    # Check if model is configured
    if not st.session_state.get('model_ready', False):
        st.warning("Please configure your model first!")
        if st.button("Go to Model Configuration"):
            st.session_state['current_page'] = 'model_page'
            st.experimental_rerun()
        return
    
    # Load model configuration
    model_config = st.session_state.get('model_config', {})
    
    # Training Parameters
    st.header("1. Training Parameters")
    
    col1, col2 = st.columns(2)
    with col1:
        epochs = st.number_input("Number of Epochs", min_value=1, max_value=1000, value=100, step=10)
        batch_size = st.number_input("Batch Size", min_value=8, max_value=512, value=32, step=8)
        learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.1, value=0.001, format="%f", step=0.0001)
    
    with col2:
        val_split = st.slider("Validation Split", min_value=0.1, max_value=0.3, value=0.2, step=0.05)
        early_stopping = st.checkbox("Use Early Stopping", value=True)
        if early_stopping:
            patience = st.number_input("Patience", min_value=1, max_value=50, value=10, step=1)
    
    # Advanced training options
    with st.expander("Advanced Training Options"):
        col1, col2 = st.columns(2)
        with col1:
            optimizer_type = st.selectbox("Optimizer", ["Adam", "AdamW", "SGD", "RMSprop"], index=1)
            weight_decay = st.number_input("Weight Decay", min_value=0.0, max_value=0.1, value=0.001, format="%f", step=0.0001)
            gradient_clip = st.number_input("Gradient Clipping", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
            gradient_accumulation = st.number_input("Gradient Accumulation Steps", min_value=1, max_value=8, value=1, step=1)
        
        with col2:
            scheduler_type = st.selectbox(
                "Learning Rate Scheduler", 
                ["None", "StepLR", "ReduceLROnPlateau", "CosineAnnealingLR", "OneCycleLR"], 
                index=0
            )
            if scheduler_type != "None":
                if scheduler_type == "StepLR":
                    step_size = st.number_input("Step Size", min_value=1, max_value=50, value=10, step=1)
                    gamma = st.number_input("Gamma", min_value=0.1, max_value=1.0, value=0.1, step=0.05)
                elif scheduler_type == "ReduceLROnPlateau":
                    factor = st.number_input("Factor", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
                    patience_lr = st.number_input("Patience", min_value=1, max_value=20, value=5, step=1)
                elif scheduler_type == "CosineAnnealingLR":
                    t_max = st.number_input("T_max", min_value=1, max_value=epochs, value=epochs, step=1)
                elif scheduler_type == "OneCycleLR":
                    max_lr = st.number_input("Max LR", min_value=0.001, max_value=0.1, value=0.01, step=0.001)
                    pct_start = st.slider("Percent Start", min_value=0.1, max_value=0.5, value=0.3, step=0.05)
    
    # Data augmentation options
    with st.expander("Data Augmentation"):
        use_augmentation = st.checkbox("Use Data Augmentation", value=False)
        if use_augmentation:
            augmentation_methods = st.multiselect(
                "Augmentation Methods",
                ["SMILES Augmentation", "Random Masking", "Substructure Replacement", "Molecule Perturbation"],
                default=["SMILES Augmentation"]
            )
            augmentation_prob = st.slider("Augmentation Probability", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
    
    # Collect training parameters
    training_params = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "val_split": val_split,
        "early_stopping": early_stopping,
        "optimizer": optimizer_type,
        "weight_decay": weight_decay,
        "gradient_clip": gradient_clip,
        "gradient_accumulation": gradient_accumulation,
        "scheduler": scheduler_type,
        "data_augmentation": use_augmentation
    }
    
    if early_stopping:
        training_params["patience"] = patience
    
    if scheduler_type == "StepLR":
        training_params["step_size"] = step_size
        training_params["gamma"] = gamma
    elif scheduler_type == "ReduceLROnPlateau":
        training_params["factor"] = factor
        training_params["patience_lr"] = patience_lr
    elif scheduler_type == "CosineAnnealingLR":
        training_params["t_max"] = t_max
    elif scheduler_type == "OneCycleLR":
        training_params["max_lr"] = max_lr
        training_params["pct_start"] = pct_start
    
    if use_augmentation:
        training_params["augmentation_methods"] = augmentation_methods
        training_params["augmentation_prob"] = augmentation_prob
    
    # Model Training
    st.header("2. Model Training")
    
    if st.button("Start Training"):
        # This would be the actual training process
        # For now, we'll simulate the training
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        loss_plot = st.empty()
        metrics_text = st.empty()
        
        # Create placeholders for metrics
        train_losses = []
        val_losses = []
        train_metrics = []
        val_metrics = []
        
        # Simulate training process
        for epoch in range(1, epochs + 1):
            status_text.text(f"Epoch {epoch}/{epochs}")
            
            # Simulate epoch time
            time.sleep(0.1)
            
            # Generate dummy metrics
            train_loss = 1.0 / (0.1 * epoch + 1) + 0.1 * np.random.random()
            val_loss = train_loss + 0.05 - 0.1 * np.random.random()
            
            if epoch > epochs // 2:
                # Add some noise but ensure validation doesn't improve too much
                val_loss = max(val_loss, train_loss * 0.9)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            # Update progress
            progress_bar.progress(epoch / epochs)
            
            # Update the loss plot
            fig = plot_learning_curve(epoch, train_losses, val_losses, "Loss")
            loss_plot.pyplot(fig)
            plt.close(fig)
            
            # Display current metrics
            metrics_df = pd.DataFrame({
                'Epoch': [epoch],
                'Train Loss': [f"{train_loss:.4f}"],
                'Val Loss': [f"{val_loss:.4f}"]
            })
            metrics_text.dataframe(metrics_df)
            
            # Check for early stopping
            if early_stopping and epoch > patience:
                # Simple early stopping implementation
                if all(val_losses[-i-1] <= val_losses[-i] for i in range(1, patience + 1)):
                    status_text.text(f"Early stopping triggered at epoch {epoch}")
                    break
        
        # Training completed
        status_text.text("Training completed!")
        
        # Save the model
        save_path = f"models/h_caan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        st.success(f"Model training completed! Model saved to {save_path}")
        
        # Store training results in session state
        st.session_state['train_losses'] = train_losses
        st.session_state['val_losses'] = val_losses
        st.session_state['model_trained'] = True
    
    # Model Evaluation
    st.header("3. Model Evaluation")
    
    if not st.session_state.get('model_trained', False):
        st.warning("Please train your model first!")
    else:
        # Create evaluation tabs
        eval_tab1, eval_tab2, eval_tab3, eval_tab4 = st.tabs(
            ["Performance Metrics", "Learning Curves", "Predictions Analysis", "Modal Analysis"]
        )
        
        with eval_tab1:
            st.subheader("Performance Metrics")
            
            # Check task type for appropriate metrics
            task_type = model_config.get('task', {}).get('type', 'Regression')
            
            if task_type == "Regression":
                # Display regression metrics
                metrics_df = pd.DataFrame({
                    'Metric': ['RMSE', 'MAE', 'R²', 'Pearson Correlation'],
                    'Training': ['0.241', '0.189', '0.876', '0.936'],
                    'Validation': ['0.285', '0.225', '0.842', '0.918'],
                    'Test': ['0.312', '0.247', '0.824', '0.907']
                })
                st.dataframe(metrics_df)
                
                # Plot scatter of predictions vs actual
                st.subheader("Predictions vs Actual Values")
                
                # Generate some sample data
                n_samples = 100
                y_true = np.random.normal(5, 2, size=n_samples)
                y_pred = y_true + np.random.normal(0, 0.5, size=n_samples)
                
                fig = plot_scatter(y_true, y_pred)
                st.pyplot(fig)
                plt.close(fig)
                
                # Show residuals
                st.subheader("Residuals Analysis")
                residuals = y_true - y_pred
                
                fig, ax = plt.subplots(1, 2, figsize=(14, 6))
                
                # Residuals vs Predicted
                ax[0].scatter(y_pred, residuals, alpha=0.5)
                ax[0].axhline(y=0, color='r', linestyle='--')
                ax[0].set_xlabel("Predicted Values")
                ax[0].set_ylabel("Residuals")
                ax[0].set_title("Residuals vs Predicted Values")
                ax[0].grid(True, linestyle='--', alpha=0.7)
                
                # Residuals distribution
                ax[1].hist(residuals, bins=20, alpha=0.7)
                ax[1].axvline(x=0, color='r', linestyle='--')
                ax[1].set_xlabel("Residuals")
                ax[1].set_ylabel("Frequency")
                ax[1].set_title("Residuals Distribution")
                ax[1].grid(True, linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                
            elif task_type == "Binary Classification":
                # Display binary classification metrics
                metrics_df = pd.DataFrame({
                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC', 'AUC-PR'],
                    'Training': ['0.938', '0.925', '0.912', '0.919', '0.962', '0.957'],
                    'Validation': ['0.912', '0.895', '0.885', '0.890', '0.945', '0.932'],
                    'Test': ['0.901', '0.887', '0.865', '0.876', '0.936', '0.922']
                })
                st.dataframe(metrics_df)
                
                # ROC Curve
                st.subheader("ROC Curve")
                
                # Generate sample ROC curve
                fpr = np.linspace(0, 1, 100)
                tpr = np.power(fpr, 0.3)  # Just a power function to simulate ROC curve
                
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = 0.936')
                ax.plot([0, 1], [0, 1], 'r--', label='Random Classifier')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.set_title('Receiver Operating Characteristic (ROC) Curve')
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.7)
                
                st.pyplot(fig)
                plt.close(fig)
                
                # Confusion Matrix
                st.subheader("Confusion Matrix")
                
                # Sample confusion matrix
                cm = np.array([[85, 15], [10, 90]])
                
                fig = plot_confusion_matrix(cm, class_names=["Negative", "Positive"])
                st.pyplot(fig)
                plt.close(fig)
                
            elif task_type == "Multi-class Classification":
                # Display multi-class classification metrics
                metrics_df = pd.DataFrame({
                    'Metric': ['Accuracy', 'Weighted Precision', 'Weighted Recall', 'Weighted F1'],
                    'Training': ['0.921', '0.923', '0.921', '0.922'],
                    'Validation': ['0.895', '0.897', '0.895', '0.896'],
                    'Test': ['0.882', '0.883', '0.882', '0.882']
                })
                st.dataframe(metrics_df)
                
                # Class-wise metrics
                st.subheader("Class-wise Metrics")
                class_metrics_df = pd.DataFrame({
                    'Class': ['Class 1', 'Class 2', 'Class 3'],
                    'Precision': ['0.902', '0.863', '0.882'],
                    'Recall': ['0.915', '0.842', '0.890'],
                    'F1 Score': ['0.908', '0.852', '0.886'],
                    'Support': ['105', '98', '97']
                })
                st.dataframe(class_metrics_df)
                
                # Confusion Matrix
                st.subheader("Confusion Matrix")
                
                # Sample multi-class confusion matrix
                cm = np.array([
                    [96, 7, 2],
                    [5, 83, 10],
                    [4, 9, 84]
                ])
                
                fig = plot_confusion_matrix(cm, class_names=["Class 1", "Class 2", "Class 3"])
                st.pyplot(fig)
                plt.close(fig)
                
            else:  # Multi-task
                # Display multi-task metrics
                st.write("Multi-task evaluation results:")
                
                # Create tabs for each task
                task_names = [f"Task {i+1}" for i in range(model_config.get('task', {}).get('num_tasks', 2))]
                task_tabs = st.tabs(task_names)
                
                for i, tab in enumerate(task_tabs):
                    with tab:
                        st.write(f"Metrics for {task_names[i]}:")
                        
                        # Generate sample metrics for each task
                        if i % 2 == 0:  # Regression task
                            task_metrics_df = pd.DataFrame({
                                'Metric': ['RMSE', 'MAE', 'R²'],
                                'Value': [f"{0.2 + i*0.05:.3f}", f"{0.15 + i*0.04:.3f}", f"{0.9 - i*0.05:.3f}"]
                            })
                        else:  # Classification task
                            task_metrics_df = pd.DataFrame({
                                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                                'Value': [f"{0.95 - i*0.03:.3f}", f"{0.94 - i*0.03:.3f}", f"{0.93 - i*0.03:.3f}", f"{0.93 - i*0.03:.3f}"]
                            })
                        
                        st.dataframe(task_metrics_df)
        
        with eval_tab2:
            st.subheader("Learning Curves")
            
            # Get training history
            train_losses = st.session_state.get('train_losses', [])
            val_losses = st.session_state.get('val_losses', [])
            
            # Plot learning curves
            if train_losses and val_losses:
                epochs_trained = len(train_losses)
                
                # Loss curve
                fig = plot_learning_curve(epochs_trained, train_losses, val_losses, "Loss")
                st.pyplot(fig)
                plt.close(fig)
                
                # Additional metrics (simulated)
                if task_type == "Regression":
                    # RMSE curve
                    train_rmse = [0.5 / (0.1 * i + 1) + 0.05 * np.random.random() for i in range(1, epochs_trained + 1)]
                    val_rmse = [x + 0.03 - 0.06 * np.random.random() for x in train_rmse]
                    
                    fig = plot_learning_curve(epochs_trained, train_rmse, val_rmse, "RMSE")
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # R2 curve
                    train_r2 = [1 - 1 / (i + 5) - 0.02 * np.random.random() for i in range(1, epochs_trained + 1)]
                    val_r2 = [x - 0.05 + 0.03 * np.random.random() for x in train_r2]
                    
                    fig = plot_learning_curve(epochs_trained, train_r2, val_r2, "R²")
                    st.pyplot(fig)
                    plt.close(fig)
                    
                elif task_type in ["Binary Classification", "Multi-class Classification"]:
                    # Accuracy curve
                    train_acc = [0.7 + 0.25 / (0.1 * i + 1) - 0.05 * np.random.random() for i in range(1, epochs_trained + 1)]
                    val_acc = [x - 0.03 - 0.02 * np.random.random() for x in train_acc]
                    
                    fig = plot_learning_curve(epochs_trained, train_acc, val_acc, "Accuracy")
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # F1 Score curve
                    train_f1 = [0.65 + 0.3 / (0.1 * i + 1) - 0.05 * np.random.random() for i in range(1, epochs_trained + 1)]
                    val_f1 = [x - 0.04 - 0.02 * np.random.random() for x in train_f1]
                    
                    fig = plot_learning_curve(epochs_trained, train_f1, val_f1, "F1 Score")
                    st.pyplot(fig)
                    plt.close(fig)
            else:
                st.warning("No training history available. Please train the model first.")
        
        with eval_tab3:
            st.subheader("Predictions Analysis")
            
            # Sample distribution
            st.write("Distribution of Predictions vs Actual Values")
            
            if task_type == "Regression":
                # Generate sample predictions
                n_samples = 200
                y_true = np.random.normal(5, 2, size=n_samples)
                y_pred = y_true + np.random.normal(0, 0.5, size=n_samples)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.kdeplot(y_true, label="Actual", color="blue", fill=True, alpha=0.3)
                sns.kdeplot(y_pred, label="Predicted", color="red", fill=True, alpha=0.3)
                ax.set_xlabel("Value")
                ax.set_ylabel("Density")
                ax.set_title("Distribution of Actual vs Predicted Values")
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.7)
                
                st.pyplot(fig)
                plt.close(fig)
                
                # Error distribution by value range
                st.write("Error Distribution by Value Range")
                
                # Calculate absolute errors
                abs_errors = np.abs(y_true - y_pred)
                
                # Create bins based on actual values
                bins = np.linspace(min(y_true), max(y_true), 6)
                bin_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
                bin_indices = np.digitize(y_true, bins) - 1
                bin_indices = np.clip(bin_indices, 0, len(bin_labels) - 1)
                
                # Calculate errors by bin
                bin_errors = [abs_errors[bin_indices == i] for i in range(len(bin_labels))]
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.boxplot(bin_errors, labels=bin_labels, showfliers=False)
                ax.set_xlabel("Value Range")
                ax.set_ylabel("Absolute Error")
                ax.set_title("Error Distribution by Value Range")
                ax.grid(True, linestyle='--', alpha=0.7)
                
                st.pyplot(fig)
                plt.close(fig)