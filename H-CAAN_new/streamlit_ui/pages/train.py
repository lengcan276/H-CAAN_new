# streamlit_ui/pages/train.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os
import sys
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from streamlit_ui.workflow import (
    mark_step_completed, 
    save_step_data, 
    get_step_data,
    WorkflowStep,
    navigate_to_next_step,
    update_step_progress
)

def plot_learning_curve(epochs, train_metrics, val_metrics, metric_name="Loss"):
    """Plot learning curves for training and validation"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_metrics, 'b-', label=f'Training {metric_name}', linewidth=2)
    plt.plot(range(1, epochs + 1), val_metrics, 'r-', label=f'Validation {metric_name}', linewidth=2)
    plt.title(f'Training and Validation {metric_name}', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    return plt

def render_train_page():
    """Render training page"""
    st.title("🚀 Model Training")
    
    # Check prerequisites
    model_config = get_step_data(WorkflowStep.MODEL_CONFIGURATION)
    if not model_config:
        st.warning("Please configure your model first!")
        return
    
    # Initialize progress
    progress = 0
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Training Setup",
        "Training Process",
        "Monitoring",
        "Checkpoints"
    ])
    
    with tab1:
        st.header("Training Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Basic Parameters")
            
            epochs = st.number_input(
                "Number of epochs",
                min_value=1,
                max_value=1000,
                value=100,
                step=10
            )
            
            batch_size = st.number_input(
                "Batch size",
                min_value=8,
                max_value=512,
                value=32,
                step=8
            )
            
            learning_rate = st.number_input(
                "Learning rate",
                min_value=0.00001,
                max_value=0.1,
                value=0.001,
                format="%f",
                step=0.0001
            )
            
            val_split = st.slider(
                "Validation split",
                min_value=0.1,
                max_value=0.3,
                value=0.2,
                step=0.05
            )
        
        with col2:
            st.subheader("Early Stopping")
            
            early_stopping = st.checkbox("Use early stopping", value=True)
            
            if early_stopping:
                patience = st.number_input(
                    "Patience (epochs)",
                    min_value=5,
                    max_value=50,
                    value=10,
                    step=1
                )
                
                min_delta = st.number_input(
                    "Minimum delta",
                    min_value=0.0001,
                    max_value=0.01,
                    value=0.001,
                    format="%f"
                )
                
                restore_best = st.checkbox("Restore best weights", value=True)
            else:
                patience = None
                min_delta = None
                restore_best = False
        
        # Advanced options
        with st.expander("Advanced Training Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Optimizer")
                
                optimizer_type = st.selectbox(
                    "Optimizer",
                    ["Adam", "AdamW", "SGD", "RMSprop", "RAdam"],
                    index=1
                )
                
                weight_decay = st.number_input(
                    "Weight decay",
                    min_value=0.0,
                    max_value=0.1,
                    value=0.01,
                    format="%f",
                    step=0.001
                )
                
                gradient_clip = st.number_input(
                    "Gradient clipping",
                    min_value=0.0,
                    max_value=10.0,
                    value=1.0,
                    step=0.1
                )
                
                gradient_accumulation = st.number_input(
                    "Gradient accumulation steps",
                    min_value=1,
                    max_value=16,
                    value=1,
                    step=1
                )
            
            with col2:
                st.subheader("Learning Rate Schedule")
                
                scheduler_type = st.selectbox(
                    "LR Scheduler",
                    ["None", "StepLR", "ReduceLROnPlateau", "CosineAnnealingLR", 
                     "OneCycleLR", "PolynomialLR"],
                    index=2
                )
                
                if scheduler_type == "StepLR":
                    step_size = st.number_input("Step size", 10, 100, 30)
                    gamma = st.number_input("Gamma", 0.1, 1.0, 0.1)
                    scheduler_params = {"step_size": step_size, "gamma": gamma}
                    
                elif scheduler_type == "ReduceLROnPlateau":
                    factor = st.number_input("Factor", 0.1, 0.9, 0.5)
                    patience_lr = st.number_input("Patience", 3, 20, 5)
                    scheduler_params = {"factor": factor, "patience": patience_lr}
                    
                elif scheduler_type == "CosineAnnealingLR":
                    t_max = st.number_input("T_max", 10, epochs, epochs)
                    scheduler_params = {"T_max": t_max}
                    
                elif scheduler_type == "OneCycleLR":
                    max_lr = st.number_input("Max LR", 0.001, 0.1, 0.01)
                    pct_start = st.slider("Percent start", 0.1, 0.5, 0.3)
                    scheduler_params = {"max_lr": max_lr, "pct_start": pct_start}
                    
                else:
                    scheduler_params = {}
        
        # Data augmentation
        with st.expander("Data Augmentation"):
            use_augmentation = st.checkbox("Use data augmentation", value=True)
            
            if use_augmentation:
                col1, col2 = st.columns(2)
                
                with col1:
                    augmentation_methods = st.multiselect(
                        "Augmentation methods",
                        ["SMILES Augmentation", "Random Masking", 
                         "Substructure Dropout", "Noise Injection"],
                        default=["SMILES Augmentation"]
                    )
                
                with col2:
                    augmentation_prob = st.slider(
                        "Augmentation probability",
                        0.0, 1.0, 0.5, 0.1
                    )
        
        # Mixed precision training
        with st.expander("Performance Optimization"):
            col1, col2 = st.columns(2)
            
            with col1:
                use_mixed_precision = st.checkbox("Mixed precision training", value=True)
                use_distributed = st.checkbox("Distributed training", value=False)
                
                if use_distributed:
                    num_gpus = st.number_input("Number of GPUs", 1, 8, 1)
                else:
                    num_gpus = 1
            
            with col2:
                compile_model = st.checkbox("Compile model (PyTorch 2.0)", value=True)
                use_checkpointing = st.checkbox("Gradient checkpointing", value=False)
        
        # Collect all training parameters
        training_params = {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "val_split": val_split,
            "early_stopping": early_stopping,
            "patience": patience,
            "min_delta": min_delta,
            "restore_best": restore_best,
            "optimizer": optimizer_type,
            "weight_decay": weight_decay,
            "gradient_clip": gradient_clip,
            "gradient_accumulation": gradient_accumulation,
            "scheduler": scheduler_type,
            "scheduler_params": scheduler_params,
            "use_augmentation": use_augmentation,
            "augmentation_methods": augmentation_methods if use_augmentation else [],
            "augmentation_prob": augmentation_prob if use_augmentation else 0,
            "use_mixed_precision": use_mixed_precision,
            "use_distributed": use_distributed,
            "num_gpus": num_gpus,
            "compile_model": compile_model,
            "use_checkpointing": use_checkpointing
        }
        
        if st.button("💾 Save Training Configuration", type="primary"):
            st.session_state['training_params'] = training_params
            st.success("✅ Training configuration saved!")
            progress = 25
    
    with tab2:
        st.header("Training Process")
        
        if 'training_params' not in st.session_state:
            st.warning("Please configure training parameters first!")
        else:
            training_params = st.session_state['training_params']
            
            # Display training summary
            st.subheader("Training Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Epochs", training_params['epochs'])
                st.metric("Batch Size", training_params['batch_size'])
            
            with col2:
                st.metric("Learning Rate", f"{training_params['learning_rate']:.6f}")
                st.metric("Optimizer", training_params['optimizer'])
            
            with col3:
                dataset_info = get_step_data(WorkflowStep.DATA_PREPARATION)
                if dataset_info:
                    st.metric("Dataset", dataset_info.get('dataset_name', 'Unknown'))
                    st.metric("Total Samples", len(dataset_info.get('dataset', [])))
            
            # Training control
            st.markdown("---")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if st.button("🚀 Start Training", type="primary", use_container_width=True):
                    # Training simulation
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Create placeholders for metrics
                    metric_container = st.container()
                    plot_container = st.empty()
                    
                    # Initialize metrics storage
                    train_losses = []
                    val_losses = []
                    train_metrics = []
                    val_metrics = []
                    
                    # Training loop (simulated)
                    for epoch in range(1, training_params['epochs'] + 1):
                        status_text.text(f"Epoch {epoch}/{training_params['epochs']}")
                        
                        # Simulate training time
                        time.sleep(0.1)
                        
                        # Generate realistic metrics
                        train_loss = 1.0 / (0.1 * epoch + 1) + 0.1 * np.random.random()
                        val_loss = train_loss + 0.05 + 0.1 * (np.random.random() - 0.5)
                        
                        # Ensure validation doesn't improve too much
                        if epoch > training_params['epochs'] // 2:
                            val_loss = max(val_loss, train_loss * 0.95)
                        
                        train_losses.append(train_loss)
                        val_losses.append(val_loss)
                        
                        # Update progress
                        progress_bar.progress(epoch / training_params['epochs'])
                        
                        # Update plot every 5 epochs
                        if epoch % 5 == 0 or epoch == training_params['epochs']:
                            fig = plot_learning_curve(epoch, train_losses, val_losses, "Loss")
                            plot_container.pyplot(fig)
                            plt.close(fig)
                        
                        # Display current metrics
                        with metric_container:
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Epoch", f"{epoch}/{training_params['epochs']}")
                            with col2:
                                st.metric("Train Loss", f"{train_loss:.4f}")
                            with col3:
                                st.metric("Val Loss", f"{val_loss:.4f}")
                            with col4:
                                # Learning rate with scheduler
                                current_lr = training_params['learning_rate'] * (0.9 ** (epoch // 30))
                                st.metric("Learning Rate", f"{current_lr:.6f}")
                        
                        # Check early stopping
                        if training_params['early_stopping'] and epoch > training_params['patience']:
                            # Simple early stopping check
                            recent_losses = val_losses[-training_params['patience']:]
                            if all(recent_losses[i] <= recent_losses[i+1] + training_params['min_delta'] 
                                  for i in range(len(recent_losses)-1)):
                                status_text.text(f"Early stopping triggered at epoch {epoch}")
                                break
                    
                    # Training completed
                    status_text.text("✅ Training completed!")
                    
                    # Save training results
                    training_results = {
                        'train_losses': train_losses,
                        'val_losses': val_losses,
                        'final_train_loss': train_losses[-1],
                        'final_val_loss': val_losses[-1],
                        'epochs_trained': len(train_losses),
                        'early_stopped': len(train_losses) < training_params['epochs']
                    }
                    
                    # Save results
                    save_step_data(WorkflowStep.TRAINING, {
                        'training_params': training_params,
                        'training_results': training_results,
                        'model_config': model_config
                    })
                    
                    mark_step_completed(WorkflowStep.TRAINING)
                    update_step_progress(WorkflowStep.TRAINING, 100)
                    
                    st.success("✅ Model training completed successfully!")
                    
                    # Show final metrics
                    st.subheader("Final Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Final Train Loss", f"{training_results['final_train_loss']:.4f}")
                    with col2:
                        st.metric("Final Val Loss", f"{training_results['final_val_loss']:.4f}")
                    with col3:
                        st.metric("Epochs Trained", training_results['epochs_trained'])
                    
                    progress = 100
    
    with tab3:
        st.header("Training Monitoring")
        
        # Check if training has been done
        training_data = get_step_data(WorkflowStep.TRAINING)
        
        if training_data and 'training_results' in training_data:
            results = training_data['training_results']
            
            # Loss curves
            st.subheader("Loss Curves")
            
            fig = plot_learning_curve(
                len(results['train_losses']),
                results['train_losses'],
                results['val_losses'],
                "Loss"
            )
            st.pyplot(fig)
            plt.close()
            
            # Training statistics
            st.subheader("Training Statistics")
            
            stats_df = pd.DataFrame({
                'Metric': ['Min Train Loss', 'Min Val Loss', 'Final Train Loss', 
                          'Final Val Loss', 'Convergence Epoch'],
                'Value': [
                    f"{min(results['train_losses']):.4f}",
                    f"{min(results['val_losses']):.4f}",
                    f"{results['final_train_loss']:.4f}",
                    f"{results['final_val_loss']:.4f}",
                    str(np.argmin(results['val_losses']) + 1)
                ]
            })
            
            st.dataframe(stats_df)
            
            # Learning rate schedule visualization
            if training_data['training_params']['scheduler'] != "None":
                st.subheader("Learning Rate Schedule")
                
                # Simulate LR schedule
                epochs = len(results['train_losses'])
                base_lr = training_data['training_params']['learning_rate']
                
                if training_data['training_params']['scheduler'] == "StepLR":
                    step_size = training_data['training_params']['scheduler_params']['step_size']
                    gamma = training_data['training_params']['scheduler_params']['gamma']
                    lrs = [base_lr * (gamma ** (i // step_size)) for i in range(epochs)]
                elif training_data['training_params']['scheduler'] == "CosineAnnealingLR":
                    lrs = [base_lr * (1 + np.cos(np.pi * i / epochs)) / 2 for i in range(epochs)]
                else:
                    lrs = [base_lr] * epochs
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(range(1, epochs + 1), lrs, 'g-', linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Learning Rate')
                ax.set_title('Learning Rate Schedule')
                ax.grid(True, alpha=0.3)
                ax.set_yscale('log')
                
                st.pyplot(fig)
                plt.close()
        else:
            st.info("No training results available yet. Start training to see monitoring data.")
    
    with tab4:
        st.header("Model Checkpoints")
        
        if training_data and 'training_results' in training_data:
            st.subheader("Available Checkpoints")
            
            # Simulated checkpoints
            checkpoints = [
                {
                    'name': 'best_model.pt',
                    'epoch': np.argmin(training_data['training_results']['val_losses']) + 1,
                    'val_loss': min(training_data['training_results']['val_losses']),
                    'size': '245 MB',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                },
                {
                    'name': 'final_model.pt',
                    'epoch': len(training_data['training_results']['train_losses']),
                    'val_loss': training_data['training_results']['final_val_loss'],
                    'size': '245 MB',
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
            ]
            
            for checkpoint in checkpoints:
                with st.expander(f"📁 {checkpoint['name']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Epoch:** {checkpoint['epoch']}")
                        st.write(f"**Val Loss:** {checkpoint['val_loss']:.4f}")
                    
                    with col2:
                        st.write(f"**Size:** {checkpoint['size']}")
                        st.write(f"**Saved:** {checkpoint['timestamp']}")
                    
                    if st.button(f"Load {checkpoint['name']}", key=checkpoint['name']):
                        st.success(f"✅ Loaded checkpoint: {checkpoint['name']}")
            
            # Export options
            st.subheader("Export Model")
            
            export_format = st.selectbox(
                "Export format:",
                ["PyTorch (.pt)", "ONNX (.onnx)", "TorchScript (.pts)", "SavedModel (TF)"]
            )
            
            if st.button("📥 Export Model", type="primary"):
                st.success(f"✅ Model exported as {export_format}")
        else:
            st.info("No checkpoints available yet. Train the model to create checkpoints.")
    
    # Update overall progress
    update_step_progress(WorkflowStep.TRAINING, progress)
    
    # Navigation
    if progress >= 100:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if st.button("📊 Proceed to Results Analysis", type="primary", use_container_width=True):
                navigate_to_next_step()