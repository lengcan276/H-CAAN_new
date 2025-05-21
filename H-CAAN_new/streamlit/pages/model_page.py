import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.encoders.smiles_encoder import SmilesTransformerEncoder
from models.encoders.ecfp_encoder import ECFPEncoder
from models.encoders.gcn_encoder import GCNEncoder
from models.encoders.mfbert_encoder import MFBERTEncoder
from models.fusion.gcau import GatedCrossModalAttentionUnit as GatedCrossAttentionUnit
from models.fusion.hierarchical_fusion import HierarchicalFusion
from models.attention.chemical_aware_attention import ChemicalAwareAttention
from models.modal_importance.task_specific_weights import TaskSpecificWeightGenerator


def model_page():
    st.title("H-CAAN: Model Configuration")
    
    # Check if data is ready
    if not st.session_state.get('data_ready', False):
        st.warning("Please process your data first!")
        if st.button("Go to Data Preparation"):
            st.session_state['current_page'] = 'data_page'
            st.experimental_rerun()
        return
    
    # Model Components Configuration
    st.header("1. Encoder Configuration")
    
    # Create tabs for each encoder
    tab1, tab2, tab3, tab4 = st.tabs(["SMILES Encoder", "ECFP Encoder", "GCN Encoder", "MFBERT Encoder"])
    
    with tab1:
        st.subheader("SMILES Transformer Encoder")
        st.write("""
        The SMILES Transformer Encoder processes the tokenized SMILES representations using a
        Transformer architecture with self-attention mechanisms.
        """)
        
        # Transformer encoder parameters
        col1, col2 = st.columns(2)
        with col1:
            smiles_hidden_dim = st.number_input("Hidden Dimension", min_value=16, max_value=1024, value=256, step=16, key="smiles_hidden_dim")
            smiles_num_layers = st.number_input("Number of Layers", min_value=1, max_value=12, value=3, step=1, key="smiles_num_layers")
        
        with col2:
            smiles_num_heads = st.number_input("Number of Attention Heads", min_value=1, max_value=16, value=4, step=1, key="smiles_num_heads")
            smiles_dropout = st.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.1, step=0.05, key="smiles_dropout")
        
        smiles_use_positional_encoding = st.checkbox("Use Positional Encoding", value=True, key="smiles_use_pos_encoding")
        
        # Advanced options
        with st.expander("Advanced Options"):
            smiles_use_chemical_attention = st.checkbox("Use Chemical-Aware Attention", value=True, key="smiles_use_chem_attn")
            smiles_ff_dim = st.number_input("Feed-Forward Dimension", min_value=64, max_value=2048, value=512, step=64, key="smiles_ff_dim")
            smiles_normalization = st.selectbox("Normalization Type", ["Layer Norm", "Batch Norm", "None"], key="smiles_norm")
    
    with tab2:
        st.subheader("ECFP Encoder (BiGRU)")
        st.write("""
        The ECFP Encoder processes molecular fingerprints using a Bidirectional Gated Recurrent Unit (BiGRU)
        network with multi-head attention.
        """)
        
        # BiGRU parameters
        col1, col2 = st.columns(2)
        with col1:
            ecfp_hidden_dim = st.number_input("Hidden Dimension", min_value=16, max_value=1024, value=256, step=16, key="ecfp_hidden_dim")
            ecfp_num_layers = st.number_input("Number of Layers", min_value=1, max_value=4, value=2, step=1, key="ecfp_num_layers")
        
        with col2:
            ecfp_num_heads = st.number_input("Number of Attention Heads", min_value=1, max_value=16, value=2, step=1, key="ecfp_num_heads")
            ecfp_dropout = st.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.1, step=0.05, key="ecfp_dropout")
        
        # Advanced options
        with st.expander("Advanced Options"):
            ecfp_bidirectional = st.checkbox("Bidirectional", value=True, key="ecfp_bidirectional")
            ecfp_attention_pooling = st.checkbox("Use Attention Pooling", value=True, key="ecfp_attn_pooling")

    with tab3:
        st.subheader("Graph Convolutional Network Encoder")
        st.write("""
        The GCN Encoder processes molecular graphs using Graph Convolutional Networks to capture
        the structural information of molecules.
        """)
        
        # GCN parameters
        col1, col2 = st.columns(2)
        with col1:
            gcn_hidden_dim = st.number_input("Hidden Dimension", min_value=16, max_value=1024, value=256, step=16, key="gcn_hidden_dim")
            gcn_num_layers = st.number_input("Number of Layers", min_value=1, max_value=6, value=2, step=1, key="gcn_num_layers")
        
        with col2:
            gcn_dropout = st.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.1, step=0.05, key="gcn_dropout")
            gcn_pooling = st.selectbox("Pooling Method", ["Mean", "Sum", "Max"], key="gcn_pooling")
        
        # Advanced options
        with st.expander("Advanced Options"):
            gcn_residual = st.checkbox("Use Residual Connections", value=True, key="gcn_residual")
            gcn_use_batch_norm = st.checkbox("Use Batch Normalization", value=True, key="gcn_batch_norm")
            gcn_aggregation = st.selectbox("Aggregation Function", ["Sum", "Mean", "Max"], key="gcn_aggregation")

    with tab4:
        st.subheader("MFBERT Encoder")
        st.write("""
        The MFBERT Encoder utilizes a pre-trained MFBERT model to generate molecular embeddings
        from SMILES strings.
        """)
        
        # MFBERT parameters
        col1, col2 = st.columns(2)
        with col1:
            mfbert_output_dim = st.number_input("Output Dimension", min_value=64, max_value=1024, value=256, step=16, key="mfbert_output_dim")
            mfbert_pooling = st.selectbox("Pooling Strategy", ["CLS", "Mean"], key="mfbert_pooling")
        
        with col2:
            mfbert_dropout = st.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.1, step=0.05, key="mfbert_dropout")
            mfbert_fine_tune = st.checkbox("Fine-tune MFBERT", value=False, key="mfbert_fine_tune")
        
        # Advanced options
        with st.expander("Advanced Options"):
            mfbert_layers_to_use = st.multiselect("Layers to Use", ["Last", "Last 2", "Last 4", "All"], default=["Last"], key="mfbert_layers")
            mfbert_use_attention = st.checkbox("Use Attention Mechanism", value=True, key="mfbert_use_attention")

    # Fusion Configuration
    st.header("2. Fusion Architecture")
    
    fusion_type = st.selectbox(
        "Select Fusion Strategy",
        ["Hierarchical Fusion", "Gated Cross-Attention", "Simple Concatenation"],
        key="fusion_type"
    )
    
    st.write(f"Selected fusion strategy: **{fusion_type}**")
    
    if fusion_type == "Hierarchical Fusion":
        st.write("""
        Hierarchical Fusion progressively combines information from different modalities in a 
        hierarchical manner, starting with the most similar modalities.
        """)
        
        # Hierarchical fusion parameters
        fusion_hierarchy = st.selectbox(
            "Select Fusion Hierarchy",
            ["(SMILES+ECFP) + (Graph+MFBERT)", "(SMILES+MFBERT) + (ECFP+Graph)", "Automatic (complexity-based)"],
            key="fusion_hierarchy"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            hierarchical_hidden_dim = st.number_input("Fusion Hidden Dimension", min_value=32, max_value=1024, value=256, step=32, key="hierarchical_hidden_dim")
        
        with col2:
            hierarchical_dropout = st.slider("Fusion Dropout Rate", min_value=0.0, max_value=0.5, value=0.2, step=0.05, key="hierarchical_dropout")
        
        hierarchical_use_attention = st.checkbox("Use Cross-Modal Attention", value=True, key="hierarchical_use_attention")
        
        # Advanced options
        with st.expander("Advanced Options"):
            hierarchical_learnable_weights = st.checkbox("Use Learnable Weights", value=True, key="hierarchical_learnable_weights")
            hierarchical_residual = st.checkbox("Use Residual Connections", value=True, key="hierarchical_residual")
    
    elif fusion_type == "Gated Cross-Attention":
        st.write("""
        Gated Cross-Attention uses attention mechanisms to dynamically weight the importance of 
        different modalities based on their relevance to the task.
        """)
        
        # Gated cross-attention parameters
        col1, col2 = st.columns(2)
        with col1:
            gca_hidden_dim = st.number_input("Attention Hidden Dimension", min_value=32, max_value=1024, value=256, step=32, key="gca_hidden_dim")
            gca_num_heads = st.number_input("Number of Attention Heads", min_value=1, max_value=16, value=4, step=1, key="gca_num_heads")
        
        with col2:
            gca_dropout = st.slider("Attention Dropout Rate", min_value=0.0, max_value=0.5, value=0.1, step=0.05, key="gca_dropout")
            gca_gating_type = st.selectbox("Gating Mechanism", ["Sigmoid", "Softmax", "Tanh"], key="gca_gating")
        
        # Advanced options
        with st.expander("Advanced Options"):
            gca_use_layernorm = st.checkbox("Use Layer Normalization", value=True, key="gca_layernorm")
            gca_temperature = st.slider("Attention Temperature", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key="gca_temperature")
    
    else:  # Simple Concatenation
        st.write("""
        Simple Concatenation combines the feature vectors from all modalities by concatenating them
        and passing through a feed-forward network.
        """)
        
        # Simple concatenation parameters
        col1, col2 = st.columns(2)
        with col1:
            concat_hidden_dim = st.number_input("Hidden Dimension", min_value=32, max_value=1024, value=256, step=32, key="concat_hidden_dim")
            concat_num_layers = st.number_input("Number of Layers", min_value=1, max_value=3, value=2, step=1, key="concat_num_layers")
        
        with col2:
            concat_dropout = st.slider("Dropout Rate", min_value=0.0, max_value=0.5, value=0.2, step=0.05, key="concat_dropout")
            concat_activation = st.selectbox("Activation Function", ["ReLU", "LeakyReLU", "GELU"], key="concat_activation")
    
    # Modal Importance Configuration
    st.header("3. Modal Importance Weighting")
    
    weight_method = st.selectbox(
        "Select Weighting Method",
        ["Task-Specific", "Complexity-Aware", "Uncertainty-Based", "Equal Weights"],
        key="weight_method"
    )
    
    st.write(f"Selected weighting method: **{weight_method}**")
    
    if weight_method == "Task-Specific":
        st.write("""
        Task-Specific weighting learns the importance of each modality based on their relevance
        to the specific prediction task.
        """)
        
        # Task-specific parameters
        col1, col2 = st.columns(2)
        with col1:
            ts_hidden_dim = st.number_input("Weight Generator Hidden Dim", min_value=16, max_value=256, value=64, step=16, key="ts_hidden_dim")
        
        with col2:
            ts_temperature = st.slider("Softmax Temperature", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key="ts_temperature")
        
        ts_use_gumbel = st.checkbox("Use Gumbel Softmax", value=False, key="ts_use_gumbel")
    
    elif weight_method == "Complexity-Aware":
        st.write("""
        Complexity-Aware weighting assigns importance based on the molecular complexity,
        giving more weight to modalities that better represent complex molecules.
        """)
        
        # Complexity-aware parameters
        col1, col2 = st.columns(2)
        with col1:
            ca_complexity_metrics = st.multiselect(
                "Complexity Metrics",
                ["Molecular Weight", "Number of Atoms", "Number of Bonds", "Number of Rings", "Rotatable Bonds"],
                default=["Molecular Weight", "Number of Rings"],
                key="ca_metrics"
            )
        
        with col2:
            ca_normalization = st.selectbox("Normalization Method", ["Min-Max", "Z-Score", "Robust"], key="ca_normalization")
        
        ca_learnable = st.checkbox("Use Learnable Mapping", value=True, key="ca_learnable")
    
    elif weight_method == "Uncertainty-Based":
        st.write("""
        Uncertainty-Based weighting estimates the prediction uncertainty of each modality
        and assigns higher weight to more confident predictions.
        """)
        
        # Uncertainty-based parameters
        col1, col2 = st.columns(2)
        with col1:
            ub_ensemble_size = st.number_input("Ensemble Size", min_value=2, max_value=10, value=5, step=1, key="ub_ensemble_size")
            ub_dropout_samples = st.number_input("MC Dropout Samples", min_value=1, max_value=20, value=10, step=1, key="ub_dropout_samples")
        
        with col2:
            ub_uncertainty_type = st.selectbox(
                "Uncertainty Type",
                ["Variance", "Entropy", "Mutual Information"],
                key="ub_uncertainty_type"
            )
            ub_calibration = st.checkbox("Apply Uncertainty Calibration", value=True, key="ub_calibration")
    
    # Task Type and Output Configuration
    st.header("4. Task Configuration")
    
    task_type = st.selectbox(
        "Select Task Type",
        ["Regression", "Binary Classification", "Multi-class Classification", "Multi-task"],
        key="task_type"
    )
    
    st.write(f"Selected task type: **{task_type}**")
    
    if task_type == "Regression":
        # Regression parameters
        col1, col2 = st.columns(2)
        with col1:
            reg_loss = st.selectbox("Loss Function", ["MSE", "MAE", "Huber", "SmoothL1"], key="reg_loss")
            reg_output_activation = st.selectbox("Output Activation", ["None", "ReLU", "Sigmoid", "Tanh"], key="reg_output_activation")
        
        with col2:
            reg_metrics = st.multiselect(
                "Evaluation Metrics",
                ["RMSE", "MAE", "R²", "Pearson Correlation", "Spearman Correlation"],
                default=["RMSE", "R²", "Pearson Correlation"],
                key="reg_metrics"
            )
    
    elif task_type == "Binary Classification":
        # Binary classification parameters
        col1, col2 = st.columns(2)
        with col1:
            bin_loss = st.selectbox("Loss Function", ["BCE", "Focal Loss", "Weighted BCE"], key="bin_loss")
            bin_threshold = st.slider("Classification Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05, key="bin_threshold")
        
        with col2:
            bin_metrics = st.multiselect(
                "Evaluation Metrics",
                ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC", "AUC-PR"],
                default=["Accuracy", "F1 Score", "AUC-ROC"],
                key="bin_metrics"
            )
            bin_class_weights = st.checkbox("Use Class Weights", value=False, key="bin_class_weights")
    
    elif task_type == "Multi-class Classification":
        # Multi-class classification parameters
        col1, col2 = st.columns(2)
        with col1:
            mc_loss = st.selectbox("Loss Function", ["Cross Entropy", "Focal Loss", "Label Smoothing"], key="mc_loss")
            mc_num_classes = st.number_input("Number of Classes", min_value=2, max_value=100, value=3, step=1, key="mc_num_classes")
        
        with col2:
            mc_metrics = st.multiselect(
                "Evaluation Metrics",
                ["Accuracy", "Precision", "Recall", "F1 Score", "Confusion Matrix"],
                default=["Accuracy", "F1 Score"],
                key="mc_metrics"
            )
            mc_class_weights = st.checkbox("Use Class Weights", value=False, key="mc_class_weights")
    
    else:  # Multi-task
        # Multi-task parameters
        col1, col2 = st.columns(2)
        with col1:
            mt_num_tasks = st.number_input("Number of Tasks", min_value=2, max_value=10, value=2, step=1, key="mt_num_tasks")
            mt_task_types = st.multiselect(
                "Task Types",
                ["Regression", "Binary Classification", "Multi-class Classification"],
                default=["Regression", "Binary Classification"],
                key="mt_task_types"
            )
        
        with col2:
            mt_loss_weighting = st.selectbox(
                "Loss Weighting Strategy",
                ["Equal", "Uncertainty", "Task Specific", "Adaptive"],
                key="mt_loss_weighting"
            )
            mt_shared_layers = st.number_input("Number of Shared Layers", min_value=1, max_value=5, value=2, step=1, key="mt_shared_layers")
    
    # Model Summary
    st.header("5. Model Summary")
    
    # Collect all parameters
    model_config = {
        "model_name": "H-CAAN",
        "encoders": {
            "smiles_encoder": {
                "type": "TransformerEncoder",
                "hidden_dim": smiles_hidden_dim,
                "num_layers": smiles_num_layers,
                "num_heads": smiles_num_heads,
                "dropout": smiles_dropout,
                "use_positional_encoding": smiles_use_positional_encoding
            },
            "ecfp_encoder": {
                "type": "BiGRU",
                "hidden_dim": ecfp_hidden_dim,
                "num_layers": ecfp_num_layers,
                "num_heads": ecfp_num_heads,
                "dropout": ecfp_dropout
            },
            "gcn_encoder": {
                "type": "GCN",
                "hidden_dim": gcn_hidden_dim,
                "num_layers": gcn_num_layers,
                "dropout": gcn_dropout,
                "pooling": gcn_pooling
            },
            "mfbert_encoder": {
                "type": "MFBERT",
                "output_dim": mfbert_output_dim,
                "pooling": mfbert_pooling,
                "dropout": mfbert_dropout,
                "fine_tune": mfbert_fine_tune
            }
        },
        "fusion": {
            "type": fusion_type
        },
        "weighting": {
            "method": weight_method
        },
        "task": {
            "type": task_type
        }
    }
    
    # Add fusion-specific parameters
    if fusion_type == "Hierarchical Fusion":
        model_config["fusion"].update({
            "hierarchy": fusion_hierarchy,
            "hidden_dim": hierarchical_hidden_dim,
            "dropout": hierarchical_dropout,
            "use_attention": hierarchical_use_attention
        })
    elif fusion_type == "Gated Cross-Attention":
        model_config["fusion"].update({
            "hidden_dim": gca_hidden_dim,
            "num_heads": gca_num_heads,
            "dropout": gca_dropout,
            "gating_type": gca_gating_type
        })
    else:  # Simple Concatenation
        model_config["fusion"].update({
            "hidden_dim": concat_hidden_dim,
            "num_layers": concat_num_layers,
            "dropout": concat_dropout,
            "activation": concat_activation
        })
    
    # Add weighting-specific parameters
    if weight_method == "Task-Specific":
        model_config["weighting"].update({
            "hidden_dim": ts_hidden_dim,
            "temperature": ts_temperature,
            "use_gumbel": ts_use_gumbel
        })
    elif weight_method == "Complexity-Aware":
        model_config["weighting"].update({
            "complexity_metrics": ca_complexity_metrics,
            "normalization": ca_normalization,
            "learnable": ca_learnable
        })
    elif weight_method == "Uncertainty-Based":
        model_config["weighting"].update({
            "ensemble_size": ub_ensemble_size,
            "dropout_samples": ub_dropout_samples,
            "uncertainty_type": ub_uncertainty_type,
            "calibration": ub_calibration
        })
    
    # Add task-specific parameters
    if task_type == "Regression":
        model_config["task"].update({
            "loss": reg_loss,
            "output_activation": reg_output_activation,
            "metrics": reg_metrics
        })
    elif task_type == "Binary Classification":
        model_config["task"].update({
            "loss": bin_loss,
            "threshold": bin_threshold,
            "metrics": bin_metrics,
            "class_weights": bin_class_weights
        })
    elif task_type == "Multi-class Classification":
        model_config["task"].update({
            "loss": mc_loss,
            "num_classes": mc_num_classes,
            "metrics": mc_metrics,
            "class_weights": mc_class_weights
        })
    else:  # Multi-task
        model_config["task"].update({
            "num_tasks": mt_num_tasks,
            "task_types": mt_task_types,
            "loss_weighting": mt_loss_weighting,
            "shared_layers": mt_shared_layers
        })
    
    # Display model config as JSON
    st.json(json.dumps(model_config, indent=2))
    
    # Save configuration button
    col1, col2 = st.columns(2)
    with col1:
        config_name = st.text_input("Configuration Name", "h_caan_config")
    
    with col2:
        save_path = st.text_input("Save Path", "config/")
    
    if st.button("Save Configuration"):
        # This would save the configuration to a file
        st.success(f"Configuration saved as {save_path}{config_name}.json")
        st.session_state['model_config'] = model_config
        st.session_state['model_ready'] = True

    # Generate a diagram of the model architecture
    if st.checkbox("Show Model Architecture Diagram"):
        st.write("Model Architecture Diagram:")
        # This would be generated based on the configured model
        st.image("https://via.placeholder.com/800x400?text=H-CAAN+Model+Architecture", 
                caption="H-CAAN Model Architecture")
    
    # Navigation
    st.header("6. Next Steps")
    st.write("Once your model is configured, you can proceed to train and evaluate it.")
    
    # Only enable the next button if model is configured
    next_disabled = not st.session_state.get('model_ready', False)
    if st.button("Go to Training", disabled=next_disabled):
        st.session_state['current_page'] = 'training_page'
        st.experimental_rerun()

if __name__ == "__main__":
    model_page()