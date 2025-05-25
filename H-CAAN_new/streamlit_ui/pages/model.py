# streamlit_ui/pages/model.py
import streamlit as st
import json
import os
import sys
from typing import Dict, Any

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
from streamlit_ui.api import configure_model, get_model_templates
from streamlit_ui.components.charts import render_architecture_diagram

def get_default_templates() -> Dict[str, Any]:
    """Get default model templates"""
    return {
        "H-CAAN Standard": {
            "description": "Standard H-CAAN configuration for general molecular property prediction",
            "smiles_encoder": {
                "num_layers": 6,
                "num_heads": 8,
                "hidden_dim": 256,
                "use_chemical_attention": True
            },
            "ecfp_encoder": {
                "num_layers": 3,
                "hidden_dim": 256,
                "bidirectional": True
            },
            "gcn_encoder": {
                "num_layers": 4,
                "hidden_dim": 256,
                "dropout": 0.1
            },
            "mfbert_encoder": {
                "use_mfbert": True,
                "hidden_dim": 768,
                "freeze_backbone": False
            },
            "fusion": {
                "levels": ["Low-level (Feature)", "Mid-level (Semantic)", "High-level (Decision)"],
                "use_adaptive_gating": True,
                "use_multi_scale": True,
                "attention_heads": 8
            },
            "modal_importance": {
                "use_task_specific": True,
                "use_complexity_aware": True,
                "use_uncertainty": True
            },
            "general": {
                "dropout": 0.1,
                "output_dim": 128,
                "activation": "gelu"
            }
        },
        "H-CAAN Lite": {
            "description": "Lightweight version for faster training and inference",
            "smiles_encoder": {
                "num_layers": 3,
                "num_heads": 4,
                "hidden_dim": 128,
                "use_chemical_attention": False
            },
            "ecfp_encoder": {
                "num_layers": 2,
                "hidden_dim": 128,
                "bidirectional": True
            },
            "gcn_encoder": {
                "num_layers": 3,
                "hidden_dim": 128,
                "dropout": 0.1
            },
            "mfbert_encoder": {
                "use_mfbert": False,
                "hidden_dim": 0,
                "freeze_backbone": True
            },
            "fusion": {
                "levels": ["Low-level (Feature)", "High-level (Decision)"],
                "use_adaptive_gating": True,
                "use_multi_scale": False,
                "attention_heads": 4
            },
            "modal_importance": {
                "use_task_specific": False,
                "use_complexity_aware": True,
                "use_uncertainty": False
            },
            "general": {
                "dropout": 0.2,
                "output_dim": 64,
                "activation": "relu"
            }
        },
        "H-CAAN Heavy": {
            "description": "High-capacity model for complex molecular systems",
            "smiles_encoder": {
                "num_layers": 12,
                "num_heads": 16,
                "hidden_dim": 512,
                "use_chemical_attention": True
            },
            "ecfp_encoder": {
                "num_layers": 4,
                "hidden_dim": 512,
                "bidirectional": True
            },
            "gcn_encoder": {
                "num_layers": 6,
                "hidden_dim": 512,
                "dropout": 0.05
            },
            "mfbert_encoder": {
                "use_mfbert": True,
                "hidden_dim": 1024,
                "freeze_backbone": False
            },
            "fusion": {
                "levels": ["Low-level (Feature)", "Mid-level (Semantic)", "High-level (Decision)"],
                "use_adaptive_gating": True,
                "use_multi_scale": True,
                "attention_heads": 16
            },
            "modal_importance": {
                "use_task_specific": True,
                "use_complexity_aware": True,
                "use_uncertainty": True
            },
            "general": {
                "dropout": 0.05,
                "output_dim": 256,
                "activation": "gelu"
            }
        }
    }

def render_model_page():
    """Render model configuration page"""
    st.title("🔧 Model Configuration")
    
    # Initialize progress
    progress = 0
    
    # Check if we have existing configuration
    saved_config = get_step_data(WorkflowStep.MODEL_CONFIGURATION)
    has_config = saved_config is not None
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Template Selection",
        "Encoder Configuration", 
        "Fusion Configuration",
        "Task Configuration"
    ])
    
    with tab1:
        st.header("Select Model Template")
        
        templates = get_default_templates()
        template_names = list(templates.keys())
        
        # Template selection
        selected_template = st.selectbox(
            "Choose a pre-configured template:",
            template_names,
            index=0 if not has_config else template_names.index(
                saved_config.get("template_name", template_names[0])
            )
        )
        
        # Display template details
        template_info = templates[selected_template]
        st.info(template_info["description"])
        
        # Display template specifications
        with st.expander("Template Specifications"):
            st.json(template_info)
        
        # Custom configuration option
        use_custom = st.checkbox("Customize configuration", value=has_config)
        
        if not use_custom:
            # Use template as-is
            st.session_state['model_config'] = template_info
            st.session_state['template_name'] = selected_template
            progress = 100
        else:
            st.session_state['base_template'] = template_info
            st.session_state['template_name'] = selected_template
            progress = 25
    
    with tab2:
        st.header("Encoder Configuration")
        
        if use_custom or has_config:
            base_config = saved_config if has_config else st.session_state.get('base_template', templates[selected_template])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("SMILES Encoder (Transformer)")
                
                smiles_config = base_config.get("smiles_encoder", {})
                
                smiles_layers = st.slider(
                    "Number of Transformer layers",
                    min_value=1,
                    max_value=12,
                    value=smiles_config.get("num_layers", 6)
                )
                
                smiles_heads = st.slider(
                    "Number of attention heads",
                    min_value=1,
                    max_value=16,
                    value=smiles_config.get("num_heads", 8)
                )
                
                smiles_dim = st.number_input(
                    "Hidden dimension",
                    min_value=64,
                    max_value=1024,
                    step=64,
                    value=smiles_config.get("hidden_dim", 256)
                )
                
                use_chemical_attention = st.checkbox(
                    "Use chemical-aware attention",
                    value=smiles_config.get("use_chemical_attention", True)
                )
                
                st.subheader("ECFP Encoder (BiGRU)")
                
                ecfp_config = base_config.get("ecfp_encoder", {})
                
                ecfp_layers = st.slider(
                    "Number of BiGRU layers",
                    min_value=1,
                    max_value=6,
                    value=ecfp_config.get("num_layers", 3)
                )
                
                ecfp_dim = st.number_input(
                    "BiGRU hidden dimension",
                    min_value=64,
                    max_value=1024,
                    step=64,
                    value=ecfp_config.get("hidden_dim", 256)
                )
                
                ecfp_bidirectional = st.checkbox(
                    "Bidirectional",
                    value=ecfp_config.get("bidirectional", True)
                )
            
            with col2:
                st.subheader("Graph Encoder (GCN)")
                
                gcn_config = base_config.get("gcn_encoder", {})
                
                gcn_layers = st.slider(
                    "Number of GCN layers",
                    min_value=1,
                    max_value=8,
                    value=gcn_config.get("num_layers", 4)
                )
                
                gcn_dim = st.number_input(
                    "GCN hidden dimension",
                    min_value=64,
                    max_value=1024,
                    step=64,
                    value=gcn_config.get("hidden_dim", 256)
                )
                
                gcn_dropout = st.slider(
                    "GCN dropout rate",
                    min_value=0.0,
                    max_value=0.5,
                    step=0.05,
                    value=gcn_config.get("dropout", 0.1)
                )
                
                st.subheader("MFBERT Encoder")
                
                mfbert_config = base_config.get("mfbert_encoder", {})
                
                use_mfbert = st.checkbox(
                    "Use MFBERT encoder",
                    value=mfbert_config.get("use_mfbert", True)
                )
                
                if use_mfbert:
                    mfbert_dim = st.number_input(
                        "MFBERT output dimension",
                        min_value=128,
                        max_value=1024,
                        step=128,
                        value=mfbert_config.get("hidden_dim", 768)
                    )
                    
                    freeze_backbone = st.checkbox(
                        "Freeze MFBERT backbone",
                        value=mfbert_config.get("freeze_backbone", False)
                    )
                else:
                    mfbert_dim = 0
                    freeze_backbone = False
            
            # Update configuration
            encoder_config = {
                "smiles_encoder": {
                    "num_layers": smiles_layers,
                    "num_heads": smiles_heads,
                    "hidden_dim": smiles_dim,
                    "use_chemical_attention": use_chemical_attention
                },
                "ecfp_encoder": {
                    "num_layers": ecfp_layers,
                    "hidden_dim": ecfp_dim,
                    "bidirectional": ecfp_bidirectional
                },
                "gcn_encoder": {
                    "num_layers": gcn_layers,
                    "hidden_dim": gcn_dim,
                    "dropout": gcn_dropout
                },
                "mfbert_encoder": {
                    "use_mfbert": use_mfbert,
                    "hidden_dim": mfbert_dim,
                    "freeze_backbone": freeze_backbone
                }
            }
            
            st.session_state['encoder_config'] = encoder_config
            progress += 25
        else:
            st.info("Select 'Customize configuration' in Template Selection to modify encoders")
    
    with tab3:
        st.header("Fusion Configuration")
        
        if use_custom or has_config:
            base_config = saved_config if has_config else st.session_state.get('base_template', templates[selected_template])
            fusion_config = base_config.get("fusion", {})
            
            # Fusion levels
            st.subheader("Fusion Levels")
            
            fusion_levels = st.multiselect(
                "Select fusion levels:",
                ["Low-level (Feature)", "Mid-level (Semantic)", "High-level (Decision)"],
                default=fusion_config.get("levels", ["Low-level (Feature)", "Mid-level (Semantic)", "High-level (Decision)"])
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Fusion Mechanisms")
                
                use_adaptive_gating = st.checkbox(
                    "Use adaptive gating",
                    value=fusion_config.get("use_adaptive_gating", True)
                )
                
                use_multi_scale = st.checkbox(
                    "Use multi-scale attention",
                    value=fusion_config.get("use_multi_scale", True)
                )
                
                if use_multi_scale:
                    attention_heads = st.slider(
                        "Number of attention heads",
                        min_value=1,
                        max_value=16,
                        value=fusion_config.get("attention_heads", 8)
                    )
                else:
                    attention_heads = 1
                
                use_residual = st.checkbox(
                    "Use residual connections",
                    value=fusion_config.get("use_residual", True)
                )
            
            with col2:
                st.subheader("Modal Importance")
                
                modal_config = base_config.get("modal_importance", {})
                
                use_task_specific = st.checkbox(
                    "Task-specific weights",
                    value=modal_config.get("use_task_specific", True)
                )
                
                use_complexity_aware = st.checkbox(
                    "Molecular complexity-aware",
                    value=modal_config.get("use_complexity_aware", True)
                )
                
                use_uncertainty = st.checkbox(
                    "Uncertainty estimation",
                    value=modal_config.get("use_uncertainty", True)
                )
                
                if use_uncertainty:
                    uncertainty_method = st.selectbox(
                        "Uncertainty method:",
                        ["Monte Carlo Dropout", "Deep Ensembles", "Bayesian"],
                        index=0
                    )
                else:
                    uncertainty_method = None
            
            # General configuration
            st.subheader("General Configuration")
            
            general_config = base_config.get("general", {})
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                dropout_rate = st.slider(
                    "Global dropout rate",
                    min_value=0.0,
                    max_value=0.5,
                    step=0.05,
                    value=general_config.get("dropout", 0.1)
                )
            
            with col2:
                output_dim = st.number_input(
                    "Output dimension",
                    min_value=32,
                    max_value=512,
                    step=32,
                    value=general_config.get("output_dim", 128)
                )
            
            with col3:
                activation = st.selectbox(
                    "Activation function",
                    ["relu", "gelu", "swish", "mish"],
                    index=["relu", "gelu", "swish", "mish"].index(
                        general_config.get("activation", "gelu")
                    )
                )
            
            # Update fusion configuration
            fusion_config = {
                "fusion": {
                    "levels": fusion_levels,
                    "use_adaptive_gating": use_adaptive_gating,
                    "use_multi_scale": use_multi_scale,
                    "attention_heads": attention_heads,
                    "use_residual": use_residual
                },
                "modal_importance": {
                    "use_task_specific": use_task_specific,
                    "use_complexity_aware": use_complexity_aware,
                    "use_uncertainty": use_uncertainty,
                    "uncertainty_method": uncertainty_method
                },
                "general": {
                    "dropout": dropout_rate,
                    "output_dim": output_dim,
                    "activation": activation
                }
            }
            
            st.session_state['fusion_config'] = fusion_config
            progress += 25
        else:
            st.info("Select 'Customize configuration' in Template Selection to modify fusion settings")
    
    with tab4:
        st.header("Task Configuration")
        
        # Get data information
        data_info = get_step_data(WorkflowStep.DATA_PREPARATION)
        
        if data_info:
            st.info(f"Dataset: {data_info.get('dataset_name', 'Unknown')}")
            st.info(f"Target property: {data_info.get('property_col', 'Unknown')}")
        
        # Task type selection
        task_type = st.selectbox(
            "Task type:",
            ["Regression", "Binary Classification", "Multi-class Classification", "Multi-task"],
            index=0
        )
        
        task_config = {"type": task_type}
        
        if task_type == "Regression":
            col1, col2 = st.columns(2)
            
            with col1:
                loss_function = st.selectbox(
                    "Loss function:",
                    ["MSE", "MAE", "Huber", "LogCosh"],
                    index=0
                )
                task_config["loss"] = loss_function
            
            with col2:
                metrics = st.multiselect(
                    "Evaluation metrics:",
                    ["RMSE", "MAE", "R2", "Pearson", "Spearman"],
                    default=["RMSE", "MAE", "R2", "Pearson"]
                )
                task_config["metrics"] = metrics
                
        elif task_type == "Binary Classification":
            col1, col2 = st.columns(2)
            
            with col1:
                loss_function = st.selectbox(
                    "Loss function:",
                    ["Binary Cross-Entropy", "Focal Loss", "Weighted BCE"],
                    index=0
                )
                task_config["loss"] = loss_function
                
                class_weight = st.selectbox(
                    "Class weighting:",
                    ["None", "Balanced", "Custom"],
                    index=0
                )
                task_config["class_weight"] = class_weight
            
            with col2:
                metrics = st.multiselect(
                    "Evaluation metrics:",
                    ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC", "AUC-PR"],
                    default=["Accuracy", "Precision", "Recall", "F1", "AUC-ROC"]
                )
                task_config["metrics"] = metrics
                
        elif task_type == "Multi-class Classification":
            num_classes = st.number_input(
                "Number of classes:",
                min_value=3,
                max_value=100,
                value=3,
                step=1
            )
            task_config["num_classes"] = num_classes
            
            col1, col2 = st.columns(2)
            
            with col1:
                loss_function = st.selectbox(
                    "Loss function:",
                    ["Cross-Entropy", "Focal Loss", "Label Smoothing CE"],
                    index=0
                )
                task_config["loss"] = loss_function
            
            with col2:
                metrics = st.multiselect(
                    "Evaluation metrics:",
                    ["Accuracy", "Macro F1", "Micro F1", "Weighted F1", "Cohen's Kappa"],
                    default=["Accuracy", "Macro F1", "Weighted F1"]
                )
                task_config["metrics"] = metrics
                
        else:  # Multi-task
            num_tasks = st.number_input(
                "Number of tasks:",
                min_value=2,
                max_value=10,
                value=2,
                step=1
            )
            task_config["num_tasks"] = num_tasks
            
            task_types = []
            for i in range(num_tasks):
                task_type_i = st.selectbox(
                    f"Task {i+1} type:",
                    ["Regression", "Binary Classification"],
                    key=f"task_type_{i}"
                )
                task_types.append(task_type_i)
            
            task_config["task_types"] = task_types
            task_config["multi_task_strategy"] = st.selectbox(
                "Multi-task strategy:",
                ["Hard parameter sharing", "Soft parameter sharing", "Task-specific layers"],
                index=0
            )
        
        st.session_state['task_config'] = task_config
        progress += 25
    
    # Update progress
    update_step_progress(WorkflowStep.MODEL_CONFIGURATION, progress)
    
    # Save configuration button
    st.markdown("---")
    
    if progress >= 100:
        if st.button("💾 Save Model Configuration", type="primary", use_container_width=True):
            # Combine all configurations
            if use_custom:
                model_config = {
                    **st.session_state.get('encoder_config', {}),
                    **st.session_state.get('fusion_config', {}),
                    "task": st.session_state.get('task_config', {})
                }
            else:
                model_config = {
                    **st.session_state.get('model_config', {}),
                    "task": st.session_state.get('task_config', {})
                }
            
            # Add template info
            model_config["template_name"] = st.session_state.get('template_name')
            
            # Save configuration
            save_step_data(WorkflowStep.MODEL_CONFIGURATION, model_config)
            mark_step_completed(WorkflowStep.MODEL_CONFIGURATION)
            
            # Configure model (call API)
            success, message = configure_model(model_config)
            
            if success:
                st.success("✅ Model configuration saved successfully!")
                
                # Show architecture visualization
                st.subheader("Model Architecture Visualization")
                render_architecture_diagram(model_config)
                
                # Navigation
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🚀 Proceed to Training", type="primary", use_container_width=True):
                        navigate_to_next_step()
            else:
                st.error(f"Error configuring model: {message}")
    else:
        st.info("Complete all configuration steps to proceed")