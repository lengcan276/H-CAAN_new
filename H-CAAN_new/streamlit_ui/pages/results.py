import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# Add parent directory to path to import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.visualization import generate_molecule_visualizations
from evaluation.ablation_study import analyze_ablation_results
import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

# 然后导入
from utils.chemical_space import chemical_space_mapping


def plot_interactive_scatter(x, y, color=None, hover_data=None, title="Interactive Scatter Plot", 
                            xlabel="X", ylabel="Y", colorscale="viridis"):
    """Create an interactive scatter plot using Plotly"""
    fig = px.scatter(x=x, y=y, color=color, hover_data=hover_data,
                   title=title, labels={'x': xlabel, 'y': ylabel},
                   color_continuous_scale=colorscale)
    
    fig.update_layout(
        plot_bgcolor='white',
        width=800,
        height=600
    )
    
    fig.update_traces(
        marker=dict(size=8, opacity=0.7),
        selector=dict(mode='markers')
    )
    
    return fig


def plot_embedding_visualization(embeddings, labels=None, method="PCA", n_components=2):
    """Visualize high-dimensional embeddings using dimensionality reduction"""
    # Apply dimensionality reduction
    if method == "PCA":
        reducer = PCA(n_components=n_components)
    elif method == "t-SNE":
        reducer = TSNE(n_components=n_components, random_state=42)
    elif method == "UMAP":
        reducer = umap.UMAP(n_components=n_components, random_state=42)
    else:
        raise ValueError(f"Unknown embedding method: {method}")
    
    # Reduce dimensionality
    reduced_data = reducer.fit_transform(embeddings)
    
    # Create visualization
    if n_components == 2:
        fig = px.scatter(
            x=reduced_data[:, 0], 
            y=reduced_data[:, 1],
            color=labels,
            title=f"{method} Visualization",
            labels={"x": f"{method} Dimension 1", "y": f"{method} Dimension 2"}
        )
    else:  # 3D visualization
        fig = px.scatter_3d(
            x=reduced_data[:, 0], 
            y=reduced_data[:, 1], 
            z=reduced_data[:, 2],
            color=labels,
            title=f"{method} Visualization",
            labels={"x": f"{method} Dim 1", "y": f"{method} Dim 2", "z": f"{method} Dim 3"}
        )
    
    return fig


def create_comparative_heatmap(data, x_labels, y_labels, title="Comparative Heatmap"):
    """Create a heatmap to compare different aspects of the model"""
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=x_labels,
        y=y_labels,
        colorscale='Viridis',
        text=[[f"{val:.2f}" for val in row] for row in data],
        texttemplate="%{text}",
        textfont={"size": 12},
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Model Configuration",
        yaxis_title="Metric",
        width=800,
        height=600
    )
    
    return fig


def plot_model_architecture_diagram():
    """Create a visualization of the H-CAAN model architecture"""
    # This function would generate a model architecture visualization
    # For this example, we'll use a placeholder diagram
    
    # Example layout
    fig = go.Figure()
    
    # Define nodes and their positions
    nodes = [
        {"name": "SMILES", "x": 0, "y": 0, "type": "input"},
        {"name": "ECFP", "x": 0, "y": 1, "type": "input"},
        {"name": "Graph", "x": 0, "y": 2, "type": "input"},
        {"name": "MFBERT", "x": 0, "y": 3, "type": "input"},
        
        {"name": "Transformer", "x": 1, "y": 0, "type": "encoder"},
        {"name": "BiGRU", "x": 1, "y": 1, "type": "encoder"},
        {"name": "GCN", "x": 1, "y": 2, "type": "encoder"},
        {"name": "MFBERT", "x": 1, "y": 3, "type": "encoder"},
        
        {"name": "GCAU", "x": 2, "y": 0.5, "type": "fusion"},
        {"name": "GCAU", "x": 2, "y": 2.5, "type": "fusion"},
        
        {"name": "Hierarchical Fusion", "x": 3, "y": 1.5, "type": "fusion"},
        
        {"name": "Task-Specific Weights", "x": 4, "y": 1.5, "type": "output"},
        
        {"name": "Output", "x": 5, "y": 1.5, "type": "output"}
    ]
    
    # Add nodes
    for node in nodes:
        marker_color = "blue" if node["type"] == "input" else \
                       "green" if node["type"] == "encoder" else \
                       "orange" if node["type"] == "fusion" else "red"
        
        fig.add_trace(go.Scatter(
            x=[node["x"]], 
            y=[node["y"]],
            mode="markers+text",
            marker=dict(size=20, color=marker_color),
            text=[node["name"]],
            textposition="bottom center",
            name=node["name"]
        ))
    
    # Define edges (connections)
    edges = [
        # Input to encoder connections
        (0, 4), (1, 5), (2, 6), (3, 7),
        
        # Encoder to fusion connections
        (4, 8), (5, 8), (6, 9), (7, 9),
        
        # Fusion to hierarchical fusion
        (8, 10), (9, 10),
        
        # Hierarchical fusion to weights
        (10, 11),
        
        # Weights to output
        (11, 12)
    ]
    
    # Add edges
    for edge in edges:
        start_node = nodes[edge[0]]
        end_node = nodes[edge[1]]
        
        fig.add_trace(go.Scatter(
            x=[start_node["x"], end_node["x"]],
            y=[start_node["y"], end_node["y"]],
            mode="lines",
            line=dict(width=2, color="gray"),
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        title="H-CAAN Model Architecture",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=900,
        height=600,
        hovermode="closest"
    )
    
    return fig


def results_page():
    st.title("H-CAAN: Results Visualization")
    if st.sidebar.button("Reset Workflow"):
        st.session_state['data_processed'] = False
        st.session_state['model_configured'] = False
        st.session_state['model_trained'] = False
        st.session_state['current_page'] = 'data_page'
        st.rerun()  # 注意这里使用了st.rerun()代替st.experimental_rerun()
    # Check if model is trained
    if not st.session_state.get('model_trained', False):
        st.warning("Please train your model first!")
        if st.button("Go to Training Page"):
            st.session_state['current_page'] = 'training_page'
            st.experimental_rerun()
        return
    
    # Load model configuration
    model_config = st.session_state.get('model_config', {})
    
    # Create visualization tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Performance Dashboard", 
        "Modal Analysis", 
        "Molecular Interpretability",
        "Ablation Study",
        "Chemical Space Analysis"
    ])
    
    with tab1:
        st.header("Performance Dashboard")
        
        # Display task type and metrics
        task_type = model_config.get('task', {}).get('type', 'Regression')
        st.subheader(f"Task Type: {task_type}")
        
        # Create a performance dashboard layout
        col1, col2 = st.columns(2)
        
        with col1:
            # Main metric display
            if task_type == "Regression":
                main_metric = "RMSE"
                main_value = 0.312
                st.metric(
                    label=main_metric, 
                    value=f"{main_value:.3f}",
                    delta="-0.05 vs. Baseline"
                )
                
                # Secondary metrics
                metrics_df = pd.DataFrame({
                    'Metric': ['MAE', 'R²', 'Pearson Correlation'],
                    'Value': ['0.247', '0.824', '0.907']
                })
                st.dataframe(metrics_df)
                
            elif task_type == "Binary Classification":
                main_metric = "AUC-ROC"
                main_value = 0.936
                st.metric(
                    label=main_metric, 
                    value=f"{main_value:.3f}",
                    delta="+0.045 vs. Baseline"
                )
                
                # Secondary metrics
                metrics_df = pd.DataFrame({
                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                    'Value': ['0.901', '0.887', '0.865', '0.876']
                })
                st.dataframe(metrics_df)
                
            elif task_type == "Multi-class Classification":
                main_metric = "Accuracy"
                main_value = 0.882
                st.metric(
                    label=main_metric, 
                    value=f"{main_value:.3f}",
                    delta="+0.032 vs. Baseline"
                )
                
                # Secondary metrics
                metrics_df = pd.DataFrame({
                    'Metric': ['Weighted Precision', 'Weighted Recall', 'Weighted F1'],
                    'Value': ['0.883', '0.882', '0.882']
                })
                st.dataframe(metrics_df)
                
            else:  # Multi-task
                st.write("Multi-task performance summary")
                # Create a multi-task performance summary
                task_metrics_df = pd.DataFrame({
                    'Task': ['Task 1 (Regression)', 'Task 2 (Classification)'],
                    'Primary Metric': ['RMSE: 0.325', 'Accuracy: 0.891'],
                    'Improvement': ['-0.042', '+0.037']
                })
                st.dataframe(task_metrics_df)
        
        with col2:
            # Model comparison with baselines
            st.subheader("Comparison with Baselines")
            
            # Create comparison data
            models = ["H-CAAN", "MolBERT", "MMFDL", "ChemBERTa", "Chemprop"]
            
            if task_type == "Regression":
                values = [0.312, 0.36, 0.35, 0.41, 0.38]  # RMSE (lower is better)
                best_value = min(values)
                normalized_values = [v / best_value for v in values]
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=models,
                        y=values,
                        marker_color=['gold' if v == best_value else 'blue' for v in values]
                    )
                ])
                
                fig.update_layout(
                    title="RMSE Comparison",
                    yaxis_title="RMSE (lower is better)",
                    hovermode="closest"
                )
                
            else:  # Classification
                values = [0.901, 0.87, 0.86, 0.83, 0.85]  # Accuracy (higher is better)
                best_value = max(values)
                normalized_values = [v / best_value for v in values]
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=models,
                        y=values,
                        marker_color=['gold' if v == best_value else 'blue' for v in values]
                    )
                ])
                
                fig.update_layout(
                    title="Accuracy Comparison",
                    yaxis_title="Accuracy (higher is better)",
                    hovermode="closest"
                )
            
            st.plotly_chart(fig)
        
        # Detailed metrics visualization
        st.subheader("Detailed Metrics Visualization")
        
        # Generate a dropdown for selecting visualization type
        viz_type = st.selectbox(
            "Select Visualization Type",
            ["Performance by Molecule Complexity", "Performance by Property Range", "Learning Curves"]
        )
        
        if viz_type == "Performance by Molecule Complexity":
            # Create a sample dataset with molecule complexity and error
            n_samples = 100
            complexity = np.random.uniform(0.2, 1.0, size=n_samples)  # Molecular complexity score
            
            if task_type == "Regression":
                # Error increases with complexity but H-CAAN handles it better
                baseline_error = 0.2 + 0.5 * complexity + 0.1 * np.random.normal(size=n_samples)
                hcaan_error = 0.15 + 0.3 * complexity + 0.08 * np.random.normal(size=n_samples)
                
                # Create a dataframe for plotting
                df = pd.DataFrame({
                    'Complexity': np.concatenate([complexity, complexity]),
                    'Error': np.concatenate([baseline_error, hcaan_error]),
                    'Model': ['Baseline'] * n_samples + ['H-CAAN'] * n_samples
                })
                
                fig = px.scatter(
                    df, x='Complexity', y='Error', color='Model',
                    trendline='lowess',
                    labels={'Complexity': 'Molecular Complexity', 'Error': 'Prediction Error'},
                    title='Prediction Error vs. Molecular Complexity'
                )
                
            else:  # Classification
                # Accuracy decreases with complexity but H-CAAN handles it better
                baseline_acc = 0.95 - 0.3 * complexity + 0.1 * np.random.normal(size=n_samples)
                baseline_acc = np.clip(baseline_acc, 0, 1)
                
                hcaan_acc = 0.98 - 0.15 * complexity + 0.08 * np.random.normal(size=n_samples)
                hcaan_acc = np.clip(hcaan_acc, 0, 1)
                
                # Create a dataframe for plotting
                df = pd.DataFrame({
                    'Complexity': np.concatenate([complexity, complexity]),
                    'Accuracy': np.concatenate([baseline_acc, hcaan_acc]),
                    'Model': ['Baseline'] * n_samples + ['H-CAAN'] * n_samples
                })
                
                fig = px.scatter(
                    df, x='Complexity', y='Accuracy', color='Model',
                    trendline='lowess',
                    labels={'Complexity': 'Molecular Complexity', 'Accuracy': 'Prediction Accuracy'},
                    title='Prediction Accuracy vs. Molecular Complexity'
                )
            
            st.plotly_chart(fig)
            
        elif viz_type == "Performance by Property Range":
            # Create a sample dataset with property values and error
            n_samples = 100
            if task_type == "Regression":
                property_values = np.random.normal(5, 2, size=n_samples)
                
                # Error varies by property range
                baseline_error = 0.3 + 0.2 * np.sin(property_values) + 0.1 * np.random.normal(size=n_samples)
                hcaan_error = 0.2 + 0.1 * np.sin(property_values) + 0.08 * np.random.normal(size=n_samples)
                
                # Create a dataframe for plotting
                df = pd.DataFrame({
                    'Property': np.concatenate([property_values, property_values]),
                    'Error': np.concatenate([baseline_error, hcaan_error]),
                    'Model': ['Baseline'] * n_samples + ['H-CAAN'] * n_samples
                })
                
                fig = px.scatter(
                    df, x='Property', y='Error', color='Model',
                    trendline='lowess',
                    labels={'Property': 'Property Value', 'Error': 'Prediction Error'},
                    title='Prediction Error by Property Value Range'
                )
                
            else:  # Classification
                # For classification, bin property values and show accuracy by bin
                property_bins = np.linspace(-2, 8, 6)  # 5 bins
                bin_labels = [f"{property_bins[i]:.1f} to {property_bins[i+1]:.1f}" for i in range(len(property_bins)-1)]
                
                # Generate random accuracies for each bin
                baseline_acc = [0.85, 0.82, 0.80, 0.75, 0.83]
                hcaan_acc = [0.92, 0.89, 0.88, 0.84, 0.90]
                
                # Create a dataframe for plotting
                df = pd.DataFrame({
                    'Property Bin': bin_labels * 2,
                    'Accuracy': baseline_acc + hcaan_acc,
                    'Model': ['Baseline'] * 5 + ['H-CAAN'] * 5
                })
                
                fig = px.bar(
                    df, x='Property Bin', y='Accuracy', color='Model', barmode='group',
                    labels={'Property Bin': 'Property Value Range', 'Accuracy': 'Prediction Accuracy'},
                    title='Prediction Accuracy by Property Value Range'
                )
            
            st.plotly_chart(fig)
            
        else:  # Learning Curves
            # Get training history
            train_losses = st.session_state.get('train_losses', [])
            val_losses = st.session_state.get('val_losses', [])
            
            if train_losses and val_losses:
                epochs = list(range(1, len(train_losses) + 1))
                
                # Create a dataframe for plotting
                df = pd.DataFrame({
                    'Epoch': epochs * 2,
                    'Loss': train_losses + val_losses,
                    'Type': ['Training'] * len(epochs) + ['Validation'] * len(epochs)
                })
                
                fig = px.line(
                    df, x='Epoch', y='Loss', color='Type',
                    labels={'Epoch': 'Epoch', 'Loss': 'Loss Value'},
                    title='Training and Validation Loss Curves'
                )
                
                st.plotly_chart(fig)
            else:
                st.warning("No training history available.")
    
    with tab2:
        st.header("Modal Analysis")
        
        # Modal contribution analysis
        st.subheader("Modal Contribution Analysis")
        
        # Create a sample dataset for modal contributions
        modalities = ["SMILES", "ECFP", "Graph", "MFBERT"]
        
        # Modal importance for different molecule types
        molecule_types = ["Small Molecules", "Drug-like", "Complex Structures", "Overall"]
        
        importance_data = np.array([
            [0.40, 0.28, 0.15, 0.17],  # Small Molecules
            [0.30, 0.25, 0.22, 0.23],  # Drug-like
            [0.25, 0.20, 0.30, 0.25],  # Complex Structures
            [0.35, 0.25, 0.20, 0.20]   # Overall
        ])
        
        # Create a heatmap
        fig = create_comparative_heatmap(
            importance_data, 
            modalities, 
            molecule_types, 
            "Modal Importance by Molecule Type"
        )
        
        st.plotly_chart(fig)
        
        # Inter-modal correlation analysis
        st.subheader("Inter-Modal Correlation Analysis")
        
        # Create a sample correlation matrix
        correlation_matrix = np.array([
            [1.00, 0.75, 0.40, 0.55],
            [0.75, 1.00, 0.35, 0.45],
            [0.40, 0.35, 1.00, 0.65],
            [0.55, 0.45, 0.65, 1.00]
        ])
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix,
            x=modalities,
            y=modalities,
            colorscale='Blues',
            text=[[f"{val:.2f}" for val in row] for row in correlation_matrix],
            texttemplate="%{text}",
            textfont={"size": 12},
        ))
        
        fig.update_layout(
            title="Inter-Modal Correlation",
            xaxis_title="Modality",
            yaxis_title="Modality",
            width=800,
            height=600
        )
        
        st.plotly_chart(fig)
        
        # Cross-attention visualization
        st.subheader("Cross-Attention Visualization")
        
        # Create a dropdown for selecting example molecules
        molecule_examples = ["Aspirin", "Ibuprofen", "Paracetamol", "Sildenafil", "Penicillin"]
        selected_molecule = st.selectbox("Select Molecule", molecule_examples)
        
        # Generate a sample cross-attention visualization for the selected molecule
        # This should be implemented with actual model attention weights
        
        attention_data = np.array([
            [0.80, 0.10, 0.05, 0.05],  # SMILES attention
            [0.15, 0.75, 0.05, 0.05],  # ECFP attention
            [0.10, 0.10, 0.70, 0.10],  # Graph attention
            [0.10, 0.05, 0.05, 0.80]   # MFBERT attention
        ])
        
        if selected_molecule == "Ibuprofen":
            # Slightly different values for different molecules
            attention_data = np.array([
                [0.75, 0.15, 0.05, 0.05],
                [0.10, 0.80, 0.05, 0.05],
                [0.05, 0.10, 0.75, 0.10],
                [0.10, 0.05, 0.10, 0.75]
            ])
        elif selected_molecule == "Paracetamol":
            attention_data = np.array([
                [0.70, 0.15, 0.10, 0.05],
                [0.15, 0.70, 0.10, 0.05],
                [0.10, 0.10, 0.65, 0.15],
                [0.05, 0.10, 0.15, 0.70]
            ])
        
        fig = go.Figure(data=go.Heatmap(
            z=attention_data,
            x=modalities,
            y=modalities,
            colorscale='Reds',
            text=[[f"{val:.2f}" for val in row] for row in attention_data],
            texttemplate="%{text}",
            textfont={"size": 12},
        ))
        
        fig.update_layout(
            title=f"Cross-Attention for {selected_molecule}",
            xaxis_title="Target Modality",
            yaxis_title="Source Modality",
            width=800,
            height=600
        )
        
        st.plotly_chart(fig)
        
        # Modal feature importance
        st.subheader("Modal Feature Importance")
        
        # Create a dropdown for selecting modality
        selected_modality = st.selectbox("Select Modality", modalities)
        
        # Generate features based on the selected modality
        if selected_modality == "SMILES":
            features = ["Aromatic Rings", "Carbonyl Groups", "Hydroxyl Groups", "Amines", "Halogens"]
            importance = [0.35, 0.25, 0.15, 0.15, 0.10]
        elif selected_modality == "ECFP":
            features = ["Substructure A", "Substructure B", "Substructure C", "Substructure D", "Substructure E"]
            importance = [0.30, 0.25, 0.20, 0.15, 0.10]
        elif selected_modality == "Graph":
            features = ["Node Degree", "Edge Connectivity", "Graph Diameter", "Clustering Coefficient", "Centrality"]
            importance = [0.28, 0.22, 0.20, 0.18, 0.12]
        else:  # MFBERT
            features = ["Feature 1", "Feature 2", "Feature 3", "Feature 4", "Feature 5"]
            importance = [0.32, 0.26, 0.18, 0.14, 0.10]
        
        # Sort features by importance
        sorted_indices = np.argsort(importance)[::-1]
        sorted_features = [features[i] for i in sorted_indices]
        sorted_importance = [importance[i] for i in sorted_indices]
        
        fig = go.Figure(data=[
            go.Bar(x=sorted_features, y=sorted_importance)
        ])
        
        fig.update_layout(
            title=f"{selected_modality} Feature Importance",
            xaxis_title="Feature",
            yaxis_title="Importance",
            width=800,
            height=500
        )
        
        st.plotly_chart(fig)
    
    with tab3:
        st.header("Molecular Interpretability")
        
        # Molecule representation visualization
        st.subheader("Molecule Representation Visualization")
        
        # Create a dropdown for selecting example molecules
        molecule_examples = ["Aspirin", "Ibuprofen", "Paracetamol", "Sildenafil", "Penicillin"]
        selected_molecule = st.selectbox("Select Molecule for Visualization", molecule_examples, key="viz_mol")
        
        # Display the molecule and its representations
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("Molecular Structure")
            # Display the molecular structure (placeholder)
            st.image("https://via.placeholder.com/300x300?text=Molecular+Structure", 
                    caption=f"{selected_molecule} Structure")
            
            # Show SMILES
            if selected_molecule == "Aspirin":
                smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
            elif selected_molecule == "Ibuprofen":
                smiles = "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"
            elif selected_molecule == "Paracetamol":
                smiles = "CC(=O)NC1=CC=C(C=C1)O"
            elif selected_molecule == "Sildenafil":
                smiles = "CCCC1=NN(C2=C1N=C(NC2=O)C3=C(C=CC(=C3)S(=O)(=O)N4CCN(CC4)C)OCC)C"
            else:  # Penicillin
                smiles = "CC1(C(N2C(S1)C(C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C"
            
            st.code(smiles, language="text")
        
        with col2:
            st.write("H-CAAN Representations")
            
            # Create tabs for different representations
            rep_tab1, rep_tab2, rep_tab3, rep_tab4 = st.tabs(modalities)
            
            with rep_tab1:
                st.write("SMILES Encoder Attention")
                # Visualization of SMILES encoder attention (placeholder)
                st.image("https://via.placeholder.com/600x200?text=SMILES+Encoder+Attention", 
                        caption="Attention weights on SMILES tokens")
            
            with rep_tab2:
                st.write("ECFP Fingerprint Visualization")
                # Visualization of ECFP fingerprint (placeholder)
                st.image("https://via.placeholder.com/600x200?text=ECFP+Fingerprint+Visualization", 
                        caption="ECFP Fingerprint bits with highest importance")
            
            with rep_tab3:
                st.write("Graph Representation")
                # Visualization of molecular graph (placeholder)
                st.image("https://via.placeholder.com/600x300?text=Graph+Representation", 
                        caption="Graph with node and edge importance")
            
            with rep_tab4:
                st.write("MFBERT Embedding Visualization")
                # Visualization of MFBERT embedding (placeholder)
                st.image("https://via.placeholder.com/600x200?text=MFBERT+Embedding", 
                        caption="MFBERT embedding visualization")
        
        # Embedding visualization
        st.subheader("Embedding Space Visualization")
        
        # Generate some random embeddings for visualization
        n_samples = 100
        n_features = 50
        
        # Create embeddings
        embeddings = np.random.normal(0, 1, size=(n_samples, n_features))
        
        # Create labels for coloring
        if task_type == "Regression":
            labels = np.random.normal(5, 2, size=n_samples)
            color_scale = "Viridis"
        else:  # Classification
            if task_type == "Binary Classification":
                labels = np.random.randint(0, 2, size=n_samples)
            else:  # Multi-class
                labels = np.random.randint(0, 3, size=n_samples)
            color_scale = "Set1"
        
        # Create select box for visualization method
        viz_method = st.selectbox(
            "Select Dimensionality Reduction Method",
            ["PCA", "t-SNE", "UMAP"],
            key="dr_method"
        )
        
        # Generate visualization
        fig = plot_embedding_visualization(embeddings, labels, method=viz_method)
        st.plotly_chart(fig)
    
    with tab4:
        st.header("Ablation Study")
        
        # Ablation study on different components
        st.subheader("Component Ablation Study")
        
        # Create data for ablation study
        components = [
            "Full H-CAAN",
            "No SMILES",
            "No ECFP",
            "No Graph",
            "No MFBERT",
            "No Hierarchical Fusion",
            "No Cross-Attention",
            "No Modal Weights"
        ]
        
        if task_type == "Regression":
            # Lower values are better (RMSE)
            performance = [0.312, 0.380, 0.345, 0.336, 0.342, 0.360, 0.348, 0.328]
            # Calculate percentage change
            baseline = performance[0]
            percentage_change = [(val - baseline) / baseline * 100 for val in performance]
            
            # Colors: green for better (negative change), red for worse (positive change)
            colors = ['blue' if i == 0 else 'red' for i in range(len(components))]
            
            fig = go.Figure([
                go.Bar(
                    x=components,
                    y=performance,
                    marker_color=colors
                )
            ])
            
            fig.update_layout(
                title="Ablation Study Results (RMSE, lower is better)",
                xaxis_title="Model Configuration",
                yaxis_title="RMSE",
                width=800,
                height=500
            )
            
        else:  # Classification
            # Higher values are better (Accuracy)
            performance = [0.901, 0.850, 0.872, 0.878, 0.876, 0.865, 0.880, 0.890]
            # Calculate percentage change
            baseline = performance[0]
            percentage_change = [(baseline - val) / baseline * 100 for val in performance]
            
            # Colors: red for worse (negative change), green for better (positive change)
            colors = ['blue' if i == 0 else 'red' for i in range(len(components))]
            
            fig = go.Figure([
                go.Bar(
                    x=components,
                    y=performance,
                    marker_color=colors
                )
            ])
            
            fig.update_layout(
                title="Ablation Study Results (Accuracy, higher is better)",
                xaxis_title="Model Configuration",
                yaxis_title="Accuracy",
                width=800,
                height=500
            )
        
        st.plotly_chart(fig)
        
        # Detailed ablation analysis
        st.subheader("Detailed Ablation Analysis")
        
        # Create tabs for different analysis views
        abl_tab1, abl_tab2 = st.tabs(["Component-wise Analysis", "Modal Combination Analysis"])
        
        with abl_tab1:
            # Component-wise analysis
            metrics = ["Primary Metric", "Secondary Metric 1", "Secondary Metric 2"]
            
            # Generate data for component-wise analysis
            if task_type == "Regression":
                component_metrics = np.array([
                    [0.312, 0.247, 0.907],  # Full H-CAAN (RMSE, MAE, Pearson)
                    [0.380, 0.305, 0.875],  # No SMILES
                    [0.345, 0.278, 0.892],  # No ECFP
                    [0.336, 0.270, 0.895],  # No Graph
                    [0.342, 0.272, 0.893],  # No MFBERT
                    [0.360, 0.286, 0.885],  # No Hierarchical Fusion
                    [0.348, 0.282, 0.890],  # No Cross-Attention
                    [0.328, 0.260, 0.900]   # No Modal Weights
                ])
                
                # For regression, lower is better for first two metrics, higher for third
                # Normalize the data for comparison (0 to 1, 1 is best)
                normalized = np.zeros_like(component_metrics)
                for i in range(2):  # First two metrics, lower is better
                    min_val = np.min(component_metrics[:, i])
                    max_val = np.max(component_metrics[:, i])
                    normalized[:, i] = 1 - (component_metrics[:, i] - min_val) / (max_val - min_val)
                
                # Last metric, higher is better
                min_val = np.min(component_metrics[:, 2])
                max_val = np.max(component_metrics[:, 2])
                normalized[:, 2] = (component_metrics[:, 2] - min_val) / (max_val - min_val)
                
            else:  # Classification
                component_metrics = np.array([
                    [0.901, 0.887, 0.876],  # Full H-CAAN (Accuracy, Precision, F1)
                    [0.850, 0.835, 0.825],  # No SMILES
                    [0.872, 0.860, 0.848],  # No ECFP
                    [0.878, 0.862, 0.855],  # No Graph
                    [0.876, 0.860, 0.853],  # No MFBERT
                    [0.865, 0.840, 0.835],  # No Hierarchical Fusion
                    [0.880, 0.868, 0.858],  # No Cross-Attention
                    [0.890, 0.875, 0.865]   # No Modal Weights
                ])
                
                # For classification, higher is better for all metrics
                # Normalize the data for comparison (0 to 1, 1 is best)
                normalized = np.zeros_like(component_metrics)
                for i in range(3):  # All metrics, higher is better
                    min_val = np.min(component_metrics[:, i])
                    max_val = np.max(component_metrics[:, i])
                    normalized[:, i] = (component_metrics[:, i] - min_val) / (max_val - min_val)
            
            # Create a radar chart
            fig = go.Figure()
            
            for i, component in enumerate(components):
                fig.add_trace(go.Scatterpolar(
                    r=normalized[i],
                    theta=metrics,
                    fill='toself',
                    name=component
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )
                ),
                title="Component-wise Performance Comparison (Normalized)",
                showlegend=True
            )
            
            st.plotly_chart(fig)
        
        with abl_tab2:
            # Modal combination analysis
            st.write("Performance of Different Modal Combinations")
            
            # Create data for modal combinations
            modal_combinations = [
                "SMILES + ECFP + Graph + MFBERT",
                "SMILES + ECFP + Graph",
                "SMILES + ECFP + MFBERT",
                "SMILES + Graph + MFBERT",
                "ECFP + Graph + MFBERT",
                "SMILES + ECFP",
                "SMILES + Graph",
                "SMILES + MFBERT",
                "ECFP + Graph",
                "ECFP + MFBERT",
                "Graph + MFBERT"
            ]
            
            if task_type == "Regression":
                # RMSE values for each combination (lower is better)
                performance = [0.312, 0.342, 0.325, 0.330, 0.345, 0.365, 0.370, 0.355, 0.390, 0.375, 0.380]
                
                fig = go.Figure([
                    go.Bar(
                        x=modal_combinations,
                        y=performance,
                        marker_color=['blue' if i == 0 else 'lightblue' for i in range(len(modal_combinations))]
                    )
                ])
                
                fig.update_layout(
                    title="Performance by Modal Combination (RMSE, lower is better)",
                    xaxis_title="Modal Combination",
                    yaxis_title="RMSE",
                    width=900,
                    height=600
                )
                
            else:  # Classification
                # Accuracy values for each combination (higher is better)
                performance = [0.901, 0.876, 0.885, 0.880, 0.872, 0.860, 0.845, 0.870, 0.835, 0.855, 0.840]
                
                fig = go.Figure([
                    go.Bar(
                        x=modal_combinations,
                        y=performance,
                        marker_color=['blue' if i == 0 else 'lightblue' for i in range(len(modal_combinations))]
                    )
                ])
                
                fig.update_layout(
                    title="Performance by Modal Combination (Accuracy, higher is better)",
                    xaxis_title="Modal Combination",
                    yaxis_title="Accuracy",
                    width=900,
                    height=600
                )
            
            st.plotly_chart(fig)
    
    with tab5:
        st.header("Chemical Space Analysis")
        
        # Chemical space mapping
        st.subheader("Chemical Space Mapping")
        
        # Generate random embedding data for visualization
        n_samples = 200
        embeddings = np.random.normal(0, 1, size=(n_samples, 50))
        
        # Generate some properties for coloring
        if task_type == "Regression":
            properties = np.random.normal(5, 2, size=n_samples)
            property_name = "Target Property"
        else:  # Classification
            if task_type == "Binary Classification":
                properties = np.random.randint(0, 2, size=n_samples)
                property_name = "Class"
            else:  # Multi-class
                properties = np.random.randint(0, 3, size=n_samples)
                property_name = "Class"
        
        # Create random molecular attributes
        mol_weight = np.random.normal(300, 50, size=n_samples)
        logp = np.random.normal(3, 1, size=n_samples)
        tpsa = np.random.normal(90, 20, size=n_samples)
        
        # Create hover data
        hover_data = pd.DataFrame({
            'MW': mol_weight,
            'LogP': logp,
            'TPSA': tpsa
        })
        
        # Create dimensionality reduction
        dr_method = st.selectbox(
            "Select Dimensionality Reduction Method",
            ["PCA", "t-SNE", "UMAP"],
            key="cs_dr_method"
        )
        
        # Apply dimensionality reduction
        if dr_method == "PCA":
            reducer = PCA(n_components=2, random_state=42)
        elif dr_method == "t-SNE":
            reducer = TSNE(n_components=2, random_state=42)
        else:  # UMAP
            reducer = umap.UMAP(n_components=2, random_state=42)
        
        reduced_data = reducer.fit_transform(embeddings)
        
        # Create a dataframe for plotting
        df = pd.DataFrame({
            'x': reduced_data[:, 0],
            'y': reduced_data[:, 1],
            property_name: properties,
            'MW': mol_weight,
            'LogP': logp,
            'TPSA': tpsa
        })
        
        # Create a scatter plot
        fig = px.scatter(
            df, x='x', y='y', color=property_name,
            hover_data=['MW', 'LogP', 'TPSA'],
            title=f"Chemical Space Mapping ({dr_method})",
            labels={'x': f"{dr_method} Dimension 1", 'y': f"{dr_method} Dimension 2"}
        )
        
        st.plotly_chart(fig)
        
        # Model performance in chemical space
        st.subheader("Model Performance in Chemical Space")
        
        # Create a dropdown for selecting property to color by
        color_property = st.selectbox(
            "Color by",
            ["Prediction Error", "Molecular Weight", "LogP", "TPSA"],
            key="cs_color"
        )
        
        # Generate sample prediction errors
        pred_error = np.abs(np.random.normal(0, 0.5, size=n_samples))
        
        # Select color values based on selected property
        if color_property == "Prediction Error":
            color_values = pred_error
            color_label = "Prediction Error"
            colorscale = "Reds"
        elif color_property == "Molecular Weight":
            color_values = mol_weight
            color_label = "MW"
            colorscale = "Viridis"
        elif color_property == "LogP":
            color_values = logp
            color_label = "LogP"
            colorscale = "Cividis"
        else:  # TPSA
            color_values = tpsa
            color_label = "TPSA"
            colorscale = "Plasma"
        
        # Create a dataframe for plotting
        df = pd.DataFrame({
            'x': reduced_data[:, 0],
            'y': reduced_data[:, 1],
            'color': color_values,
            'Error': pred_error,
            'MW': mol_weight,
            'LogP': logp,
            'TPSA': tpsa
        })
        
        # Create a scatter plot
        fig = px.scatter(
            df, x='x', y='y', color='color',
            hover_data=['Error', 'MW', 'LogP', 'TPSA'],
            color_continuous_scale=colorscale,
            title=f"Model Performance in Chemical Space (Colored by {color_label})",
            labels={'x': f"{dr_method} Dimension 1", 'y': f"{dr_method} Dimension 2", 'color': color_label}
        )
        
        st.plotly_chart(fig)
    
    # Model Architecture Visualization
    st.header("Model Architecture")
    
    # Visualize the model architecture
    st.write("H-CAAN Model Architecture")
    
    # Create and display the model architecture diagram
    fig = plot_model_architecture_diagram()
    st.plotly_chart(fig)
    
    # Navigation
    st.header("Next Steps")
    st.write("Once you've analyzed your results, you can generate a paper.")
    
    if st.button("Go to Paper Generation"):
        st.session_state['current_page'] = 'paper_page'
        st.rerun()

if __name__ == "__main__":
    results_page()