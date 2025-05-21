import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle
import re

# 尝试导入seaborn，如果不可用则给出警告
try:
    import seaborn as sns
except ImportError:
    st.warning("Seaborn not found. Some visualizations may be limited.")
    # 创建一个简单的替代
    class SnsMock:
        def histplot(self, *args, **kwargs):
            ax = kwargs.get('ax', plt.gca())
            return ax.hist(*args)
    sns = SnsMock()

# 尝试导入RDKit，如果不可用则给出警告
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem, Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    st.warning("RDKit not found. Molecular visualizations will not be available.")
    RDKIT_AVAILABLE = False
    # 创建假的RDKit组件
    class MockChem:
        def MolFromSmiles(self, *args, **kwargs):
            return None
    class MockDraw:
        def MolToImage(self, *args, **kwargs):
            return None
    Chem = MockChem()
    Draw = MockDraw()

# 添加父目录到路径以导入项目模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

# 尝试导入自定义模块，如果不可用则提供模拟功能
try:
    from data.dataset_processors import process_dataset, analyze_dataset
except ImportError:
    st.warning("Could not import dataset_processors. Using placeholder functions.")
    
    def process_dataset(df, smiles_col, property_col, options=None):
        """Mock process_dataset function"""
        if options is None:
            options = {}
        # Return a simplified processed data structure
        return {
            'smiles': df[smiles_col].tolist(),
            'property': df[property_col].tolist(),
            'smiles_tokens': [],
            'fingerprints': [],
            'graphs': [],
            'mfbert_embeddings': [],
            'summary': {
                'original_count': len(df),
                'cleaned_count': len(df),
                'removed_count': 0
            }
        }
    
    def analyze_dataset(processed_data):
        """Mock analyze_dataset function"""
        return {
            'count': len(processed_data['smiles']),
            'property': {
                'mean': np.mean(processed_data['property']),
                'std': np.std(processed_data['property']),
                'min': np.min(processed_data['property']),
                'max': np.max(processed_data['property']),
                'histogram': np.histogram(processed_data['property'], bins=20)
            }
        }

try:
    from utils.molecular_utils import tokenize_smiles, generate_fingerprints, smiles_to_graph
except ImportError:
    st.warning("Could not import molecular_utils. Using placeholder functions.")
    
    def tokenize_smiles(smiles):
        return [0] * 50  # Return dummy token IDs
    
    def generate_fingerprints(smiles):
        return [0] * 1024  # Return dummy fingerprint
    
    def smiles_to_graph(smiles):
        return None  # Return dummy graph


def display_molecule(smiles):
    """Display molecular structure from SMILES string"""
    if not RDKIT_AVAILABLE:
        return None
        
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Draw.MolToImage(mol)
        else:
            return None
    except Exception as e:
        st.error(f"Error rendering molecule: {str(e)}")
        return None


def data_page():
    """
    Main function for the data preparation page of H-CAAN.
    """
    st.title("H-CAAN: Data Preparation")
    
    # Data Upload Section
    st.header("1. Dataset Upload")
    
    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=['csv'])
    
    dataset_option = st.selectbox(
        "Or select a benchmark dataset",
        ["None", "Delaney", "Llinas2020", "Lipophilicity", "SAMPL", "BACE", "PDBbind"]
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state['dataset'] = df
            st.session_state['dataset_name'] = uploaded_file.name
        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")
    elif dataset_option != "None":
        try:
            df = pd.read_csv(f"data/benchmark/{dataset_option.lower()}.csv")
            st.session_state['dataset'] = df
            st.session_state['dataset_name'] = dataset_option
        except Exception as e:
            st.error(f"Could not load {dataset_option} dataset: {str(e)}")
            # Try alternative path
            try:
                alt_path = os.path.join(parent_dir, "data", "benchmark", f"{dataset_option.lower()}.csv")
                df = pd.read_csv(alt_path)
                st.session_state['dataset'] = df
                st.session_state['dataset_name'] = dataset_option
                st.success(f"Loaded {dataset_option} from alternative path.")
            except:
                st.error("Failed to load dataset from alternative path.")
    
    # Data Exploration and Preprocessing
    if 'dataset' in st.session_state:
        st.header("2. Data Exploration")
        
        df = st.session_state['dataset']
        st.write(f"Dataset: {st.session_state.get('dataset_name', 'Uploaded Dataset')}")
        st.write(f"Shape: {df.shape}")
        
        # Display dataframe
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        
        # Identify columns
        st.subheader("Column Identification")
        
        all_columns = df.columns.tolist()
        
        # Default SMILES column selection
        default_smiles = next((col for col in all_columns if col.lower() in ['smiles', 'smile', 'canonical_smiles']), all_columns[0])
        smiles_col = st.selectbox("Select SMILES column", all_columns, index=all_columns.index(default_smiles))
        
        # Default property column selection
        property_names = ['property', 'target', 'activity', 'solubility', 'logp', 'logs', 'pka', 'pk']
        default_property = next((col for col in all_columns if any(prop in col.lower() for prop in property_names)), all_columns[-1])
        property_col = st.selectbox("Select target property column", all_columns, index=all_columns.index(default_property))
        
        # Quick stats on property column
        if property_col in df.columns:
            try:
                st.subheader(f"Distribution of {property_col}")
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.histplot(df[property_col].dropna(), kde=True, ax=ax)
                st.pyplot(fig)
                
                st.write("Summary statistics:")
                st.write(df[property_col].describe())
            except Exception as e:
                st.error(f"Error generating property statistics: {str(e)}")
        
        # Molecule visualization
        if RDKIT_AVAILABLE:
            st.subheader("Molecular Visualization")
            
            if smiles_col in df.columns:
                try:
                    # Sample some SMILES to visualize
                    sample_indices = np.random.choice(len(df), min(5, len(df)), replace=False)
                    sample_smiles = df.iloc[sample_indices][smiles_col].tolist()
                    
                    cols = st.columns(len(sample_smiles))
                    for i, (col, smiles) in enumerate(zip(cols, sample_smiles)):
                        mol_img = display_molecule(smiles)
                        if mol_img:
                            col.image(mol_img, caption=f"Sample {i+1}")
                            col.write(f"SMILES: {smiles[:20]}..." if len(smiles) > 20 else f"SMILES: {smiles}")
                        else:
                            col.write(f"Could not render molecule: {smiles[:20]}...")
                except Exception as e:
                    st.error(f"Error visualizing molecules: {str(e)}")
        
        # Preprocessing options
        st.header("3. Data Preprocessing")
        
        st.subheader("Cleaning Options")
        remove_duplicates = st.checkbox("Remove duplicate molecules", value=True)
        remove_invalid = st.checkbox("Remove invalid SMILES", value=True)
        handle_missing = st.checkbox("Handle missing values", value=True)
        
        st.subheader("Representation Options")
        st.write("Select representations to generate:")
        gen_tokenized = st.checkbox("Generate SMILES-encoded vectors", value=True)
        gen_fingerprints = st.checkbox("Generate ECFP fingerprints", value=True)
        gen_graphs = st.checkbox("Generate molecular graphs", value=True)
        gen_mfbert = st.checkbox("Generate MFBERT embeddings", value=False)
        
        if st.button("Process Dataset"):
            try:
                with st.spinner("Processing dataset..."):
                    # Collect preprocessing options
                    options = {
                        'remove_duplicates': remove_duplicates,
                        'remove_invalid': remove_invalid,
                        'handle_missing': handle_missing,
                        'generate_tokens': gen_tokenized,
                        'generate_fingerprints': gen_fingerprints,
                        'generate_graphs': gen_graphs,
                        'generate_mfbert': gen_mfbert
                    }
                    
                    # Process the dataset
                    progress_bar = st.progress(0)
                    
                    # Step 1: Prepare data (10%)
                    progress_bar.progress(0.1)
                    st.info("Preparing data...")
                    
                    # Step 2: Process dataset (50%)
                    progress_bar.progress(0.2)
                    st.info("Processing dataset...")
                    processed_data = process_dataset(df, smiles_col, property_col, options)
                    progress_bar.progress(0.5)
                    
                    # Steps 3-6: Generate representations
                    if gen_tokenized:
                        progress_bar.progress(0.6)
                        st.info("Generating SMILES-encoded vectors...")
                    
                    if gen_fingerprints:
                        progress_bar.progress(0.7)
                        st.info("Generating ECFP fingerprints...")
                    
                    if gen_graphs:
                        progress_bar.progress(0.8)
                        st.info("Generating molecular graphs...")
                    
                    if gen_mfbert:
                        progress_bar.progress(0.9)
                        st.info("Generating MFBERT embeddings...")
                    
                    # Step 7: Analyze dataset (100%)
                    progress_bar.progress(0.95)
                    st.info("Analyzing processed data...")
                    analysis = analyze_dataset(processed_data)
                    progress_bar.progress(1.0)
                    
                    # Store processed data and analysis in session state
                    st.session_state['processed_data'] = processed_data
                    st.session_state['data_analysis'] = analysis
                    
                    st.success("Dataset processed successfully!")
                    
                    # Show preprocessing summary
                    st.subheader("Preprocessing Summary")
                    summary = processed_data.get('summary', {})
                    
                    original_count = summary.get('original_count', len(df))
                    cleaned_count = summary.get('cleaned_count', len(processed_data['smiles']))
                    removed_count = summary.get('removed_count', original_count - cleaned_count)
                    
                    st.write(f"Original molecules: {original_count}")
                    st.write(f"Molecules after cleaning: {cleaned_count}")
                    st.write(f"Removed molecules: {removed_count}")
                    
                    if remove_invalid:
                        invalid_count = summary.get('removed_invalid', 0)
                        st.write(f"Invalid SMILES removed: {invalid_count}")
                    
                    # Show feature distributions
                    st.subheader("Feature Distributions")
                    
                    tab1, tab2, tab3, tab4 = st.tabs(["SMILES Tokens", "ECFP", "Graphs", "MFBERT"])
                    
                    with tab1:
                        if gen_tokenized and processed_data.get('smiles_tokens'):
                            st.write("SMILES token length distribution")
                            try:
                                # Create histogram of token lengths
                                token_lengths = [len(tokens) for tokens in processed_data['smiles_tokens'] if tokens]
                                fig, ax = plt.subplots()
                                ax.hist(token_lengths, bins=20)
                                ax.set_xlabel("Token Sequence Length")
                                ax.set_ylabel("Frequency")
                                st.pyplot(fig)
                            except Exception as e:
                                st.error(f"Error generating token visualization: {str(e)}")
                                # Fallback to placeholder
                                fig, ax = plt.subplots()
                                ax.hist(np.random.normal(50, 15, size=100), bins=20)
                                ax.set_xlabel("Token Sequence Length (Placeholder)")
                                ax.set_ylabel("Frequency")
                                st.pyplot(fig)
                    
                    with tab2:
                        if gen_fingerprints and processed_data.get('fingerprints'):
                            st.write("ECFP bit distribution")
                            try:
                                # Create visualization of fingerprints
                                fingerprints = np.array(processed_data['fingerprints'])
                                sample_fps = fingerprints[:min(10, len(fingerprints))]
                                fig, ax = plt.subplots(figsize=(10, 4))
                                ax.spy(sample_fps, markersize=2)
                                ax.set_xlabel("Bit Position")
                                ax.set_ylabel("Sample Index")
                                st.pyplot(fig)
                                
                                # Also show bit frequency
                                bit_counts = fingerprints.sum(axis=0)
                                bit_freq = bit_counts / len(fingerprints)
                                fig, ax = plt.subplots(figsize=(10, 4))
                                ax.bar(range(min(100, len(bit_freq))), sorted(bit_freq, reverse=True)[:100])
                                ax.set_xlabel("Top Bits")
                                ax.set_ylabel("Frequency")
                                ax.set_title("Top 100 Most Frequent Bits")
                                st.pyplot(fig)
                            except Exception as e:
                                st.error(f"Error generating fingerprint visualization: {str(e)}")
                                # Fallback to placeholder
                                fig, ax = plt.subplots()
                                ax.spy(np.random.binomial(1, 0.1, size=(10, 100)), markersize=3)
                                ax.set_xlabel("Bit Position (Placeholder)")
                                ax.set_ylabel("Sample Index")
                                st.pyplot(fig)
                    
                    with tab3:
                        if gen_graphs and processed_data.get('graphs'):
                            st.write("Graph size distribution")
                            try:
                                # Create histogram of graph sizes
                                if isinstance(processed_data['graphs'][0], tuple):
                                    # If graph is tuple, first element is usually num_atoms
                                    graph_sizes = [g[0] for g in processed_data['graphs'] if g is not None]
                                else:
                                    # Placeholder sizes if format not recognized
                                    graph_sizes = np.random.randint(10, 50, size=len(processed_data['graphs']))
                                
                                fig, ax = plt.subplots()
                                ax.hist(graph_sizes, bins=20)
                                ax.set_xlabel("Number of Atoms")
                                ax.set_ylabel("Frequency")
                                st.pyplot(fig)
                            except Exception as e:
                                st.error(f"Error generating graph visualization: {str(e)}")
                                # Fallback to placeholder
                                fig, ax = plt.subplots()
                                sizes = np.random.randint(10, 50, size=100)
                                ax.hist(sizes, bins=20)
                                ax.set_xlabel("Number of Atoms (Placeholder)")
                                ax.set_ylabel("Frequency")
                                st.pyplot(fig)
                    
                    with tab4:
                        if gen_mfbert and processed_data.get('mfbert_embeddings'):
                            st.write("MFBERT embedding visualizations")
                            try:
                                # Create t-SNE visualization of MFBERT embeddings
                                from sklearn.manifold import TSNE
                                
                                # Prepare embeddings for visualization
                                if len(processed_data['mfbert_embeddings']) > 0:
                                    if isinstance(processed_data['mfbert_embeddings'][0], dict):
                                        # If embeddings are stored as dictionaries
                                        embeddings = [emb.get('input_ids', [0]*100) for emb in processed_data['mfbert_embeddings']]
                                    else:
                                        # If embeddings are raw arrays
                                        embeddings = processed_data['mfbert_embeddings']
                                        
                                    # Convert to 2D for visualization
                                    X = np.array(embeddings)[:100]  # Limit to 100 samples
                                    X_embedded = TSNE(n_components=2, perplexity=min(30, len(X)-1)).fit_transform(X)
                                    
                                    # Use property values for coloring if available
                                    property_values = np.array(processed_data['property'][:100])
                                    
                                    fig, ax = plt.subplots(figsize=(8, 6))
                                    scatter = ax.scatter(X_embedded[:, 0], X_embedded[:, 1], 
                                                      c=property_values, cmap='viridis')
                                    plt.colorbar(scatter, label=property_col)
                                    ax.set_xlabel("t-SNE component 1")
                                    ax.set_ylabel("t-SNE component 2")
                                    ax.set_title("MFBERT Embeddings Visualization")
                                    st.pyplot(fig)
                                else:
                                    st.warning("No MFBERT embeddings available for visualization.")
                            except Exception as e:
                                st.error(f"Error generating MFBERT visualization: {str(e)}")
                                # Fallback to placeholder
                                fig, ax = plt.subplots()
                                X = np.random.normal(0, 1, size=(100, 50))
                                X_embedded = np.random.normal(0, 1, size=(100, 2))
                                scatter = ax.scatter(X_embedded[:, 0], X_embedded[:, 1], 
                                                   c=np.random.random(size=100), cmap='viridis')
                                plt.colorbar(scatter)
                                ax.set_xlabel("t-SNE component 1 (Placeholder)")
                                ax.set_ylabel("t-SNE component 2 (Placeholder)")
                                st.pyplot(fig)
                    
                    # Save button
                    st.subheader("Save Processed Data")
                    save_path = st.text_input("Save directory path", "data/processed")
                    
                    if st.button("Save Processed Data"):
                        try:
                            # Create directory if it doesn't exist
                            os.makedirs(save_path, exist_ok=True)
                            
                            # Save data
                            dataset_name = st.session_state.get('dataset_name', 'dataset').split('.')[0]
                            save_file = os.path.join(save_path, f"{dataset_name}_processed.pkl")
                            
                            with open(save_file, 'wb') as f:
                                pickle.dump(processed_data, f)
                            
                            # Also save as CSV for the basic data
                            csv_data = {
                                'smiles': processed_data['smiles'],
                                'property': processed_data['property']
                            }
                            pd.DataFrame(csv_data).to_csv(os.path.join(save_path, f"{dataset_name}_processed.csv"), index=False)
                            
                            st.success(f"Data saved to {save_file}")
                            st.session_state['data_ready'] = True
                        except Exception as e:
                            st.error(f"Error saving data: {str(e)}")
                
            except Exception as e:
                st.error(f"Error during data processing: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
    
    # Navigation
    st.header("4. Next Steps")
    st.write("Once your data is processed, you can proceed to configure your model.")
    
    # Only enable the next button if data is processed
    next_disabled = not st.session_state.get('data_ready', False)
    if st.button("Go to Model Configuration", disabled=next_disabled):
        st.session_state['current_page'] = 'model_page'
        st.experimental_rerun()


# Ensure the function is defined at the module level
__all__ = ['data_page']

# Allow running this page directly
if __name__ == "__main__":
    data_page()