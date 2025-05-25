# streamlit_ui/pages/2_Data_Preparation.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle
import re

# Page configuration
st.set_page_config(
    page_title="Data Preparation - H-CAAN",
    page_icon="📊",
    layout="wide"
)

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
streamlit_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(streamlit_dir)
sys.path.insert(0, project_root)

# Try to import seaborn
try:
    import seaborn as sns
except ImportError:
    st.warning("Seaborn not found. Some visualizations may be limited.")
    class SnsMock:
        def histplot(self, *args, **kwargs):
            ax = kwargs.get('ax', plt.gca())
            return ax.hist(*args)
    sns = SnsMock()

# Try to import RDKit
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem, Descriptors
    RDKIT_AVAILABLE = True
except ImportError:
    st.warning("RDKit not found. Molecular visualizations will not be available.")
    RDKIT_AVAILABLE = False
    class MockChem:
        def MolFromSmiles(self, *args, **kwargs):
            return None
    class MockDraw:
        def MolToImage(self, *args, **kwargs):
            return None
    Chem = MockChem()
    Draw = MockDraw()

# Try to import custom modules
try:
    from data.dataset_processors import process_dataset, analyze_dataset
except ImportError:
    st.warning("Could not import dataset_processors. Using placeholder functions.")
    
    def process_dataset(df, smiles_col, property_col, options=None):
        """Mock process_dataset function"""
        if options is None:
            options = {}
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
        return [0] * 50
    
    def generate_fingerprints(smiles):
        return [0] * 1024
    
    def smiles_to_graph(smiles):
        return None


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


def render():
    """Main function for the data preparation page"""
    st.title("Data Preparation")
    
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
            # Try multiple paths
            possible_paths = [
                f"data/benchmark/{dataset_option.lower()}.csv",
                os.path.join(project_root, "data", "benchmark", f"{dataset_option.lower()}.csv"),
                os.path.join(streamlit_dir, "data", "benchmark", f"{dataset_option.lower()}.csv")
            ]
            
            loaded = False
            for path in possible_paths:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    st.session_state['dataset'] = df
                    st.session_state['dataset_name'] = dataset_option
                    loaded = True
                    break
            
            if not loaded:
                st.error(f"Could not find {dataset_option} dataset in any expected location.")
        except Exception as e:
            st.error(f"Error loading dataset: {str(e)}")
    
    # Data Exploration and Preprocessing
    if 'dataset' in st.session_state:
        st.header("2. Data Exploration")
        
        df = st.session_state['dataset']
        st.write(f"**Dataset:** {st.session_state.get('dataset_name', 'Uploaded Dataset')}")
        st.write(f"**Shape:** {df.shape}")
        
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
                
                st.write("**Summary statistics:**")
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
        
        if st.button("Process Dataset", type="primary"):
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
                    
                    # Step 1: Prepare data
                    progress_bar.progress(0.1)
                    st.info("Preparing data...")
                    
                    # Step 2: Process dataset
                    progress_bar.progress(0.2)
                    st.info("Processing dataset...")
                    
                    try:
                        processed_data = process_dataset(df, smiles_col, property_col, options)
                    except AssertionError as e:
                        error_msg = str(e)
                        if "could not be joined after tokenization" in error_msg:
                            smiles_with_error = re.search(r'(.*?) could not be joined', error_msg)
                            problematic_smiles = smiles_with_error.group(1) if smiles_with_error else "Unknown"
                            st.error(f"SMILES tokenization error: {problematic_smiles}")
                            st.info("This is likely due to special characters in the SMILES string.")
                            
                            if st.button("Continue with simplified tokenization"):
                                options['use_simplified_tokenization'] = True
                                try:
                                    processed_data = process_dataset(df, smiles_col, property_col, options)
                                except Exception as inner_e:
                                    st.error(f"Processing failed: {str(inner_e)}")
                                    return
                        else:
                            st.error(f"Assertion error: {error_msg}")
                            return
                    except Exception as e:
                        st.error(f"Error during processing: {str(e)}")
                        import traceback
                        st.error(traceback.format_exc())
                        return
                    
                    progress_bar.progress(0.5)
                    
                    # Generate representations
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
                    
                    # Analyze dataset
                    progress_bar.progress(0.95)
                    st.info("Analyzing processed data...")
                    analysis = analyze_dataset(processed_data)
                    progress_bar.progress(1.0)
                    
                    # Store processed data and analysis
                    st.session_state['processed_data'] = processed_data
                    st.session_state['data_analysis'] = analysis
                    st.session_state['data_processed'] = True
                    
                    st.success("Dataset processed successfully!")
                    
                    # Show preprocessing summary
                    st.subheader("Preprocessing Summary")
                    summary = processed_data.get('summary', {})
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Original molecules", summary.get('original_count', len(df)))
                    with col2:
                        st.metric("Cleaned molecules", summary.get('cleaned_count', len(processed_data['smiles'])))
                    with col3:
                        st.metric("Removed molecules", summary.get('removed_count', 0))
                    
                    # Show feature distributions
                    st.subheader("Feature Distributions")
                    
                    tab1, tab2, tab3, tab4 = st.tabs(["SMILES Tokens", "ECFP", "Graphs", "MFBERT"])
                    
                    with tab1:
                        if gen_tokenized and processed_data.get('smiles_tokens'):
                            try:
                                token_lengths = [len(tokens) for tokens in processed_data['smiles_tokens'] if tokens]
                                fig, ax = plt.subplots()
                                ax.hist(token_lengths, bins=20)
                                ax.set_xlabel("Token Sequence Length")
                                ax.set_ylabel("Frequency")
                                st.pyplot(fig)
                            except:
                                st.info("Token visualization not available")
                    
                    with tab2:
                        if gen_fingerprints and processed_data.get('fingerprints'):
                            try:
                                fingerprints = np.array(processed_data['fingerprints'])
                                bit_counts = fingerprints.sum(axis=0)
                                bit_freq = bit_counts / len(fingerprints)
                                
                                fig, ax = plt.subplots(figsize=(10, 4))
                                ax.bar(range(min(100, len(bit_freq))), sorted(bit_freq, reverse=True)[:100])
                                ax.set_xlabel("Top Bits")
                                ax.set_ylabel("Frequency")
                                ax.set_title("Top 100 Most Frequent Bits")
                                st.pyplot(fig)
                            except:
                                st.info("Fingerprint visualization not available")
                    
                    with tab3:
                        if gen_graphs and processed_data.get('graphs'):
                            st.info("Graph statistics generated")
                    
                    with tab4:
                        if gen_mfbert and processed_data.get('mfbert_embeddings'):
                            st.info("MFBERT embeddings generated")
                    
                    # Save button
                    st.subheader("Save Processed Data")
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        save_path = st.text_input("Save directory path", "data/processed")
                    with col2:
                        if st.button("Save Data"):
                            try:
                                os.makedirs(save_path, exist_ok=True)
                                
                                dataset_name = st.session_state.get('dataset_name', 'dataset').split('.')[0]
                                save_file = os.path.join(save_path, f"{dataset_name}_processed.pkl")
                                
                                with open(save_file, 'wb') as f:
                                    pickle.dump(processed_data, f)
                                
                                csv_data = {
                                    'smiles': processed_data['smiles'],
                                    'property': processed_data['property']
                                }
                                pd.DataFrame(csv_data).to_csv(
                                    os.path.join(save_path, f"{dataset_name}_processed.csv"), 
                                    index=False
                                )
                                
                                st.success(f"Data saved to {save_file}")
                                st.session_state['data_ready'] = True
                                
                            except Exception as e:
                                st.error(f"Error saving data: {str(e)}")
                
            except Exception as e:
                st.error(f"Error during processing: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
    
    # Navigation
    st.header("4. Next Steps")
    st.write("Once your data is processed, you can proceed to configure your model.")
    
    if st.session_state.get('data_processed', False):
        if st.button("Go to Model Configuration", type="primary"):
            st.switch_page("pages/3_Model_Configuration.py")
    else:
        st.info("Please process the data first")
        st.button("Go to Model Configuration", disabled=True)

# Main execution
render()