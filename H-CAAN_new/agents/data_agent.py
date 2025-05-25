import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, QED
from rdkit.Chem import PandasTools
#from sklearn.manifold import UMAP
from umap import UMAP
import logging
from tqdm import tqdm
import tempfile
import seaborn as sns
from umap import UMAP

class DataAgent:
    """
    Agent responsible for molecular dataset processing, analysis,
    and preparation for the H-CAAN model.
    """
    
    def __init__(self, knowledge_base=None, openai_api_key=None, verbose=True):
        """
        Initialize the Data Agent.
        
        Args:
            knowledge_base (dict, optional): Shared knowledge base
            openai_api_key (str, optional): OpenAI API key for LLM integration
            verbose (bool): Whether to output detailed logs
        """
        self.knowledge_base = knowledge_base or {}
        self.openai_api_key = openai_api_key
        self.verbose = verbose
        
        # Set up logging
        self.logger = self._setup_logging()
        
        # Initialize required directories
        self.data_dir = os.path.join(os.getcwd(), 'data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Predefined datasets
        self.predefined_datasets = {
            "Delaney (ESOL)": self._load_delaney,
            "Lipophilicity": self._load_lipophilicity,
            "BACE": self._load_bace,
            "BBBP": self._load_bbbp,
            "ClinTox": self._load_clintox,
            "HIV": self._load_hiv,
            "SIDER": self._load_sider
        }
        
        self.logger.info("Data Agent initialized successfully")
    
    def _setup_logging(self):
        """Set up logging configuration"""
        logger = logging.getLogger("DataAgent")
        
        if self.verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        
        # Create console handler
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(ch)
        
        return logger
    
    def process_dataset(self, dataset_path):
        """
        Process the specified dataset.
        
        Args:
            dataset_path (str): Path to the dataset or name of predefined dataset
            
        Returns:
            dict: Dictionary containing dataset statistics, splits, and visualizations
        """
        self.logger.info(f"Processing dataset: {dataset_path}")
        
        # Check if it's a predefined dataset
        if dataset_path in self.predefined_datasets:
            self.logger.info(f"Loading predefined dataset: {dataset_path}")
            df = self.predefined_datasets[dataset_path]()
        else:
            # Load from file
            try:
                if isinstance(dataset_path, str) and os.path.isfile(dataset_path):
                    if dataset_path.endswith('.csv'):
                        df = pd.read_csv(dataset_path)
                    elif dataset_path.endswith('.sdf'):
                        df = self._load_sdf(dataset_path)
                    else:
                        raise ValueError(f"Unsupported file format: {dataset_path}")
                else:
                    # Handle uploaded file (e.g., from Streamlit)
                    temp_path = tempfile.NamedTemporaryFile(delete=False)
                    temp_path.write(dataset_path.read())
                    temp_path.close()
                    
                    if dataset_path.name.endswith('.csv'):
                        df = pd.read_csv(temp_path.name)
                    elif dataset_path.name.endswith('.sdf'):
                        df = self._load_sdf(temp_path.name)
                    else:
                        raise ValueError(f"Unsupported file format: {dataset_path.name}")
                    
                    # Clean up
                    os.unlink(temp_path.name)
            except Exception as e:
                self.logger.error(f"Error loading dataset: {str(e)}")
                return {"error": str(e)}
        
        # Process the dataset
        processed_data = self._process_dataframe(df)
        
        # Get dataset statistics
        stats = self._get_dataset_statistics(processed_data)
        
        # Split the dataset
        splits = self._split_dataset(processed_data)
        
        # Generate visualizations
        visualizations = self._generate_visualizations(processed_data)
        
        # Return results
        results = {
            "data": processed_data,
            "stats": stats,
            "splits": splits,
            "visualizations": visualizations
        }
        
        # Update knowledge base
        if self.knowledge_base is not None:
            self.knowledge_base.update({
                "dataset": {
                    "name": dataset_path if isinstance(dataset_path, str) else dataset_path.name,
                    "stats": stats,
                    "splits": {k: v.shape for k, v in splits.items()}
                }
            })
        
        self.logger.info(f"Dataset processing completed. Size: {len(processed_data)}")
        
        return results
    
    def _load_sdf(self, file_path):
        """
        Load SDF file using RDKit.
        
        Args:
            file_path (str): Path to the SDF file
            
        Returns:
            pd.DataFrame: DataFrame containing the molecules
        """
        df = pd.DataFrame()
        PandasTools.LoadSDF(df, file_path)
        return df
    
    def _process_dataframe(self, df):
        """
        Process the DataFrame to ensure it has the required columns.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: Processed DataFrame
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Check for SMILES column
        smiles_columns = [col for col in df.columns if col.lower() in ['smiles', 'canonical_smiles', 'canonical smiles']]
        if smiles_columns:
            df['smiles'] = df[smiles_columns[0]]
        elif 'ROMol' in df.columns:
            # Generate SMILES from RDKit molecules
            df['smiles'] = df['ROMol'].apply(lambda x: Chem.MolToSmiles(x) if x is not None else None)
        else:
            self.logger.error("No SMILES information found in the dataset")
            return df
        
        # Validate SMILES and create RDKit molecules
        if 'ROMol' not in df.columns:
            df['ROMol'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x) if isinstance(x, str) else None)
        
        # Filter out invalid molecules
        original_size = len(df)
        df = df[df['ROMol'].notnull()]
        filtered_size = len(df)
        
        if filtered_size < original_size:
            self.logger.warning(f"Filtered out {original_size - filtered_size} invalid molecules")
        
        # Generate molecular descriptors
        self.logger.info("Generating molecular descriptors...")
        
        # Calculate basic RDKit descriptors
        descriptors = []
        for mol in tqdm(df['ROMol'], desc="Calculating descriptors"):
            if mol is not None:
                # Calculate basic properties
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                tpsa = Descriptors.TPSA(mol)
                hbd = Lipinski.NumHDonors(mol)
                hba = Lipinski.NumHAcceptors(mol)
                rotbonds = Descriptors.NumRotatableBonds(mol)
                rings = Descriptors.RingCount(mol)
                aromatic_rings = Chem.Mol.GetSSSR(mol)
                heavy_atoms = mol.GetNumHeavyAtoms()
                qed_value = QED.qed(mol)
                
                descriptors.append({
                    'MolWt': mw,
                    'LogP': logp,
                    'TPSA': tpsa,
                    'HBD': hbd,
                    'HBA': hba,
                    'RotBonds': rotbonds,
                    'Rings': rings,
                    'AromaticRings': len(aromatic_rings),
                    'HeavyAtoms': heavy_atoms,
                    'QED': qed_value
                })
            else:
                descriptors.append({
                    'MolWt': None, 'LogP': None, 'TPSA': None,
                    'HBD': None, 'HBA': None, 'RotBonds': None,
                    'Rings': None, 'AromaticRings': None,
                    'HeavyAtoms': None, 'QED': None
                })
        
        # Add descriptors to DataFrame
        descriptors_df = pd.DataFrame(descriptors)
        for col in descriptors_df.columns:
            df[col] = descriptors_df[col]
        
        # Generate ECFP fingerprints
        self.logger.info("Generating ECFP fingerprints...")
        df['ECFP'] = df['ROMol'].apply(
            lambda x: list(AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=1024)) if x is not None else None
        )
        
        # Identify target property column (if applicable)
        target_columns = []
        for col in df.columns:
            if col.lower() in ['target', 'property', 'activity', 'label', 'y', 'value', 'ic50', 'ec50', 
                              'ki', 'solubility', 'logp', 'logs', 'pka']:
                # Check if the column has numeric values
                if pd.api.types.is_numeric_dtype(df[col]):
                    target_columns.append(col)
        
        if target_columns:
            # Use the first identified target column
            df['Property'] = df[target_columns[0]]
            self.logger.info(f"Identified target property column: {target_columns[0]}")
        else:
            self.logger.warning("No target property column identified")
        
        return df
    
    def _get_dataset_statistics(self, df):
        """
        Calculate statistics for the dataset.
        
        Args:
            df (pd.DataFrame): Processed DataFrame
            
        Returns:
            dict: Dictionary of dataset statistics
        """
        # Basic statistics
        stats = {
            "num_molecules": len(df),
            "num_valid_molecules": df['ROMol'].notnull().sum(),
            "descriptor_stats": {}
        }
        
        # Calculate descriptor statistics
        descriptor_columns = ['MolWt', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds', 'Rings', 'AromaticRings', 'HeavyAtoms', 'QED']
        for col in descriptor_columns:
            if col in df.columns:
                stats["descriptor_stats"][col] = {
                    "mean": float(df[col].mean()),
                    "std": float(df[col].std()),
                    "min": float(df[col].min()),
                    "25%": float(df[col].quantile(0.25)),
                    "50%": float(df[col].quantile(0.5)),
                    "75%": float(df[col].quantile(0.75)),
                    "max": float(df[col].max())
                }
        
        # Target property statistics (if available)
        if 'Property' in df.columns:
            stats["property_stats"] = {
                "mean": float(df['Property'].mean()),
                "std": float(df['Property'].std()),
                "min": float(df['Property'].min()),
                "25%": float(df['Property'].quantile(0.25)),
                "50%": float(df['Property'].quantile(0.5)),
                "75%": float(df['Property'].quantile(0.75)),
                "max": float(df['Property'].max())
            }
        
        # Get example molecules
        if len(df) > 0:
            stats["example_mols"] = df['smiles'].head(5).tolist()
        
        return stats
    
    def _split_dataset(self, df, test_size=0.1, valid_size=0.1, random_state=42):
        """
        Split the dataset into training, validation, and test sets.
        
        Args:
            df (pd.DataFrame): Processed DataFrame
            test_size (float): Proportion of dataset to use for test set
            valid_size (float): Proportion of dataset to use for validation set
            random_state (int): Random seed for reproducibility
            
        Returns:
            dict: Dictionary containing the dataset splits
        """
        # Check if Property column exists
        if 'Property' not in df.columns:
            self.logger.warning("No Property column found. Creating dummy Property column.")
            df['Property'] = 0.0
        
        # Split into train+valid and test
        train_valid_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state
        )
        
        # Split train+valid into train and valid
        train_df, valid_df = train_test_split(
            train_valid_df, test_size=valid_size/(1-test_size), random_state=random_state
        )
        
        # Log split sizes
        self.logger.info(f"Dataset split: Train={len(train_df)}, Validation={len(valid_df)}, Test={len(test_df)}")
        
        # Save splits
        train_df.to_csv(os.path.join(self.data_dir, 'train.csv'), index=False)
        valid_df.to_csv(os.path.join(self.data_dir, 'valid.csv'), index=False)
        test_df.to_csv(os.path.join(self.data_dir, 'test.csv'), index=False)
        
        return {
            "train": train_df,
            "valid": valid_df,
            "test": test_df
        }
    
    def _generate_visualizations(self, df):
        """
        Generate visualizations for the dataset.
        
        Args:
            df (pd.DataFrame): Processed DataFrame
            
        Returns:
            dict: Dictionary of visualization figures
        """
        visualizations = {}
        
        # Create figure for molecular weight distribution
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df['MolWt'], bins=50, kde=True, ax=ax)
            ax.set_title('Molecular Weight Distribution')
            ax.set_xlabel('Molecular Weight (Da)')
            ax.set_ylabel('Count')
            plt.tight_layout()
            visualizations['mol_weight_dist'] = fig
        except Exception as e:
            self.logger.error(f"Error generating molecular weight distribution: {str(e)}")
        
        # Create figure for property distribution (if available)
        if 'Property' in df.columns:
            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(df['Property'], bins=50, kde=True, ax=ax)
                ax.set_title('Property Distribution')
                ax.set_xlabel('Property Value')
                ax.set_ylabel('Count')
                plt.tight_layout()
                visualizations['property_dist'] = fig
            except Exception as e:
                self.logger.error(f"Error generating property distribution: {str(e)}")
        
        # Create chemical space visualization using UMAP on ECFP fingerprints
        try:
            # Get valid fingerprints
            valid_fps = df['ECFP'].dropna().tolist()
            
            if len(valid_fps) > 10:  # Need at least a few samples for UMAP
                # Convert to numpy array
                X = np.array(valid_fps)
                
                # Apply UMAP
                umap_model = UMAP(n_components=2, random_state=42)
                X_umap = umap_model.fit_transform(X)
                
                # Create plot
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Color by property if available
                if 'Property' in df.columns:
                    property_values = df.loc[df['ECFP'].notna(), 'Property'].values
                    scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], c=property_values, 
                                        cmap='viridis', alpha=0.7, s=10)
                    plt.colorbar(scatter, ax=ax, label='Property Value')
                else:
                    ax.scatter(X_umap[:, 0], X_umap[:, 1], alpha=0.7, s=10)
                
                ax.set_title('Chemical Space (UMAP of ECFP Fingerprints)')
                ax.set_xlabel('UMAP 1')
                ax.set_ylabel('UMAP 2')
                plt.tight_layout()
                visualizations['chemical_space'] = fig
            else:
                self.logger.warning("Too few valid fingerprints for UMAP visualization")
        except Exception as e:
            self.logger.error(f"Error generating chemical space visualization: {str(e)}")
        
        return visualizations
    
    # Methods to load predefined datasets
    def _load_delaney(self):
        """Load the Delaney (ESOL) solubility dataset"""
        try:
            # Try to load from local data directory first
            file_path = os.path.join(self.data_dir, 'delaney.csv')
            if os.path.isfile(file_path):
                df = pd.read_csv(file_path)
            else:
                # Download from MoleculeNet
                url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv'
                df = pd.read_csv(url)
                df.to_csv(file_path, index=False)  # Save for future use
            
            # Rename columns to match our schema
            df = df.rename(columns={'SMILES': 'smiles', 'measured log solubility in mols per litre': 'Property'})
            return df
        except Exception as e:
            self.logger.error(f"Error loading Delaney dataset: {str(e)}")
            return pd.DataFrame()
    
    def _load_lipophilicity(self):
        """Load the Lipophilicity dataset"""
        try:
            # Try to load from local data directory first
            file_path = os.path.join(self.data_dir, 'lipophilicity.csv')
            if os.path.isfile(file_path):
                df = pd.read_csv(file_path)
            else:
                # Download from MoleculeNet
                url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv'
                df = pd.read_csv(url)
                df.to_csv(file_path, index=False)  # Save for future use
            
            # Rename columns to match our schema
            df = df.rename(columns={'smiles': 'smiles', 'exp': 'Property'})
            return df
        except Exception as e:
            self.logger.error(f"Error loading Lipophilicity dataset: {str(e)}")
            return pd.DataFrame()
    
    def _load_bace(self):
        """Load the BACE dataset"""
        try:
            # Try to load from local data directory first
            file_path = os.path.join(self.data_dir, 'bace.csv')
            if os.path.isfile(file_path):
                df = pd.read_csv(file_path)
            else:
                # Download from MoleculeNet
                url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv'
                df = pd.read_csv(url)
                df.to_csv(file_path, index=False)  # Save for future use
            
            # Rename columns to match our schema
            df = df.rename(columns={'mol': 'smiles', 'Class': 'Property'})
            return df
        except Exception as e:
            self.logger.error(f"Error loading BACE dataset: {str(e)}")
            return pd.DataFrame()
    
    def _load_bbbp(self):
        """Load the Blood-Brain Barrier Penetration (BBBP) dataset"""
        try:
            # Try to load from local data directory first
            file_path = os.path.join(self.data_dir, 'bbbp.csv')
            if os.path.isfile(file_path):
                df = pd.read_csv(file_path)
            else:
                # Download from MoleculeNet
                url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv'
                df = pd.read_csv(url)
                df.to_csv(file_path, index=False)  # Save for future use
            
            # Rename columns to match our schema
            df = df.rename(columns={'smiles': 'smiles', 'p_np': 'Property'})
            return df
        except Exception as e:
            self.logger.error(f"Error loading BBBP dataset: {str(e)}")
            return pd.DataFrame()
    
    def _load_clintox(self):
        """Load the ClinTox dataset"""
        try:
            # Try to load from local data directory first
            file_path = os.path.join(self.data_dir, 'clintox.csv')
            if os.path.isfile(file_path):
                df = pd.read_csv(file_path)
            else:
                # Download from MoleculeNet
                url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/clintox.csv.gz'
                df = pd.read_csv(url, compression='gzip')
                df.to_csv(file_path, index=False)  # Save for future use
            
            # Rename columns to match our schema
            df = df.rename(columns={'smiles': 'smiles', 'FDA_APPROVED': 'Property'})
            return df
        except Exception as e:
            self.logger.error(f"Error loading ClinTox dataset: {str(e)}")
            return pd.DataFrame()
    
    def _load_hiv(self):
        """Load the HIV dataset"""
        try:
            # Try to load from local data directory first
            file_path = os.path.join(self.data_dir, 'hiv.csv')
            if os.path.isfile(file_path):
                df = pd.read_csv(file_path)
            else:
                # Download from MoleculeNet
                url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/HIV.csv'
                df = pd.read_csv(url)
                df.to_csv(file_path, index=False)  # Save for future use
            
            # Rename columns to match our schema
            df = df.rename(columns={'smiles': 'smiles', 'HIV_active': 'Property'})
            return df
        except Exception as e:
            self.logger.error(f"Error loading HIV dataset: {str(e)}")
            return pd.DataFrame()
    
    def _load_sider(self):
        """Load the SIDER dataset"""
        try:
            # Try to load from local data directory first
            file_path = os.path.join(self.data_dir, 'sider.csv')
            if os.path.isfile(file_path):
                df = pd.read_csv(file_path)
            else:
                # Download from MoleculeNet
                url = 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz'
                df = pd.read_csv(url, compression='gzip')
                df.to_csv(file_path, index=False)  # Save for future use
            
            # Rename columns to match our schema
            df = df.rename(columns={'smiles': 'smiles'})
            # Use the first side effect as the property
            if 'Hepatobiliary disorders' in df.columns:
                df['Property'] = df['Hepatobiliary disorders']
            return df
        except Exception as e:
            self.logger.error(f"Error loading SIDER dataset: {str(e)}")
            return pd.DataFrame()
    
    def generate_molecular_features(self, smiles_list):
        """
        Generate features for a list of SMILES strings.
        
        Args:
            smiles_list (list): List of SMILES strings
            
        Returns:
            dict: Dictionary containing generated features
        """
        self.logger.info(f"Generating features for {len(smiles_list)} molecules")
        
        # Convert SMILES to RDKit molecules
        mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
        
        # Filter valid molecules
        valid_mols = [mol for mol in mols if mol is not None]
        valid_smiles = [smiles for mol, smiles in zip(mols, smiles_list) if mol is not None]
        
        if len(valid_mols) < len(mols):
            self.logger.warning(f"Filtered out {len(mols) - len(valid_mols)} invalid molecules")
        
        # Generate features
        features = {}
        
        # ECFP fingerprints
        ecfp_features = []
        for mol in valid_mols:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            ecfp_features.append(list(fp))
        features['ecfp'] = np.array(ecfp_features)
        
        # SMILES encoding will be handled by the model
        features['smiles'] = valid_smiles
        
        # Generate molecular graphs for GCN
        graph_features = []
        for mol in valid_mols:
            # Get atom features
            atom_features = []
            for atom in mol.GetAtoms():
                # Basic atom features (type, degree, formal charge, etc.)
                atom_feature = [
                    atom.GetAtomicNum(),
                    atom.GetDegree(),
                    atom.GetFormalCharge(),
                    atom.GetNumRadicalElectrons(),
                    atom.GetHybridization(),
                    atom.GetIsAromatic(),
                    atom.IsInRing()
                ]
                atom_features.append(atom_feature)
            
            # Get bond features and connectivity
            bond_features = []
            connectivity = []
            for bond in mol.GetBonds():
                # Get atoms connected by this bond
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                
                # Add connectivity
                connectivity.append((begin_idx, end_idx))
                connectivity.append((end_idx, begin_idx))  # Add both directions
                
                # Basic bond features (type, conjugated, in ring)
                bond_feature = [
                    bond.GetBondType(),
                    bond.GetIsConjugated(),
                    bond.IsInRing()
                ]
                # Add bond feature twice (for both directions)
                bond_features.append(bond_feature)
                bond_features.append(bond_feature)
            
            graph_features.append({
                'atom_features': atom_features,
                'bond_features': bond_features,
                'connectivity': connectivity
            })
        
        features['graph'] = graph_features
        
        return features
    
    def analyze_chemical_space(self, df):
        """
        Analyze the chemical space coverage of the dataset.
        
        Args:
            df (pd.DataFrame): Processed DataFrame
            
        Returns:
            dict: Dictionary containing analysis results
        """
        self.logger.info("Analyzing chemical space coverage...")
        
        # Get molecular descriptors
        descriptors = df[['MolWt', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds', 'Rings']].dropna()
        
        # Calculate Lipinski Rule of 5 compliance
        lipinski_violations = []
        for i, row in descriptors.iterrows():
            violations = 0
            if row['MolWt'] > 500:
                violations += 1
            if row['LogP'] > 5:
                violations += 1
            if row['HBD'] > 5:
                violations += 1
            if row['HBA'] > 10:
                violations += 1
            lipinski_violations.append(violations)
        
        ro5_compliant = sum(v <= 1 for v in lipinski_violations) / len(lipinski_violations) if lipinski_violations else 0
        
        # Analyze diversity using ECFP fingerprints
        diversity = self._analyze_diversity(df)
        
        # Analyze property distribution
        property_coverage = {}
        if 'Property' in df.columns:
            property_values = df['Property'].dropna()
            property_coverage = {
                'min': float(property_values.min()),
                'max': float(property_values.max()),
                'range': float(property_values.max() - property_values.min()),
                'std': float(property_values.std())
            }
        
        return {
            'ro5_compliance': ro5_compliant,
            'diversity': diversity,
            'property_coverage': property_coverage
        }
    
    def _analyze_diversity(self, df):
        """
        Analyze molecular diversity using ECFP fingerprints.
        
        Args:
            df (pd.DataFrame): Processed DataFrame
            
        Returns:
            dict: Dictionary containing diversity metrics
        """
        # Get valid fingerprints
        valid_fps = [fp for fp in df['ECFP'].dropna().tolist() if len(fp) > 0]
        
        if len(valid_fps) < 2:
            return {'average_similarity': None, 'median_similarity': None}
        
        # Convert to numpy array
        fp_array = np.array(valid_fps)
        
        # Calculate pairwise Tanimoto similarities for a sample (full matrix is too large)
        max_sample = 1000
        if len(fp_array) > max_sample:
            indices = np.random.choice(len(fp_array), max_sample, replace=False)
            sample_fps = fp_array[indices]
        else:
            sample_fps = fp_array
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(sample_fps)):
            for j in range(i+1, len(sample_fps)):
                # Tanimoto similarity
                intersection = np.sum(np.logical_and(sample_fps[i], sample_fps[j]))
                union = np.sum(np.logical_or(sample_fps[i], sample_fps[j]))
                similarity = intersection / union if union > 0 else 0
                similarities.append(similarity)
        
        return {
            'average_similarity': float(np.mean(similarities)),
            'median_similarity': float(np.median(similarities)),
            'min_similarity': float(np.min(similarities)),
            'max_similarity': float(np.max(similarities))
        }
