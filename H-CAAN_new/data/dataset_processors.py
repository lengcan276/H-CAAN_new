# data/dataset_processors.py

import os
import pickle
import numpy as np
import pandas as pd
import torch
import re
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from typing import Dict, List, Tuple, Optional, Union, Any
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data, Dataset, InMemoryDataset
import torch.nn.functional as F
from tqdm import tqdm


class SMILESTokenizer:
    """Tokenizer for SMILES strings"""
    
    def __init__(self, vocab_path: Optional[str] = None):
        self.pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|_|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
        self.regex = re.compile(self.pattern)
        self.vocab = {}
        self.vocab_size = 0
        
        if vocab_path and os.path.exists(vocab_path):
            self.load_vocab(vocab_path)
    
    def tokenize(self, smiles: str) -> List[str]:
        """Tokenize a SMILES string into a list of tokens."""
        tokens = [token for token in self.regex.findall(smiles)]
        assert smiles == ''.join(tokens), f"{smiles} could not be joined after tokenization"
        return tokens
    
    def build_vocab(self, smiles_list: List[str], min_freq: int = 1) -> Dict[str, int]:
        """Build vocabulary from a list of SMILES strings."""
        token_counts = {}
        for smiles in tqdm(smiles_list, desc="Building vocabulary"):
            tokens = self.tokenize(smiles)
            for token in tokens:
                if token in token_counts:
                    token_counts[token] += 1
                else:
                    token_counts[token] = 1
        
        # Filter tokens by frequency and assign indices
        self.vocab = {"<pad>": 0, "<unk>": 1, "<mask>": 2}
        idx = 3
        for token, count in sorted(token_counts.items(), key=lambda x: -x[1]):
            if count >= min_freq:
                self.vocab[token] = idx
                idx += 1
        
        self.vocab_size = len(self.vocab)
        return self.vocab
    
    def save_vocab(self, vocab_path: str):
        """Save vocabulary to a file."""
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        with open(vocab_path, 'wb') as f:
            pickle.dump(self.vocab, f)
    
    def load_vocab(self, vocab_path: str):
        """Load vocabulary from a file."""
        with open(vocab_path, 'rb') as f:
            self.vocab = pickle.load(f)
        self.vocab_size = len(self.vocab)
    
    def encode(self, smiles: str, max_length: int = 512, padding: bool = True, truncation: bool = True) -> List[int]:
        """Convert a SMILES string to a sequence of token IDs."""
        tokens = self.tokenize(smiles)
        
        # Truncate if necessary
        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length]
        
        # Convert tokens to IDs
        token_ids = [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]
        
        # Pad if necessary
        if padding and len(token_ids) < max_length:
            token_ids.extend([self.vocab["<pad>"]] * (max_length - len(token_ids)))
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """Convert a sequence of token IDs back to a SMILES string."""
        # Create a reverse mapping
        id_to_token = {idx: token for token, idx in self.vocab.items()}
        
        # Convert IDs back to tokens
        tokens = [id_to_token.get(id, "<unk>") for id in token_ids if id != self.vocab["<pad>"]]
        
        # Join tokens
        return ''.join(tokens)


class MolecularGraphBuilder:
    """Builder for molecular graphs from SMILES strings."""
    
    def __init__(self, add_hydrogen: bool = False, use_3d_coords: bool = False):
        self.add_hydrogen = add_hydrogen
        self.use_3d_coords = use_3d_coords
        
        # Atom feature parameters
        self.atom_symbols = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 
                             'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 
                             'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 
                             'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 
                             'Cr', 'Pt', 'Hg', 'Pb', 'Unknown']
        self.atom_degrees = list(range(11))  # 0-10
        self.atom_total_hs = list(range(11))  # 0-10
        self.atom_implicit_valence = list(range(11))  # 0-10
        self.atom_feature_dim = len(self.atom_symbols) + len(self.atom_degrees) + \
                               len(self.atom_total_hs) + len(self.atom_implicit_valence) + 1
        
        # Bond feature parameters
        self.bond_types = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, 
                          Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
        self.bond_stereo = [Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOZ, 
                           Chem.rdchem.BondStereo.STEREOE, Chem.rdchem.BondStereo.STEREOCIS, 
                           Chem.rdchem.BondStereo.STEREOTRANS]
        self.bond_feature_dim = len(self.bond_types) + len(self.bond_stereo) + 2  # bond types + stereo + in_ring + conjugated
    
    def _one_hot_encoding(self, value, allowable_set):
        """One-hot encoding for a value in an allowable set."""
        if value not in allowable_set:
            value = allowable_set[-1]
        return [int(value == s) for s in allowable_set]
    
    def _get_atom_features(self, atom):
        """Get features for an atom."""
        features = self._one_hot_encoding(atom.GetSymbol(), self.atom_symbols) + \
                  self._one_hot_encoding(atom.GetDegree(), self.atom_degrees) + \
                  self._one_hot_encoding(atom.GetTotalNumHs(), self.atom_total_hs) + \
                  self._one_hot_encoding(atom.GetImplicitValence(), self.atom_implicit_valence) + \
                  [atom.GetIsAromatic()]
        
        # Normalize features
        features_sum = sum(features)
        if features_sum > 0:
            features = [f / features_sum for f in features]
        
        return features
    
    def _get_bond_features(self, bond):
        """Get features for a bond."""
        features = self._one_hot_encoding(bond.GetBondType(), self.bond_types) + \
                  self._one_hot_encoding(bond.GetStereo(), self.bond_stereo) + \
                  [bond.GetIsConjugated(), bond.IsInRing()]
        
        return features
    
    def _get_mol_3d_coords(self, mol):
        """Generate 3D coordinates for a molecule."""
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        
        coords = []
        for atom in mol.GetAtoms():
            pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            coords.append([pos.x, pos.y, pos.z])
        
        return np.array(coords)
    
    def smiles_to_graph(self, smiles: str) -> Tuple[int, List[List[float]], List[List[int]], Optional[List[List[float]]], Optional[np.ndarray]]:
        """Convert a SMILES string to a molecular graph."""
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            return None
            
        if self.add_hydrogen:
            mol = Chem.AddHs(mol)
        
        # Get atom features
        features = []
        for atom in mol.GetAtoms():
            features.append(self._get_atom_features(atom))
        
        # Get bond features and edge indices
        edge_indices = []
        edge_features = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            edge_indices.extend([[i, j], [j, i]])  # Add both directions
            
            bond_feats = self._get_bond_features(bond)
            edge_features.extend([bond_feats, bond_feats])  # Same features for both directions
        
        # Get 3D coordinates if requested
        coords = None
        if self.use_3d_coords:
            try:
                coords = self._get_mol_3d_coords(mol)
            except:
                # If 3D coordinate generation fails, return None for coords
                coords = None
        
        return mol.GetNumAtoms(), features, edge_indices, edge_features, coords
    
    def smiles_to_pyg_data(self, smiles: str, y: Optional[float] = None) -> Optional[Data]:
        """Convert a SMILES string to a PyTorch Geometric Data object."""
        graph_data = self.smiles_to_graph(smiles)
        
        if graph_data is None:
            return None
            
        num_atoms, features, edge_indices, edge_features, coords = graph_data
        
        # Convert to PyTorch tensors
        x = torch.tensor(features, dtype=torch.float)
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        
        # Create Data object
        data = Data(x=x, edge_index=edge_index)
        
        # Add edge features if available
        if edge_features:
            data.edge_attr = torch.tensor(edge_features, dtype=torch.float)
        
        # Add coordinates if available
        if coords is not None:
            data.pos = torch.tensor(coords, dtype=torch.float)
        
        # Add label if available
        if y is not None:
            data.y = torch.tensor([y], dtype=torch.float)
        
        return data


class ECFPGenerator:
    """Generator for Extended Connectivity Fingerprints (ECFP)."""
    
    def __init__(self, radius: int = 2, nBits: int = 1024, use_features: bool = False):
        self.radius = radius
        self.nBits = nBits
        self.use_features = use_features
    
    def generate(self, smiles: str) -> np.ndarray:
        """Generate ECFP fingerprint for a SMILES string."""
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            return np.zeros(self.nBits)
        
        if self.use_features:
            fingerprint = AllChem.GetMorganFeaturesFingerprint(mol, self.radius, nBits=self.nBits)
        else:
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.nBits)
        
        return np.array(fingerprint)
    
    def generate_batch(self, smiles_list: List[str]) -> np.ndarray:
        """Generate ECFP fingerprints for a batch of SMILES strings."""
        return np.array([self.generate(smiles) for smiles in smiles_list])


class MolecularComplexityCalculator:
    """Calculator for molecular complexity features."""
    
    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        self.scalers = {}
    
    def calculate_complexity(self, smiles: str) -> Dict[str, float]:
        """Calculate complexity features for a SMILES string."""
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            return {
                "num_atoms": 0,
                "num_bonds": 0,
                "num_rings": 0,
                "num_aromatic_rings": 0,
                "num_rotatable_bonds": 0,
                "num_h_donors": 0,
                "num_h_acceptors": 0,
                "tpsa": 0,
                "mw": 0,
                "qed": 0,
                "fsp3": 0
            }
        
        # Calculate basic features
        features = {
            "num_atoms": mol.GetNumAtoms(),
            "num_bonds": mol.GetNumBonds(),
            "num_rings": mol.GetRingInfo().NumRings(),
            "num_aromatic_rings": sum(1 for ring in mol.GetSSSR() if Chem.MolToSmiles(ring)),
            "num_rotatable_bonds": Descriptors.NumRotatableBonds(mol),
            "num_h_donors": Descriptors.NumHDonors(mol),
            "num_h_acceptors": Descriptors.NumHAcceptors(mol),
            "tpsa": Descriptors.TPSA(mol),
            "mw": Descriptors.MolWt(mol),
            "qed": Descriptors.qed(mol),
            "fsp3": Descriptors.FractionCSP3(mol)
        }
        
        return features
    
    def calculate_batch(self, smiles_list: List[str]) -> pd.DataFrame:
        """Calculate complexity features for a batch of SMILES strings."""
        features_list = [self.calculate_complexity(smiles) for smiles in smiles_list]
        return pd.DataFrame(features_list)
    
    def fit_scalers(self, features_df: pd.DataFrame):
        """Fit scalers for normalizing features."""
        if not self.normalize:
            return
        
        for column in features_df.columns:
            scaler = StandardScaler()
            scaler.fit(features_df[[column]])
            self.scalers[column] = scaler
    
    def normalize_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features using fitted scalers."""
        if not self.normalize or not self.scalers:
            return features_df
        
        normalized_df = features_df.copy()
        for column in features_df.columns:
            if column in self.scalers:
                normalized_df[column] = self.scalers[column].transform(features_df[[column]])
        
        return normalized_df
    
    def save_scalers(self, filepath: str):
        """Save fitted scalers to file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.scalers, f)
    
    def load_scalers(self, filepath: str):
        """Load fitted scalers from file."""
        with open(filepath, 'rb') as f:
            self.scalers = pickle.load(f)


class MolecularDataset(InMemoryDataset):
    """PyTorch Geometric dataset for molecular data with multiple modalities."""
    
    def __init__(self, 
                 root: str,
                 name: str,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 tokenizer: Optional[SMILESTokenizer] = None,
                 graph_builder: Optional[MolecularGraphBuilder] = None,
                 ecfp_generator: Optional[ECFPGenerator] = None,
                 complexity_calculator: Optional[MolecularComplexityCalculator] = None,
                 mfbert_tokenizer=None):
        self.name = name
        self.tokenizer = tokenizer or SMILESTokenizer()
        self.graph_builder = graph_builder or MolecularGraphBuilder()
        self.ecfp_generator = ecfp_generator or ECFPGenerator()
        self.complexity_calculator = complexity_calculator or MolecularComplexityCalculator()
        self.mfbert_tokenizer = mfbert_tokenizer
        super(MolecularDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_file_names(self):
        return [f"{self.name}.csv"]
    
    @property
    def processed_file_names(self):
        return [f"{self.name}_processed.pt"]
    
    def download(self):
        # Download is handled externally or dataset should already exist
        pass
    
    def process(self):
        # Load raw data
        raw_data_path = os.path.join(self.raw_dir, f"{self.name}.csv")
        df = pd.read_csv(raw_data_path)
        
        # Ensure necessary columns exist
        assert "smiles" in df.columns, "CSV must contain 'smiles' column"
        assert "target" in df.columns, "CSV must contain 'target' column"
        
        data_list = []
        invalid_smiles = 0
        
        # Process each molecule
        for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {self.name} dataset"):
            smiles = row["smiles"]
            target = row["target"]
            
            # Skip invalid SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                invalid_smiles += 1
                continue
            
            try:
                # 1. Create PyG graph data
                graph_data = self.graph_builder.smiles_to_pyg_data(smiles, y=target)
                if graph_data is None:
                    invalid_smiles += 1
                    continue
                
                # 2. Add SMILES encoding
                smiles_encoding = self.tokenizer.encode(smiles)
                graph_data.smiles_encoding = torch.tensor(smiles_encoding, dtype=torch.long)
                
                # 3. Add ECFP fingerprint
                ecfp = self.ecfp_generator.generate(smiles)
                graph_data.ecfp = torch.tensor(ecfp, dtype=torch.float)
                
                # 4. Add complexity features
                complexity = self.complexity_calculator.calculate_complexity(smiles)
                complexity_features = torch.tensor(list(complexity.values()), dtype=torch.float)
                graph_data.complexity = complexity_features
                
                # 5. Add MFBERT tokenization if available
                if self.mfbert_tokenizer:
                    mfbert_encoded = self.mfbert_tokenizer(smiles, padding='max_length', 
                                                           truncation=True, max_length=512, 
                                                           return_tensors='pt')
                    graph_data.mfbert_input_ids = mfbert_encoded['input_ids'].squeeze(0)
                    graph_data.mfbert_attention_mask = mfbert_encoded['attention_mask'].squeeze(0)
                
                # 6. Add raw SMILES for reference
                graph_data.smiles = smiles
                
                # 7. Apply pre-transform if defined
                if self.pre_transform is not None:
                    graph_data = self.pre_transform(graph_data)
                
                # 8. Apply pre-filter if defined
                if self.pre_filter is not None and not self.pre_filter(graph_data):
                    continue
                
                data_list.append(graph_data)
                
            except Exception as e:
                print(f"Error processing SMILES {smiles}: {e}")
                invalid_smiles += 1
        
        print(f"Processed {len(data_list)} valid molecules. Skipped {invalid_smiles} invalid SMILES.")
        
        # Save processed data
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    
    def get_smiles(self, idx: int) -> str:
        """Get the raw SMILES string for a given index."""
        return self.data.smiles[idx]
    
    def get_complexity_stats(self) -> Dict[str, Tuple[float, float]]:
        """Get statistics (mean, std) of molecular complexity features."""
        complexity_data = torch.stack([data.complexity for data in self])
        means = complexity_data.mean(dim=0)
        stds = complexity_data.std(dim=0)
        
        # Map to feature names
        feature_names = [
            "num_atoms", "num_bonds", "num_rings", "num_aromatic_rings",
            "num_rotatable_bonds", "num_h_donors", "num_h_acceptors",
            "tpsa", "mw", "qed", "fsp3"
        ]
        
        return {name: (means[i].item(), stds[i].item()) for i, name in enumerate(feature_names)}


def create_dataset_splits(dataset_path: str, dataset_name: str, config, 
                          tokenizer=None, graph_builder=None, ecfp_generator=None, 
                          complexity_calculator=None, mfbert_tokenizer=None,
                          test_dataset_path: Optional[str] = None):
    """Create train, validation, and test dataset splits."""
    # Load data
    df = pd.read_csv(dataset_path)
    
    # If test dataset is provided, load it
    if test_dataset_path and os.path.exists(test_dataset_path):
        test_df = pd.read_csv(test_dataset_path)
        has_separate_test = True
    else:
        test_df = None
        has_separate_test = False
    
    # Split dataset if no separate test set
    if not has_separate_test:
        # Shuffle data
        df = df.sample(frac=1, random_state=config.data.random_seed).reset_index(drop=True)
        
        # Calculate split indices
        train_size = int(len(df) * config.data.train_ratio)
        val_size = int(len(df) * config.data.val_ratio)
        
        # Split data
        train_df = df[:train_size]
        val_df = df[train_size:train_size + val_size]
        test_df = df[train_size + val_size:]
    else:
        # If we have a separate test set, just split between train and val
        df = df.sample(frac=1, random_state=config.data.random_seed).reset_index(drop=True)
        val_size = int(len(df) * (config.data.val_ratio / (config.data.train_ratio + config.data.val_ratio)))
        train_df = df[val_size:]
        val_df = df[:val_size]
    
    # Save splits to disk
    os.makedirs(os.path.join(config.data.data_dir, "processed"), exist_ok=True)
    train_df.to_csv(os.path.join(config.data.data_dir, "processed", f"{dataset_name}_train.csv"), index=False)
    val_df.to_csv(os.path.join(config.data.data_dir, "processed", f"{dataset_name}_val.csv"), index=False)
    test_df.to_csv(os.path.join(config.data.data_dir, "processed", f"{dataset_name}_test.csv"), index=False)
    
    # Create datasets
    train_dataset = MolecularDataset(
        root=os.path.join(config.data.data_dir, "processed"),
        name=f"{dataset_name}_train",
        tokenizer=tokenizer,
        graph_builder=graph_builder,
        ecfp_generator=ecfp_generator,
        complexity_calculator=complexity_calculator,
        mfbert_tokenizer=mfbert_tokenizer
    )
    
    val_dataset = MolecularDataset(
        root=os.path.join(config.data.data_dir, "processed"),
        name=f"{dataset_name}_val",
        tokenizer=tokenizer,
        graph_builder=graph_builder,
        ecfp_generator=ecfp_generator,
        complexity_calculator=complexity_calculator,
        mfbert_tokenizer=mfbert_tokenizer
    )
    
    test_dataset = MolecularDataset(
        root=os.path.join(config.data.data_dir, "processed"),
        name=f"{dataset_name}_test",
        tokenizer=tokenizer,
        graph_builder=graph_builder,
        ecfp_generator=ecfp_generator,
        complexity_calculator=complexity_calculator,
        mfbert_tokenizer=mfbert_tokenizer
    )
    
    return train_dataset, val_dataset, test_dataset

def process_dataset(dataset_df, smiles_column, target_column, options=None):
    """
    Process a dataset by cleaning and generating molecular representations.
    
    Args:
        dataset_df (pd.DataFrame): The dataset to process
        smiles_column (str): Name of the column containing SMILES strings
        target_column (str): Name of the column containing target values
        options (dict, optional): Processing options
            - remove_duplicates (bool): Whether to remove duplicate molecules
            - remove_invalid (bool): Whether to remove invalid SMILES
            - handle_missing (bool): Whether to handle missing values
            - generate_tokens (bool): Whether to generate SMILES-encoded vectors
            - generate_fingerprints (bool): Whether to generate ECFP fingerprints
            - generate_graphs (bool): Whether to generate molecular graphs
            - generate_mfbert (bool): Whether to generate MFBERT embeddings
            
    Returns:
        dict: A dictionary containing processed data
    """
    if options is None:
        options = {
            'remove_duplicates': True,
            'remove_invalid': True,
            'handle_missing': True,
            'generate_tokens': True,
            'generate_fingerprints': True,
            'generate_graphs': True,
            'generate_mfbert': False
        }
    
    # Create a copy of the dataset
    df = dataset_df.copy()
    
    # 1. Clean the dataset
    original_count = len(df)
    
    # Handle missing values
    if options.get('handle_missing', True):
        df = df.dropna(subset=[smiles_column, target_column])
    
    # Remove invalid SMILES
    if options.get('remove_invalid', True):
        valid_smiles = []
        for smile in df[smiles_column]:
            mol = Chem.MolFromSmiles(smile)
            valid_smiles.append(mol is not None)
        df = df[valid_smiles]
    
    # Remove duplicate molecules
    if options.get('remove_duplicates', True):
        df = df.drop_duplicates(subset=[smiles_column])
    
    cleaned_count = len(df)
    
    # 2. Generate molecular representations
    processed_data = {
        'smiles': df[smiles_column].tolist(),
        'property': df[target_column].tolist(),
        'smiles_tokens': [],
        'fingerprints': [],
        'graphs': [],
        'mfbert_embeddings': []
    }
    
    # Generate SMILES-encoded vectors
    if options.get('generate_tokens', True):
        tokenizer = SMILESTokenizer()
        tokenizer.build_vocab(processed_data['smiles'])
        for smiles in processed_data['smiles']:
            processed_data['smiles_tokens'].append(tokenizer.encode(smiles))
    
    # Generate ECFP fingerprints
    if options.get('generate_fingerprints', True):
        ecfp_generator = ECFPGenerator()
        processed_data['fingerprints'] = ecfp_generator.generate_batch(processed_data['smiles']).tolist()
    
    # Generate molecular graphs
    if options.get('generate_graphs', True):
        graph_builder = MolecularGraphBuilder()
        for smiles in processed_data['smiles']:
            graph_data = graph_builder.smiles_to_graph(smiles)
            if graph_data:
                processed_data['graphs'].append(graph_data)
            else:
                processed_data['graphs'].append(None)
    
    # Generate MFBERT embeddings
    if options.get('generate_mfbert', False) and 'mfbert_tokenizer' in globals():
        for smiles in processed_data['smiles']:
            try:
                # This is just a placeholder - actual MFBERT tokenization would depend on the model
                mfbert_encoded = {'input_ids': [0], 'attention_mask': [0]}
                processed_data['mfbert_embeddings'].append(mfbert_encoded)
            except:
                processed_data['mfbert_embeddings'].append(None)
    
    # 3. Add processing summary
    processed_data['summary'] = {
        'original_count': original_count,
        'cleaned_count': cleaned_count,
        'removed_count': original_count - cleaned_count,
        'removed_invalid': sum([1 for valid in valid_smiles if not valid]) if options.get('remove_invalid', True) else 0,
        'has_tokens': options.get('generate_tokens', True),
        'has_fingerprints': options.get('generate_fingerprints', True),
        'has_graphs': options.get('generate_graphs', True),
        'has_mfbert': options.get('generate_mfbert', False)
    }
    
    return processed_data

def analyze_dataset(processed_data):
    """
    Analyze the processed dataset to extract useful information.
    
    Args:
        processed_data (dict): The processed dataset
        
    Returns:
        dict: A dictionary containing analysis results
    """
    analysis = {}
    
    # Basic statistics
    analysis['count'] = len(processed_data['smiles'])
    
    # Property distribution
    if processed_data['property']:
        property_values = np.array(processed_data['property'])
        analysis['property'] = {
            'mean': np.mean(property_values),
            'std': np.std(property_values),
            'min': np.min(property_values),
            'max': np.max(property_values),
            'median': np.median(property_values),
            'histogram': np.histogram(property_values, bins=20)
        }
    
    # SMILES token length distribution
    if processed_data['smiles_tokens']:
        token_lengths = [len(tokens) for tokens in processed_data['smiles_tokens']]
        analysis['token_length'] = {
            'mean': np.mean(token_lengths),
            'std': np.std(token_lengths),
            'min': np.min(token_lengths),
            'max': np.max(token_lengths),
            'histogram': np.histogram(token_lengths, bins=20)
        }
    
    # ECFP fingerprint analysis
    if processed_data['fingerprints']:
        fingerprints = np.array(processed_data['fingerprints'])
        bit_counts = fingerprints.sum(axis=0)
        analysis['fingerprints'] = {
            'mean_bits_on': np.mean(fingerprints.sum(axis=1)),
            'std_bits_on': np.std(fingerprints.sum(axis=1)),
            'most_common_bits': np.argsort(bit_counts)[-10:].tolist(),
            'least_common_bits': np.argsort(bit_counts)[:10].tolist()
        }
    
    # Graph analysis
    if processed_data['graphs']:
        valid_graphs = [g for g in processed_data['graphs'] if g is not None]
        if valid_graphs:
            atom_counts = [g[0] for g in valid_graphs]  # g[0] is num_atoms
            analysis['graphs'] = {
                'mean_atoms': np.mean(atom_counts),
                'std_atoms': np.std(atom_counts),
                'min_atoms': np.min(atom_counts),
                'max_atoms': np.max(atom_counts),
                'histogram': np.histogram(atom_counts, bins=20)
            }
    
    return analysis