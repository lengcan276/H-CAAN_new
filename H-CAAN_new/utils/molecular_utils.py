import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
import logging
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional, Any

logger = logging.getLogger(__name__)

class MolecularUtils:
    """
    Utility class for molecular data processing and feature extraction.
    """
    
    @staticmethod
    def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
        """
        Convert SMILES string to RDKit molecule.
        
        Args:
            smiles (str): SMILES string
            
        Returns:
            Chem.Mol or None: RDKit molecule object or None if invalid
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
            return mol
        except Exception as e:
            logger.error(f"Error converting SMILES to molecule: {str(e)}")
            return None
    
    @staticmethod
    def canonicalize_smiles(smiles: str) -> Optional[str]:
        """
        Convert SMILES to canonical form.
        
        Args:
            smiles (str): Input SMILES string
            
        Returns:
            str or None: Canonical SMILES or None if invalid
        """
        mol = MolecularUtils.smiles_to_mol(smiles)
        if mol is not None:
            return Chem.MolToSmiles(mol, isomericSmiles=True)
        return None
    
    @staticmethod
    def get_molecular_descriptors(mol: Chem.Mol) -> Dict[str, float]:
        """
        Calculate molecular descriptors for a molecule.
        
        Args:
            mol (Chem.Mol): RDKit molecule
            
        Returns:
            dict: Dictionary of descriptors
        """
        if mol is None:
            return {}
        
        try:
            descriptors = {
                'MolWt': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'TPSA': Descriptors.TPSA(mol),
                'HBD': Lipinski.NumHDonors(mol),
                'HBA': Lipinski.NumHAcceptors(mol),
                'RotBonds': Descriptors.NumRotatableBonds(mol),
                'Rings': Descriptors.RingCount(mol),
                'AromaticRings': rdMolDescriptors.CalcNumAromaticRings(mol),
                'HeavyAtoms': mol.GetNumHeavyAtoms(),
                'Charge': Chem.GetFormalCharge(mol),
                'QED': QED.qed(mol),
                'FractionCSP3': Descriptors.FractionCSP3(mol),
                'NumHeteroatoms': Descriptors.NumHeteroatoms(mol)
            }
            return descriptors
        except Exception as e:
            logger.error(f"Error calculating descriptors: {str(e)}")
            return {}
    
    @staticmethod
    def generate_ecfp(mol: Chem.Mol, radius: int = 2, nBits: int = 1024) -> np.ndarray:
        """
        Generate Extended Connectivity Fingerprint (ECFP) for a molecule.
        
        Args:
            mol (Chem.Mol): RDKit molecule
            radius (int): Radius for Morgan fingerprint
            nBits (int): Number of bits in fingerprint
            
        Returns:
            np.ndarray: Fingerprint as numpy array
        """
        if mol is None:
            return np.zeros(nBits)
        
        try:
            fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
            return np.array(fingerprint)
        except Exception as e:
            logger.error(f"Error generating ECFP: {str(e)}")
            return np.zeros(nBits)
    
    @staticmethod
    def get_molecular_scaffold(mol: Chem.Mol) -> Optional[str]:
        """
        Get Bemis-Murcko scaffold for a molecule.
        
        Args:
            mol (Chem.Mol): RDKit molecule
            
        Returns:
            str or None: Scaffold SMILES or None if invalid
        """
        if mol is None:
            return None
        
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold, isomericSmiles=True)
        except Exception as e:
            logger.error(f"Error getting scaffold: {str(e)}")
            return None
    
    @staticmethod
    def identify_functional_groups(mol: Chem.Mol) -> Dict[str, int]:
        """
        Identify functional groups in a molecule.
        
        Args:
            mol (Chem.Mol): RDKit molecule
            
        Returns:
            dict: Dictionary of functional groups and their counts
        """
        if mol is None:
            return {}
        
        functional_groups = {
            'Carboxylic Acid': Chem.MolFromSmarts('C(=O)[OH]'),
            'Ester': Chem.MolFromSmarts('C(=O)O[C,c]'),
            'Amide': Chem.MolFromSmarts('C(=O)[NH]'),
            'Amine': Chem.MolFromSmarts('[NH2,NH1,NH0]'),
            'Hydroxyl': Chem.MolFromSmarts('[OH]'),
            'Ketone': Chem.MolFromSmarts('[#6][CX3](=O)[#6]'),
            'Aldehyde': Chem.MolFromSmarts('[CX3H1](=O)'),
            'Ether': Chem.MolFromSmarts('[OD2]([#6])[#6]'),
            'Nitrile': Chem.MolFromSmarts('C#N'),
            'Halogen': Chem.MolFromSmarts('[F,Cl,Br,I]'),
            'Nitro': Chem.MolFromSmarts('[N+](=O)[O-]'),
            'Sulfonamide': Chem.MolFromSmarts('S(=O)(=O)[NH]'),
            'Phosphate': Chem.MolFromSmarts('P(=O)([O,OH])'),
            'Thiol': Chem.MolFromSmarts('[SH]'),
            'Aromatic Ring': Chem.MolFromSmarts('c1ccccc1'),
            'Aliphatic Ring': Chem.MolFromSmarts('[R]')
        }
        
        results = {}
        for name, smarts in functional_groups.items():
            matches = mol.GetSubstructMatches(smarts)
            results[name] = len(matches)
        
        return results
    
    @staticmethod
    def calculate_lipinski_violations(mol: Chem.Mol) -> int:
        """
        Calculate Lipinski's Rule of Five violations.
        
        Args:
            mol (Chem.Mol): RDKit molecule
            
        Returns:
            int: Number of violations (0-4)
        """
        if mol is None:
            return 0
        
        violations = 0
        
        # Molecular weight > 500
        if Descriptors.MolWt(mol) > 500:
            violations += 1
        
        # LogP > 5
        if Descriptors.MolLogP(mol) > 5:
            violations += 1
        
        # H-bond donors > 5
        if Lipinski.NumHDonors(mol) > 5:
            violations += 1
        
        # H-bond acceptors > 10
        if Lipinski.NumHAcceptors(mol) > 10:
            violations += 1
        
        return violations
    
    @staticmethod
    def smiles_to_graph(smiles: str) -> Dict[str, Any]:
        """
        Convert SMILES to graph representation for GNNs.
        
        Args:
            smiles (str): SMILES string
            
        Returns:
            dict: Graph representation with nodes, edges and features
        """
        mol = MolecularUtils.smiles_to_mol(smiles)
        if mol is None:
            return {'nodes': [], 'edges': [], 'node_features': [], 'edge_features': []}
        
        # Get atoms and bonds
        atoms = mol.GetAtoms()
        bonds = mol.GetBonds()
        
        # Atom features
        node_features = []
        for atom in atoms:
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                atom.GetChiralTag(),
                atom.GetIsAromatic(),
                atom.GetHybridization(),
                atom.GetImplicitValence(),
                atom.GetNoImplicit(),
                atom.GetNumExplicitHs(),
                atom.GetNumImplicitHs(),
                atom.GetNumRadicalElectrons(),
                atom.IsInRing()
            ]
            node_features.append(features)
        
        # Edge indices
        edges = []
        edge_features = []
        for bond in bonds:
            # Add edge in both directions
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edges.extend([[i, j], [j, i]])
            
            # Bond features
            features = [
                bond.GetBondType(),
                bond.GetIsAromatic(),
                bond.GetIsConjugated(),
                bond.IsInRing()
            ]
            edge_features.extend([features, features])
        
        return {
            'nodes': list(range(len(atoms))),
            'edges': edges,
            'node_features': node_features,
            'edge_features': edge_features
        }
    
    @staticmethod
    def compute_molecular_similarity(mol1: Chem.Mol, mol2: Chem.Mol, method: str = 'tanimoto') -> float:
        """
        Compute similarity between two molecules.
        
        Args:
            mol1 (Chem.Mol): First molecule
            mol2 (Chem.Mol): Second molecule
            method (str): Similarity method ('tanimoto', 'dice', etc.)
            
        Returns:
            float: Similarity score (0-1)
        """
        if mol1 is None or mol2 is None:
            return 0.0
        
        try:
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
            
            if method == 'tanimoto':
                return DataStructs.TanimotoSimilarity(fp1, fp2)
            elif method == 'dice':
                return DataStructs.DiceSimilarity(fp1, fp2)
            else:
                return DataStructs.TanimotoSimilarity(fp1, fp2)
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}")
            return 0.0

    @staticmethod
    def calculate_complexity_score(mol: Chem.Mol) -> float:
        """
        Calculate a molecular complexity score.
        
        Args:
            mol (Chem.Mol): RDKit molecule
            
        Returns:
            float: Complexity score (0-1)
        """
        if mol is None:
            return 0.0
        
        try:
            # Calculate components of complexity
            num_atoms = mol.GetNumAtoms()
            num_rings = Descriptors.RingCount(mol)
            num_stereocenters = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
            num_rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            num_heteroatoms = Descriptors.NumHeteroatoms(mol)
            
            # Normalize each component
            norm_atoms = min(1.0, num_atoms / 50.0)  # Normalize up to 50 atoms
            norm_rings = min(1.0, num_rings / 6.0)   # Normalize up to 6 rings
            norm_stereo = min(1.0, num_stereocenters / 4.0)  # Normalize up to 4 stereocenters
            norm_rot = min(1.0, num_rotatable_bonds / 10.0)  # Normalize up to 10 rotatable bonds
            norm_hetero = min(1.0, num_heteroatoms / 10.0)   # Normalize up to 10 heteroatoms
            
            # Calculate weighted complexity score
            complexity = (
                0.25 * norm_atoms +
                0.25 * norm_rings +
                0.2 * norm_stereo +
                0.15 * norm_rot +
                0.15 * norm_hetero
            )
            
            return complexity
        except Exception as e:
            logger.error(f"Error calculating complexity: {str(e)}")
            return 0.0
    
    @staticmethod
    def preprocess_dataset(df: pd.DataFrame, smiles_col: str = 'smiles') -> pd.DataFrame:
        """
        Preprocess molecular dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            smiles_col (str): Column containing SMILES strings
            
        Returns:
            pd.DataFrame: Processed dataframe with added features
        """
        if smiles_col not in df.columns:
            logger.error(f"SMILES column '{smiles_col}' not found in dataframe")
            return df
        
        # Create a copy to avoid modifying the original
        processed_df = df.copy()
        
        # Convert SMILES to molecules
        processed_df['ROMol'] = processed_df[smiles_col].apply(MolecularUtils.smiles_to_mol)
        
        # Filter invalid molecules
        valid_mask = processed_df['ROMol'].notnull()
        if not valid_mask.all():
            logger.warning(f"Filtered out {(~valid_mask).sum()} invalid molecules")
            processed_df = processed_df[valid_mask].reset_index(drop=True)
        
        # Calculate molecular descriptors
        descriptors = processed_df['ROMol'].apply(MolecularUtils.get_molecular_descriptors)
        
        # Add descriptor columns
        for desc in ['MolWt', 'LogP', 'TPSA', 'HBD', 'HBA', 'RotBonds', 'Rings', 'AromaticRings', 'HeavyAtoms', 'QED']:
            processed_df[desc] = descriptors.apply(lambda x: x.get(desc, None))
        
        # Calculate ECFP fingerprints
        processed_df['ECFP'] = processed_df['ROMol'].apply(lambda x: MolecularUtils.generate_ecfp(x).tolist())
        
        # Calculate Lipinski violations
        processed_df['Lipinski_Violations'] = processed_df['ROMol'].apply(MolecularUtils.calculate_lipinski_violations)
        
        # Get scaffolds
        processed_df['Scaffold'] = processed_df['ROMol'].apply(MolecularUtils.get_molecular_scaffold)
        
        # Calculate complexity scores
        processed_df['Complexity'] = processed_df['ROMol'].apply(MolecularUtils.calculate_complexity_score)
        
        return processed_df