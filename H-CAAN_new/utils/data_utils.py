"""
数据处理工具函数
"""
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def validate_smiles(smiles: str) -> bool:
    """验证SMILES字符串是否有效"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False

def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    """将SMILES转换为分子对象"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except Exception as e:
        logger.error(f"SMILES转换失败: {smiles}, 错误: {str(e)}")
        return None

def calculate_molecular_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    """计算分子描述符"""
    descriptors = {
        'molecular_weight': Descriptors.MolWt(mol),
        'logp': Crippen.MolLogP(mol),
        'hbd': Lipinski.NumHDonors(mol),
        'hba': Lipinski.NumHAcceptors(mol),
        'tpsa': Descriptors.TPSA(mol),
        'rotatable_bonds': Lipinski.NumRotatableBonds(mol),
        'num_rings': mol.GetRingInfo().NumRings(),
        'num_aromatic_rings': Descriptors.NumAromaticRings(mol),
        'num_heteroatoms': Lipinski.NumHeteroatoms(mol),
        'num_heavy_atoms': Lipinski.HeavyAtomCount(mol)
    }
    return descriptors

def preprocess_dataset(df: pd.DataFrame, smiles_col: str = 'smiles',
                      target_col: Optional[str] = None) -> pd.DataFrame:
    """预处理数据集"""
    # 验证SMILES
    valid_mask = df[smiles_col].apply(validate_smiles)
    df_valid = df[valid_mask].copy()
    
    logger.info(f"有效SMILES: {valid_mask.sum()}/{len(df)}")
    
    # 计算分子描述符
    mols = df_valid[smiles_col].apply(smiles_to_mol)
    descriptors = mols.apply(lambda mol: calculate_molecular_descriptors(mol) if mol else {})
    
    # 添加描述符列
    descriptor_df = pd.DataFrame(list(descriptors))
    df_valid = pd.concat([df_valid, descriptor_df], axis=1)
    
    # 处理缺失值
    df_valid = df_valid.fillna(df_valid.mean(numeric_only=True))
    
    return df_valid

def split_dataset(df: pd.DataFrame, test_size: float = 0.2,
                 random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """划分数据集"""
    from sklearn.model_selection import train_test_split
    
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    
    return train_df, test_df

def augment_smiles(smiles: str, num_augmented: int = 5) -> List[str]:
    """SMILES数据增强"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]
    
    augmented = [smiles]
    for _ in range(num_augmented - 1):
        # 随机化原子顺序
        new_smiles = Chem.MolToSmiles(mol, doRandom=True)
        if new_smiles not in augmented:
            augmented.append(new_smiles)
    
    return augmented

def normalize_features(features: np.ndarray, method: str = 'standard') -> np.ndarray:
    """特征归一化"""
    if method == 'standard':
        mean = features.mean(axis=0)
        std = features.std(axis=0) + 1e-8
        return (features - mean) / std
    elif method == 'minmax':
        min_val = features.min(axis=0)
        max_val = features.max(axis=0)
        return (features - min_val) / (max_val - min_val + 1e-8)
    else:
        return features

def load_molecular_dataset(file_path: str) -> pd.DataFrame:
    """加载分子数据集"""
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith('.pkl'):
        return pd.read_pickle(file_path)
    elif file_path.endswith('.xlsx'):
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {file_path}")