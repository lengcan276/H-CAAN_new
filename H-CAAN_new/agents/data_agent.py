"""
数据感知与预处理智能体
负责分子数据的加载、解析、特征提取和预处理
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
import torch
from torch_geometric.data import Data
import logging

logger = logging.getLogger(__name__)

class DataAgent:
    """数据处理智能体"""
    
    def __init__(self):
        self.supported_formats = ['sdf', 'mol2', 'csv', 'smi']
        self.feature_calculators = {
            'molecular_weight': Descriptors.MolWt,
            'logp': Crippen.MolLogP,
            'hbd': Lipinski.NumHDonors,
            'hba': Lipinski.NumHAcceptors,
            'tpsa': Descriptors.TPSA,
            'rotatable_bonds': Lipinski.NumRotatableBonds
        }
        
    def load_raw_data(self, data_path: str) -> Dict:
        """
        读取原始多模态分子数据
        
        Args:
            data_path: 原始数据文件路径
            
        Returns:
            包含SMILES、图结构、指纹等的字典
        """
        logger.info(f"开始加载数据: {data_path}")
        
        raw_data = {
            'smiles': [],
            'molecules': [],
            'properties': {},
            'metadata': {}
        }
        
        # 判断文件格式并加载
        file_ext = os.path.splitext(data_path)[1].lower()
        
        if file_ext == '.csv':
            df = pd.read_csv(data_path)
            if 'smiles' in df.columns:
                raw_data['smiles'] = df['smiles'].tolist()
                # 解析分子对象
                raw_data['molecules'] = [Chem.MolFromSmiles(smi) for smi in raw_data['smiles']]
                # 提取其他属性列
                for col in df.columns:
                    if col != 'smiles':
                        raw_data['properties'][col] = df[col].tolist()
                        
        elif file_ext in ['.sdf', '.mol2']:
            supplier = Chem.SDMolSupplier(data_path) if file_ext == '.sdf' else Chem.Mol2MolSupplier(data_path)
            for mol in supplier:
                if mol is not None:
                    raw_data['molecules'].append(mol)
                    raw_data['smiles'].append(Chem.MolToSmiles(mol))
                    
        logger.info(f"成功加载 {len(raw_data['molecules'])} 个分子")
        return raw_data
        
    def preprocess_data(self, raw_data: Dict) -> Dict:
        """
        预处理原始数据，进行模态特征提取
        
        Args:
            raw_data: load_raw_data返回的原始数据
            
        Returns:
            包含归一化特征、结构信息、复杂度评分的字典
        """
        logger.info("开始预处理数据...")
        
        processed_data = {
            'smiles_features': [],
            'graph_features': [],
            'fingerprints': [],
            'molecular_descriptors': [],
            'structure_info': [],
            'complexity_scores': [],
            'labels': raw_data.get('properties', {})
        }
        
        for idx, mol in enumerate(raw_data['molecules']):
            if mol is None:
                continue
                
            # 1. SMILES特征提取
            smiles = raw_data['smiles'][idx]
            smiles_features = self._extract_smiles_features(smiles)
            processed_data['smiles_features'].append(smiles_features)
            
            # 2. 图结构特征提取
            graph_data = self._mol_to_graph(mol)
            processed_data['graph_features'].append(graph_data)
            
            # 3. 分子指纹
            fp = GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            processed_data['fingerprints'].append(np.array(fp))
            
            # 4. 分子描述符
            descriptors = self._calculate_descriptors(mol)
            processed_data['molecular_descriptors'].append(descriptors)
            
            # 5. 结构信息和复杂度
            structure_info = self._analyze_structure(mol)
            processed_data['structure_info'].append(structure_info)
            
            complexity = self._calculate_complexity(mol)
            processed_data['complexity_scores'].append(complexity)
            
        # 归一化处理
        processed_data = self._normalize_features(processed_data)
        
        logger.info("数据预处理完成")
        return processed_data
        
    def _extract_smiles_features(self, smiles: str) -> np.ndarray:
        """提取SMILES字符串特征"""
        # 简单的字符编码，实际应用中可以使用更复杂的方法
        char_dict = {c: i for i, c in enumerate('CNOSFClBrIP()=[]#+-0123456789')}
        features = []
        
        for char in smiles[:100]:  # 限制长度
            if char in char_dict:
                features.append(char_dict[char])
            else:
                features.append(0)
                
        # 填充到固定长度
        features = features + [0] * (100 - len(features))
        return np.array(features)
        
    def _mol_to_graph(self, mol) -> Data:
        """将分子转换为图结构"""
        # 节点特征
        atom_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                int(atom.GetHybridization()),
                int(atom.GetIsAromatic())
            ]
            atom_features.append(features)
            
        # 边索引
        edge_indices = []
        edge_attrs = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_indices.extend([[i, j], [j, i]])
            
            bond_type = bond.GetBondType()
            edge_attr = [
                int(bond_type == Chem.rdchem.BondType.SINGLE),
                int(bond_type == Chem.rdchem.BondType.DOUBLE),
                int(bond_type == Chem.rdchem.BondType.TRIPLE),
                int(bond_type == Chem.rdchem.BondType.AROMATIC)
            ]
            edge_attrs.extend([edge_attr, edge_attr])
            
        # 创建PyG数据对象
        x = torch.FloatTensor(atom_features)
        edge_index = torch.LongTensor(edge_indices).t().contiguous()
        edge_attr = torch.FloatTensor(edge_attrs)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
    def _calculate_descriptors(self, mol) -> np.ndarray:
        """计算分子描述符"""
        descriptors = []
        for name, func in self.feature_calculators.items():
            try:
                value = func(mol)
                descriptors.append(value)
            except:
                descriptors.append(0)
                
        return np.array(descriptors)
        
    def _analyze_structure(self, mol) -> Dict:
        """分析分子结构信息"""
        rings = mol.GetRingInfo()
        return {
            'num_atoms': mol.GetNumAtoms(),
            'num_bonds': mol.GetNumBonds(),
            'num_rings': rings.NumRings(),
            'num_aromatic_rings': sum(1 for ring in rings.AtomRings() 
                                    if all(mol.GetAtomWithIdx(idx).GetIsAromatic() 
                                          for idx in ring))
        }
        
    def _calculate_complexity(self, mol) -> float:
        """计算分子复杂度得分"""
        # 简单的复杂度计算
        num_atoms = mol.GetNumAtoms()
        num_rings = mol.GetRingInfo().NumRings()
        num_rotatable = Lipinski.NumRotatableBonds(mol)
        
        complexity = (num_atoms * 0.5 + num_rings * 2 + num_rotatable * 1.5) / 10
        return min(complexity, 10.0)  # 归一化到0-10
        
    def _normalize_features(self, data: Dict) -> Dict:
        """归一化特征"""
        # 对数值特征进行标准化
        if data['molecular_descriptors']:
            descriptors = np.array(data['molecular_descriptors'])
            mean = descriptors.mean(axis=0)
            std = descriptors.std(axis=0) + 1e-8
            data['molecular_descriptors'] = ((descriptors - mean) / std).tolist()
            
        return data
    
    def split_data(self, processed_data: Dict, train_ratio: float = 0.6, 
               val_ratio: float = 0.2, test_ratio: float = 0.2, 
               random_state: int = 42) -> Dict:
        """
        将数据划分为训练集、验证集和测试集
        
        Args:
            processed_data: 预处理后的数据
            train_ratio: 训练集比例
            val_ratio: 验证集比例  
            test_ratio: 测试集比例
            random_state: 随机种子
            
        Returns:
            包含训练集、验证集、测试集的字典
        """
        # 确保比例之和为1
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "训练集、验证集、测试集比例之和必须为1"
        
        # 获取样本数量
        n_samples = len(processed_data['fingerprints'])
        indices = np.arange(n_samples)
        
        # 设置随机种子
        np.random.seed(random_state)
        np.random.shuffle(indices)
        
        # 计算划分点
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        # 划分索引
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        # 划分数据
        split_data = {
            'train': self._extract_subset(processed_data, train_indices),
            'val': self._extract_subset(processed_data, val_indices),
            'test': self._extract_subset(processed_data, test_indices),
            'indices': {
                'train': train_indices.tolist(),
                'val': val_indices.tolist(),
                'test': test_indices.tolist()
            },
            'ratios': {
                'train': train_ratio,
                'val': val_ratio,
                'test': test_ratio
            }
        }
        
        logger.info(f"数据集划分完成: 训练集 {len(train_indices)} 样本, "
                    f"验证集 {len(val_indices)} 样本, "
                    f"测试集 {len(test_indices)} 样本")
        
        return split_data

    def _extract_subset(self, data: Dict, indices: np.ndarray) -> Dict:
        """提取数据子集"""
        subset = {}
        
        # 处理列表类型的数据
        for key in ['smiles_features', 'graph_features', 'fingerprints', 
                    'molecular_descriptors', 'structure_info', 'complexity_scores']:
            if key in data and data[key]:
                subset[key] = [data[key][i] for i in indices]
        
        # 处理标签数据
        if 'labels' in data:
            subset['labels'] = {}
            for label_name, label_values in data['labels'].items():
                subset['labels'][label_name] = [label_values[i] for i in indices]
        
        return subset