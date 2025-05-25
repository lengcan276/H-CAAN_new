"""
特征融合工具函数
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class AttentionFusion(nn.Module):
    """注意力融合模块"""
    
    def __init__(self, input_dims: List[int], hidden_dim: int = 256):
        super().__init__()
        self.attention_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for dim in input_dims
        ])
        
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        # 计算注意力权重
        attention_scores = []
        for i, feat in enumerate(features):
            score = self.attention_layers[i](feat)
            attention_scores.append(score)
        
        attention_scores = torch.cat(attention_scores, dim=1)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # 加权融合
        fused = torch.zeros_like(features[0])
        for i, feat in enumerate(features):
            weight = attention_weights[:, i:i+1]
            fused += weight * feat
            
        return fused

def hierarchical_fusion(modal_features: Dict[str, np.ndarray],
                       fusion_method: str = 'attention') -> np.ndarray:
    """层次化特征融合"""
    if fusion_method == 'attention':
        # 使用注意力机制
        return attention_based_fusion(modal_features)
    elif fusion_method == 'weighted':
        # 加权平均
        return weighted_average_fusion(modal_features)
    elif fusion_method == 'concatenate':
        # 直接拼接
        return concatenate_fusion(modal_features)
    else:
        raise ValueError(f"未知融合方法: {fusion_method}")

def attention_based_fusion(modal_features: Dict[str, np.ndarray]) -> np.ndarray:
    """基于注意力的融合"""
    # 转换为张量
    tensors = {k: torch.FloatTensor(v) for k, v in modal_features.items()}
    
    # 计算自注意力
    query = torch.stack(list(tensors.values())).mean(dim=0)
    keys = torch.stack(list(tensors.values()))
    
    # 注意力得分
    scores = torch.matmul(query, keys.transpose(-2, -1))
    weights = F.softmax(scores / np.sqrt(query.shape[-1]), dim=-1)
    
    # 加权融合
    fused = torch.zeros_like(query)
    for i, (name, tensor) in enumerate(tensors.items()):
        fused += weights[..., i:i+1] * tensor
        
    return fused.numpy()

def weighted_average_fusion(modal_features: Dict[str, np.ndarray],
                          weights: Optional[Dict[str, float]] = None) -> np.ndarray:
    """加权平均融合"""
    if weights is None:
        # 默认等权重
        n_modals = len(modal_features)
        weights = {k: 1.0 / n_modals for k in modal_features.keys()}
    
    # 归一化权重
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}
    
    # 加权融合
    fused = None
    for name, features in modal_features.items():
        weight = weights.get(name, 0)
        if fused is None:
            fused = weight * features
        else:
            fused += weight * features
            
    return fused

def concatenate_fusion(modal_features: Dict[str, np.ndarray]) -> np.ndarray:
    """拼接融合"""
    return np.concatenate(list(modal_features.values()), axis=-1)

def gated_fusion(modal_features: Dict[str, np.ndarray]) -> np.ndarray:
    """门控融合"""
    # 简化实现
    n_features = list(modal_features.values())[0].shape[-1]
    
    # 计算门控值
    gates = {}
    for name, features in modal_features.items():
        # 使用sigmoid作为门控函数
        gate = 1 / (1 + np.exp(-features.mean(axis=-1, keepdims=True)))
        gates[name] = gate
    
    # 归一化门控值
    total_gate = sum(gates.values())
    gates = {k: v / total_gate for k, v in gates.items()}
    
    # 门控融合
    fused = np.zeros((features.shape[0], n_features))
    for name, features in modal_features.items():
        fused += gates[name] * features
        
    return fused

def compute_fusion_metrics(original_features: Dict[str, np.ndarray],
                         fused_features: np.ndarray) -> Dict[str, float]:
    """计算融合指标"""
    metrics = {}
    
    # 信息保留率
    for name, features in original_features.items():
        correlation = np.corrcoef(
            features.flatten(),
            fused_features.flatten()
        )[0, 1]
        metrics[f'{name}_correlation'] = correlation
    
    # 融合效率（压缩率）
    original_size = sum(f.size for f in original_features.values())
    fused_size = fused_features.size
    metrics['compression_ratio'] = fused_size / original_size
    
    # 特征多样性
    metrics['feature_diversity'] = np.std(fused_features)
    
    return metrics