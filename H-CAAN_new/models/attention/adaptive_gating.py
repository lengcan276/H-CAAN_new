# models/attention/adaptive_gating.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveGating(nn.Module):
    """
    自适应门控机制，用于控制不同模态间的信息交换
    """
    
    def __init__(self, hidden_dim, dropout=0.1):
        """
        初始化自适应门控模块
        
        Args:
            hidden_dim (int): 隐藏层维度
            dropout (float): Dropout概率
        """
        super(AdaptiveGating, self).__init__()
        
        # 特征转换
        self.transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 门控网络
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 任务适应性转换
        self.task_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 化学结构感知
        self.chem_awareness = ChemicalStructureAwareness(hidden_dim)
    
    def forward(self, feature_list):
        """
        前向传播
        
        Args:
            feature_list (list): 不同模态的特征列表
            
        Returns:
            增强的特征列表
        """
        num_modalities = len(feature_list)
        batch_size = feature_list[0].shape[0]
        hidden_dim = feature_list[0].shape[1]
        device = feature_list[0].device
        
        # 转换所有特征
        transformed_features = []
        for feature in feature_list:
            transformed_features.append(self.transform(feature))
        
        # 计算任务特定表示
        task_repr = torch.zeros(batch_size, hidden_dim, device=device)
        for feature in feature_list:
            task_repr += feature
        task_repr = task_repr / num_modalities
        task_repr = self.task_transform(task_repr)
        
        # 应用化学结构感知
        chem_enhanced_features = self.chem_awareness(transformed_features)
        
        # 计算模态间门控
        gated_features = []
        for i, feature in enumerate(chem_enhanced_features):
            # 初始化增强特征
            enhanced = torch.zeros_like(feature)
            
            # 从其他模态获取信息
            for j, other_feature in enumerate(chem_enhanced_features):
                if i != j:
                    # 连接当前特征和其他特征
                    combined = torch.cat([feature, other_feature], dim=1)
                    
                    # 计算门控值
                    gate = self.gate_net(combined)
                    
                    # 使用门控值控制信息流
                    flow = gate * other_feature
                    
                    # 累积信息
                    enhanced += flow
            
            # 平均累积的信息
            enhanced = enhanced / (num_modalities - 1)
            
            # 残差连接
            final_feature = feature + enhanced
            
            # 融合任务特定信息
            final_feature = final_feature + 0.1 * task_repr
            
            gated_features.append(final_feature)
        
        return gated_features

class ChemicalStructureAwareness(nn.Module):
    """
    化学结构感知模块，用于增强特征的化学相关性
    """
    
    def __init__(self, hidden_dim):
        super(ChemicalStructureAwareness, self).__init__()
        
        # 化学基团检测器
        self.functional_group_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 32),  # 32种常见功能基团
            nn.Sigmoid()
        )
        
        # 环系统检测器
        self.ring_system_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 16),  # 16种环系统类型
            nn.Sigmoid()
        )
        
        # 化学增强器
        self.chemical_enhancer = nn.Sequential(
            nn.Linear(hidden_dim + 32 + 16, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, feature_list):
        """
        前向传播
        
        Args:
            feature_list (list): 不同模态的特征列表
            
        Returns:
            化学增强的特征列表
        """
        enhanced_features = []
        
        for feature in feature_list:
            # 检测功能基团
            functional_groups = self.functional_group_detector(feature)
            
            # 检测环系统
            ring_systems = self.ring_system_detector(feature)
            
            # 连接原始特征和化学感知信息
            combined = torch.cat([feature, functional_groups, ring_systems], dim=1)
            
            # 增强特征
            enhanced = self.chemical_enhancer(combined)
            
            # 残差连接
            enhanced = feature + enhanced
            
            enhanced_features.append(enhanced)
        
        return enhanced_features