# models/fusion/hierarchical_fusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalFusion(nn.Module):
    """
    层次融合模块，结合低层、中层和高层表示
    """
    
    def __init__(self, common_dim, num_modalities, use_low_level=True, use_mid_level=True, use_high_level=True, dropout=0.1):
        """
        初始化层次融合模块
        
        Args:
            common_dim (int): 共同特征维度
            num_modalities (int): 模态数量
            use_low_level (bool): 是否使用低层级融合
            use_mid_level (bool): 是否使用中层级融合
            use_high_level (bool): 是否使用高层级融合
            dropout (float): Dropout概率
        """
        super(HierarchicalFusion, self).__init__()
        
        self.common_dim = common_dim
        self.num_modalities = num_modalities
        self.use_low_level = use_low_level
        self.use_mid_level = use_mid_level
        self.use_high_level = use_high_level
        
        # 低层级融合 - 特征层面
        if use_low_level:
            self.low_level_fusion = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(common_dim * 2, common_dim),
                    nn.LayerNorm(common_dim),
                    nn.GELU(),
                    nn.Dropout(dropout)
                ) for _ in range(num_modalities * (num_modalities - 1) // 2)
            ])
        
        # 中层级融合 - 语义层面
        if use_mid_level:
            self.mid_level_attention = nn.MultiheadAttention(
                embed_dim=common_dim,
                num_heads=8,
                dropout=dropout,
                batch_first=True
            )
            self.mid_level_gate = nn.Sequential(
                nn.Linear(common_dim * 2, common_dim),
                nn.Sigmoid()
            )
        
        # 高层级融合 - 决策层面
        if use_high_level:
            self.high_level_fusion = nn.Sequential(
                nn.Linear(common_dim * num_modalities, common_dim * 2),
                nn.LayerNorm(common_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(common_dim * 2, common_dim),
                nn.LayerNorm(common_dim)
            )
            
            # 专家混合系统
            self.experts = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(common_dim, common_dim // 2),
                    nn.GELU(),
                    nn.Linear(common_dim // 2, common_dim)
                ) for _ in range(3)  # 使用3个专家
            ])
            
            self.expert_gate = nn.Linear(common_dim, 3)  # 3个专家的门控
        
        # 特征金字塔
        self.feature_pyramid = FeaturePyramid(common_dim, dropout)
    
    def forward(self, feature_list, weights=None):
        """
        前向传播
        
        Args:
            feature_list (list): 不同模态的特征列表
            weights (torch.Tensor, optional): 模态权重
            
        Returns:
            融合的表示 [batch_size, common_dim]
        """
        batch_size = feature_list[0].shape[0]
        device = feature_list[0].device
        
        # 如果没有提供权重，使用均匀权重
        if weights is None:
            weights = torch.ones(self.num_modalities, device=device) / self.num_modalities
        
        # 低层级融合
        if self.use_low_level:
            low_level_features = self._apply_low_level_fusion(feature_list)
        else:
            low_level_features = feature_list
        
        # 中层级融合
        if self.use_mid_level:
            mid_level_features = self._apply_mid_level_fusion(low_level_features)
        else:
            mid_level_features = low_level_features
        
        # 高层级融合
        if self.use_high_level:
            fused_representation = self._apply_high_level_fusion(mid_level_features, weights)
        else:
            # 如果不使用高层级融合，简单加权求和
            fused_representation = torch.zeros(batch_size, self.common_dim, device=device)
            for i, feature in enumerate(mid_level_features):
                fused_representation += weights[i] * feature
        
        # 应用特征金字塔
        final_representation = self.feature_pyramid(fused_representation)
        
        return final_representation
    
    def _apply_low_level_fusion(self, feature_list):
        """应用低层级融合"""
        enhanced_features = []
        
        # 复制原始特征
        for feature in feature_list:
            enhanced_features.append(feature.clone())
        
        # 在每对模态之间应用融合
        fusion_idx = 0
        for i in range(len(feature_list)):
            for j in range(i+1, len(feature_list)):
                # 特征i和特征j之间的融合
                fi = feature_list[i]
                fj = feature_list[j]
                
                # 连接两个特征
                combined = torch.cat([fi, fj], dim=1)
                
                # 应用融合网络
                fusion_result = self.low_level_fusion[fusion_idx](combined)
                
                # 更新特征
                enhanced_features[i] = enhanced_features[i] + fusion_result
                enhanced_features[j] = enhanced_features[j] + fusion_result
                
                fusion_idx += 1
        
        return enhanced_features
    
    def _apply_mid_level_fusion(self, feature_list):
        """应用中层级融合"""
        enhanced_features = []
        
        # 将特征堆叠为序列
        stacked_features = torch.stack(feature_list, dim=1)  # [batch_size, num_modalities, common_dim]
        
        # 应用多头自注意力
        attn_output, _ = self.mid_level_attention(
            stacked_features, stacked_features, stacked_features
        )
        
        # 应用门控机制
        for i, feature in enumerate(feature_list):
            # 提取对应模态的注意力输出
            modal_attn = attn_output[:, i, :]
            
            # 计算门控值
            gate_input = torch.cat([feature, modal_attn], dim=1)
            gate = self.mid_level_gate(gate_input)
            
            # 应用门控
            enhanced = feature * (1 - gate) + modal_attn * gate
            
            enhanced_features.append(enhanced)
        
        return enhanced_features
    
    def _apply_high_level_fusion(self, feature_list, weights):
        """应用高层级融合"""
        # 连接所有特征
        concat_features = torch.cat(feature_list, dim=1)
        
        # 应用融合网络
        fused = self.high_level_fusion(concat_features)
        
        # 应用专家混合系统
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(fused))
        
        # 计算专家门控
        gate_logits = self.expert_gate(fused)
        gate_probs = F.softmax(gate_logits, dim=1)
        
        # 组合专家输出
        combined_expert = torch.zeros_like(expert_outputs[0])
        for i, expert_output in enumerate(expert_outputs):
            combined_expert += gate_probs[:, i:i+1] * expert_output
        
        return combined_expert

class FeaturePyramid(nn.Module):
    """
    特征金字塔模块，处理多尺度特征
    """
    
    def __init__(self, hidden_dim, dropout=0.1):
        super(FeaturePyramid, self).__init__()
        
        # 不同尺度的特征提取器
        self.scale1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        self.scale2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        self.scale3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )
        
        self.scale4 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
        )
        
        # 将多尺度特征合并回原始维度
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, hidden_dim]
            
        Returns:
            多尺度增强特征 [batch_size, hidden_dim]
        """
        # 提取不同尺度的特征
        s1 = self.scale1(x)
        s2 = self.scale2(x)
        s3 = self.scale3(x)
        s4 = self.scale4(x)
        
        # 连接多尺度特征
        multi_scale = torch.cat([s1, s2, s3, s4], dim=1)
        
        # 合并回原始维度
        enhanced = self.combine(multi_scale)
        
        # 残差连接
        output = x + enhanced
        
        return output