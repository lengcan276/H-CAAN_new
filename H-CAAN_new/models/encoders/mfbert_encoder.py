# models/encoders/mfbert_encoder.py

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaConfig

class MFBERTEncoder(nn.Module):
    """
    MFBERT编码器，使用预训练的分子语言模型
    """
    
    def __init__(self, hidden_dim, dropout=0.1, pretrained_model_path=None):
        """
        初始化MFBERT编码器
        
        Args:
            hidden_dim (int): 隐藏层维度
            dropout (float): Dropout概率
            pretrained_model_path (str, optional): 预训练模型路径
        """
        super(MFBERTEncoder, self).__init__()
        
        # 加载MFBERT预训练模型
        if pretrained_model_path:
            self.mfbert = RobertaModel.from_pretrained(pretrained_model_path)
        else:
            # 如果没有预训练模型，则使用默认配置初始化
            config = RobertaConfig(
                vocab_size=50265,
                hidden_size=768,
                num_hidden_layers=6,
                num_attention_heads=12,
                intermediate_size=3072
            )
            self.mfbert = RobertaModel(config)
        
        # MFBERT输出维度
        self.mfbert_dim = self.mfbert.config.hidden_size
        
        # 投影到目标隐藏维度
        self.projection = nn.Sequential(
            nn.Linear(self.mfbert_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 是否冻结预训练参数
        self.freeze_pretrained = False
    
    def freeze_pretrained_params(self):
        """冻结预训练MFBERT参数"""
        self.freeze_pretrained = True
        for param in self.mfbert.parameters():
            param.requires_grad = False
    
    def unfreeze_pretrained_params(self):
        """解冻预训练MFBERT参数"""
        self.freeze_pretrained = False
        for param in self.mfbert.parameters():
            param.requires_grad = True
    
    def forward(self, inputs):
        """
        前向传播
        
        Args:
            inputs: MFBERT输入，包括input_ids和attention_mask
            
        Returns:
            输出的MFBERT表示 [batch_size, hidden_dim]
        """
        # 提取输入
        input_ids = inputs.get('input_ids')
        attention_mask = inputs.get('attention_mask')
        
        # 通过MFBERT模型
        if self.freeze_pretrained:
            with torch.no_grad():
                outputs = self.mfbert(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=True
                )
        else:
            outputs = self.mfbert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
        
        # 获取[CLS]表示和所有隐藏状态
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS]表示
        hidden_states = outputs.last_hidden_state         # 所有隐藏状态
        
        # 计算SMILES注意力池化
        # 在所有除了填充的token上进行池化
        if attention_mask is not None:
            # 扩展attention_mask以适应hidden_states的形状
            extended_attention_mask = attention_mask.unsqueeze(-1)
            
            # 应用mask并计算平均池化
            masked_hidden = hidden_states * extended_attention_mask
            sum_hidden = masked_hidden.sum(dim=1)
            sum_mask = attention_mask.sum(dim=1, keepdim=True)
            
            # 避免除以零
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            
            # 平均池化
            mean_pooled = sum_hidden / sum_mask
        else:
            # 如果没有mask，直接平均池化
            mean_pooled = hidden_states.mean(dim=1)
        
        # 组合[CLS]表示和平均池化表示
        combined = (cls_output + mean_pooled) / 2
        
        # 投影到目标维度
        output = self.projection(combined)
        
        return output