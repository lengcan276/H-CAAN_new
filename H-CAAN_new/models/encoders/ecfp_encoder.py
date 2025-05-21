# models/encoders/ecfp_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ECFPEncoder(nn.Module):
    """
    为ECFP指纹设计的编码器，使用BiGRU结构
    ECFP (Extended Connectivity Fingerprint)是一种广泛使用的分子指纹表示方法
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.1):
        """
        初始化ECFP编码器
        
        Args:
            input_dim (int): 输入指纹维度
            hidden_dim (int): 隐藏层维度
            num_layers (int): GRU层数
            dropout (float): Dropout概率
        """
        super(ECFPEncoder, self).__init__()
        
        # 输入投影
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # BiGRU层
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,  # 双向，所以是一半
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 自注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入的ECFP指纹 [batch_size, input_dim]
            
        Returns:
            输出的ECFP表示 [batch_size, hidden_dim]
        """
        batch_size = x.size(0)
        
        # 扩展指纹为序列以适应GRU
        # 通过复制和切片将指纹变为伪序列
        seq_len = 32  # 假设将指纹分为32段
        chunk_size = x.size(1) // seq_len
        x_seq = x.view(batch_size, seq_len, chunk_size)
        
        # 投影到隐藏维度
        x_projected = self.input_projection(x_seq)
        
        # 通过BiGRU
        gru_output, _ = self.gru(x_projected)
        
        # 应用自注意力
        attn_output, _ = self.attention(gru_output, gru_output, gru_output)
        
        # 残差连接
        combined = gru_output + attn_output
        
        # 全局池化
        global_repr = combined.mean(dim=1)
        
        # 输出层
        output = self.output_layer(global_repr)
        
        return output