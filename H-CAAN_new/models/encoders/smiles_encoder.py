# models/encoders/smiles_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """位置编码，用于Transformer编码器
    
    将序列中token的位置信息注入到embedding中，使模型能够感知序列顺序
    """
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # 注册为缓冲区，不作为模型参数
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: [batch_size, seq_len, embedding_dim]
            
        Returns:
            带有位置编码的嵌入向量 [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class SmilesEncoder(nn.Module):
    """
    使用Transformer架构的SMILES编码器
    专门为捕获化学结构中的长距离关系而设计
    """
    
    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads, dropout=0.1, 
                 use_positional_encoding=True, ff_dim=None):
        """
        初始化SMILES编码器
        
        Args:
            vocab_size (int): 词汇表大小
            hidden_dim (int): 隐藏层维度
            num_layers (int): Transformer层数
            num_heads (int): 注意力头数
            dropout (float): Dropout概率
            use_positional_encoding (bool): 是否使用位置编码
            ff_dim (int): 前馈网络维度，如不指定则为hidden_dim*4
        """
        super(SmilesEncoder, self).__init__()
        
        self.use_positional_encoding = use_positional_encoding
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        
        # 位置编码
        if use_positional_encoding:
            self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)
        
        # 前馈网络维度
        if ff_dim is None:
            ff_dim = hidden_dim * 4
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        
        # 堆叠多层Transformer编码器
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # 输出层
        self.output_layer = nn.LayerNorm(hidden_dim)
        
        # 输出池化
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.global_repr = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, src_mask=None, return_attention=False):
        """
        前向传播
        
        Args:
            src: 输入的SMILES序列 [batch_size, seq_len]
            src_mask: 掩码 [batch_size, seq_len]
            return_attention (bool): 是否返回注意力权重
            
        Returns:
            output: 输出的SMILES表示 [batch_size, seq_len, hidden_dim]
            global_repr: 全局表示 [batch_size, hidden_dim]
            attn_weights: 如果return_attention为True则返回注意力权重
        """
        # 嵌入
        src = self.embedding(src) * math.sqrt(self.hidden_dim)
        
        # 应用位置编码
        if self.use_positional_encoding:
            src = self.pos_encoder(src)
        
        # 创建掩码以处理填充
        key_padding_mask = None
        if src_mask is not None:
            key_padding_mask = (src_mask == 0)  # 转换为布尔掩码
        
        # 通过Transformer编码器
        if return_attention:
            output, attn_weights = self._get_attention_weights(src, key_padding_mask)
        else:
            output = self.transformer_encoder(src, src_key_padding_mask=key_padding_mask)
        
        # 应用输出层归一化
        output = self.output_layer(output)
        
        # 计算全局表示
        # 首先转置以适应池化层的输入格式 [batch, hidden_dim, seq_len]
        transposed = output.transpose(1, 2)
        # 池化得到 [batch, hidden_dim, 1]
        pooled = self.pool(transposed)
        # 调整维度得到 [batch, hidden_dim]
        global_repr = pooled.squeeze(-1)
        # 应用额外的变换
        global_repr = self.global_repr(global_repr)
        
        if return_attention:
            return output, global_repr, attn_weights
        return output, global_repr
    
    def _get_attention_weights(self, src, key_padding_mask):
        """
        获取注意力权重
        
        Args:
            src: 输入的SMILES嵌入 [batch_size, seq_len, hidden_dim]
            key_padding_mask: 掩码 [batch_size, seq_len]
            
        Returns:
            output: 输出的SMILES表示 [batch_size, seq_len, hidden_dim]
            attn_weights: 注意力权重 list of [batch_size, num_heads, seq_len, seq_len]
        """
        attn_weights = []
        
        # 手动通过每一层以获取注意力权重
        x = src
        for layer in self.transformer_encoder.layers:
            # 保存当前层的注意力权重
            layer.self_attn.need_weights = True
            x, weights = layer.self_attn(
                x, x, x,
                key_padding_mask=key_padding_mask,
                need_weights=True
            )
            attn_weights.append(weights)
            
            # 完成该层的其余部分
            x = layer.norm1(x)
            x = x + layer._ff_block(x)
            x = layer.norm2(x)
        
        return x, attn_weights

class ChemicallyAwareSmilesEncoder(SmilesEncoder):
    """
    化学感知的SMILES编码器，通过额外的功能增强标准的SmilesEncoder
    识别化学结构中的功能基团并利用这一信息提高表示能力
    """
    
    def __init__(self, vocab_size, hidden_dim, num_layers, num_heads, dropout=0.1, 
                 use_positional_encoding=True, ff_dim=None, num_functional_groups=32):
        """
        初始化化学感知的SMILES编码器
        
        Args:
            vocab_size (int): 词汇表大小
            hidden_dim (int): 隐藏层维度
            num_layers (int): Transformer层数
            num_heads (int): 注意力头数
            dropout (float): Dropout概率
            use_positional_encoding (bool): 是否使用位置编码
            ff_dim (int): 前馈网络维度，如不指定则为hidden_dim*4
            num_functional_groups (int): 要识别的功能基团数量
        """
        super(ChemicallyAwareSmilesEncoder, self).__init__(
            vocab_size, hidden_dim, num_layers, num_heads, dropout,
            use_positional_encoding, ff_dim
        )
        
        self.num_functional_groups = num_functional_groups
        
        # 添加用于识别常见化学基团的功能
        self.functional_group_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_functional_groups),
            nn.Sigmoid()
        )
        
        # 用于加权功能基团的重要性
        self.functional_group_weighter = nn.Sequential(
            nn.Linear(hidden_dim + num_functional_groups, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 功能基团集中度检测
        self.group_concentration = nn.Sequential(
            nn.Linear(num_functional_groups, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.Sigmoid()
        )
    
    def forward(self, src, src_mask=None, return_attention=False, return_functional_groups=False):
        """
        前向传播，包含化学感知功能
        
        Args:
            src: 输入的SMILES序列 [batch_size, seq_len]
            src_mask: 掩码 [batch_size, seq_len]
            return_attention (bool): 是否返回注意力权重
            return_functional_groups (bool): 是否返回检测到的功能基团
            
        Returns:
            final_output: 增强后的序列表示 [batch_size, seq_len, hidden_dim]
            enhanced_global_repr: 增强后的全局表示 [batch_size, hidden_dim]
            attn_weights: 如果return_attention为True则返回注意力权重
            functional_groups: 如果return_functional_groups为True则返回功能基团检测结果
        """
        # 获取基本的SMILES表示
        if return_attention:
            base_output, global_repr, attn_weights = super().forward(src, src_mask, return_attention=True)
        else:
            base_output, global_repr = super().forward(src, src_mask)
        
        # 检测功能基团
        functional_groups = self.functional_group_detector(global_repr)  # [batch_size, num_functional_groups]
        
        # 对每个位置增强表示
        batch_size, seq_len, _ = base_output.shape
        
        # 在序列维度上重复功能基团表示
        expanded_fg = functional_groups.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, num_functional_groups]
        
        # 连接原始表示和功能基团
        combined = torch.cat([base_output, expanded_fg], dim=-1)  # [batch_size, seq_len, hidden_dim+num_functional_groups]
        
        # 应用权重器生成增强表示
        enhanced_output = self.functional_group_weighter(combined)
        
        # 计算功能基团集中度，用于全局表示增强
        group_concentration = self.group_concentration(functional_groups)
        
        # 增强全局表示
        enhanced_global_repr = global_repr * group_concentration
        
        # 残差连接
        final_output = base_output + enhanced_output
        
        # 根据参数返回不同的结果
        if return_attention and return_functional_groups:
            return final_output, enhanced_global_repr, attn_weights, functional_groups
        elif return_attention:
            return final_output, enhanced_global_repr, attn_weights
        elif return_functional_groups:
            return final_output, enhanced_global_repr, functional_groups
        
        return final_output, enhanced_global_repr

class SmilesTransformerEncoder(ChemicallyAwareSmilesEncoder):
    """
    SMILES Transformer Encoder - 与ChemicallyAwareSmilesEncoder功能相同的别名类
    以保持与H-CAAN项目其他代码的兼容性
    
    这个类在内部使用化学感知的编码器，能够识别化学功能基团，
    并利用该信息增强分子表示，适用于各种分子性质预测任务。
    """
    def __init__(self, vocab_size, hidden_dim=256, num_layers=3, num_heads=8, dropout=0.1,
                 use_positional_encoding=True, ff_dim=None, num_functional_groups=32):
        """
        初始化SMILES Transformer编码器
        
        Args:
            vocab_size (int): 词汇表大小
            hidden_dim (int): 隐藏层维度
            num_layers (int): Transformer层数
            num_heads (int): 注意力头数
            dropout (float): Dropout概率
            use_positional_encoding (bool): 是否使用位置编码
            ff_dim (int): 前馈网络维度，如不指定则为hidden_dim*4
            num_functional_groups (int): 要识别的功能基团数量
        """
        super(SmilesTransformerEncoder, self).__init__(
            vocab_size, hidden_dim, num_layers, num_heads, dropout,
            use_positional_encoding, ff_dim, num_functional_groups
        )
        # 添加H-CAAN项目特有的属性或方法
        self.supports_hierarchical_fusion = True
        self.output_dim = hidden_dim
        self.modality_name = "smiles"
        
    def get_pooled_representation(self, src, src_mask=None):
        """
        获取分子的池化表示，便于与其他模态融合
        
        Args:
            src: 输入的SMILES序列 [batch_size, seq_len]
            src_mask: 掩码 [batch_size, seq_len]
            
        Returns:
            pooled_repr: 分子的池化表示 [batch_size, hidden_dim]
        """
        _, pooled_repr = self.forward(src, src_mask)
        return pooled_repr
    
    def extract_features(self, src, src_mask=None, level="high"):
        """
        提取不同层次的特征，用于分层融合
        
        Args:
            src: 输入的SMILES序列 [batch_size, seq_len]
            src_mask: 掩码 [batch_size, seq_len]
            level: 特征层次，可选值为"low"、"mid"、"high"
            
        Returns:
            特定层次的特征
        """
        seq_output, global_repr = self.forward(src, src_mask)
        
        if level == "low":
            # 低层次特征: token级别的嵌入
            return self.embedding(src) * math.sqrt(self.hidden_dim)
        elif level == "mid":
            # 中层次特征: Transformer编码器的中间层输出
            mid_layer_idx = len(self.transformer_encoder.layers) // 2
            x = self.embedding(src) * math.sqrt(self.hidden_dim)
            if self.use_positional_encoding:
                x = self.pos_encoder(x)
                
            # 创建掩码
            key_padding_mask = None
            if src_mask is not None:
                key_padding_mask = (src_mask == 0)
                
            # 通过前半部分的编码器层
            for i, layer in enumerate(self.transformer_encoder.layers):
                if i > mid_layer_idx:
                    break
                x = layer(x, src_key_padding_mask=key_padding_mask)
            return x
        else:  # "high"
            # 高层次特征: 全局分子表示
            return global_repr