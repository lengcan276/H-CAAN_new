# models/encoders/gcn_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool

class GCNEncoder(nn.Module):
    """
    使用图卷积网络(GCN)的分子图编码器
    """
    
    def __init__(self, input_dim, hidden_dim, num_layers=3, dropout=0.1):
        """
        初始化GCN编码器
        
        Args:
            input_dim (int): 节点特征维度
            hidden_dim (int): 隐藏层维度
            num_layers (int): GCN层数
            dropout (float): Dropout概率
        """
        super(GCNEncoder, self).__init__()
        
        # 初始节点特征转换
        self.node_embedding = nn.Linear(input_dim, hidden_dim)
        
        # GCN层
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            # 第一层的输入维度是hidden_dim，后续层的输入和输出维度都是hidden_dim
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data):
        """
        前向传播
        
        Args:
            data: PyG数据对象，包含x（节点特征）和edge_index（边索引）
            
        Returns:
            输出的图表示 [batch_size, hidden_dim]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 初始节点嵌入
        x = self.node_embedding(x)
        
        # 应用GCN层
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            # 残差连接
            identity = x
            
            # 图卷积
            x = conv(x, edge_index)
            x = batch_norm(x)
            x = F.relu(x)
            x = self.dropout(x)
            
            # 残差连接
            x = x + identity
        
        # 全局池化：将节点特征聚合为分子级表示
        x = global_mean_pool(x, batch)
        
        # 输出层
        x = self.output_layer(x)
        
        return x

class EnhancedGCNEncoder(nn.Module):
    """
    增强版GCN编码器，结合了GCN和GAT，并添加了边特征
    """
    
    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers=3, num_heads=4, dropout=0.1):
        """
        初始化增强版GCN编码器
        
        Args:
            node_dim (int): 节点特征维度
            edge_dim (int): 边特征维度
            hidden_dim (int): 隐藏层维度
            num_layers (int): 层数
            num_heads (int): GAT头数
            dropout (float): Dropout概率
        """
        super(EnhancedGCNEncoder, self).__init__()
        
        # 节点嵌入
        self.node_embedding = nn.Linear(node_dim, hidden_dim)
        
        # 边嵌入
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ) if edge_dim > 0 else None
        
        # GCN和GAT交替层
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            # 偶数层使用GCN，奇数层使用GAT
            if i % 2 == 0:
                self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
            else:
                self.conv_layers.append(GATConv(
                    hidden_dim, 
                    hidden_dim // num_heads, 
                    heads=num_heads,
                    dropout=dropout
                ))
            
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # 多尺度池化
        self.pool_mean = global_mean_pool
        self.pool_add = global_add_pool
        
        # 组合池化结果
        self.combine_pools = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data):
        """
        前向传播
        
        Args:
            data: PyG数据对象，包含x（节点特征）、edge_index（边索引）和edge_attr（边属性）
            
        Returns:
            输出的图表示 [batch_size, hidden_dim]
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # 初始节点嵌入
        x = self.node_embedding(x)
        
        # 边特征处理
        edge_features = None
        if self.edge_embedding is not None and edge_attr is not None:
            edge_features = self.edge_embedding(edge_attr)
        
        # 应用卷积层
        for i, (conv, batch_norm) in enumerate(zip(self.conv_layers, self.batch_norms)):
            # 残差连接
            identity = x
            
            # 根据层类型应用不同的卷积
            if i % 2 == 0:  # GCN层
                x = conv(x, edge_index)
            else:  # GAT层
                if edge_features is not None:
                    x = conv(x, edge_index, edge_attr=edge_features)
                else:
                    x = conv(x, edge_index)
            
            x = batch_norm(x)
            x = F.relu(x)
            x = self.dropout(x)
            
            # 残差连接
            if x.shape == identity.shape:
                x = x + identity
        
        # 多尺度池化
        x_mean = self.pool_mean(x, batch)
        x_add = self.pool_add(x, batch)
        
        # 组合池化结果
        x_combined = torch.cat([x_mean, x_add], dim=1)
        x = self.combine_pools(x_combined)
        
        # 输出层
        x = self.output_layer(x)
        
        return x