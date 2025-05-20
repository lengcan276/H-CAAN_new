import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool as gmp, global_add_pool as gap
from transformers import RobertaForMaskedLM
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, max_length):
        super(PositionalEncoding, self).__init__()
        self.embedding_size = embedding_size
        self.max_length = max_length

        pe = torch.zeros(max_length, embedding_size)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * (-math.log(10000.0) / embedding_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.embedding_size)
        seq_len = x.size(1)
        pe = self.pe[:, :seq_len, :]
        return x + pe

class comModel(nn.Module):
    def __init__(self, args):
        super(comModel, self).__init__()
        # 基础配置
        self.num_features_smi = args['num_features_smi']
        self.num_features_ecfp = args['num_features_ecfp']
        self.num_features_x = args['num_features_x']
        self.dropout = args['dropout']
        self.num_layer = args['num_layer']
        self.num_heads = args['num_heads']
        self.hidden_dim = args['hidden_dim']
        self.output_dim = args['output_dim']
        self.n_output = args['n_output']

        # SMILES Encoder
        self.smi_embedding = nn.Embedding(self.num_features_smi, self.hidden_dim)
        self.pos_encoder = PositionalEncoding(self.hidden_dim, 1000)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.num_heads,
                dim_feedforward=self.hidden_dim*4,
                dropout=self.dropout,
                batch_first=True
            ),
            num_layers=self.num_layer
        )

        # ECFP Encoder
        self.ep_gru = nn.GRU(self.num_features_ecfp, self.hidden_dim,
                            self.num_layer, batch_first=True, bidirectional=True)
        self.ep_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim*2,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )

        # Graph Encoder
        self.gcn_conv1 = GCNConv(self.num_features_x, self.hidden_dim)
        self.gcn_conv2 = GCNConv(self.hidden_dim, self.hidden_dim)
        self.graph_norm1 = nn.LayerNorm(self.hidden_dim)
        self.graph_norm2 = nn.LayerNorm(self.hidden_dim)

        # MFBERT
        self.mfbert = RobertaForMaskedLM.from_pretrained('Model/pre-trained')
        self.mfbert.config.output_hidden_states = True
        self.mfbert_pooler = nn.Sequential(
            nn.Linear(self.mfbert.config.hidden_size, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )

        # Feature Fusion
        self.feature_transform = nn.ModuleDict({
            'smi': nn.Linear(self.hidden_dim, self.output_dim),
            'ep': nn.Linear(self.hidden_dim*2, self.output_dim),
            'graph': nn.Linear(self.hidden_dim, self.output_dim),
            'mfbert': nn.Linear(self.hidden_dim, self.output_dim)
        })

        self.feature_norms = nn.ModuleDict({
            'smi': nn.LayerNorm(self.output_dim),
            'ep': nn.LayerNorm(self.output_dim),
            'graph': nn.LayerNorm(self.output_dim),
            'mfbert': nn.LayerNorm(self.output_dim)
        })

        # Output Layers
        self.output_layers = nn.ModuleDict({
            'smi': nn.Linear(self.hidden_dim, self.n_output),
            'ep': nn.Linear(self.hidden_dim*2, self.n_output),
            'graph': nn.Linear(self.hidden_dim, self.n_output),
            'fusion': nn.Linear(self.output_dim, self.n_output)
        })

    def forward(self, encodedSmi, encodedSmi_mask, ecfp, x, edge_index, batch, input_ids, attention_mask):
        # SMILES Processing
        smi_embedded = self.smi_embedding(encodedSmi)
        smi_embedded = self.pos_encoder(smi_embedded)

        # Convert mask to boolean
        smi_mask = encodedSmi_mask.bool()  # 转换为布尔型
        smi_encoded = self.transformer_encoder(smi_embedded, src_key_padding_mask=smi_mask)
        smi_features = torch.mean(smi_encoded, dim=1)  # Global average pooling

        # ECFP Processing
        ecfp = ecfp.unsqueeze(1)
        ep_out, _ = self.ep_gru(ecfp)
        ep_features, _ = self.ep_attention(ep_out, ep_out, ep_out)
        ep_features = torch.mean(ep_features, dim=1)  # Global average pooling

        # Graph Processing
        x = self.gcn_conv1(x, edge_index)
        x = F.relu(self.graph_norm1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn_conv2(x, edge_index)
        x = F.relu(self.graph_norm2(x))
        graph_features = gap(x, batch)

        # MFBERT Processing
        mfbert_outputs = self.mfbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True
        )
        mfbert_features = self.mfbert_pooler(mfbert_outputs.hidden_states[-1][:, 0, :])

        # Feature Transformation
        transformed_features = {
            'smi': self.feature_transform['smi'](smi_features),
            'ep': self.feature_transform['ep'](ep_features),
            'graph': self.feature_transform['graph'](graph_features),
            'mfbert': self.feature_transform['mfbert'](mfbert_features)
        }

        # Apply Layer Normalization
        normalized_features = {
            name: self.feature_norms[name](feature)
            for name, feature in transformed_features.items()
        }

        # Combine features using weighted sum
        weights = torch.softmax(torch.stack([
            torch.mean(feat) for feat in normalized_features.values()
        ]), dim=0)

        combined_features = sum(w * f for w, f in zip(weights, normalized_features.values()))

        # Generate outputs
        smi_out = self.output_layers['smi'](smi_features)
        ep_out = self.output_layers['ep'](ep_features)
        graph_out = self.output_layers['graph'](graph_features)
        fusion_out = self.output_layers['fusion'](combined_features)

        return smi_out, ep_out, graph_out, fusion_out.squeeze()

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

