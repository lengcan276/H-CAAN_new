import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool as gap
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig
import pandas as ps
import os

class MFBERT(nn.Module):
    def __init__(self, weights_dir='Model/pre-trained', return_attention=False, inference_method='mean'):
        super(MFBERT, self).__init__()
        self.return_attention = return_attention
        if inference_method not in ['cls', 'mean']:
            raise ValueError('Please Enter a valid inference method from {"cls", "mean"}')
        else:
            self.inference = inference_method.lower()

        if os.path.isdir(weights_dir) and weights_dir != '':
            self.base = RobertaModel.from_pretrained(weights_dir, output_attentions=return_attention)
            print('Loaded Pre-trained weights...')
        else:
            print('No Pre-trained weights found, initializing...')
            config = RobertaConfig.from_pretrained('Model/config.json')
            self.base = RobertaModel(config)

    def forward(self, inputs):
        all_output = self.base(**inputs)
        if self.return_attention:
            if self.inference == 'cls':
                return all_output[1], all_output[2]  # CLS + Attention
            elif self.inference == 'mean':
                return torch.mean(all_output[0], dim=1), all_output[2]  # Mean of hidden states + Attention
        else:
            if self.inference == 'cls':
                return all_output[1]  # CLS token output
            elif self.inference == 'mean':
                return torch.mean(all_output[0], dim=1)  # Mean of hidden states

