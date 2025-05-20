import os
import numpy as np
from math import sqrt
from scipy import stats
import torch_geometric 
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA
import torch
from torch.serialization import add_safe_globals

add_safe_globals([torch_geometric.data.data.DataEdgeAttr])
# from utils_smiecfp import *
# from data_gen_modify import *
import pdb
class formDataset(InMemoryDataset):
    def __init__(self, root='../',dataset='data_train',
                 encodedSmi=None, ecfp=None, y=None, smile_graph=None, mfbert_features=None):
        super(formDataset, self).__init__(root)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(encodedSmi, ecfp, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, encodedSmi, ecfp, log, smile_graph):
        assert (len(encodedSmi) == len(ecfp) and len(ecfp) == len(log)), "The three lists must be the same length!"

        data_list = []

        for idx, (smi, ep, y_) in enumerate(zip(encodedSmi, ecfp, log)):
            smi = torch.LongTensor([smi.tolist()])
            ep = torch.FloatTensor([ep.tolist()])
            y = torch.FloatTensor([y_.tolist()])

            mfbert_inputs=self.tokenizer(smi, return_tensors='pt', padding=True, truncation=True)
            input_ids=mfbert_inputs['input_ids']
            attention_mask=mfbert_inputs['attention_mask']

            with torch.no_grad():
                mfbert_output=self.mfbert_model(input_ids=input_ids, attention_mask=attention_mask)
                mfbert_pooled_output=mfbert_output.last_hidden_state[:, 0, :]

            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[idx]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                )
            GCNData.smi = smi
            GCNData.ep = ep
            GCNData.y = y
            GCNData.mfbert_out=mfbert_pooled_output
            # GCNData.encodedSmi = torch.LongTensor([smi])
            # GCNData.ecfp = torch.IntTensor([ep])
            # GCNData.y = torch.FloatTensor([y_])
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])

class formDataset_Single(InMemoryDataset):
    def __init__(self, root='../',dataset='data_train',
                 encodedSmi=None, ecfp=None, y=None, smile_graph=None,mfbert_features=None):
        super(formDataset_Single, self).__init__(root)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(encodedSmi, ecfp, y, smile_graph,mfbert_features)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)



    def process(self, encodedSmi, ecfp, log, smile_graph,mfbert_features):

        #assert (len(encodedSmi) == len(ecfp) and len(ecfp) == len(y)==len(smile_graph)==len(mfbert_features)), "The three lists must be the same length!"
        #print(len(encodedSmi),len(ecfp), len(y),len(smile_graph),len(mfbert_features))
        data_list = []
#         计算需要填充的最大长度
        for idx, (enSmi, ep, y_) in enumerate(zip(encodedSmi, ecfp, log)):
            smi = torch.LongTensor([enSmi])
            ep = torch.FloatTensor([ep])
            pros = torch.FloatTensor([float(y_)])
    # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[idx]
            #input_ids, attention_masks=mfbert_features[idx]
            input_id, attention_mask=mfbert_features[idx]
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                               )
            GCNData.smi = smi
            GCNData.ep = ep
            GCNData.y = pros
          #  GCNData.input_id = torch.LongTensor(input_id)  # 添加 MFBERT input_ids
           # GCNData.attention_mask = torch.LongTensor(attention_mask)   # 添加 MFBERT attention_mask
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # 打印调试信息，检查每个输入的大小
            GCNData.input_id = torch.LongTensor(input_id)  # 添加 MFBERT input_ids
            GCNData.attention_mask = torch.LongTensor(attention_mask)   # 添加 MFBERT attention_mask
            data_list.append(GCNData)
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
       # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])
