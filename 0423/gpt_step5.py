import matplotlib.pyplot as plt
import pickle
import pdb
import os
import numpy as np
import pandas as pd
import re
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pickle
import random
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data, DataLoader
from transformers import BertTokenizer, BertModel
import csv
from rdkit import Chem
from rdkit.Chem import AllChem
import random
sys.path.append('../../util')
from utils_smiecfp import *
from data_gen_modify import *
from utils_MFBERT import *

import os
import json,pickle
from collections import OrderedDict
from rdkit.Chem import MolFromSmiles
import networkx as nx

from MFBERTmodel import MFBERT
from MFBERT_Tokenizer import MFBERTTokenizer
MFBERT_PATH = 'Model'
#tokenizer = BertTokenizer.from_pretrained(MFBERT_PATH)
tokenizer = MFBERTTokenizer.from_pretrained('Model/',
                                                dict_file = 'Model/dict.txt')

#model = BertModel.from_pretrained(MFBERT_PATH)
model = MFBERT()

def pad_tensor(tensor, target_size, dim=0, padding_value=0):
    """
    对输入的 tensor 进行 padding 以确保在目标维度上大小一致
    """
    padding_size = target_size - tensor.size(dim)
    if padding_size > 0:
        pad = [0] * (tensor.dim() * 2)  # 创建一个 pad 参数
        pad[dim * 2 + 1] = padding_size
        tensor = torch.nn.functional.pad(tensor, pad, value=padding_value)
    return tensor

def generate_mfbert_embedding(smiles):
    """使用MFBERT生成SMILES字符串的特征"""
    inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True, max_length=512)
#    input_ids=inputs['input_ids']
 #   attention_mask=inputs.get('attention_mask',None)
    with torch.no_grad():
        outputs=model(inputs)
    if isinstance(outputs, dict):  # 如果返回的是字典
        if model.inference == 'cls':
            embeddings = outputs['CLS_FINGERPRINT']
        elif model.inference == 'mean':
            embeddings = outputs['MEAN_FINGERPRINT']
    else:  # 如果返回的是张量
        if model.inference == 'cls':
            embeddings = outputs[:, 0, :]  # 获取第一个 token 的嵌入（CLS）
        elif model.inference == 'mean':
            embeddings = outputs.mean(dim=1)  # 获取所有 token 的平均嵌入
    #print("Embeddings shape:", embeddings.shape)
    #print("Input IDs shape:", inputs['input_ids'].shape)
    #print("Attention Mask shape:", inputs['attention_mask'].shape)

    input_ids = inputs['input_ids'].squeeze()
    attention_mask = inputs['attention_mask'].squeeze()

    input_ids = pad_tensor(input_ids, 512)
    attention_mask = pad_tensor(attention_mask, 512)

 #   print(f"Embeddings shape: {embeddings.shape}")
  #  print(f"Input IDs shape: {input_ids.shape}")
   # print(f"Attention Mask shape: {attention_mask.shape}")

    return embeddings.squeeze(), input_ids, attention_mask


def atom_features(atom): #这个函数根据原子对象的特征（如原子符号、度数、氢原子数、隐含化合价和芳香性），生成一个数组，表示该原子的特征向量。
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):  #将SMILES格式的分子字符串转换为分子图
    mol = Chem.MolFromSmiles(smile)
    
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return c_size, features, edge_index

def getProcessData(label):
    if label == 'train':
        allFoldPath = '/home/nudt_cleng/ccleng/MMFDL-main/notebook/0423/results/processed/Lipophilicity_MFBERT_dataTrain.dat'
    else:
        allFoldPath = '/home/nudt_cleng/ccleng/MMFDL-main/notebook/0423/results/processed/Lipophilicity_MFBERT_dataTest.dat'

    vocab_path = 'results/Lipophilicity_smiles_char_dict.pkl'
    with open(vocab_path, 'rb') as f:
        smilesVoc = pickle.load(f)

    

    allData = getData(allFoldPath)
    smiles = []
    encodedSmi = []
    ecfp = []
    properties = []
    mfbert_features=[] #for MFBERT

    for index, item in enumerate(allData):
        smi = item[0]  # SMILES 字符串
        smiles.append(smi)
        enSmi = [int(float(val)) for val in item[1].split(',')]
     #   enSmi = item[1].split(',')
      #  enSmi = [int(float(val)) for val in enSmi]
        encodedSmi.append(enSmi)
         # 使用MFBERT生成特征
        #mfbert_embedding = generate_mfbert_embedding(item[0])
        embeddings, input_ids,attention_mask = generate_mfbert_embedding(smi)
#        if input_ids.size(0) != 44:
 #           print(f"Warning: input_ids length mismatch for {smi}. Expected 44, got {input_ids.size(0)}")
        mfbert_features.append((input_ids.tolist(), attention_mask.tolist()))

        ep = [int(val) for val in item[2].split(',')]
        ecfp.append(ep)
        properties.append(item[3])

    
    smi_to_graph = {}
    resultSmi = []
    resultEp = []
    resultY = []
    resultMF=[]
    count = 0
    max_ecfp_size = max([len(ep) for ep in ecfp])  # 找到最大ECFP长度
    for smi, enSmi, ep, y, mf in zip(smiles, encodedSmi, ecfp,properties, mfbert_features):
        c_size, features, edge_index = smile_to_graph(smi)
        if not edge_index:
            continue
        # 对 ECFP 进行填充
        padded_ecfp = pad_tensor(torch.FloatTensor(ep), max_ecfp_size)
        smi_to_graph[count] = (c_size, features, edge_index)
        resultSmi.append(enSmi)
        resultEp.append(ep)
        resultY.append(y)
        resultMF.append(mf)
        count += 1
        # 调试输出
    print(f"resultSmi length: {len(resultSmi)}")
    print(f"resultEp length: {len(resultEp)}")
    print(f"resultY length: {len(resultY)}")
    print(f"smi_to_graph length: {len(smi_to_graph)}")
    print(f"resultMF length: {len(resultMF)}")

    return resultSmi, resultEp, resultY, smi_to_graph, resultMF



processed_data_file_train = '/home/nudt_cleng/ccleng/MMFDL-main/notebook/0423/results/processed/Lipophilicity_MFBERT_dataTrain.pt'
processed_data_file_test = '/home/nudt_cleng/ccleng/MMFDL-main/notebook/0423/results/processed/Lipophilicity_MFBERT_datatest.pt'

if (not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test)):
 #   pdb.set_trace()
    train_enSmi, train_ep, train_pro, train_smiGraph,train_mfbert = getProcessData('train')
    test_enSmi, test_ep, test_pro, test_smiGraph,test_mfbert = getProcessData('test')
    train_data = formDataset_Single(root='/home/nudt_cleng/ccleng/MMFDL-main/notebook/0423/results/processed/', dataset='Lipophilicity_MFBERT_dataTrain', encodedSmi=train_enSmi, ecfp=train_ep, y=train_pro, smile_graph=train_smiGraph,mfbert_features=train_mfbert)
    test_data = formDataset_Single(root='/home/nudt_cleng/ccleng/MMFDL-main/notebook/0423/results/processed/', dataset='Lipophilicity_MFBERT_dataTest', encodedSmi=test_enSmi, ecfp=test_ep, y=test_pro, smile_graph=test_smiGraph,mfbert_features=test_mfbert)

    print('preparing data_train.pt in pytorch format!')
    print('preparing data_test.pt in pytorch format!')
    
    # 检查各个特征的维度
    print("MMFDL token维度:", len(test_enSmi[0]))
    print("ECFP维度:", len(test_ep[0]))
    print("MFBERT维度:", len(test_mfbert[0][0]))
    
else:
    print('preparing data_train.pt is already created!')
    print('preparing data_test.pt is already created!')
