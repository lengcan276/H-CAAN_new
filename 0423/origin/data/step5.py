import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
import pandas as pd
import re
import sys

import csv
from rdkit import Chem
from rdkit.Chem import AllChem
import random
sys.path.append('../../../../util')
from utils_smiecfp import *
from data_gen_modify import *
from utils import *

def tokenizer(smile):
    "Tokenizes SMILES string"
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|_|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regezz = re.compile(pattern)
    tokens = [token for token in regezz.findall(smile)]
    assert smile == ''.join(tokens), ("{} could not be joined".format(smile))
    return tokens

def build_vocab(smiles, vocab_name='char_dict', save_dir=None):
    ### Build vocab dictionary
    print('building dictionary...')
    char_dict = {}
    char_idx = 1
    mol_toks = []
    with open(smiles, 'r') as f:
        for line in f:
            line = line.split('\n')[0]
            if line.lower() in ['smile', 'smiles', 'selfie', 'selfies']:
                pass
            else:
                mol = tokenizer(line)
                for tok in mol:
                    if tok not in char_dict.keys():
                        char_dict[tok] = char_idx
                        char_idx += 1
                    else:
                        pass
                mol.append('<end>')
                mol_toks.append(mol)

    ### Write dictionary to file
    with open(os.path.join(save_dir, vocab_name+'.pkl'), 'wb') as f:
        pickle.dump(char_dict, f)
    return char_dict



def make_variable_length(smiles, letters):
    resultVec = []
    char_list = tokenizer(smiles)
    for item in char_list:
        resultVec.append(letters[item])
    return len(resultVec)

def splitData_dat(filePath, smilesVoc):
    allData = []
    with open(filePath, "r") as dat_file:
        for line in dat_file:
            line = line.strip()
            elements = line.split(" ")
            allData.append(elements)
    random.shuffle(allData)
    count = int(len(allData) / 7)
    dataTrain = []
    dataTest = []
    for i in range(count):
        for j in range(7):
            if j < 6:
                dataTrain.append(allData[i*7+j])
            else:
                dataTest.append(allData[i*7+j])


    savePath = '../dataSour/Lipophilicity_dataTrain.dat'
    with open(savePath, 'w') as dat_file:
        for row in dataTrain:
            smi = row[0]   #smi
            enSmi = make_variable_one(smi, smilesVoc, 100)
            enSmi = ','.join(map(str, enSmi))
            ep = row[1]
            exp = row[2]    #y
            line = smi + ' ' + enSmi + ' ' + ep + ' ' + exp
            dat_file.write(line + '\n')
    print('writing ' + savePath + ' finished!')

    savePath = '../dataSour/Lipophilicity_dataTest.dat'
    with open(savePath, 'w') as dat_file:
        for row in dataTest:
            smi = row[0]
            enSmi=make_variable_one(smi, smilesVoc, 100)  #smile vector 将给定的 SMILES 字符串转换为定长的数字序列表示
            enSmi = ','.join(map(str, enSmi))
            ep = row[1] 
            exp = row[2]
            line = smi + ' ' + enSmi + ' ' + ep + ' ' + exp
            dat_file.write(line + '\n')
    print('writing ' + savePath + ' finished!')


def check_ecfp(filePath):
    smiles = pd.read_csv(filePath)['smiles'].tolist()
    notGen = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol != None:
            if mol.HasSubstructMatch(Chem.MolFromSmarts("[H]")):
                mol = Chem.RemoveHs(mol)
            ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            if ecfp == None:
                notGen.append(smi)
        else:
            notGen.append(smi)


def gene_ecfp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    radius = 2
    nBits = 1024
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return ecfp

def transToDat_All(filePath, savePath):  #transfer smile and y to ecfp format
    df = pd.read_csv(filePath)
    smiles = df['SMILES'].tolist()
    y = df['expt'].tolist()
    with open(savePath, "w") as dat_file:
        for smi, y_ in zip(smiles, y):
            ecfp = gene_ecfp(smi)   #ecfp
            ep = ','.join(str(x) for x in ecfp)
            yy = str(y_)
            line = smi + ' ' + ep + ' ' + yy
            dat_file.write(line + '\n')
    print('writing ' + savePath + ' finished!')

import numpy as np
import os
import json,pickle
from collections import OrderedDict
from rdkit.Chem import MolFromSmiles
import networkx as nx


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
        allFoldPath = '/home/nudt_cleng/ccleng/MMFDL-main/notebook/0423/origin/results/Lipophilicity_dataTrain.dat'
    else:
        allFoldPath = '/home/nudt_cleng/ccleng/MMFDL-main/notebook/0423/origin/results/Lipophilicity_dataTest.dat'

    vocab_path = '/home/nudt_cleng/ccleng/MMFDL-main/notebook/0423/origin/results/Lipophilicity_smiles_char_dict.pkl'
    with open(vocab_path, 'rb') as f:
        smilesVoc = pickle.load(f)

    

    allData = getData(allFoldPath)
    smiles = []
    encodedSmi = []
    ecfp = []
    properties = []
    for index, item in enumerate(allData):
        smiles.append(item[0])
        enSmi = item[1].split(',')
        enSmi = [int(float(val)) for val in enSmi]
        encodedSmi.append(enSmi)
        ep = item[2].split(',')
        ep = [int(val) for val in ep]
        ecfp.append(ep)
        properties.append(item[3])
    
    smi_to_graph = {}
    resultSmi = []
    resultEp = []
    resultY = []
    count = 0
    for smi, enSmi, ep, y in zip(smiles, encodedSmi, ecfp,properties):
        c_size, features, edge_index = smile_to_graph(smi)
        if edge_index == []:
            continue
        smi_to_graph[count] = smile_to_graph(smi)
        resultSmi.append(enSmi)
        resultEp.append(ep)
        resultY.append(y)
        count = count + 1

    return resultSmi, resultEp, resultY, smi_to_graph

processed_data_file_train = '/home/nudt_cleng/ccleng/MMFDL-main/notebook/0423/origin/results/Lipophilicity_dataTrain.pt'
processed_data_file_test = '/home/nudt_cleng/ccleng/MMFDL-main/notebook/0423/origin/results/Lipophilicity_dataTest.pt'

if (not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test)):

    train_enSmi, train_ep, train_pro, train_smiGraph = getProcessData('train')
    test_enSmi, test_ep, test_pro, test_smiGraph = getProcessData('test')
    print(len(train_enSmi),len(train_ep),len(train_pro),len(train_smiGraph))
    train_data = formDataset_Single(root='/home/nudt_cleng/ccleng/MMFDL-main/notebook/0423/origin/results/', dataset='Lipophilicity_dataTrain', encodedSmi=train_enSmi, ecfp=train_ep, y=train_pro, smile_graph=train_smiGraph)
    test_data = formDataset_Single(root='/home/nudt_cleng/ccleng/MMFDL-main/notebook/0423/origin/results', dataset='Lipophilicity_dataTest', encodedSmi=test_enSmi, ecfp=test_ep, y=test_pro, smile_graph=test_smiGraph)
   
    print('preparing data_train.pt in pytorch format!')
    print('preparing data_test.pt in pytorch format!')

    
else:
    print('preparing data_train.pt is already created!')
    print('preparing data_test.pt is already created!')
