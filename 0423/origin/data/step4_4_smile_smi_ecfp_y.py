import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
import pandas as pd
import re
import sys
import pdb
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

def make_variable_one(smiles, letters, max_smiles_len):  #¿¿¿¿ SMILES ¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿
    resultVec = []
    char_list = tokenizer(smiles)
    for item in char_list:
       try:
           resultVec.append(letters[item])
       except KeyError:
           #print(f"Warning: '{item}' not found in the vocabulary!")  # 打印缺失的符号
           resultVec.append(letters.get("<unk>", 0))  # 使用 <unk> 代替，如果词汇表中不存在 <unk>，使用 0



    if len(resultVec) < max_smiles_len:
        resultVec.extend([0] * (max_smiles_len - len(resultVec)))
    elif len(resultVec) > max_smiles_len:
        resultVec = resultVec[:max_smiles_len]
    return resultVec

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


    savePath = '/home/nudt_cleng/ccleng/MMFDL-main/notebook/0423/origin/results/Lipophilicity_dataTrain.dat'
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

    savePath = '/home/nudt_cleng/ccleng/MMFDL-main/notebook/0423/origin/results/Lipophilicity_dataTest.dat'
    with open(savePath, 'w') as dat_file:
        for row in dataTest:
            smi = row[0]
            enSmi=make_variable_one(smi,smilevoc,100) #将给定的 SMILES 字符串转换为定长的数字序列表示
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
    #pdb.set_trace()
    mol = Chem.MolFromSmiles(smiles)
    radius = 2
    nBits = 1024
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    return ecfp

def transToDat_All(filePath, savePath):  #transfer smile and y to ecfp format
    df = pd.read_csv(filePath)
    smiles = df['smiles'].tolist()
    y = df['exp'].tolist()
    with open(savePath, "w") as dat_file:
        for smi, y_ in zip(smiles, y):
            ecfp = gene_ecfp(smi)   #ecfp
            ep = ','.join(str(x) for x in ecfp)
            yy = str(y_)
            line = smi + ' ' + ep + ' ' + yy
            dat_file.write(line + '\n')
    print('writing ' + savePath + ' finished!')


if __name__ == '__main__':
    filePath = '/home/nudt_cleng/ccleng/MMFDL-main/dataSour/Lipophilicity.csv'
    savePath = '/home/nudt_cleng/ccleng/MMFDL-main/notebook/0423/origin/results/LipophilicitydataAll.dat'
    #pdb.set_trace()
    transToDat_All(filePath, savePath)

    vocab_path = '/home/nudt_cleng/ccleng/MMFDL-main/notebook/0423/origin/results/Lipophilicity_smiles_char_dict.pkl'
    with open(vocab_path, 'rb') as f:
        smilevoc = pickle.load(f)

    filePath = '/home/nudt_cleng/ccleng/MMFDL-main/notebook/0423/origin/results/LipophilicitydataAll.dat'
    splitData_dat(filePath, smilevoc)
