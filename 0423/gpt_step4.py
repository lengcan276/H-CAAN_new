import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
import pandas as pd
import re
import sys
import pdb
from transformers import BertTokenizer, BertModel
import csv
from rdkit import Chem
from rdkit.Chem import AllChem
import random
sys.path.append('../../util')
from utils_smiecfp import *
from data_gen_modify import *
from utils import *
from MFBERTmodel import MFBERT
from MFBERT_Tokenizer import MFBERTTokenizer
MFBERT_PATH = 'Model'
#tokenizer = BertTokenizer.from_pretrained(MFBERT_PATH)
tokenizer = MFBERTTokenizer.from_pretrained('Model/',
                                                dict_file = 'Model/dict.txt')

#model = BertModel.from_pretrained(MFBERT_PATH)
model = MFBERT()
#from MFBERT_Tokenizer import MFBERTTokenizer
# Initialize MFBERT tokenizer
#fbert_tokenizer = MFBERTTokenizer.from_pretrained('Model/',
 #                                               dict_file = 'Model/dict.txt')

#tokenizer = BertTokenizer.from_pretrained(MFBERT_PATH)
#model = BertModel.from_pretrained(MFBERT_PATH)
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

    return embeddings.squeeze(), inputs['input_ids'].squeeze(), inputs['attention_mask'].squeeze()

def tokenizer_fdl(smile):
    "Tokenizes SMILES string"
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|_|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regezz = re.compile(pattern)
    tokens = [token for token in regezz.findall(smile)]
    assert smile == ''.join(tokens), ("{} could not be joined".format(smile))
    return tokens

def make_variable_one(smiles, letters, max_smiles_len):  #¿¿¿¿ SMILES ¿¿¿¿¿¿¿¿¿¿¿¿¿¿¿
    resultVec = []
    char_list = tokenizer_fdl(smiles)
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


    savePath = '/home/nudt_cleng/ccleng/MMFDL-main/notebook/0423/results/processed/Lipophilicity_MFBERT_dataTrain.dat'
    with open(savePath, 'w') as dat_file:
        for row in dataTrain:
            smi = row[0]   #smi
            enSmi = make_variable_one(smi, smilesVoc, 100)
            enSmi_str = ','.join(map(str, enSmi))

            ep = row[1]
            exp = row[2]    #y

            # 使用MFBERT生成特征
            embeddings, input_ids,attention_mask = generate_mfbert_embedding(smi)
            #mfbert_str = ','.join(map(str, mfbert_embedding.tolist()))
            mfbert_str = ','.join(map(str, input_ids.tolist())) + ' ' + ','.join(map(str, attention_mask.tolist()))

            line = smi + ' ' + enSmi_str +' ' + ep + ' ' + exp + ' ' + mfbert_str
            dat_file.write(line + '\n')
    print('writing ' + savePath + ' finished!')

    savePath = '/home/nudt_cleng/ccleng/MMFDL-main/notebook/0423/results/processed/Lipophilicity_MFBERT_dataTest.dat'
    with open(savePath, 'w') as dat_file:
        for row in dataTest:
            smi = row[0]

            enSmi=make_variable_one(smi,smilevoc,100) #将给定的 SMILES 字符串转换为定长的数字序列表示
            enSmi_str = ','.join(map(str, enSmi))

            # 使用MFBERT生成特征
            _, input_ids,attention_mask = generate_mfbert_embedding(smi)
            mfbert = ','.join(map(str, input_ids.tolist())) + ' ' + ','.join(map(str, attention_mask.tolist()))

            ep = row[1] 
            exp = row[2]
            line = smi + ' ' + enSmi_str + ' ' + ep + ' ' + exp + ' ' +mfbert_str
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
    smiles = df['smiles'].tolist()
    y = df['exp'].tolist()
    with open(savePath, "w") as dat_file:
        for smi, y_ in zip(smiles, y):
            ecfp = gene_ecfp(smi)   #ecfp
            ep_str = ','.join(str(x) for x in ecfp)

            yy = str(y_)

              # 使用MFBERT生成特征
            _, input_ids, attention_mask = generate_mfbert_embedding(smi)
            mfbert = ','.join(map(str, input_ids.tolist())) + ' ' + ','.join(map(str, attention_mask.tolist()))

            line = smi + ' ' + ep_str + ' ' +  yy + ' ' + mfbert
            dat_file.write(line + '\n')
    print('writing ' + savePath + ' finished!')


if __name__ == '__main__':
    filePath = 'results/Lipophilicity_all.csv'
    savePath = 'results/LipophilicitydataAll.dat'
    transToDat_All(filePath, savePath)

   # vocab_path = 'results/smiles_char_dict.pkl'
    vocab_path='results/Lipophilicity_smiles_char_dict.pkl'
    with open(vocab_path, 'rb') as f:
        smilevoc = pickle.load(f)

    filePath = 'results/LipophilicitydataAll.dat'
    splitData_dat(filePath, smilevoc)
