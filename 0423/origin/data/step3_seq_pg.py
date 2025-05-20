import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
import pandas as pd
import re
import sys

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

def pltSmiCurveDistribution_sample(filePath, smilesVoc, title):
    df = pd.read_csv(filePath)
    smiles = df['smiles'].tolist()
    seq_smiles = []
    for smi in smiles:
        seq_smiles.append(make_variable_length(smi, smilesVoc))
    num_bins = 8
    plt.figure(figsize=(10, 6))
    plt.title(title, fontsize=18)
    n, bins, patches = plt.hist(seq_smiles, bins=num_bins, edgecolor='black', alpha=0.7)
    plt.xlabel('sequence length', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.xticks(bins, fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, linestyle='--', color='grey', alpha=0.3)
    plt.savefig('/home/nudt_cleng/ccleng/MMFDL-main/notebook/0423/origin/results/Lipophilicity_sampl_smiles.png', dpi=600)
    plt.show()

if __name__ == '__main__':
    vocab_path = '/home/nudt_cleng/ccleng/MMFDL-main/notebook/0423/origin/results/Lipophilicity_smiles_char_dict.pkl'
    with open(vocab_path, 'rb') as f:
        smilesVoc = pickle.load(f)
    filePath = '/home/nudt_cleng/ccleng/MMFDL-main/notebook/0423/origin/results/Lipophilicity_all.csv'
    pltSmiCurveDistribution_sample(filePath, smilesVoc, 'Lipophilicity')
