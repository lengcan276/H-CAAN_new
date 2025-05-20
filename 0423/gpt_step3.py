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



def make_variable_length(smiles, letters):
    resultVec = []
    char_list = tokenizer(smiles)
    for item in char_list:
        resultVec.append(letters[item])
    return len(resultVec)

# Function to calculate the length of MFBERT tokens (input_ids)
def calculate_mfbert_length(smiles, mfbert_tokenized_data):
    return len(mfbert_tokenized_data[smiles]['input_ids'])

# Function to plot both MMFDL and MFBERT tokenized sequence length distributions

def pltSmiCurveDistribution_combined(filePath, smilesVoc, mfbert_tokenized_data, title):
    df = pd.read_csv(filePath)
    smiles = df['smiles'].tolist()
    # Calculate MMFDL tokenized sequence lengths
    seq_smiles_mmfdl = [make_variable_length(smi, smilesVoc) for smi in smiles]
    
    # Calculate MFBERT tokenized sequence lengths
    seq_smiles_mfbert = [calculate_mfbert_length(smi, mfbert_tokenized_data) for smi in smiles]
    num_bins = 8
    plt.figure(figsize=(10, 6))
    plt.title(title, fontsize=18)
    # Plot MMFDL tokenized sequence length distribution
    plt.hist(seq_smiles_mmfdl, bins=num_bins, edgecolor='black', alpha=0.7, label='MMFDL Token Lengths')
    # Plot MFBERT tokenized sequence length distribution
    plt.hist(seq_smiles_mfbert, bins=num_bins, edgecolor='blue', alpha=0.5, label='MFBERT Token Lengths')
    plt.xlabel('Sequence Length', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, linestyle='--', color='grey', alpha=0.3)
    plt.legend(fontsize=14)
    plt.savefig('results/smiles_mf_mm_distribution.png', dpi=600)
    plt.show()
if __name__ == '__main__':
     # Load MMFDL vocab
    vocab_path = 'results/Lipophilicity_smiles_char_dict.pkl'
    with open(vocab_path, 'rb') as f:
        smilesVoc = pickle.load(f)
    # Load MFBERT tokenized data
    mfbert_tokenized_path = 'results/mfbert_tokenized_data.pkl'
    with open(mfbert_tokenized_path, 'rb') as f:
        mfbert_tokenized_data = pickle.load(f)
    # File containing SMILES data
    filePath = 'results/Lipophilicity.csv'
    # Plot the combined distribution of MMFDL and MFBERT tokenized sequence lengths
    pltSmiCurveDistribution_combined(filePath, smilesVoc, mfbert_tokenized_data, 'SMILES vs MFBERT Token Length Distribution')
