import os
import pickle
import numpy as np
import pandas as pd
import re
import sys
# from scripts.parsers import vocab_parser
#from transformers import BertTokenizer
from MFBERT_Tokenizer import MFBERTTokenizer
# Initialize MFBERT tokenizer
#mfbert_tokenizer = BertTokenizer.from_pretrained('path_to_mfbert')
mfbert_tokenizer = MFBERTTokenizer.from_pretrained('Model/',
                                                dict_file = 'Model/dict.txt')


def tokenizer(smile):
    "Tokenizes SMILES string"
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|_|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regezz = re.compile(pattern)
    tokens = [token for token in regezz.findall(smile)]
    assert smile == ''.join(tokens), ("{} could not be joined".format(smile))
    return tokens

# Tokenize SMILES using MFBERT and return input_ids and attention_mask
def tokenize_smiles_with_mfbert(smiles):
    """
    Tokenize SMILES using MFBERT tokenizer.
    Args:
    - smiles (str): SMILES string.
    
    Returns:
    - input_ids: MFBERT tokenized input ids.
    - attention_mask: MFBERT attention mask.
    """
    encoded_mfbert = mfbert_tokenizer(smiles, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
    return encoded_mfbert['input_ids'], encoded_mfbert['attention_mask']



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


def save_mfbert_tokenized_data(smiles_file, save_dir):
    """
    Saves MFBERT tokenized data (input_ids and attention_mask) for each SMILES string.
    Args:
    - smiles_file (str): Path to the SMILES file.
    - save_dir (str): Directory to save the tokenized data.
    """
    print('Tokenizing SMILES with MFBERT...')
    tokenized_data = {}

    with open(smiles_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.lower() in ['smile', 'smiles', 'selfie', 'selfies']:
                continue  # Skip headers or invalid lines
            else:
                input_ids, attention_mask = tokenize_smiles_with_mfbert(line)
                tokenized_data[line] = {
                    'input_ids': input_ids.squeeze().tolist(),
                    'attention_mask': attention_mask.squeeze().tolist()
                }

    # Save tokenized data as a pickle file
    mfbert_tokenized_path = os.path.join(save_dir, 'mfbert_tokenized_data.pkl')
    with open(mfbert_tokenized_path, 'wb') as f:
        pickle.dump(tokenized_data, f)
    
    print(f"MFBERT tokenized data saved at {mfbert_tokenized_path}")



if __name__ == '__main__':
    smiles_file = 'results/LipophilicitydataSmileAll.txt'
    file_name = 'Lipophilicity_smiles_char_dict'
    savePath = 'results/'
    char_dict = build_vocab(smiles_file, file_name, savePath)

 # Generate and save MFBERT tokenized data
    save_mfbert_tokenized_data(smiles_file, savePath)

    vocab_path = 'results/Lipophilicity_smiles_char_dict.pkl'
    with open(vocab_path, 'rb') as f:
        smilesVoc = pickle.load(f)
    print(smilesVoc)

     # Load and print MFBERT tokenized data for verification
    mfbert_tokenized_path = os.path.join(savePath, 'mfbert_tokenized_data.pkl')
    with open(mfbert_tokenized_path, 'rb') as f:
        mfbert_data = pickle.load(f)
    print("MFBERT tokenized data sample:", list(mfbert_data.items())[:5])  # Print the first 5 items

