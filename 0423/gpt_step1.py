import pandas as pd

# Read the CSV file
df = pd.read_csv('results/Lipophilicity.csv')

# Strip any leading or trailing spaces from column names
df.columns = df.columns.str.strip()

# Print the column names to verify

# Initialize the dictionary to store relevant columns
resultDic = {}
resultDic['smiles'] = df['smiles']
resultDic['exp'] = df['exp']

# Create DataFrame and save it
res = pd.DataFrame(resultDic)
res.to_csv('results/Lipophilicity_all.csv', index=False)

# Extract 'smiles' into a new dictionary
smilesAll = {'smiles': []}
for item in resultDic['smiles']:
    smilesAll['smiles'].append(item)

# Create DataFrame and save 'smiles' to a text file
res = pd.DataFrame(smilesAll)
res.to_csv('results/LipophilicitydataSmileAll.txt', index=False, header=False)

# Print the result DataFrames
print(pd.DataFrame(resultDic))
print(pd.DataFrame(smilesAll))

