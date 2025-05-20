import pandas as pd

df = pd.read_csv('/home/nudt_cleng/ccleng/MMFDL-main/dataSour/Lipophilicity.csv')
df.columns = df.columns.str.strip()

resultDic = {}
resultDic['smiles'] = df['smiles']
resultDic['exp'] = df['exp']

res = pd.DataFrame(resultDic)
res.to_csv('/home/nudt_cleng/ccleng/MMFDL-main/notebook/0423/origin/results/Lipophilicity_all.csv', index=False)

smilesAll = {'smiles': []}
for item in resultDic['smiles']:
    smilesAll['smiles'].append(item)

res= pd.DataFrame(smilesAll)
res.to_csv('/home/nudt_cleng/ccleng/MMFDL-main/notebook/0423/origin/results/LipophilicitySmileAll.txt', index=False, header=False)
print(pd.DataFrame(resultDic))
print(pd.DataFrame(smilesAll))


