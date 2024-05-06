import pandas as pd

data = pd.read_csv("glossary_ref_value.csv")
df = pd.DataFrame(data)

#df.rename(columns = {"ID":"Dataset Reg ID"},inplace=True)
#df['Dataset Reg ID']=df['Dataset Reg ID'].astype(str)
#df.loc[df['Dataset Reg ID']!='DSET000114','Dataset Reg ID']='DSET000114'
#df[["Dataset Reg ID"]]=df[["Dataset Reg ID"]].replace([],["DSET000114"])
df.to_csv("mod_profile.csv",index=None,header=True)