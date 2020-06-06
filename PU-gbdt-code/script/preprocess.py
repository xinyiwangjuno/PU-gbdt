import pandas as pd
path = '/Users/junowang/Desktop/gbdt-code/data/test.csv'
df = pd.read_csv(path, header=0, encoding = "utf-8")
#df_raw[~pos_index]['label'] = -1
#df_raw.to_csv('/Users/junowang/Desktop/gbdt-code/data/data_for_tongji.csv',index=False,encoding = "utf-8")
for id in df.index:
    if df.loc[id, 'label'] == 0.0:
        df.loc[id, 'label'] = 1
    else:
        df.loc[id, 'label'] = -1
df.to_csv('/Users/junowang/Desktop/gbdt-code/data/data_for_tongji.csv',index=False,encoding = "utf-8")

path = '/Users/junowang/Desktop/gbdt-code/data/test.csv'
df = pd.read_csv(path, header=0, encoding = "utf-8")
#df_raw[~pos_index]['label'] = -1
#df_raw.to_csv('/Users/junowang/Desktop/gbdt-code/data/data_for_tongji.csv',index=False,encoding = "utf-8")
for id in df.index:
    if df.loc[id, 'label'] == 0:
        df.loc[id, 'label'] = 1
    else:
        df.loc[id, 'label'] = -1
df.to_csv('/Users/junowang/Desktop/gbdt-code/data/test_new.csv',index=False,encoding = "utf-8")