import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 数据集
path = '/Users/junowang/Desktop/gbdt-code/output/testloss.xls'
data1 = pd.read_excel(path, sheet_name='step1')
data2 = pd.read_excel(path, sheet_name= 'step2')
data1[['t1','t2','t3','t4','t5']] = np.exp(data1[['t1','t2','t3','t4','t5']]*100)
data2[['t1','t2','t3','t4','t5']] = np.exp(data2[['t1','t2','t3','t4','t5']]*100)
# 绘画折线图：
f, axes = plt.subplots(2, 1, figsize=(15, 10))
for ylabel in ['t1','t2','t3','t4','t5']:
    #axes[0].set_xscale("log")data1[['t1','t2','t3','t4','t5']] = data1[['t1','t2','t3','t4','t5']]*100
    #axes[0].set_yscale("log")
    sns.regplot("itera", ylabel, data1, ax=axes[0], scatter_kws={"s": 35}, label = ylabel)
for ylabel in ['t1', 't2', 't3', 't4', 't5']:
    #axes[0].set_yscale("log")
    sns.regplot("itera", ylabel, data2, ax=axes[1], scatter_kws={"s": 35}, label = ylabel)
axes[0].legend(loc='right')
axes[1].legend(loc='right')
axes[0].set_title('step = 0.01')
axes[1].set_title('step = 0.03')
axes[0].xaxis.set_ticks_position('bottom')
axes[1].xaxis.set_ticks_position('bottom')
axes[1].set_xlabel('Iteration')
axes[1].set_ylabel('exp(testloss)')
axes[0].set_xlabel(' ')
axes[0].set_ylabel('exp(testloss)')
axes[0].set_xticks(np.linspace(1,20,20))
axes[1].set_xticks(np.linspace(1,20,20))
plt.show()