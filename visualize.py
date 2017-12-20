from matplotlib import pyplot as plt
import numpy as np; np.seed=5
from matplotlib import pyplot as plt
import seaborn as sns; sns.set(color_codes=True)
from tqdm import tqdm
import pandas as pd



def load_dfs(paths):
    dfs = []
    for path in paths:
        dfs.append(pd.read_csv(path,index_col=0))
    return dfs


def plot(df_list, xlim=None, ylim=None):
    
    df_tot = pd.concat(df_list,axis=0)
    df_tot = df_tot.reset_index(drop=False).rename(columns={'index':'epoch'})
    
    sns.tsplot(time="epoch", value="loss_test",
                 unit="experiment", condition="name",
                 data=df_tot, ci=[68]
          ,estimator=np.median)

    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)

    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.show()

def subplot(df_list, ax, xlim=None, ylim=None):
    
    df_tot = pd.concat(df_list,axis=0)
    df_tot = df_tot.reset_index(drop=False).rename(columns={'index':'epoch'})
    
    sns.tsplot(time="epoch", value="loss_test",
                 unit="experiment", condition="name",
                 data=df_tot, ci=[68]
          ,estimator=np.median, ax=ax)

    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    
    return ax