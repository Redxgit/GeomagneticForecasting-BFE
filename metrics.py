import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error as msem
from sklearn.metrics import mean_absolute_error as maem
from sklearn.metrics import r2_score as r2m

BINWIDTH = 10


def rmse(y_true, y_pred):
    """
    Wrapper
    """
    return msem(y_true, y_pred, squared=False)

def roundup(x):
    return int(np.ceil(x / 10.0)) * 10

def rounddown(x):
    return int(np.floor(x / 10.0)) * 10

def calculate_BFE(labels, preds, binwidth = 10):    
    df = pd.DataFrame({'labels' : labels, 'preds' : preds})
    df['diff'] = df['labels'] - df['preds']
    df['diff'] = df['diff'].abs()
    min_val = rounddown(df['labels'].min())
    max_val = roundup(df['labels'].max())
    bins = np.arange(min_val, max_val + binwidth, binwidth)
    df['labels_bins'] = pd.cut(df['labels'], bins = bins, right = False)
    df['labels_bins'] = df['labels_bins'].apply(lambda x: x.left)
    bfe = df.groupby('labels_bins', observed = True)['diff'].mean()    
    bfe.index = bfe.index.astype(int)
    
    return bfe.mean()

def get_BFE_data(labels, preds, binwidth=10):
    df = pd.DataFrame({"labels": labels, "preds": preds})
    df["diff"] = df["labels"] - df["preds"]
    df["diff"] = df["diff"].abs()
    min_val = rounddown(df["labels"].min())
    max_val = roundup(df["labels"].max())
    bins = np.arange(min_val, max_val + binwidth, binwidth)
    df["labels_bins"] = pd.cut(df["labels"], bins=bins, right=False)
    df["labels_bins"] = df["labels_bins"].apply(lambda x: x.left)
    bfe = df.groupby("labels_bins", observed=True)["diff"].mean()
    bfe.index = bfe.index.astype(int)

    bfe_count = df.groupby("labels_bins", observed=True)["labels"].count()
    bfe_count.index = bfe_count.index.astype(int)

    return bfe.mean(), bfe_count, bfe, min_val, max_val