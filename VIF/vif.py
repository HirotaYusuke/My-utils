import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

def vif_search(df1):
    df_vif = pd.DataFrame()
    df_vif["VIF Factor"] = [variance_inflation_factor(df1.values, i) for i in tqdm(range(df1.shape[1]))]
    df_vif["features"] = df1.columns
    
    return df_vif


def vif_decrese_method(x, permission:int):
    x_copy = x.copy()
    perm_level = permission
    drop_list = []
    drop_fail_list = []
    df_vif = vif_search(x)

    for index, data in tqdm(df_vif.iterrows()):
        if data[0] > perm_level:
            if data [1] in x_copy.columns:
                drop_list.append(data[1])
            else:
                drop_fail_list.append(data[1])

    x_copy.drop(drop_list, axis=1, inplace=True)            

    return x_copy