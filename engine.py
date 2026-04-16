# engine.py
from ladybug.epw import EPW
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.decomposition import PCA
import os

def get_epwData(epw_path):
    epw = EPW(epw_path)
    cliVars = [epw.dry_bulb_temperature.values, epw.dew_point_temperature.values,
               epw.relative_humidity.values, epw.global_horizontal_radiation.values,
               epw.direct_normal_radiation.values, epw.diffuse_horizontal_radiation.values,
               epw.wind_speed.values, epw.wind_direction.values]
    cliNames = ['DBT','DPT','RH','GHR','DNR','DHR','WS','WD']
    
    temp_df = pd.DataFrame()
    for i,name in enumerate(cliNames):
        temp_df[name] = cliVars[i]
    
    temp_df['DateTime'] = pd.date_range(start="2018-01-01 00:00", end="2018-12-31 23:00", freq="H")
    temp_df.index = temp_df['DateTime']
    temp_df = temp_df.drop('DateTime',axis=1)
    temp_df['Day'] = temp_df.index.dayofyear
    temp_df['Hour'] = temp_df.index.hour
    temp_df = temp_df.reset_index().iloc[:,1:]
    
    dfs = [pd.pivot_table(temp_df,values=col,index='Day',columns='Hour',aggfunc=np.sum) for col in temp_df.columns[:8]]
    join_df = pd.concat(dfs,axis=1)
    col_names = []
    for col in temp_df.columns[:8]:
        col_names += [f'{col}_{i}' for i in range(0,24)]
    join_df.columns = col_names
    
    temp_df = join_df.copy().reset_index().iloc[:,1:]
    full_year_2025 = pd.date_range(start='2025-01-01', end='2025-12-31', freq='D')
    temp_df['Month'] = full_year_2025.month
    temp_df['Day_of_Month'] = full_year_2025.day
    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(temp_df)
    return temp_df, pd.DataFrame(scaled, columns=temp_df.columns)

def run_PCA(df):
    pca = PCA(n_components=0.95, random_state=42) 
    X_pca = pca.fit_transform(df.to_numpy())
    return pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])], index=df.index)

def run_kMeans(df):
    df_pca = run_PCA(df)
    X = df_pca.to_numpy()
    # Simplified for app usage: find best K and return labels
    best_k, best_score, best_rs = 2, -1, 0
    for k in range(2, 6): # Smaller range for web speed
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        score = silhouette_score(X, km.labels_)
        if score > best_score:
            best_k, best_score = k, score
    km_final = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X)
    return km_final.labels_, round(best_score, 2)

def get_rep(df):
    X = df.to_numpy()
    D = pairwise_distances(X, metric='euclidean')
    return np.argmin(D.mean(axis=1))