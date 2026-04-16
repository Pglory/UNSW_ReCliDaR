import pandas as pd
import numpy as np
from ladybug.epw import EPW
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.decomposition import PCA

class ReCliDaR:
    """
    Comprehensive tool for Representative Climate Data Recognition.
    Supports k-Means, GMM, and HAC for large-scale .epw processing.
    """
    
    @staticmethod
    def process_epw(file_path):
        """Loads and scales EPW data."""
        epw = EPW(file_path)
        cliVars = [epw.dry_bulb_temperature.values, epw.dew_point_temperature.values,
                   epw.relative_humidity.values, epw.global_horizontal_radiation.values,
                   epw.direct_normal_radiation.values, epw.diffuse_horizontal_radiation.values,
                   epw.wind_speed.values, epw.wind_direction.values]
        cliNames = ['DBT','DPT','RH','GHR','DNR','DHR','WS','WD']
        
        temp_df = pd.DataFrame({name: var for name, var in zip(cliNames, cliVars)})
        temp_df['DateTime'] = pd.date_range(start="2018-01-01 00:00", end="2018-12-31 23:00", freq="h")
        temp_df.index = temp_df['DateTime']
        temp_df = temp_df.drop('DateTime', axis=1)
        temp_df['Day'], temp_df['Hour'] = temp_df.index.dayofyear, temp_df.index.hour
        
        dfs = [pd.pivot_table(temp_df, values=col, index='Day', columns='Hour', aggfunc=np.sum) for col in cliNames]
        join_df = pd.concat(dfs, axis=1)
        join_df.columns = [f'{col}_{i}' for col in cliNames for i in range(24)]
        
        final_df = join_df.copy().reset_index().iloc[:, 1:]
        full_year = pd.date_range(start='2025-01-01', end='2025-12-31', freq='d')
        final_df['Month'], final_df['Day_of_Month'] = full_year.month, full_year.day
        
        scaled = StandardScaler().fit_transform(final_df)
        return final_df, pd.DataFrame(scaled, columns=final_df.columns)

    @staticmethod
    def run_analysis(scaled_df, method='kMeans'):
        """Runs clustering using the selected algorithm (kMeans, GMM, or HAC)."""
        X = PCA(n_components=0.95, random_state=42).fit_transform(scaled_df.to_numpy())
        
        if method == 'kMeans':
            best_k = max(range(2, 6), key=lambda k: silhouette_score(X, KMeans(n_clusters=k, n_init=10, random_state=42).fit(X).labels_))
            return KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X).labels_
        
        elif method == 'GMM':
            best_n = min(range(2, 6), key=lambda n: GaussianMixture(n_components=n, random_state=42).fit(X).bic(X))
            return GaussianMixture(n_components=best_n, random_state=42).fit(X).predict(X)
            
        elif method == 'HAC':
            best_k = max(range(2, 6), key=lambda k: silhouette_score(X, AgglomerativeClustering(n_clusters=k).fit_predict(X)))
            return AgglomerativeClustering(n_clusters=best_k).fit_predict(X)
            
        return None

    @staticmethod
    def get_representative_days(original_df, scaled_df, labels, method_name):
        """Output 1: Dataframe of Representative Days."""
        rep_days = []
        for j in np.unique(labels):
            subset_scaled = scaled_df[labels == j]
            D = pairwise_distances(subset_scaled.to_numpy(), metric='euclidean')
            actual_idx = subset_scaled.index[np.argmin(D.mean(axis=1))]
            row = original_df.loc[actual_idx]
            rep_days.append({
                'Algorithm': method_name,
                'Cluster': j,
                'Month': int(row["Month"]),
                'Day': int(row["Day_of_Month"]),
                'Weight': len(subset_scaled)
            })
        return pd.DataFrame(rep_days)

    @staticmethod
    def get_monthly_distribution(original_df, labels, method_name):
        """Output 2: Monthly Cluster counts."""
        dist_df = pd.DataFrame(index=range(1, 13))
        dist_df.index.name = 'Month'
        temp_df = original_df.copy()
        temp_df['cluster'] = labels
        for j in np.unique(labels):
            dist_df[f'{method_name}_{j}'] = [len(temp_df[(temp_df['cluster'] == j) & (temp_df['Month'] == m)]) for m in range(1, 13)]
        return dist_df