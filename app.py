import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
from ladybug.epw import EPW
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, pairwise_distances
from sklearn.decomposition import PCA

# --- 1. RESEARCH ENGINE FUNCTIONS ---

def get_epwData(epw_file):
    with open("temp.epw", "wb") as f:
        f.write(epw_file.getbuffer())
    
    epw = EPW("temp.epw")
    cliVars = [epw.dry_bulb_temperature.values, epw.dew_point_temperature.values,
               epw.relative_humidity.values, epw.global_horizontal_radiation.values,
               epw.direct_normal_radiation.values, epw.diffuse_horizontal_radiation.values,
               epw.wind_speed.values, epw.wind_direction.values]
    cliNames = ['DBT','DPT','RH','GHR','DNR','DHR','WS','WD']
    
    temp_df = pd.DataFrame()
    for i, name in enumerate(cliNames):
        temp_df[name] = cliVars[i]
        
    # Using lowercase 'h' for compatibility with newer Pandas versions
    temp_df['DateTime'] = pd.date_range(start="2018-01-01 00:00", end="2018-12-31 23:00", freq="h")
    temp_df.index = temp_df['DateTime']
    temp_df = temp_df.drop('DateTime', axis=1)
    temp_df['Day'] = temp_df.index.dayofyear
    temp_df['Hour'] = temp_df.index.hour
    temp_df = temp_df.reset_index().iloc[:, 1:]
    
    dfs = [pd.pivot_table(temp_df, values=col, index='Day', columns='Hour', aggfunc=np.sum) for col in temp_df.columns[:8]]
    join_df = pd.concat(dfs, axis=1)
    join_df.columns = [f'{col}_{i}' for col in cliNames for i in range(24)]
    
    final_df = join_df.copy().reset_index().iloc[:, 1:]
    
    # Using lowercase 'd' for daily frequency
    full_year_2025 = pd.date_range(start='2025-01-01', end='2025-12-31', freq='d')
    final_df['Month'] = full_year_2025.month
    final_df['Day_of_Month'] = full_year_2025.day
    
    scaler = StandardScaler()
    scaled = scaler.fit_transform(final_df)
    return final_df, pd.DataFrame(scaled, columns=final_df.columns)

def run_PCA(df):
    pca = PCA(n_components=0.95, random_state=42) 
    X_pca = pca.fit_transform(df.to_numpy())
    return pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])], index=df.index)

def run_kMeans(df):
    X = run_PCA(df).to_numpy()
    best_k, best_score = 2, -1
    for k in range(2, 6):
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
        score = silhouette_score(X, km.labels_)
        if score > best_score:
            best_k, best_score = k, score
    return KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X).labels_

def run_GMM(df):
    X = run_PCA(df).to_numpy()
    best_n, best_bic = 2, np.inf
    for n in range(2, 6):
        gmm = GaussianMixture(n_components=n, random_state=42).fit(X)
        if gmm.bic(X) < best_bic:
            best_n, best_bic = n, gmm.bic(X)
    return GaussianMixture(n_components=best_n, random_state=42).fit(X).predict(X)

def run_HAC(df):
    X = run_PCA(df).to_numpy()
    best_k, best_sil = 2, -1
    for k in range(2, 6):
        hac = AgglomerativeClustering(n_clusters=k).fit(X)
        score = silhouette_score(X, hac.labels_)
        if score > best_sil:
            best_k, best_sil = k, score
    return AgglomerativeClustering(n_clusters=best_k).fit_predict(X)

def get_rep(df):
    D = pairwise_distances(df.to_numpy(), metric='euclidean')
    return np.argmin(D.mean(axis=1))

# --- 2. STREAMLIT UI & SESSION STATE ---

st.set_page_config(page_title="UNSW ReCliDaR", page_icon="🌤️")

# Initialize session state so results persist after download clicks
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
    st.session_state.rep_df = None
    st.session_state.dist_df = None

# Display Logo
try:
    logo = Image.open("image_b0e079.png")
    st.image(logo, width=100)
except:
    st.warning("Logo file not found in repository.")

st.title("ReCliDaR: Representative Climate Days Recognizer")
st.markdown("Faculty of Architecture and Town Planning")

uploaded_file = st.file_uploader("Choose an EPW file", type="epw")

# Reset analysis state if a new file is uploaded
if uploaded_file is None:
    st.session_state.analysis_done = False

if uploaded_file is not None:
    if st.button("Run ML Analysis"):
        with st.spinner("Analyzing climate patterns..."):
            try:
                # 1. Process Data
                epw_df, scaled_df = get_epwData(uploaded_file)
                
                # 2. Run Clustering
                results = {
                    'kMeans': run_kMeans(scaled_df),
                    'GMM': run_GMM(scaled_df),
                    'HAC': run_HAC(scaled_df)
                }
                
                # 3. Create Outputs
                rep_days = []
                check_df = pd.DataFrame(index=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])

                for name, labels in results.items():
                    current_df = epw_df.copy()
                    current_df['cluster'] = labels
                    unique_labels = np.unique(labels)
                    for j in unique_labels:
                        subset_scaled = scaled_df[labels == j]
                        idx_in_subset = get_rep(subset_scaled)
                        actual_idx = subset_scaled.index[idx_in_subset]
                        rep_row = epw_df.loc[actual_idx]
                        rep_days.append([f'{name}_{j}', int(rep_row["Month"]), int(rep_row["Day_of_Month"]), len(subset_scaled)])
                        check_df[f'{name}_{j}'] = [len(current_df[(current_df['cluster']==j) & (current_df['Month']==m)]) for m in range(1, 13)]
                
                # Store results in session state
                st.session_state.rep_df = pd.DataFrame(rep_days, columns=['Category','Month','Day','Count'])
                st.session_state.dist_df = check_df
                st.session_state.analysis_done = True
                
                st.success("Analysis Complete!")
                
            except Exception as e:
                st.error(f"Analysis Error: {e}")

# Display results and download buttons if analysis is complete
if st.session_state.analysis_done:
    st.subheader("Representative Days Summary")
    st.dataframe(st.session_state.rep_df)
    
    st.subheader("Download Results")
    col1, col2 = st.columns(2)
    
    with col1:
        csv_rep = st.session_state.rep_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Report.csv",
            data=csv_rep,
            file_name="report.csv",
            mime="text/csv"
        )
        
    with col2:
        csv_dist = st.session_state.dist_df.to_csv().encode('utf-8')
        st.download_button(
            label="Download Distribution.csv",
            data=csv_dist,
            file_name="distribution.csv",
            mime="text/csv"
        )