# app.py
import streamlit as st
import os
import pandas as pd
from engine import get_epwData, run_kMeans, get_rep

st.set_page_config(page_title="ReCliDaR - UNSW", layout="wide")

st.title("ReCliDaR: Representative Climate Data Retriever")
st.write("An ML-based tool to determine representative climate days from EPW files.")

uploaded_file = st.file_uploader("Upload your EPW file", type=["epw"])

if uploaded_file is not None:
    # Save file temporarily
    with open("temp.epw", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with st.spinner('Processing Climate Data...'):
        # 1. Get Data
        epw_df, scaled_df = get_epwData("temp.epw")
        
        # 2. Run Clustering
        labels, score = run_kMeans(scaled_df)
        epw_df['cluster'] = labels
        
        st.success(f"Clustering Complete! Silhouette Score: {score}")
        
        # 3. Get Representative Days
        rep_days = []
        for j in range(len(set(labels))):
            cluster_subset_scaled = scaled_df[labels == j]
            cluster_subset_raw = epw_df[labels == j]
            idx = get_rep(cluster_subset_scaled)
            rep_row = cluster_subset_raw.iloc[idx]
            rep_days.append({
                "Category": f"Cluster_{j}",
                "Month": int(rep_row["Month"]),
                "Day": int(rep_row["Day_of_Month"]),
                "Days_in_Category": len(cluster_subset_raw)
            })
            
        # 4. Show Results
        df_results = pd.DataFrame(rep_days)
        st.subheader("Representative Days Results")
        st.table(df_results)
        
        # Download button
        csv = df_results.to_csv(index=False).encode('utf-8')
        st.download_button("Download Report as CSV", csv, "ReCliDaR_Report.csv", "text/csv")

    os.remove("temp.epw")