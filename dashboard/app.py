import streamlit as st
import plotly.express as px
import pandas as pd
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
st.title("Customer Segmentation Dashboard")

# Sidebar for uploading data
encoding_options = ['utf-8', 'latin1', 'ISO-8859-1']
selected_encoding = st.sidebar.selectbox("Select file encoding", encoding_options)
uploaded_file = st.sidebar.file_uploader("Upload your CSV", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, encoding=selected_encoding)
    except Exception as e:
        st.error(f"Error reading the CSV file: {e}")
    else:
        if 'CustomerID' in df.columns and 'InvoiceDate' in df.columns and 'UnitPrice' in df.columns:
            rfm = df.groupby("CustomerID").agg({
            "InvoiceDate": lambda x: (datetime.now() - pd.to_datetime(x).max()).days,
            "CustomerID": "count",
            "UnitPrice": "sum"
            })
            rfm.columns = ["Recency", "Frequency", "Monetary"]
            
            # Initialize KMeans and StandardScaler
            kmeans = KMeans(n_clusters=4, random_state=42)
            scaler = StandardScaler()

            # Scale data
            rfm_scaled = scaler.fit_transform(rfm)
            rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

            st.subheader("Cluster Summary")
            st.write(rfm.groupby("Cluster").mean())

            st.subheader("Cluster Visualization")
            fig = px.scatter(
                rfm, x="Recency", y="Monetary",
                color="Cluster", title="Customer Segments"
            )
            st.plotly_chart(fig)
        else:
            st.error("The uploaded CSV file does not contain the required columns: 'CustomerID', 'InvoiceDate', 'UnitPrice'.")