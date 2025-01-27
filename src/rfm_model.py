import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

'''
# Feature Engineering
 Features for clustering:
    Recency: Days since last purchase.
    Frequency: Total number of purchases.
    Monetary: Total purchase value.
'''
def rfm_calculus(df, scaler):
    current_date = datetime.now() # Set the current date as the reference date
    rfm = df.groupby("CustomerID").agg({ # Aggregate the data by CustomerID
        "InvoiceDate": lambda x: (current_date - x.max()).days, # Calculate Recency
        "CustomerID": "count",  # Calculate Frequency
        "UnitPrice": "sum" # Calculate Monetary
    }).rename(columns={
        "InvoiceDate": "Recency",
        "CustomerID": "Frequency",
        "UnitPrice": "Monetary"
    })
    
    ''' Normalize Data'''
    # Scale the features to a uniform range.
    rfm_scaled = scaler.fit_transform(rfm)

    '''
    Apply K-Means Clustering Algorithm
    K-Means is a popular clustering algorithm that groups similar data points into clusters.
    '''
    # Use the elbow method to determine the optimal number of clusters.

    distortions = [] # Initialize an empty list to store the distortion values
    tolerance = 0.01   # Set a tolerance value to stop the loop
    for k in range(1, 11): 
        kmeans = KMeans(n_clusters=k, random_state=42) # Initialize the KMeans model
        kmeans.fit(rfm_scaled) 
        inertia = kmeans.inertia_ 
        if k > 1 and abs(distortions[-1] - inertia) < tolerance: # Check if the difference is less than the tolerance
            break 
        distortions.append(inertia) 
    plt.plot(range(1, 11), distortions, marker='o') 
    plt.title("Elbow Method")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Distortion")
    plt.show()
    plt.show()

    # Fit the Model:
    # Choose the optimal number of clusters and fit the model.
    # In this example, we choose 4 clusters based on the elbow method.

    kmeans = KMeans(n_clusters=4, random_state=42) # Initialize the KMeans model
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled) # Fit the model and assign clusters to the data

    # Visualize the Clusters using 2D projection

    sns.scatterplot(
        x=rfm_scaled[:, 0], y=rfm_scaled[:, 1],
        hue=rfm["Cluster"], palette="Set2"
    )
    plt.title("Customer Segments")
    plt.show()
    
    '''Analyze Segments 
    Calculate mean RFM values for each cluster.
    The summary provides the average Recency, Frequency, and Monetary values for each cluster,
    which can be used to understand the characteristics of each customer segment.
    '''
    cluster_summary = rfm.groupby("Cluster").mean()
    
    return cluster_summary 
    
    # Instantiate the StandardScaler
def scaler_func(): 
    scaler = StandardScaler() 

    return scaler