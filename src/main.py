from data_preprocessing import preprocess_data
from rfm_model import rfm_calculus, scaler_func

# Load data and Preprocess data
file_path = "/path/to/your/file.csv"

df = preprocess_data(file_path)

# Instantiate the StandardScaler
scaler = scaler_func()

# Train and evaluate model
cluster_summary = rfm_calculus(df, scaler)

# Print the cluster summary
print(cluster_summary)