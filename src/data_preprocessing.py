# import required libraries
import pandas as pd
from datetime import datetime

def preprocess_data(file_path):
    '''
    This function will load and preprocess the data by:
    1. Converting the InvoiceDate to datetime
    2. Filling the Description column with 'Unknown' for missing values
    3. Making the Quantity column positive
    4. Dropping duplicates
    5. Dropping rows with missing values
    6. Returning the cleaned dataframe

    param: file_path: str: path to the file
           return: df: pd.DataFrame: cleaned dataframe
    '''
    
    df = pd.read_csv(file_path, encoding='ISO-8859-1') # encoding because of special characters

    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    df['Description'] = df['Description'].fillna('Unknown')
    df['Quantity'] = df['Quantity'].abs()
    df = df.drop_duplicates()
    df = df.dropna()
    return df