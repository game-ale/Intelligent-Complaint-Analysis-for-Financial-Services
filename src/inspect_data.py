
import pandas as pd
import os

try:
    df = pd.read_csv('c:/week5/Intelligent-Complaint-Analysis-for-Financial-Services/data/complaints.csv')
    print("Columns:", df.columns.tolist())
    print("\nUnique Products:")
    print(df['Product'].unique())
    print("\nSample rows:")
    print(df.head())
except Exception as e:
    print(f"Error: {e}")
