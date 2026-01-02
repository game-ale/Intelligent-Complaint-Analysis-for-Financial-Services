
import pandas as pd

try:
    df = pd.read_csv('c:/week5/Intelligent-Complaint-Analysis-for-Financial-Services/data/complaints.csv', nrows=5000)
    print("Columns:", df.columns.tolist())
    print("\nUnique Products found in first 5000:")
    print(df['Product'].unique())
    print("\nUnique Sub-products found in first 5000:")
    if 'Sub-product' in df.columns:
        print(df['Sub-product'].unique())
except Exception as e:
    print(f"Error: {e}")
