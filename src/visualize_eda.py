
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    print("Loading filtered data...")
    try:
        df = pd.read_csv('data/filtered_complaints.csv')
    except FileNotFoundError:
        print("Data not found. Run processing first.")
        return

    # Set style
    sns.set_theme(style="whitegrid")
    
    # 1. Product Distribution
    plt.figure(figsize=(10, 6))
    count_data = df['Category'].value_counts()
    sns.barplot(x=count_data.index, y=count_data.values, palette="viridis")
    plt.title('Distribution of Complaints by Product')
    plt.xlabel('Product Category')
    plt.ylabel('Number of Complaints')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('reports/product_distribution.png')
    print("Saved reports/product_distribution.png")
    
    # 2. Word Count Distribution
    # Calculate word counts if not present (though we did it in analysis, let's redo for plot)
    df['word_count'] = df['cleaned_narrative'].fillna("").apply(lambda x: len(str(x).split()))
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df['word_count'], bins=50, kde=True, color='blue')
    plt.title('Distribution of Consumer Complaint Narrative Length')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.xlim(0, 1000) # Limit x-axis to see the bulk of data
    plt.tight_layout()
    plt.savefig('reports/word_count_distribution.png')
    print("Saved reports/word_count_distribution.png")

if __name__ == "__main__":
    main()
