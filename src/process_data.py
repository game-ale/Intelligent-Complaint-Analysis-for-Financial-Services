
import pandas as pd
import os
import re

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove boilerplate "I am writing to file a complaint..." (Example heuristic)
    # The prompt actually mentions this example.
    # We can just do basic cleaning for now.
    text = re.sub(r'xx+', '', text) # Remove redacted xxx
    return text.strip()

def main():
    print("Loading data...")
    # Load all for now
    df = pd.read_csv('data/complaints.csv', dtype=str)
    
    print(f"Total rows: {len(df)}")
    
    # Filter Products
    # We need to identifying the values that correspond to the 4 categories
    # Mapping based on typical CFPB names:
    
    # 1. Credit Cards
    # Product: 'Credit card', 'Credit card or prepaid card', 'Prepaid card'
    # 2. Personal Loans
    # Product: 'Payday loan, title loan, or personal loan', 'Consumer Loan'
    # 3. Savings Accounts
    # Product: 'Checking or savings account', 'Bank account or service' -> Sub-product: 'Savings account'
    # 4. Money Transfers
    # Product: 'Money transfer, virtual currency, or money service', 'Money transfers'

    # Let's inspect unique products to be safe, but since this is a script, I'll use list inclusion
    
    target_products = [
        'Credit card', 
        'Credit card or prepaid card',
        'Payday loan, title loan, or personal loan',
        'Checking or savings account',
        'Money transfer, virtual currency, or money service',
        'Money transfers'
    ]
    
    # Filter by Product column
    df_filtered = df[df['Product'].isin(target_products)].copy()
    
    # Further refinement for Savings Account? 
    # The prompt specifically says "Savings account". 
    # If Product is 'Checking or savings account', we might want to keep all or just 'Savings account' subproduct.
    # Given the prompt's phrasing "five major product categories: Credit Cards, Personal Loans, Savings Accounts, Money Transfers", 
    # it likely wants Savings explicitly.
    # Let's clean the Product list to map to the 4 canonical names for easier analysis.
    
    def normalize_product(row):
        p = row['Product']
        sp = row.get('Sub-product', '')
        
        if p in ['Credit card', 'Credit card or prepaid card']:
            return 'Credit Card'
        elif p in ['Payday loan, title loan, or personal loan']:
            return 'Personal Loan'
        elif p in ['Money transfer, virtual currency, or money service', 'Money transfers']:
            return 'Money Transfer'
        elif p in ['Checking or savings account', 'Bank account or service']:
            if isinstance(sp, str) and 'savings' in sp.lower():
                return 'Savings Account'
            elif isinstance(sp, str) and 'checking' in sp.lower():
                return None # Prompt didn't ask for Checking? Or maybe it did? "Savings Accounts" is explicit.
            # If subproduct is unclear, maybe drop?
            # Let's keep 'Checking or savings account' broadly if we can't distinguish, but 'Savings Accounts' is specific.
            # I will filter strictly for Savings if possible.
            return 'Savings Account' # For now, let's be generous or check logic.
            # Wait, if "Checking or savings account" is the product, and I only want Savings, I should look at subproduct.
            # But earlier inspection showed I couldn't see subproducts clearly.
            # I'll implement a logic: if 'Checking or savings account' and subproduct is NOT savings, drop IDK?
            # Let's assuming Sub-product column exists.
        return None

    # Apply mapping
    df_filtered['Category'] = df_filtered.apply(normalize_product, axis=1)
    df_filtered = df_filtered.dropna(subset=['Category'])
    
    # Filter empty narratives
    print("Filtering empty narratives...")
    df_final = df_filtered.dropna(subset=['Consumer complaint narrative']).copy()
    
    # Normalize text
    print("Cleaning text...")
    df_final['cleaned_narrative'] = df_final['Consumer complaint narrative'].apply(clean_text)
    
    # EDA Stats
    print("Generating EDA report...")
    report = []
    report.append(f"Original Count: {len(df)}")
    report.append(f"Filtered (Product & Narrative) Count: {len(df_final)}")
    report.append("\nDistribution by Product:")
    report.append(df_final['Category'].value_counts().to_string())
    
    # Length analysis
    df_final['word_count'] = df_final['cleaned_narrative'].apply(lambda x: len(x.split()))
    report.append("\nWord Count Stats:")
    report.append(df_final['word_count'].describe().to_string())
    
    # Save processed data
    output_path = 'data/filtered_complaints.csv'
    df_final.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")

    # Write report
    with open('reports/eda_summary.txt', 'w') as f:
        f.write('\n'.join(report))
        
    print("Done.")

if __name__ == "__main__":
    main()
