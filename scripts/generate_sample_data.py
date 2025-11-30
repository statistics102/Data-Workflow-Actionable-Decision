# generate_sample_data.py
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import os

def create_sample_data():
    """Generate realistic sample customer data for analysis"""
    print("🚀 Generating sample customer data...")
    
    # Create data directory
    os.makedirs('data', exist_ok=True)

    # Generate synthetic dataset
    np.random.seed(42)
    n_samples = 1000

    # Create classification dataset
    X, y = make_classification(n_samples=n_samples, n_features=5, n_redundant=0, 
                              n_informative=5, n_clusters_per_class=1, random_state=42)

    # Create DataFrame with realistic column names
    df = pd.DataFrame(X, columns=['age', 'income', 'spending_score', 'credit_score', 'satisfaction'])
    df['target'] = y

    # Convert to realistic ranges FIRST (before adding missing values)
    df['age'] = (df['age'] * 10 + 35).astype(int)
    df['income'] = (df['income'] * 20000 + 50000).astype(int)
    df['credit_score'] = (df['credit_score'] * 100 + 650).astype(int)
    df['spending_score'] = (df['spending_score'] * 25 + 50).astype(int)
    df['satisfaction'] = (df['satisfaction'] * 2 + 3).round(1)

    # NOW add some missing values and anomalies
    df.loc[10:15, 'income'] = np.nan
    df.loc[50:55, 'credit_score'] = np.nan
    df.loc[100, 'age'] = 200  # Anomaly
    df.loc[150, 'spending_score'] = -100  # Anomaly

    # Save raw data
    df.to_csv('data/raw_customer_data.csv', index=False)
    
    print("✅ Sample data generated successfully!")
    print(f"📊 Data shape: {df.shape}")
    print("📁 File saved: 'data/raw_customer_data.csv'")
    
    # Display sample of the data
    print("\n📋 Sample of the generated data:")
    print(df.head(10))
    
    # Show data info
    print("\n🔍 Data information:")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    return df

if __name__ == "__main__":
    create_sample_data()