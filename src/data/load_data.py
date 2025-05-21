# src/data/load_data.py

import pandas as pd
import os

def load_raw_data():
    # Determine project root (two levels up from this file)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    csv_path = os.path.join(project_root, 'data', 'raw', 'Online Retail')

    # Load the CSV
    df = pd.read_csv(csv_path, encoding='ISO-8859-1')

    # Print basic info
    print("Dataset shape:", df.shape)
    print(df.head())

    return df

if __name__ == '__main__':
    load_raw_data()
