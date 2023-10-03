import pandas as pd

# Define the minimum percentage of non-zero days to keep an item
min_non_zero_percentage = 10  # Adjust as needed
total_days = 1541

def filter_byper(df):
    # Calculate the percentage of non-zero days for each item
    df['non_zero_percentage'] = (df.iloc[:, 2:] > 0).sum(axis=1) / total_days * 100

    # Filter the DataFrame to keep only items with non-zero percentage >= min_non_zero_percentage
    df_filtered = df[df['non_zero_percentage'] >= min_non_zero_percentage]

    # Drop the 'non_zero_percentage' column if no longer needed
    df_filtered = df_filtered.drop(columns=['non_zero_percentage'])
    return df_filtered