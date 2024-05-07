from __init__ import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

def analyze_correlation(data1, data2, year_col1, value_col1, year_col2, value_col2,
                        rename_col1, rename_col2, title="Correlation Plot"):

    try:
        data1[year_col1] = pd.to_numeric(data1[year_col1], errors='coerce').dropna().astype(int)
        data2[year_col2] = pd.to_numeric(data2[year_col2], errors='coerce').dropna().astype(int)
        # Aggregate data by year and sum values
        summary1 = data1.groupby(year_col1)[value_col1].sum().reset_index()
        summary2 = data2.groupby(year_col2)[value_col2].sum().reset_index()

        # Rename columns for clarity
        summary1.rename(columns={value_col1: rename_col1}, inplace=True)
        summary2.rename(columns={value_col2: rename_col2}, inplace=True)

        # Merge datasets on year column
        combined_data = pd.merge(summary1, summary2, left_on=year_col1, right_on=year_col2, how='inner')

        # Calculate correlation
        correlation_matrix = combined_data[[rename_col1, rename_col2]].corr()
        print("Correlation Matrix:")
        print(correlation_matrix)

        # Plotting
        plt.figure(figsize=(10, 6))
        sb.scatterplot(x=rename_col2, y=rename_col1, data=combined_data)
        plt.title(title)
        plt.xlabel(rename_col2)
        plt.ylabel(rename_col1)
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"An error occurred: {e}")


