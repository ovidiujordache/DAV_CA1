
from __init__ import *
def drop_rows(df, column_, start_, end_):

    exclude_ = range(start_, end_ + 1)


    filtered_df = df[~df[column_].isin(exclude_)]
    
    return filtered_df


def check_filtering(df_filtered, df_original, row_to_check):
  
    if df_filtered.shape[0] < df_original.shape[0]:
        print("Rows have been filtered.")
    else:
        print("No rows have been filtered.")

    # Check if a specific year has been removed
    if row_to_check is not None:
        try:
            # Attempt to access a row with the specified year
            row = df_filtered[df_filtered['Year'] == row_to_check].iloc[0]
            print(f"The Row {row_to_check} is still present in the DataFrame.")
        except IndexError:
            print(f"The Row {row_to_check} has been filtered out of the DataFrame.")
import pandas as pd

def aggregate_annual_data(file_path,save_path):
    try:
       
        df = pd.read_csv(file_path, names=['STATISTIC Label', 'Quarter', 'Garda Division', 'Type of Offence', 'UNIT', 'VALUE'], header=None, on_bad_lines='skip')
        df['VALUE'] = pd.to_numeric(df['VALUE'], errors='coerce')
        # Drop rows with NaN in essential columns
        df = df.dropna(subset=['STATISTIC Label', 'Quarter', 'Garda Division', 'Type of Offence', 'UNIT', 'VALUE'], how='any')
        
        # Extract year from 'Quarter' and safely convert to integer
        df['Year'] = pd.to_numeric(df['Quarter'].str[:4], errors='coerce')
        if df['Year'].isna().any():
             print("NA values found in Year extraction:", df[df['Year'].isna()]['Quarter'])        
    
        
        # Convert 'Year' to integer
        df['Year'] = df['Year'].astype(int)
        
        # Filter data for the years 2008 to 2016 inclusive
        df = df[(df['Year'] >= 2008) & (df['Year'] <= 2016)]
        print("Years in data:", sorted(df['Year'].unique()))
     
        annual_data = df.groupby(['Year', 'Garda Division'], as_index=False)['VALUE'].sum()
        if save_path:
            annual_data.to_csv(save_path, index=False)
            print(f"Data saved to {save_path}")
        
        return annual_data
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()


def analyze_data(data,label):
    # Summary Statistics
    print(data.groupby('Year')['VALUE'].describe())

    # Total annual values
    total_values = data.groupby('Year')['VALUE'].sum()
    print(total_values)

    # Visualize the trend over the years
    ax=total_values.plot(kind='line', marker='o', title=f"{label}")
    ax.title.set_color('red')
    ax.title.set_fontsize(16)
    ax.lines[0].set_color('green')
    
    ax.xaxis.label.set_color('magenta')
    ax.set_xlabel('Year', fontsize=14)
    

    ax.yaxis.label.set_color('blue')
    ax.set_ylabel('Total Value', fontsize=14) 
    



    plt.show()


def printHead(df, description=""):
    if df is not None:      
        if df.empty:
            print(f"The DataFrame {description} is empty.")
        else:
            print(df.head())
    else:
        print(f"No DataFrame provided {description}.")



