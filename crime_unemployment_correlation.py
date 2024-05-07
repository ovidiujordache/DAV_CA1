from __init__ import *

def analyze_crime_unemployment_correlation(crime_data_path, unemployment_data_path):

    try:
    
        crime_data = pd.read_csv(crime_data_path)
        crime_summary = crime_data.groupby('Year')['VALUE'].sum().reset_index()
        crime_summary.rename(columns={'VALUE': 'Total Crimes'}, inplace=True)
 
        unemployment_data = pd.read_csv(unemployment_data_path)
        unemployment_data.rename(columns={'VALUE': 'Average Unemployed'}, inplace=True)

        # Merge the dataframes on the 'Year' column
        combined_data = pd.merge(crime_summary, unemployment_data, on='Year', how='inner')

        # Calculate correlation
        correlation_matrix = combined_data[['Total Crimes', 'Average Unemployed']].corr()
        print("Correlation Matrix:")
        print(correlation_matrix)

        # Plotting
        plt.figure(figsize=(10, 6))
        sb.scatterplot(x='Average Unemployed', y='Total Crimes', data=combined_data)
        plt.title('Correlation Between Unemployment and Crime Rates')
        plt.xlabel('Average Unemployed Persons')
        plt.ylabel('Total Crime Incidents')
        plt.grid(True)
        plt.show()
      
        print(combined_data.head())
       

    except Exception as e:
        print(f"An error occurred: {e}")

# Usage example
crime_data_path = '../data/filtered_crime.csv'  # Adjust to your actual file path
unemployment_data_path = '../data/average_live_register_2008_2016.csv'  # Adjust to your actual file path
analyze_crime_unemployment_correlation(crime_data_path, unemployment_data_path)
