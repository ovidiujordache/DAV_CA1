import sys
import os




from  __init__ import *

def analyze_and_plot_crime_data(filepath):

    try:
      
        data = pd.read_csv(filepath)

       
        data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
        data['VALUE'] = pd.to_numeric(data['VALUE'], errors='coerce')
  
        # Drop any rows where conversion failed (if any)
        data = data.dropna(subset=['Year', 'VALUE'])


        data = data[(data['Year'] >= 2008) & (data['Year'] <= 2016)]


        crime_summary = data.groupby(['Year', 'Type of Offence'])['VALUE'].sum().unstack()

        # Identify the top four most prevalent crimes
        top_crimes = crime_summary.sum().nlargest(10).index

        # Select data for the top crimes
        crime_summary_selected = crime_summary[top_crimes]

        # Plotting
        crime_summary_selected.plot(kind='line', figsize=(14, 8), title='Top Crime Types from 2008 to 2016')
        plt.xlabel('Year')
        plt.ylabel('Total Incidents')
        plt.xticks(rotation=0)
        plt.legend(title='Type of Offence', bbox_to_anchor=(0.9, 1), loc='center')  
        plt.show()

    


    except Exception as e:
        print(f"An error occurred: {e}")




