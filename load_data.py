from __init__ import *
'''
Problem Statement.
When Economy is down[files:(live register, poverty rates)] the Crime[files:(crimes_in_ireland)] is UP.
Also when Economy is down, Culture[files:(third_level_entrance,expenditure_on_culture)] is UP.
When Economy is booming Culture and Art is down.As a vivid example, during the height of the Celtic Tiger in 2002, 
a controversial and widely debated monument, "The Spire of Dublin," was constructed. 
Conversely, at the decline of the Celtic Tiger around 2008-2009, the Samuel Beckett Bridge was built,
 showcasing another masterpiece from Santiago Calatrava.



'''

#2008 -2017 for ALL  data sets. Both years are inclusive.

def read_csv_file(file_path):

    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
        print(f"Error: {e}")
        return None

# dataframe raw
poverty_rates_file = '../data/poverty_rates.csv'
df_poverty_rate = read_csv_file(poverty_rates_file)

education_third_level_file = '../data/education_third_level.csv'
df_education_third_level = read_csv_file(education_third_level_file)



#dataframe filtered
df_poverty_rate_filtered = drop_rows(df_poverty_rate,'Year',2018,2019)
#dataframe filtered
df_education_third_level_filtered=drop_rows(df_education_third_level,'Year',2000,2008)


check_filtering(df_poverty_rate_filtered, df_poverty_rate,2018)
check_filtering(df_education_third_level_filtered, df_education_third_level, 2008)



def printHead(df):
	if df is not None:
   		 print(df.head())
printHead(df_poverty_rate_filtered)
printHead(df_education_third_level_filtered)



