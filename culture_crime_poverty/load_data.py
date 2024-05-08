from __init__ import *

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
poverty_rates_file = '../../data/poverty_rates.csv'
df_poverty_rate = read_csv_file(poverty_rates_file)

education_third_level_file = '../../data/education_third_level.csv'
df_education_third_level = read_csv_file(education_third_level_file)


expenditure_culture_file = '../../data/expenditure_culture.csv'

df_expenditure_culture = read_csv_file(expenditure_culture_file)

df_average_live_register=read_csv_file('../../data/live_register_2008_2016.csv')
df_average_live_register=average_live_register(df_average_live_register)
#dataframe filtered
df_poverty_rate_filtered = drop_rows(df_poverty_rate,'Year',2017,2019)
#dataframe filtered
df_education_third_level_filtered=drop_rows(df_education_third_level,'Year',2000,2007)


check_filtering(df_poverty_rate_filtered, df_poverty_rate,2018)
check_filtering(df_education_third_level_filtered, df_education_third_level, 20017)

crime_rates_file = '../../data/recorded_crime_incidents.csv'

df_crime= aggregate_annual_data(crime_rates_file,"../../data/filtered_crime.csv")

file_path = '../../data/recorded_crime_incidents.csv' 
analyze_and_plot_crime_data(file_path)

printHead(df_poverty_rate_filtered,"Poverty Rates")

printHead(df_education_third_level_filtered,"Eduction Third Level")

printHead(df_crime,"'Crime Rates'")

analyze_data(df_crime,"Total Crimes All Divisions")
analyze_data(df_poverty_rate_filtered,"Poverty Rates")
analyze_data(df_education_third_level_filtered,"Total Enrollments 3rd Level ")
analyze_data(df_expenditure_culture,"Expenditure on Culture")
analyze_data(df_average_live_register,"Live Register")



analyze_correlation(df_crime, df_average_live_register,
                    'Year', 'VALUE',   
                    'Year', 'VALUE',  
                    'Total Crimes', 'Average Unemployed',"Correlation: Crime and Unemployment")
analyze_correlation( df_crime,df_poverty_rate,
                    'Year', 'VALUE',  
                    'Year', 'VALUE',  
                    'Total Crimes', 'Poverty Rates',"Correlation: Crime and Poverty Rate ")  

analyze_correlation( df_crime,df_education_third_level_filtered,
                    'Year', 'VALUE',  
                    'Year', 'VALUE',  
                    'Total Crimes', 'College Enrollment',"Correlation: Crime and College Enrollment")
analyze_correlation( df_crime,df_expenditure_culture,
                    'Year', 'VALUE',  
                    'Year', 'VALUE',  
                    'Total Crimes', 'Expenditure On Culture',"Correlation: Crime and Expenditure on Culture")    

analyze_correlation(df_education_third_level_filtered, df_average_live_register,
                    'Year', 'VALUE',  
                    'Year', 'VALUE',  
                    'College Enrollment', 'Unemployment',"Correlation: 3rd Level Enrollment and Unemployment")


