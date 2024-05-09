from __init__ import *

tab =0
BLUE='\033[94m'
GREEN = '\033[92m'
RED = '\033[91m'
END = '\033[0m'  # default color




import pandas as pd









def correlation(file,x_,y_):
	print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
	data = pd.read_csv(file)

	global BLUE
	global GREEN
	global RED
	global END
	global tab
	tab+=1
	print(f"{tab}.Correlation between: {BLUE}{x_}{END} and {BLUE}{y_}{END}")

# Calculate the p-value

	correlation_coefficient, p_value = pearsonr(data[x_], data[y_])    
	if abs(correlation_coefficient) < 0.3:
		strength = "weak"
	elif abs(correlation_coefficient) < 0.7:
		strength = "moderate"
	else:
		strength = "strong"

    # if the correlation is statistically significant
	if p_value < 0.05:
		print(f"Correlation is statistically significant and {GREEN}{strength}{END}.")
		print(f"Correlation Coefficient{GREEN} {correlation_coefficient}{END}")
		print(f"P-value:{GREEN} {p_value}{END}")
	else:
		print(f"Correlation is not statistically significant and is considered {RED}{strength}{END}.")

	print("---------------------------------------------------------------")
    # Print the correlation coefficient and p-value

def calculate_covariance(file,columns):
 
    data = pd.read_csv(file)


    
    numerical_data = data[columns]
    covariance_matrix = numerical_data.cov()
    plt.figure(figsize=(10, 8))
    sb.heatmap(covariance_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Covariance Matrix Heatmap')
    plt.show()
    # covariance_matrix.to_csv('covariance_matrix.csv')	

def linear_regression(file, x_, y_):
    global BLUE
    global GREEN
    global RED
    global END
    global tab
    
    data = pd.read_csv(file)

    # Check if columns exist
    if x_ not in data.columns or y_ not in data.columns:
        print(f"One or both specified columns: {RED}{x_}{END} or{RED} {y_}{END} do not exist in the dataset.")
        return
    

    sb.lmplot(x=x_, y=y_, data=data, height=7, aspect=1.3)
    plt.title(f"Linear Regression: {x_} vs {y_}",color='red')
    plt.xlabel(x_,color='blue')
    plt.ylabel(y_,color='green')
    
    # Display the plot
    plt.show()





def simple_linear_regression(file,x_,y_):

    data = pd.read_csv(file)
    
   
    if x_ not in data.columns or y_ not in data.columns:
        print(f"The columns{x_} or/and {y_} are not present in the dataset.")
        return

    
    X = data[x_]  
    y = data[y_]  

  
    X = sm.add_constant(X)


    model = sm.OLS(y, X)


    results = model.fit()

    print(results.summary())
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

def remove_outliers(file, column):
	data = pd.read_csv(file)
 
	Q1 = data[column].quantile(0.25)
	Q3 = data[column].quantile(0.75)
	IQR = Q3 - Q1

    # Q1 Q3
	lower_bound = Q1 - 1.5 * IQR
	upper_bound = Q3 + 1.5 * IQR

    # Filter based on Q1 -Q3
	filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
	# output_path = '../../data/shopping_trends_filtered.csv'  
	# data.to_csv(output_path, index=False)
	return filtered_data


#Original Data file - No Salary Column -> see add_salary_column.py


#file_o is original data file
file_o = '../../data/shopping_trends.csv'



correlation(file_o,'Purchase Amount (USD)','Age')

correlation(file_o,'Purchase Amount (USD)','Previous Purchases')

linear_regression(file_o,'Purchase Amount (USD)','Age')
linear_regression(file_o,'Purchase Amount (USD)','Previous Purchases')


#Data with salary column added

#file_s is original+SALARY Column added
file_s = '../../data/shopping_trends_updated.csv'

filtered_data=remove_outliers(file_s,"Salary")

#file_f is Filtered Data No outliers.
file_f='../../data/shopping_trends_filtered.csv'
correlation(file_f,'Salary','Purchase Amount (USD)')
correlation(file_f,'Salary','Previous Purchases')


linear_regression(file_f,'Salary','Purchase Amount (USD)')
linear_regression(file_f,'Salary','Previous Purchases')

columns_covariance1 = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases', 'Salary']
calculate_covariance(file_f,columns_covariance1)

simple_linear_regression(file_f,'Purchase Amount (USD)','Purchase Amount (USD)')

simple_linear_regression(file_f,'Purchase Amount (USD)','Previous Purchases')