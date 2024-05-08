from __init__ import *

def correlation(file,x_,y_):

	data = pd.read_csv(file)
	BLUE='\033[94m'
	GREEN = '\033[92m'
	RED = '\033[91m'
	END = '\033[0m'  # Reset to default color

	print(f"Correlation between: {BLUE}{x_}{END} and {BLUE}{y_}{END}")

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

    # Print the correlation coefficient and p-value
	

#Original Data file - No Salary Column -> see add_salary_column.py

file_o = '../../data/shopping_trends.csv'



correlation(file_o,'Purchase Amount (USD)','Age')

correlation(file_o,'Purchase Amount (USD)','Previous Purchases')





#Data with salary column added

file_s = '../../data/shopping_trends_updated.csv'
correlation(file_s,'Salary','Purchase Amount (USD)')