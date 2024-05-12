import sys
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


from main import * 








file='../../data/shopping_trends_updated.csv'
columns_to_convert = ['Subscription Status', 'Discount Applied', 'Promo Code Used']
data_=pd.read_csv(file)

data_=remove_outliers(data_,"Salary")
data_= convert_to_binary(data_,columns_to_convert)


correlation(data_,"Salary","Purchase Amount (USD)")
correlation(data_,'Salary','Previous Purchases')
columns_covariance1 = ['Age', 'Purchase Amount (USD)', 'Review Rating', 'Previous Purchases', 'Salary']

# calculate_covariance(data_,columns_covariance1)


linear_regression(data_,'Salary','Purchase Amount (USD)')
linear_regression(data_,'Salary','Previous Purchases')

predictor_columns = ['Salary', 'Subscription Status']  # Independent variables
response_column = 'Purchase Amount (USD)'  # Dependent variable
    

regression_prediction_model(data_,predictor_columns,response_column)
#numerical 
plot_relationship_numerical(data_,"Salary","Purchase Amount (USD)","Promo Code Used")
#categorical
plot_relationship_categorical(data_,"Salary","Purchase Amount (USD)","Shipping Type")
