import sys
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from main import *






#run file once
# downloaded data
def add_salary_column(output_path_):
    data_path = '../data/shopping_trends.csv' 
    data = pd.read_csv(data_path)

    max_purchases = data['Previous Purchases'].max()
    max_purchase_amount = data['Purchase Amount (USD)'].max()

    data['Salary'] = ((data['Purchase Amount (USD)'] / max_purchase_amount) * 50000) + \
                 ((data['Previous Purchases'] / max_purchases) * 50000)


    np.random.seed(42)  
    random_factor = np.random.normal(1.0, 0.1, size=data.shape[0])  
    data['Salary'] *= random_factor


    data['Salary'] = data['Salary'].astype(int)


 
    data.to_csv(output_path_, index=False)

#runs once only
#uncomment method calll to run once

# add_salary_column()



file='../data/shopping_trends_updated.csv'   
columns_to_convert = ['Subscription Status', 'Discount Applied', 'Promo Code Used']
#run once
# add_salary_column(file)
