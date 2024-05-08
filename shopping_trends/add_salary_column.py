from __init__ import *






#run file once
# downloaded data
data_path = '../../data/shopping_trends.csv' 
data = pd.read_csv(data_path)

max_purchases = data['Previous Purchases'].max()
max_purchase_amount = data['Purchase Amount (USD)'].max()

data['Salary'] = ((data['Purchase Amount (USD)'] / max_purchase_amount) * 50000) + \
                 ((data['Previous Purchases'] / max_purchases) * 50000)


np.random.seed(42)  
random_factor = np.random.normal(1.0, 0.1, size=data.shape[0])  
data['Salary'] *= random_factor


data['Salary'] = data['Salary'].astype(int)


output_path = '../../data/shopping_trends_updated.csv'  
data.to_csv(output_path, index=False)



