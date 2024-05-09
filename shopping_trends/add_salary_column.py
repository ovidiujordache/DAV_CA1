from __init__ import *






#run file once
# downloaded data
def add_salary_column():
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

#runs once only
#uncomment method calll to run once

# add_salary_column()


def convert_yes_no_to_binary(file, columns_to_convert):

    try:

        data = pd.read_csv(file)


        replacement_dict = {'Yes': 1, 'No': 0}

   
        for column in columns_to_convert:
            if column in data.columns:
                data[column] = data[column].replace(replacement_dict)
            else:
                print(f"The column '{column}' does not exist in the dataset.")

        output_path = '../../data/shopping_trends_updated.csv'  
        data.to_csv(output_path, index=False)  
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


file_to_convert='../../data/shopping_trends_updated.csv'
columns_to_convert = ['Subscription Status', 'Discount Applied', 'Promo Code Used']
#run once
#uncomment method call
# convert_yes_no_to_binary(file_to_convert,columns_to_convert)