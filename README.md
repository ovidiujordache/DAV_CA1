

linkf of datasets.csv are in description.
To run code  out of the box:
1. Unzip data folder.
2. code looks for ../data/name_of_file.csv 
3. In  /main/ __init__.py  is  where imports live and code
5. list of all installed packages can be found in dav_ca1_packages_list.txt or dav_ca1_packages_list.yml
6.Link to files provided on OneDrive

HYPOTHESIS: When ECONOMY IS DOWN [files:(live register, poverty rates)]  CRIME  IS  UP [files:(crimes_in_ireland)].
Also when ECONOMY IS DOWN, CULTURE IS UP [files:(third_level_entrance,expenditure_on_culture)].
When ECONOMY IS BOOMING CULTURE AND ART IS DOWN. As a vivid example, during the height of the Celtic Tiger in 2002, 
a controversial and widely debated monument, "The Spire of Dublin," was constructed. 
Conversely, at the decline of the Celtic Tiger around 2008-2009, the Samuel Beckett Bridge was built,
 showcasing another masterpiece from Santiago Calatrava.
Aiming to find an index to show the relationship between 
Economy, Crime, Unemployment, Poverty, College Admissions, Culture and Art


Hypothesis in laymen terms:
 When poverty strikes, individuals face two options: the quick route of 
 turning to crime to make ends meet, or the slower, more sustainable route through culture and education.
 Both choices reflect personal decisions within society, rather than being driven primarly by a 
 collective moral force. Clearly  "personal decisions" involve additional context,influenced 
 by various factors(upbringing, educational background), which are beyond the scope of this report.

 There is a very poor correlation between Crime and Live Register.(Unemployed) -0.06.
 Have to look for more data or different data.

 ### Top Crimes between 2008-2016
 ![Alt text](./images/top_crimes.png?raw=true)

 ### Total No Of Crimes 2008-2016
 ![Alt text](./images/total_crimes.png?raw=true)

 ### Live Register 2008-1016
 ![Alt text](./images/live_register.png?raw=true)

 ### Poverty Rates 2008-2016
 ![Alt text](./images/poverty_rates.png?raw=true)

 ### Enrollment in 3rd Level Institutions 2008-2016
 ![Alt text](./images/total_enrollement.png?raw=true)

 ### Expenditure on Culture 2008-2016
 ![Alt text](./images/expenditure_culture.png?raw=true)

 ## Failed attempt with this data.Data has temporal dimension(Year) which is common across all datasets.
 # NEW DATA.Customer Purchases. source. Kaggle.com
 ### Added a column "Salary" , a random number created using "purchases" and "amount" with a seed of (42). 
 ### correlation salary   purchase_amount
 ![Alt text](./images/correlation_salary_purchase_amount.png?raw=true)
 ### correlation salary  previous purchase

 ![Alt text](./images/salary_previous_purchases.png?raw=true)
  ### actual vs predicted
 
 ![Alt text](./images/actual_vs_predicted.png?raw=true)

   ### salary Promo Code Used
 ![Alt text](./images/salary_promo.png?raw=true)

   ### salary   vs purchased amount shipping type
 ![Alt text](./images/salary_purchased_amount_shipping_type.png?raw=true)


 # Another Attempt. Using wine chemical components/characteristics to find clusters  


   ### Correlation Alohol Prioline
 ![Alt text](./images/correlation_alcohol_prioline.png?raw=true)


   ### Correlation Color Intensity Hue
 ![Alt text](./images/corr_color_hue.png?raw=true)


   ### Multiple Regression 1st part
 ![Alt text](./images/multiple_regression1.png?raw=true)


   ### Multiple Regression 2nd part
 ![Alt text](./images/multiple_regression2.png?raw=true)

   ### Principal Component 
 ![Alt text](./images/principal_component.png?raw=true)