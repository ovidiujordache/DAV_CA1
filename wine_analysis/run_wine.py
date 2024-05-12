
import sys
from pathlib import Path

parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from main import *



file_wine='../data/wine.csv'

data_wine=pd.read_csv(file_wine)

print(data_wine.describe())
correlation(data_wine,'Alcohol','Proline')
correlation(data_wine,"Color_Intensity","Hue")
correlation(data_wine,'Hue','Proline')


linear_regression(data_wine,'Alcohol','Proline')
linear_regression(data_wine,"Color_Intensity","Hue")
linear_regression(data_wine,'Hue','Proline')

#Based on influencial characteristics what is more prevalent in white or red wines
wine_columns_all = ['Alcohol', 'Malic_Acid', 'Ash', 'Ash_Alcanity', 'Magnesium', 'Total_Phenols', 'Flavanoids',
                    'Nonflavanoid_Phenols', 'Proanthocyanins', 'Color_Intensity', 'Hue', 'OD280', 'Proline']


white_wine_vars = ['Malic_Acid', 'Total_Phenols', 'Proanthocyanins', 'Color_Intensity', 'Hue']
red_wine_vars = ['Flavanoids', 'Nonflavanoid_Phenols', 'Color_Intensity', 'Hue', 'OD280']


calculate_covariance(data_wine,wine_columns_all)


multiple_regression(data_wine, list(filter(lambda x: x != 'Alcohol', wine_columns_all)) , 'Alcohol')


multiple_regression(data_wine,list(filter(lambda x: x != 'Proline', wine_columns_all)) , 'Proline')

multiple_regression(data_wine, list(filter(lambda x: x != 'Ash', wine_columns_all)), 'Ash')



hierarchical_clustering_all(data_wine, wine_columns_all)


hierarchical_clustering_2columns(data_wine, white_wine_vars, red_wine_vars)


kmean_PCA_2columns(data_wine, white_wine_vars, red_wine_vars)



cluster_and_visualize_all_data_3d(data_wine, wine_columns_all)
