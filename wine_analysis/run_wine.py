
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


covariance_matrix=calculate_covariance(data_wine,wine_columns_all)
eigenvalues,eigenvectors=eigenvalues_eigenvectors(covariance_matrix)
feature_vector= create_feature_vector(eigenvalues,eigenvectors,2)

multiple_regression(data_wine, list(filter(lambda x: x != 'Alcohol', wine_columns_all)) , 'Alcohol')


multiple_regression(data_wine,list(filter(lambda x: x != 'Proline', wine_columns_all)) , 'Proline')

multiple_regression(data_wine, list(filter(lambda x: x != 'Ash', wine_columns_all)), 'Ash')



hierarchical_clustering_all(data_wine, wine_columns_all)




pca_data = pca_reduction(data_wine, n_components=2)

kmean_clusters=kmean_clustering(data_wine,n_clusters=3)

# 2. Hierarchical clustering
clusters = hierarchical_clustering(pca_data, n_clusters=3)

# 3. PCA for visualization
plot_data_pca(pca_data, clusters)
plot_hierarchical(data_wine)


# 1. PCA Reduction
reduced_data = pca_reduction_feature_vectors(data_wine, feature_vector)
plot_data_pca_reduced(reduced_data)

clusters = kmeans_clustering_feature_vectors(data_wine, feature_vector, 3)
plot_kmeans_clusters(reduced_data, clusters)



hierarchical_clustering_2columns(data_wine, white_wine_vars, red_wine_vars)

kmean_PCA_2columns(data_wine, white_wine_vars, red_wine_vars)
cluster_and_visualize_all_data_3d(data_wine, wine_columns_all)

red_wine_subindices = {
    'Phenolic Composition': ['Flavanoids', 'Nonflavanoid_Phenols'],
  'Color Intensity and Hue': ['Color_Intensity', 'Hue'],
    'OD280': ['Hue', 'OD280']
}


white_wine_subindices = {
    'Malic Acid and Total Phenols': ['Malic_Acid', 'Total_Phenols'],
    'Proanthocyanins': ['Proanthocyanins'],
  
}

    # 5 Components. Number of components taken from reserch paper.
    # See references.txt    
rotated_comp_matrix=rotated_component_matrix (component_matrix(data_wine))
apply_weights(rotated_comp_matrix)
red_wine_index = apply_weights(rotated_comp_matrix)
white_wine_index = apply_weights(rotated_comp_matrix)
bitter_wine_index = apply_weights(rotated_comp_matrix)