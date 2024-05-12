
from pydrive.drive import GoogleDrive

# Data Visualization
from matplotlib import pyplot as plt  # Importing pyplot from matplotlib for plotting
import seaborn as sb  # Importing seaborn for enhanced visualization capabilities

# Data Manipulation
import numpy as np  # Importing numpy for numerical computations
import pandas as pd  # Importing pandas for data manipulation

# Statistical Analysis
import pingouin as pg  # Importing pingouin for statistical analysis

# 3D Plotting
from mpl_toolkits import mplot3d  # Importing mplot3d for 3D plotting capabilities

# Hierarchical Clustering and Dendrogram
import scipy.cluster.hierarchy as sch  # Importing scipy.cluster.hierarchy for hierarchical clustering
from scipy.spatial.distance import squareform  # Importing squareform for distance calculation
from scipy.stats import pearsonr  # Importing pearsonr for Pearson correlation coefficient calculation
from scipy.cluster.hierarchy import linkage, dendrogram, cut_tree  # Importing linkage, dendrogram, and cut_tree for dendrogram plotting

# Machine Learning - Clustering
from sklearn.datasets import make_blobs  # Importing make_blobs for generating sample data
from sklearn.cluster import AgglomerativeClustering   # Importing AgglomerativeClustering from sklearn for hierarchical clustering
from sklearn.preprocessing import StandardScaler   # Importing StandardScaler for data scaling
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans   # Importing KMeans from sklearn.cluster for KMeans clustering

from sklearn.preprocessing import MinMaxScaler  # Importing MinMaxScaler from sklearn.preprocessing for data scaling
from sklearn.metrics import adjusted_rand_score


import statsmodels.api as sm


__all__=[	"GoogleDrive",
			"plt","sb",
			"np","pd",
			"pg",
			"mplot3d","sm",
			"sch",
			"squareform",
			"pearsonr",
			"linkage",
			"dendrogram",
			"cut_tree",
			"make_blobs",
			"AgglomerativeClustering",
			"StandardScaler",
			"KMeans",
			"MinMaxScaler",
			"adjusted_rand_score",
			"PCA",
			"remove_outliers",
			"convert_to_binary",
			"correlation",
			"calculate_covariance",
			"linear_regression",
			"regression_prediction_model",
			"plot_relationship_numerical"
			,"plot_relationship_categorical",
			"multiple_regression",
			"hierarchical_clustering_all",
			"hierarchical_clustering_2columns",
			"kmean_PCA_all",
			"kmean_PCA_2columns",
			"cluster_and_visualize_all_data_3d",

			];









tab =0
BLUE='\033[94m'
GREEN = '\033[92m'
RED = '\033[91m'
END = '\033[0m'  # default color









def remove_outliers(data_, column):
    data = data_
 
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1

    # Q1 Q3
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter based on Q1 -Q3
    filtered_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    # data.to_csv(output_path_, index=False)
    return filtered_data


def convert_to_binary(data_,columns_to_convert):

    try:

        data = data_


        replacement_dict = {'Yes': 1, 'No': 0,"Male":0,"Female":1}

   
        for column in columns_to_convert:
            if column in data.columns:
                data[column] = data[column].replace(replacement_dict)
            else:
                print(f"The column '{column}' does not exist in the dataset.")

            
        # data.to_csv(output_path_, index=False)  
        return data_
        
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def correlation(data_,x_,y_):
	print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
	data = data_
    
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


def calculate_covariance(data_,columns):
 
    data = data_

    
    numerical_data = data[columns]
    covariance_matrix = numerical_data.cov()
    plt.figure(figsize=(10, 8))
    sb.heatmap(covariance_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Covariance Matrix Heatmap')
    plt.show()
    # covariance_matrix.to_csv('covariance_matrix.csv')	

def linear_regression(data_, x_, y_):
    global BLUE
    global GREEN
    global RED
    global END
    global tab
    
    data = data_

    # Check if columns exist
    if x_ not in data.columns or y_ not in data.columns:
        print(f"One or both specified columns: {RED}{x_}{END} or{RED} {y_}{END} do not exist in the dataset.")
        return
    

    X = data[x_]  
    y = data[y_]  

  
    X = sm.add_constant(X)


    model = sm.OLS(y, X)


    results = model.fit()

    print(results.summary())
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

    sb.lmplot(x=x_, y=y_, data=data, height=7, aspect=1.3,  line_kws={'color': 'red'})
    plt.title(f"Linear Regression: {x_} vs {y_}",color='red')
    plt.xlabel(x_,color='blue')
    plt.ylabel(y_,color='green')
    
    # Display the plot
    plt.show()



def multiple_regression(data, predictors, response):
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    END = '\033[0m'
    
    # Check if all specified columns exist in the data
    missing_columns = [col for col in predictors + [response] if col not in data.columns]
    if missing_columns:
        print(f"Specified columns do not exist in the dataset: {RED}{', '.join(missing_columns)}{END}")
        return
    
    # Prepare predictor variables and add a constant for the intercept
    X = data[predictors]
    X = sm.add_constant(X)
    y = data[response]
    
    # Create and fit the regression model
    model = sm.OLS(y, X)
    results = model.fit()
    
    print(results.summary())
    print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    
    # Plotting: Create subplots for each predictor against the response
    plt.figure(figsize=(14, len(predictors) * 4))
    for i, predictor in enumerate(predictors, start=1):
        plt.subplot(len(predictors), 1, i)
        sb.regplot(x=predictor, y=response, data=data, line_kws={'color': 'red'})
        plt.title(f"Linear Regression: {predictor} vs {response}", color='red')
        plt.xlabel(predictor, color='blue')
        plt.ylabel(response, color='green')
    plt.tight_layout()
    plt.show()




# def simple_linear_regression(file,x_,y_):

#     data = pd.read_csv(file)
    
   
#     if x_ not in data.columns or y_ not in data.columns:
#         print(f"The columns{x_} or/and {y_} are not present in the dataset.")
#         return

    
#     X = data[x_]  
#     y = data[y_]  

  
#     X = sm.add_constant(X)


#     model = sm.OLS(y, X)


#     results = model.fit()

#     print(results.summary())
#     print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")








def regression_prediction_model(data_, predictor_columns, response_column):

    data = data_[predictor_columns + [response_column]].dropna()

    X = data[predictor_columns]  # Independent variables
    y = data[response_column]  # Dependent variable
    
    # Add a constant to the predictor variables (for the intercept)
    X = sm.add_constant(X)
    
  # Ordinary Least Squares (OLS) regression model is fitted
    model = sm.OLS(y, X).fit()

    # Prediction vs Actual plot
    predictions = model.predict(X)
    plt.figure(figsize=(10, 6))
    plt.scatter(y, predictions, alpha=0.3)  # Plotting actual vs. predicted values
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Line showing perfect predictions
    plt.xlabel(f'Actual {response_column}')
    plt.ylabel(f'Predicted {response_column}')
    plt.title('Actual vs Predicted ' + response_column)
    plt.show()

    # Print the summary of the regression model
    print(model.summary())






def plot_relationship_numerical(data_, x_, y_, label_column):
    data = data_
    
    # Extract necessary columns
    x_data = data[x_].values
    y_data = data[y_].values
    labels = data[label_column].values
    
    # Check if labels are numeric or need encoding
    if labels.dtype == 'object':
        # Convert categorical labels to numeric codes
        unique_labels, labels = np.unique(labels, return_inverse=True)
    
    # Plot setup
    plt.title(f"{x_} vs {y_} by {label_column}")
    plt.xlabel(x_)
    plt.ylabel(y_)
    
    # Scatter plot with color based on the specified label
    scatter = plt.scatter(x_data, y_data, c=labels, cmap='viridis', alpha=0.6)
    plt.colorbar(scatter, label=label_column)

    # Fitting and plotting regression lines for each category
    for label in np.unique(labels):
        label_mask = (labels == label)
        if np.sum(label_mask) > 1:  # Ensure there are enough points to fit
            label_x = x_data[label_mask]
            label_y = y_data[label_mask]
            plt.plot(np.unique(label_x), np.poly1d(np.polyfit(label_x, label_y, 1))(np.unique(label_x)), label=f'{label} Fit')

    # Adding a legend to explain the plots
    plt.legend(title=label_column)
    plt.show()





def plot_relationship_categorical(data_, x_, y_, category_column):
    data = data_
    
    # Extract necessary columns
    x_data = data[x_].values
    y_data = data[y_].values
    categories = data[category_column].values
    
    # Encoding categories to numeric values if they are categorical
    if categories.dtype == 'object':
        # Convert categorical data to numeric codes
        unique_categories, categories = np.unique(categories, return_inverse=True)
    
    # Split the data based on category
    category_indices = {cat_id: np.where(categories == cat_id)[0] for cat_id in np.unique(categories)}
    
    # Plot setup
    plt.title(f"{x_} vs {y_} by {category_column}")
    plt.xlabel(x_)
    plt.ylabel(y_)
    
    # Scatter plot with color based on the specified category
    scatter = plt.scatter(x_data, y_data, c=categories, cmap='viridis')
    colorbar = plt.colorbar(scatter, label=category_column)
    colorbar.set_ticks(np.arange(len(unique_categories)))
    colorbar.set_ticklabels(unique_categories)
    
    # Fitting and plotting regression lines for each category
    for cat_id, indices in category_indices.items():
        if len(indices) > 1:  # Ensure there are enough points to fit
            cat_x = x_data[indices]
            cat_y = y_data[indices]
            plt.plot(np.unique(cat_x), np.poly1d(np.polyfit(cat_x, cat_y, 1))(np.unique(cat_x)), label=f'{unique_categories[cat_id]} Fit')
    
    # Adding a legend to explain the plots
    plt.legend(title=category_column)
    plt.show()




















def kmean_PCA_all(data, columns, n_clusters=2):

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[columns])
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    
    # Assign cluster labels to the main DataFrame
    data['Cluster'] = 'Cluster_' + pd.Series(clusters).astype(str)
    
    # Perform PCA
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(scaled_data)
    
    # Prepare data for plotting
    pca_df = pd.DataFrame(pca_results, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = data['Cluster']
    
    # Plotting the PCA results
    plt.figure(figsize=(12, 6))
    sb.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Cluster', style='Cluster', s=100, palette='viridis')
    plt.title('PCA Plot of Wine Data Clustered by All Variables')
    plt.legend(title='Cluster')
    plt.show()





def kmean_PCA_2columns(data, white_vars, red_vars, n_clusters=2):

    scaler = StandardScaler()
    white_scaled = scaler.fit_transform(data[white_vars])
    red_scaled = scaler.fit_transform(data[red_vars])
    
    # Perform KMeans clustering
    kmeans_white = KMeans(n_clusters=n_clusters, random_state=42)
    white_clusters = kmeans_white.fit_predict(white_scaled)
    
    kmeans_red = KMeans(n_clusters=n_clusters, random_state=42)
    red_clusters = kmeans_red.fit_predict(red_scaled)
    
    # Assign cluster labels to the main DataFrame
    data['White_Wine_Cluster'] = 'White_' + pd.Series(white_clusters).astype(str)
    data['Red_Wine_Cluster'] = 'Red_' + pd.Series(red_clusters).astype(str)
    
    # Combine all features and perform PCA
    all_features = list(set(white_vars + red_vars))
    combined_scaled = scaler.fit_transform(data[all_features].dropna())
    pca = PCA(n_components=2)
    combined_pca = pca.fit_transform(combined_scaled)
    
    # Prepare data for plotting
    pca_df = pd.DataFrame(combined_pca, columns=['PC1', 'PC2'], index=data.dropna(subset=all_features).index)
    pca_df['White_Wine_Cluster'] = data.loc[pca_df.index, 'White_Wine_Cluster']
    pca_df['Red_Wine_Cluster'] = data.loc[pca_df.index, 'Red_Wine_Cluster']
    
    # Plotting the PCA results
    plt.figure(figsize=(12, 6))
    sb.scatterplot(data=pca_df, x='PC1', y='PC2', hue='White_Wine_Cluster', style='Red_Wine_Cluster', s=100)
    plt.title('PCA Plot of Combined Wine Data Clustered by Wine Type Variables')
    plt.legend(title='Cluster Type')
    plt.show()




def hierarchical_clustering_2columns(data, white_vars, red_vars):

    white_wine_data = data[white_vars]
    scaler = StandardScaler()
    white_wine_scaled = scaler.fit_transform(white_wine_data)

    # Subset and scale the data for red wine
    red_wine_data = data[red_vars]
    red_wine_scaled = scaler.fit_transform(red_wine_data)

    # Perform hierarchical clustering for white wine
    white_linkage = linkage(white_wine_scaled, method='ward')

    # Perform hierarchical clustering for red wine
    red_linkage = linkage(red_wine_scaled, method='ward')

    # Plotting the dendrogram for white wine data
    plt.figure(figsize=(10, 7))
    plt.title("Hierarchical Clustering Dendrogram - White Wine")
    dendrogram(white_linkage, labels=white_wine_data.index, leaf_rotation=90, leaf_font_size=8)
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    plt.show()

    # Plotting the dendrogram for red wine data
    plt.figure(figsize=(10, 7))
    plt.title("Hierarchical Clustering Dendrogram - Red Wine")
    dendrogram(red_linkage, labels=red_wine_data.index, leaf_rotation=90, leaf_font_size=8)
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    plt.show()






def hierarchical_clustering_all(data, columns, method='ward', figsize=(15, 10), p=50):

    # Extract and scale the data
    wine_data = data[columns]
    scaler = StandardScaler()
    wine_scaled = scaler.fit_transform(wine_data)
    
    # Perform hierarchical clustering
    linkage_matrix = linkage(wine_scaled, method=method)
    
    # Plotting the dendrogram
    plt.figure(figsize=figsize)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    dendrogram(
        linkage_matrix,
        leaf_rotation=90,  
        leaf_font_size=8, 
        truncate_mode='lastp',  
        p=p, 
        show_contracted=True,  
    )
    plt.show()

def cluster_and_visualize_all_data_3d(data, columns, n_clusters=3):

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[columns])
    
    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    
    # Assign cluster labels to the main DataFrame
    data['Cluster'] = 'Cluster_' + pd.Series(clusters).astype(str)
    
    # Perform PCA with 3 components
    pca = PCA(n_components=3)
    pca_results = pca.fit_transform(scaled_data)
    
    # Prepare data for plotting
    pca_df = pd.DataFrame(pca_results, columns=['PC1', 'PC2', 'PC3'])
    pca_df['Cluster'] = data['Cluster']
    

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], c=pd.Categorical(pca_df['Cluster']).codes, cmap='viridis', s=100, label=pca_df['Cluster'])
    ax.set_title('3D PCA Plot of Wine Data Clustered by All Variables')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    legend = ax.legend(*scatter.legend_elements(), title="Clusters")
    ax.add_artist(legend)
    plt.show()



