
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

from sklearn.cluster import KMeans   # Importing KMeans from sklearn.cluster for KMeans clustering

from sklearn.preprocessing import MinMaxScaler  # Importing MinMaxScaler from sklearn.preprocessing for data scaling
from sklearn.metrics import adjusted_rand_score



from filter_data import *
from crime_charts import *

__all__=[	"GoogleDrive",
			"plt","sb",
			"np","pd",
			"pg",
			"mplot3d",
			"sch","squareform","pearsonr","linkage","dendrogram","cut_tree",
			"make_blobs","AgglomerativeClustering","StandardScaler","KMeans","MinMaxScaler","adjusted_rand_score",
			"check_filtering","drop_rows","aggregate_annual_data","analyze_data","printHead","average_live_register","analyze_and_plot_crime_data"];