import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

### IMORTING DATA

prices = pd.read_excel('Price.xlsx')

### CLAENING DATA
# Only taking data from DK2 zone
dk2_prices = prices[prices['PriceArea'] == 'DK2'].reset_index(drop=True)

#Check missing hours
dk2_prices['HourDK'] = pd.to_datetime(dk2_prices['HourDK'])

start_date = dk2_prices['HourDK'].min()
end_date = dk2_prices['HourDK'].max()
expected_hours = pd.date_range(start=start_date, end=end_date, freq='H')

actual_hours = dk2_prices['HourDK'].sort_values()
missing_hours = expected_hours.difference(actual_hours)

missing_days = pd.to_datetime(missing_hours).to_series().dt.date
days_with_missing_hours = missing_days.value_counts().sort_index()


#TRAINING AND TESTING
dk2_prices = dk2_prices.sort_values(by='HourDK')

split_index = int(0.8 * len(dk2_prices))

train_prices = dk2_prices.iloc[:split_index]
test_prices = dk2_prices.iloc[split_index:]


#CLASSIFYING DATA
#Classifying by cluster

features = ['PriceEUR']
clustering_data = train_prices[features].dropna()

# Standardize the features (important for K-Means)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(clustering_data)

# Apply K-Means clustering
n_clusters = 10  # Specify the number of clusters you want
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clustering_data['Cluster'] = kmeans.fit_predict(scaled_data)

# Add the cluster labels back to the original dataset
train_prices['Cluster'] = None
train_prices.loc[clustering_data.index, 'Cluster'] = clustering_data['Cluster']

cluster_means = train_prices.groupby('Cluster')['PriceEUR'].mean()
train_prices['ClusterMean'] = train_prices['Cluster'].map(cluster_means)

cluster_centers = kmeans.cluster_centers_

#STATE SPACE

cluster_means = cluster_means.sort_values(ascending=True).values

SoC=np.array([0,100,200,300,400,500])

states = [[bat, price] for bat in SoC for price in cluster_means]

#TRANSITION PROBABILITY MATRIX

prob_matrix = np.zeros((n_clusters,n_clusters))

for from_state_index in range(len(cluster_means)):
    mask_from = train_prices['ClusterMean'] == cluster_means[from_state_index]
    total = np.sum(mask_from)
    for to_state_index in range(len(cluster_means)):
        mask_to = train_prices['ClusterMean'] == cluster_means[to_state_index]
        shifted_mask_to = mask_to.shift(-1, fill_value=False)
        mask_from = mask_from[:-1]# Shift "to" mask upwards
        count = np.sum(mask_from & shifted_mask_to)  # Valid transitions
        prob_matrix[from_state_index][to_state_index] = count / total



print('End')