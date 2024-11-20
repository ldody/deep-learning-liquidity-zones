import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os, sys

path = os.path.dirname(os.path.abspath(__file__))
root_path = path
while os.path.basename(root_path) != 'PhD_article_2':
	root_path = os.path.dirname(root_path)
	
processed_path = os.path.join(root_path,'data','processed','FOB')
processed_path_LOB = os.path.join(processed_path,'LOB')


df_lob = pd.read_parquet(os.path.join(processed_path_LOB, 'NL0000226223_final_LOB.parquet.gzip')).reset_index().drop_duplicates()

df_lob = df_lob[df_lob['index'] == '2023-10-31 15:01:00']
df_lob['side'] = df_lob['side'].apply(lambda x: 1 if x == 'Buy' else -1)

data = df_lob[['price','size','side']].to_numpy()

scaler = MinMaxScaler(feature_range=(-1, 1))
data_scaled = scaler.fit_transform(data)
data_scaled[:,2] = data[:,2]

bic_scores = []

n_clusters_range = range(1, 20)

for n_clusters in n_clusters_range:
	gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=42)
	clusters = gmm.fit(data_scaled)
	bic_scores.append(gmm.bic(data_scaled))

n_components = n_clusters_range[np.argmin(bic_scores)]

gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
clusters = gmm.fit_predict(data_scaled)

# Ajouter les clusters au dataset
data_with_clusters = np.hstack((data, clusters.reshape(-1, 1)))

# Visualisation des résultats
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df_lob['price'].to_numpy(), df_lob['size'].to_numpy(), c=clusters, cmap='viridis', s=10)
plt.colorbar(scatter, label='Cluster Label')
plt.title(f'Clustering des zones de liquidité avec GMM avec {n_components} clusters')
plt.xlabel('Prix')
plt.ylabel('Volumes')
plt.grid(True)
plt.show()
plt.savefig('fig_gmm.png')