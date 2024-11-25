import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KernelDensity
import scipy
from scipy.ndimage import gaussian_filter1d
import pandas as pd
import os, sys

path = os.path.dirname(os.path.abspath(__file__))
root_path = path
while os.path.basename(root_path) != 'PhD_article_2':
	root_path = os.path.dirname(root_path)
	
processed_path = os.path.join(root_path,'data','processed','FOB')
processed_path_LOB = os.path.join(processed_path,'LOB')


df_lob = pd.read_parquet(os.path.join(processed_path_LOB, 'NL0000235190_final_LOB.parquet.gzip')).reset_index().drop_duplicates()

df_lob = df_lob[df_lob['side'] == 'Buy']
df_lob = df_lob[df_lob['index'] == '2023-10-02 09:38:00']
df_lob = df_lob[df_lob['price'] >= df_lob['price'].max() - 1.2]
df_lob['color'] = df_lob['side'].apply(lambda x: 'green' if x == 'Buy' else 'red')
df_lob['side'] = df_lob['side'].apply(lambda x: 1 if x == 'Buy' else -1)

# KDE
weight = np.repeat(df_lob['price'], df_lob['size']).to_numpy()
price_range = np.arange(df_lob['price'].min(), df_lob['price'].max(), 0.02)[:, np.newaxis]

kde = KernelDensity(kernel="gaussian", bandwidth=0.03).fit(weight.reshape(-1, 1))
log_density = kde.score_samples(df_lob['price'].to_numpy().reshape(-1, 1))
dens = np.exp(log_density)

df_lob['smoothed_size'] = np.exp(log_density)
df_lob['smoothed_size'] *= df_lob['size']

#df_lob['smoothed_size'] = gaussian_filter1d(df_lob['size'], sigma=6)

data = df_lob.copy()

data = data[['price','smoothed_size','side']].to_numpy()

data_scaled = data.copy()

#data_scaled[:, 1] = np.log(data_scaled[:, 1])
data_scaled[:, 1] = MinMaxScaler(feature_range=(0, 1)).fit_transform(data_scaled[:, 1].reshape(-1, 1)).squeeze()

bic_scores = []

n_clusters_range = range(1, 40)

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
scatter = plt.scatter(df_lob['price'].to_numpy(), df_lob['size'].to_numpy(), c=clusters, cmap='Accent', s=10)
plt.colorbar(scatter, label='Cluster Label')
plt.title(f'Clustering des zones de liquidité avec GMM avec {n_components} clusters')
plt.xlabel('Prix')
plt.ylabel('Volumes')
plt.xlim(35,45)
plt.ylim(-50, 6000)
plt.grid(True)
plt.show()
plt.savefig('fig_gmm.png')

df_res = pd.DataFrame(data_with_clusters, columns=['price','smoothed_size','side','cluster'])
df_res['size'] = df_lob['size'].reset_index(drop=True)
print(df_lob['size'])
df_res.to_csv('results_GMM.csv')