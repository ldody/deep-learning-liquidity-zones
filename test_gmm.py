import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Génération de données simulées
np.random.seed(42)

# Simuler des prix et des volumes (zones de liquidité denses et bruit)
prices = np.concatenate([
	np.random.normal(100, 1, 500),  # Cluster autour de 100
	np.random.normal(105, 1, 300),  # Cluster autour de 105
	np.random.uniform(90, 110, 200) # Bruit uniforme
])
volumes = np.concatenate([
	np.random.randint(1000, 10000, 500),  # Volumes élevés autour de 100
	np.random.randint(5000, 20000, 300), # Volumes élevés autour de 105
	np.random.randint(0, 200, 200)       # Bruit de faible volume
])

# Construire le tableau des données
data = np.array([prices, volumes]).T

# Standardiser les données (important pour GMM)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Appliquer GMM
n_components = 3  # Définir le nombre de clusters supposés
gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
clusters = gmm.fit_predict(data_scaled)

# Ajouter les clusters au dataset
data_with_clusters = np.hstack((data, clusters.reshape(-1, 1)))

# Visualisation des résultats
plt.figure(figsize=(10, 6))
scatter = plt.scatter(prices, volumes, c=clusters, cmap='viridis', s=10)
plt.colorbar(scatter, label='Cluster Label')
plt.title('Clustering des zones de liquidité avec GMM')
plt.xlabel('Prix')
plt.ylabel('Volumes')
plt.grid(True)
plt.show()
plt.savefig('fig_gmm.png')