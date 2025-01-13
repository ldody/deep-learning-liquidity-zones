import numpy as np
import random
import pandas as pd

class TanhNormalizer:
	def __init__(self):
		self.mean_ = None
		self.std_ = None

	def fit(self, df):
		# Convertir le DataFrame en un tableau NumPy
		try:
			X = df.to_numpy()
		except: None

		# Calculer les moyennes et écarts types de chaque colonne
		self.mean_ = np.mean(X, axis=0)
		self.std_ = np.std(X, axis=0)
		return self

	def transform(self, df):
		
		X = df
		# Appliquer la normalisation tanh avec les moyennes et écarts types
		X_normalized = (np.tanh(0.01*(X+self.mean_)/self.std_)+1)/2
		
		if (self.mean_ == 0).all() and (self.std_ == 0).all():
			X_normalized = np.nan_to_num(X_normalized)
		
		return X_normalized

	def fit_transform(self, df):
		# Adapter et transformer les données en une seule étape
		self.fit(df)
		return self.transform(df)

	def inverse_transform(self, df):
		# Convertir le DataFrame en un tableau NumPy
		#X = df.values
		X = df

		# Inverser la normalisation tanh avec les moyennes et écarts types
		X_original = np.arctanh(2*X-1)*self.std_/0.01-self.mean_
		return X_original

