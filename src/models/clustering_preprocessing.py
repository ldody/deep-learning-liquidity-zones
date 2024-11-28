#packages
import os, sys
import zipfile
import pandas as pd
import numpy as np
import argparse
from fastparquet import write
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KernelDensity


class ClusteringPreprocess:
	"""
	Preprocessing FOB to create LOB.

	Args:
		job_id (int, optionnal): Slurm job ID.
	"""
	def __init__(self, job_id: int = 0, resampling_unit: str = 'min'):
		"""
		Initializes the FOBPreprocessor instance.
		
		Attributes:
			path (str): Path of the current script.
			root_path (str): Root path of the project.
			job_id (int): Slurm job ID.
			raw_path (str): Path of the repository with raw data of FOB /data/raw/FOB/.
			processed_path (str): Path of the repository with processed data of FOB /data/processed/FOB/.
			processed_path (str): Path of the repository with processed data of LOB /data/processed/FOB/LOB/.
			processed_path (str): Path of the repository with processed data of FO /data/processed/FOB/FO/.
			fobdm (Class): Class from the FOB database management.
			FOB (DataFrame): FOB DataFrame.
			LOB (DataFrame): LOB DataFrame.
			FO (DataFrame): FO DataFrame.
			filename_tmp (str): Name of the temporary parquet file with LOB dataframe.
			filename_zip (str): Name of the gzip file with the final parquet file with LOB/FO dataframe.
			file (str): Name of the FOB file in process.
			isin (str): Name of the ISIN in process.
			resampling_unit (str): Rule of resampling for the FOB.
			
		Args:
			job_id (int, optionnal): Slurm job ID, Default=0.
		"""
		self.path = os.path.dirname(os.path.abspath(__file__))
		self.root_path = self.path
		while os.path.basename(self.root_path) != 'PhD_article_2':
			self.root_path =  os.path.dirname(self.root_path)
		
	def LOB_preprocessing(self, data, asset_char):
		
		data[['liquidity_ref','smoothed_size']] = 0.0

		for _, chunk in data.groupby('index'):
			if chunk.empty: continue
			
			lower_bound = chunk.loc[chunk['side'] == 'Buy','price'].max() - asset_char['range'] 
			upper_bound = chunk.loc[chunk['side'] == 'Sell','price'].min() + asset_char['range']
			
			chunk['price'] = chunk['price'].apply(lambda x: x if lower_bound <= x <= upper_bound else np.nan)
			chunk.dropna(inplace=True)
			
			buy_liq = chunk[chunk['side'] == 'Buy']['size'].sum()
			sell_liq = chunk[chunk['side'] == 'Sell']['size'].sum()

			chunk['liquidity_ref'] = chunk.apply(lambda row: buy_liq if row['side'] == 'Buy' else sell_liq, axis=1)

			chunk['side'] = chunk['side'].apply(lambda x: 1 if x == 'Buy' else -1)
			'''
			weight = np.repeat(chunk['price'], chunk['size']).to_numpy()
			price_range = np.arange(chunk['price'].min(), chunk['price'].max(), asset_char['Tick_step'])[:, np.newaxis]
			
			kde = KernelDensity(kernel="gaussian", bandwidth=asset_char['Tick_step'] * 1.5).fit(weight.reshape(-1, 1))
			log_density = kde.score_samples(chunk['price'].to_numpy().reshape(-1, 1))

			chunk['smoothed_size'] = np.exp(log_density)
			chunk['smoothed_size'] *= chunk['size']
			
			'''
			for _, c in chunk.groupby('side'):
				weight = np.repeat(c['price'], c['size']).to_numpy()
				price_range = np.arange(c['price'].min(), c['price'].max(), asset_char['Tick_step'])[:, np.newaxis]
				
				kde = KernelDensity(kernel="gaussian", bandwidth=asset_char['Tick_step'] * 1.5).fit(weight.reshape(-1, 1))
				log_density = kde.score_samples(c['price'].to_numpy().reshape(-1, 1))

				c['smoothed_size'] = np.exp(log_density)
				c['smoothed_size'] *= c['size']
				
				chunk.loc[c.index, 'smoothed_size'] = c['smoothed_size'].astype(float)
			
			           
			try:
				chunk['smoothed_size'] = MinMaxScaler(feature_range=(0, 1)).fit_transform(chunk['smoothed_size'].to_numpy().reshape(-1, 1)).squeeze()
				
			except:
				pass

			data.loc[chunk.index, ['side', 'liquidity_ref', 'smoothed_size']] = chunk[['side', 'liquidity_ref', 'smoothed_size']]
		data = data[data['liquidity_ref'] != 0]
		
		return data
