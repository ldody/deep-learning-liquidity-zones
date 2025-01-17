#packages
import os, sys
import zipfile
import pandas as pd
import numpy as np
import argparse
from fastparquet import write
from tqdm import tqdm, tqdm_pandas


class ClusteringAnalysisPreprocess:
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
		self.results_path_analysis = os.path.join(self.root_path,'results','clustering_analysis')
		
	def preprocessing(self, OHLCV, clusters, filename):
		"""
		Preprocessing clusters for analysis.
		"""
		#clusters['cross_time'] = clusters.apply(lambda clust: self.cross_detection(clust, OHLCV), axis=1)
		if filename in os.listdir(self.results_path_analysis):
			clusters = pd.read_parquet(os.path.join(self.results_path_analysis, filename))
			return clusters
		
		
		clusters['cross_time'] = np.nan
		
		for i, row in tqdm(clusters.iterrows(), desc='Preprocessing data for clustering analysis', total=len(clusters), ncols=100, mininterval=600):
			clusters.loc[i, 'cross_time'] = self.cross_detection(row, OHLCV)
		
		write(os.path.join(self.results_path_analysis, filename), clusters, compression='GZIP', append=False)
		
		return clusters
		
	def cross_detection(self, cluster, df_price):
		"""
		Detecting price crossing with cluster.
		"""
		bound = max(cluster['price_min'], cluster['side'] * cluster['price_max'])
		
		df_price['price'] = df_price.apply(lambda row: min(row['Low'], cluster['side'] * row['Close']), axis=1)
		df_price = df_price[['Local Time', 'price']]
		
		cond = ((cluster['index'] <= df_price['Local Time']) & (cluster['side'] * (df_price['price'] - bound) <= 0))
		
		try:
			return df_price.loc[cond, 'Local Time'].iloc[0]
		except:
			return np.nan