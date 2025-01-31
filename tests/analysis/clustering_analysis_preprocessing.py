#packages
import os, sys
import zipfile
import pandas as pd
import numpy as np
import argparse
from fastparquet import write
from tqdm import tqdm, tqdm_pandas
import ta


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
		
		
		clusters['cross_time'] = pd.NaT
		
		for i, row in tqdm(clusters.iterrows(), desc='Preprocessing data for clustering analysis', total=len(clusters), ncols=100, mininterval=600):
			clusters.loc[i, 'cross_time'] = self.cross_detection(row, OHLCV)
		
		OHLCV['MACD'] = ta.trend.macd_signal(OHLCV['Close'], window_slow=26, window_fast=12).apply(lambda x: 1 if x > 0 else -1 if x < 0 else 0 if x == 0 else np.nan)
		
		clusters['delta'] = clusters.apply(lambda row: self.calc_delta(row['index'], row['cross_time']) if not pd.isna(row['cross_time']) else np.nan, axis=1)
		clusters = clusters.merge(OHLCV[['Local Time','Close','MACD']], left_on='index', right_on='Local Time')
		clusters['dist'] = abs(clusters['Close'] - clusters[['price_min','price_max']].mean(axis=1))
		clusters = clusters[(clusters['dist'] != 0) & (clusters['ratio'] <= 1)]
		clusters = clusters.dropna(subset=['MACD'])
		
		
		write(os.path.join(self.results_path_analysis, filename), clusters, compression='GZIP', append=False)
		
		return clusters
		
	def cross_detection(self, cluster, df_price):
		"""
		Detecting price crossing with cluster.
		"""
		bound = max(cluster['price_min'], cluster['side'] * cluster['price_max'])
		
		df_price['price'] = df_price.apply(lambda row: abs(min(row['Low'], cluster['side'] * row['High'])), axis=1)
		df_price = df_price[['Local Time', 'price']]
		cond = ((cluster['index'] < df_price['Local Time']) & (cluster['side'] * (df_price['price'] - bound) <= 0))
		
		try:
			return df_price.loc[cond, 'Local Time'].iloc[0]
		except:
			return pd.NaT
			
	def calc_delta(self, start, end):
		"""
		Calculating time delta considering only open market hours and days.
		"""
		time_range = pd.date_range(start=start, end=end, freq='min')
		time_range = time_range.to_series().between_time("09:00", "17:30")
		time_range = time_range[time_range.index.weekday < 5]
		
		return len(time_range) - 1