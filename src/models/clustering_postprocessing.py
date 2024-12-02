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
from tqdm import tqdm


class ClusteringPostprocess:
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
			filename_data (str): Name of the temporary parquet file with LOB dataframe.
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
		
	def LOB_postprocessing(self, data, asset_char):
		
		tick_step = asset_char['Tick_step']
		
		data = data.groupby(['index','side','cluster'], as_index=False).agg({'size': 'sum', 
																  'liquidity_ref': 'last', 
																  'price': ['min', 'max']})

		data.columns = ['_'.join(col) if (isinstance(col, tuple)) & ('price' in col) else col[0] for col in data.columns]
		data['range'] = data[['price_min','price_max']].apply(lambda row: abs(row['price_min'] - row['price_max']) if (row['price_min'] - row['price_max']) != 0 else tick_step, axis=1)
		data['density'] = data['size'] / data['range']
		data['ratio'] = data['size'] / data['liquidity_ref']

		data['stick_to_prev'] = False

		for _, c in data.groupby(['index','side']):
			inter = []
			for i, row in c.iterrows():
				val_min = row['price_min']
				val_max = row['price_max']

				if (any(r_min - tick_step <= val_min <= r_max + tick_step for r_min, r_max in inter) | 
					any(r_min - tick_step <= val_max <= r_max + tick_step for r_min, r_max in inter)):

					data.at[i, 'stick_to_prev'] = True

				else:
					inter.append((row['price_min'], row['price_max']))
					
		data = data[(~data['stick_to_prev']) & (data['cluster'] != -1)]
		
		return data