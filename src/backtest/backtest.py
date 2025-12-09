# import packages
import os, sys
import logging
import warnings
import pandas as pd
import numpy as np
import argparse
import pickle
import time
from joblib import Parallel, delayed

warnings.simplefilter(action='ignore', category=Warning)
warnings.simplefilter(action='ignore', category=FutureWarning)
		
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_log import Base, log_execution

class ohlcv_bid_ask(Base):
	"""
	Clustering LOB and FO data.

	Args:
		job_id (int, optionnal): Slurm job ID.
	"""
	def __init__(self, job_id: int = 0, resampling_unit: str = 'min'):
		"""
		Initializing the regression instance.
		
		Attributes:

			
		Args:
			job_id (int, optionnal): Slurm job ID, Default=0.
		"""
		self.path = os.path.dirname(os.path.abspath(__file__))
		self.root_path = self.path
		while os.path.basename(self.root_path) != 'PhD_article_2':
			self.root_path =  os.path.dirname(self.root_path)
		self.job_id = job_id
		self.path_model = os.path.join(self.root_path,'model')
		self.data_path = os.path.join(self.root_path,'data')
		self.ohlcv_path = os.path.join(self.data_path,'raw','OHLCV')
		self.processed_path = os.path.join(self.data_path,'processed','FOB')
		self.processed_path_LOB = os.path.join(self.processed_path,'LOB')
		self.results_path = os.path.join(self.root_path,'results','clustering')
		self.df_assets = pd.read_csv(os.path.join(self.data_path, 'assets.csv'), index_col=0)#[['ISIN','RIC']]
		self.files_input = pd.DataFrame(columns=['ISIN','data'])

	def get_files(self):
		"""
		Retrieving clustering files and OHLCV files.
		
		
		
		"""
		for path, f_type in zip([self.processed_path_LOB, self.ohlcv_path], ['LOB', 'OHLCV']):
			self.files[f_type] = [os.path.join(path, f) for f in os.listdir(path) if (f.split('_')[0] in self.df_assets.values.flatten().tolist())]
			
			if f_type == 'OHLCV':
				df_tmp = pd.DataFrame({f_type: self.files[f_type]})
				df_tmp['RIC'] = df_tmp[f_type].apply(lambda x: os.path.basename(x).split('_')[0])
				self.df_assets = self.df_assets.merge(df_tmp, on='RIC', how='left')
				
			else:
				df_tmp = pd.DataFrame({f_type: self.files[f_type]})
				df_tmp['ISIN'] = df_tmp[f_type].apply(lambda x: os.path.basename(x).split('_')[0])
				self.df_assets = self.df_assets.merge(df_tmp, on='ISIN', how='left')
				
		self.df_assets.set_index(['ISIN','RIC','OHLCV'], inplace=True)
		t = self.df_assets.loc[:,['LOB']]
		a = self.df_assets.loc[:,~self.df_assets.columns.isin(['LOB', 'FO'])].reset_index()
		t = t.stack().to_frame().reset_index()
		t.columns = ['ISIN', 'RIC', 'OHLCV', 'data_type', 'path']
		self.df_assets = t.merge(a, on=['ISIN','RIC','OHLCV'], how='left').reset_index()
		
	def load_data(self):
		"""
		Loading data.
		"""
		self.df_ohlcv = pd.read_csv(self.to_process['OHLCV'])
		self.df_ohlcv['Local Time'] = pd.to_datetime(self.df_ohlcv['Local Time'])
		self.df_ohlcv = self.df_ohlcv.set_index('Local Time').between_time('9:00', '17:00').reset_index()
		#self.df_ohlcv = self.df_ohlcv['Volume' not in self.df_ohlcv.columns]
		
		self.df_data = pd.read_parquet(self.to_process['path'])
		self.df_data = self.df_data.set_index('index').between_time('9:00', '17:00').reset_index()
		self.df_data = self.df_data[self.df_data['err'].isin([0,1])]
		
		def attrib(data):
			if 'Buy' in data['side'].values:
				return data.loc[data['price'] == data['price'].max()]
			
			elif 'Sell' in data['side'].values:
				return data.loc[data['price'] == data['price'].min()]
				
		self.df_data = self.df_data.reset_index().groupby(['index','side'], as_index=False).apply(attrib)
		self.df_data = self.df_data.reset_index(drop=True)
		self.df_data = self.df_data.groupby(['index','side']).last()
		self.df_data = self.df_data.unstack(level=1)
		self.df_data.columns = [f"{lvl0}_{lvl1}" for (lvl0, lvl1) in self.df_data.columns]
		self.df_data = self.df_data.rename(columns={
			"price_Buy": "bid",
			"price_Sell": "ask",
			"size_Buy": "bid_size",
			"size_Sell": "ask_size",
		})
		
		print(self.df_ohlcv, self.df_data)
		
		self.merged_df = self.df_ohlcv.join(self.df_data, how="inner").sort_index()
		self.merged_df["mid"] = (self.merged_df["bid"] + self.merged_df["ask"]) / 2.0
		
	def main(self):
		"""
		Launch BT.
		"""
		self.get_files()
		
		for i in range(len(self.to_process)):
			self.to_process = self.df_assets.loc[i]
			print(self.to_process)
			self.load_data()
			print(self.merge)
			break
			

if __name__ == "__main__":
	
	ohlcv_bid_ask().main()