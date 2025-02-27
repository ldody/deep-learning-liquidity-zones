#packages
import os, sys
import zipfile
import pandas as pd
import numpy as np
import argparse
from fastparquet import write
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
#from tanh_normalizer import TanhNormaliser
from base_log import Base, log_execution


class RegressionPreprocess(Base):
	"""
	Preprocessing FOB to create LOB.

	Args:
		job_id (int, optionnal): Slurm job ID.
	"""
	def __init__(self, job_id: int = 0, var: float = 1.0, n_pred: int = 10):
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
			
		self.n_pred = n_pred
		self.var = var
		
	def preprocessing(self, df_data, df_ohlcv, new_var):
		'''
		Preprocessing LOB/FO data and OHLCV for CNN 2D model.
		'''
		self.var = new_var
		df_data = df_data.loc[df_data['rank_size'] <= 10, ['index', 'price_min', 'price_max', 'rank_size']]
		
		def fill_lines(group):
			missing_lines = self.n_pred - len(group)
			if missing_lines > 0:
				additional_lines = pd.DataFrame({
					'index': [group['index'].iloc[0]] * missing_lines
				})
				additional_lines[['price_min', 'price_max', 'rank_size']] = 0
				return pd.concat([group, additional_lines], ignore_index=True)
			return group
		
		df_data = df_data.groupby('index', group_keys=False).apply(fill_lines)
		df_ohlcv['Volume'] = df_ohlcv['Volume'].fillna(0)
		df_ohlcv['Close'] = df_ohlcv['Close'].ffill()
		df_ohlcv = df_ohlcv.bfill(axis=1)
		
		ohlcv, data, n_interval = self.prepare_sequences(df_ohlcv, df_data)
		
		scaled_ohlcv, scaled_data, ls_scaler_p, ls_scaler_v, scaler_r, scaler_nb = self.scaling(ohlcv, data)
		
		n_interval = scaler_nb.transform(n_interval.reshape(-1, 1))
		
		return scaled_data, scaled_ohlcv, n_interval, ls_scaler_p, ls_scaler_v, scaler_r, scaler_nb
		

	def prepare_sequences(self, df_ohlcv, df_data, window_size: int=100):
		'''
		Preparing the sequences.
		'''
		sequences_ohlcv = []
		
		for i in range(len(df_ohlcv) - window_size + 1):
			sequences_ohlcv.append(df_ohlcv.to_numpy()[i : i + window_size])
		
		sequences_ohlcv = np.array(sequences_ohlcv)
		
		sequences_data = []
		sequences_n_interval = []
		index_todelete = []
		
		for i, t in enumerate([date[-1,0] for date in sequences_ohlcv]):
			
			if len(df_data[df_data['index'] == t]) == 0:
				index_todelete.append(i)
				continue

			sequences_data.append(df_data[df_data['index'] == t].to_numpy())
			sequences_n_interval.append(float(df_data.loc[df_data['index'] == t, 'rank_size'].max()))
		
		sequences_ohlcv = np.delete(sequences_ohlcv, index_todelete, axis=0)
		
		try:
			sequences_data = np.array(sequences_data)
		except:
			print('ERROR !!!!!', len(sequences_data))
			sys.exit()
			
			
		sequences_n_interval = np.array(sequences_n_interval)
		
		return sequences_ohlcv, sequences_data, sequences_n_interval
		
	def scaling(self, ohlcv, data):
		'''
		Scaling data.
		'''
		ls_scaler_price = []
		ls_scaler_volume = []
		
		scaled_ohlcv = []
		scaled_data = []
		
		scaler_rank = MinMaxScaler(feature_range=(0, 1)).fit(np.array([1,self.n_pred]).reshape(-1,1))
		scaler_nb = MinMaxScaler(feature_range=(0, 1)).fit(np.array([0,self.n_pred]).reshape(-1,1))
		
		for seq_ohlcv, seq_data in zip(ohlcv, data):
			scaler_price = MinMaxScaler(feature_range=(0, 1))
			scaler_volume = MinMaxScaler(feature_range=(0, 1))
			
			price_offset = np.full(3, seq_ohlcv[-1,1])
			price_offset[0] += self.var
			price_offset[1] -= self.var
			
			scaler_price.fit(price_offset.reshape(-1, 1))
			scaler_volume.fit(seq_ohlcv[:,5].reshape(-1, 1))
			
			ls_scaler_price.append(scaler_price)
			ls_scaler_volume.append(scaler_volume)
			
			scaled_seq_ohlcv = np.empty_like(seq_ohlcv)
			scaled_seq_data = np.empty_like(seq_data)		
			
			for col in range(1,5):
				scaled_seq_ohlcv[:,col] = scaler_price.transform(seq_ohlcv[:,col].reshape(-1,1)).squeeze()
				
			scaled_seq_ohlcv[:,5] = scaler_volume.transform(seq_ohlcv[:,5].reshape(-1,1)).squeeze()
			
			for col in range(1,3):
				scaled_seq_data[:,col] = scaler_price.transform(seq_data[:,col].reshape(-1,1)).squeeze()
				
			scaled_seq_data[:,3] = scaler_rank.transform(seq_data[:,3].reshape(-1,1)).squeeze()
			
			scaled_ohlcv.append(scaled_seq_ohlcv)
			scaled_data.append(scaled_seq_data)
			
		scaled_ohlcv = np.array(scaled_ohlcv, dtype=np.float32)[:,:,1:]
		scaled_data = np.array(scaled_data, dtype=np.float32)[:,:,1:]
		
		return scaled_ohlcv, scaled_data, ls_scaler_price, ls_scaler_volume, scaler_rank, scaler_nb