#packages
import os, sys
import zipfile
import pandas as pd
import numpy as np
import argparse
from fastparquet import write
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from clustering import HDBSCAN_model
from clustering_preprocessing import ClusteringPreprocess as cprepro
from clustering_postprocessing import ClusteringPostprocess as cpostpro

from base_log import Base, log_execution

class clustering(Base):
	"""
	Clustering LOB and FO data.

	Args:
		job_id (int, optionnal): Slurm job ID.
	"""
	def __init__(self, job_id: int, resampling_unit: str = 'min'):
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
		self.job_id = job_id
		self.data_path = os.path.join(self.root_path,'data')
		self.raw_path = os.path.join(self.data_path,'raw','FOB')
		self.processed_path = os.path.join(self.data_path,'processed','FOB')
		self.processed_path_LOB = os.path.join(self.processed_path,'LOB')
		self.processed_path_FO = os.path.join(self.processed_path,'FO')
		self.results_path = os.path.join(self.root_path,'results','clustering')
		self.results_path_LOB = os.path.join(self.results_path,'LOB')
		self.results_path_FO = os.path.join(self.results_path,'FO')
		self.LOB = None
		self.FO = None
		self.filename_results = None
		self.file = ''
		self.isin = ''
		self.resampling_unit = resampling_unit
		self.data = None
		self.row = ''
		self.prepro = cprepro()
		self.postpro = cpostpro()
		self.files_input = pd.DataFrame(columns=['ISIN','data'])
		
	def get_files(self):
		"""
		Retrieving built LOB and FO files.
		"""		
		for d in ['LOB', 'FO']:
			tmp_df = pd.DataFrame({'ISIN': [f for f in os.listdir(os.path.join(self.processed_path,f'{d}')) if 'final' in f], 
			'data': f'{d}'})
			
			self.files_input = pd.concat([self.files_input, tmp_df], ignore_index=True)
		
		self.files_input['ISIN'] = self.files_input['ISIN'].apply(lambda x: x.split('_')[0])
		
		
	def load_asset_characteristics(self):
		"""
		Loading characteristics of assets.
		"""
		df_assets = pd.read_csv(os.path.join(self.data_path, 'assets.csv'), index_col=0)
		df_assets.set_index('ISIN', inplace=True)
		t = df_assets.loc[:,df_assets.columns.str.contains('min')]
		a = df_assets.loc[:,~df_assets.columns.str.contains('min')]
		t = t.stack().to_frame().reset_index()
		t.columns = ['ISIN', 'timestep', 'range']
		t = t.merge(a, on='ISIN', how='left')
		
		self.assets = pd.concat([t, t], ignore_index=True)
		self.assets['data'] = [i for i in ['LOB', 'FO'] for _ in range(len(t))]
		self.assets = self.assets[self.assets['timestep'] == '1min']
		
	def load_data(self, asset, data_type):
		"""
		Loading data to cluster.
		"""
		tmp_path = os.path.join(self.processed_path, data_type)
		filename = [f for f in os.listdir(tmp_path) if f'{asset}_final_{data_type}' in f][0]
		self.data = pd.read_parquet(os.path.join(tmp_path, filename)).reset_index().drop_duplicates()
		self.data = self.data.set_index('index').between_time('9:00', '17:30').reset_index()

		# to remove after testing
		#self.data = self.data.loc[self.data['index'].isin(self.data['index'].unique()[:40])]
		
	def checking_file(self):
		"""
		Checking if file already processed.
		"""
		if self.filename_results in os.listdir(self.results_path_LOB) + os.listdir(self.results_path_FO):
			return False
			
		else: return True
	
	def resample_data(self, data):
		"""
		Resampling data.
		"""
		def func_resamp(chunk):
			if chunk.empty:
				return chunk
				
			chunk = chunk.loc[chunk.index.max()]
			chunk = chunk.groupby(chunk.index).agg(list)
			return chunk
			
		data = data.set_index('index').resample(self.row['timestep']).apply(lambda col: func_resamp(col))

		data = data.explode(data.columns.to_list()).dropna().reset_index()
		
		return data
	
	def HDBSCAN_clustering(self, data, progress_bar=None):
		"""
		Processing data with DBSCAN.
		"""
		
		data.loc[:, 'cluster'] = 0
		
		DBI_ls, dunn_ls, silhouette_ls = [], [], []
		
		for _, chunk in tqdm(data.groupby('index'), desc='Processing data with DBSCAN', total=len(data['index'].unique()), ncols=100, mininterval=10):
			if chunk.empty: continue
			
			data_scaled = chunk.copy()[['price','smoothed_size','side']]

			clusterer = HDBSCAN_model(self.row['Tick_step']).model_build()
			
			try:
				clusterer.fit(data_scaled[['price','side','smoothed_size']])
				
				DBI_ls.append(HDBSCAN_model.evaluation.DBI(data_scaled[['price','side','smoothed_size']], clusterer.labels_))
				dunn_ls.append(HDBSCAN_model.evaluation.dunn_index(data_scaled[['price','side','smoothed_size']], clusterer.labels_))
				silhouette_ls.append(HDBSCAN_model.evaluation.silhouette_score(data_scaled[['price','side','smoothed_size']], clusterer.labels_))

				chunk['cluster'] = clusterer.labels_
				
				data.loc[chunk.index, 'cluster'] = chunk['cluster']
				
			except:
				print(data_scaled[['price','side','smoothed_size']])
			
		DBI_df = pd.DataFrame(DBI_ls, columns=['DBI'])
		dunn_df = pd.DataFrame(dunn_ls, columns=['dunn'])
		silhouette_df = pd.DataFrame(silhouette_ls, columns=['silhouette'])
		
		score_df = pd.concat([DBI_df, dunn_df, silhouette_df], axis=1, ignore_index=True)
		
		write(os.path.join(self.results_path_LOB, 'score_' + self.filename_results), score_df, compression='GZIP', append=False)
		
		return data
		
		
	def array_process(self):
		"""
		Running script with slurm array jobs
		"""
		self.get_files()
		self.load_asset_characteristics()
		self.assets = self.assets.merge(self.files_input, on=['ISIN', 'data'], how='inner')
		
		self.row = self.assets.loc[self.job_id]
		print(self.row)
		self.filename_results = f'{self.row["ISIN"]}_clustering_{self.row["data"]}_{self.row["timestep"]}.parquet.gzip'
		
		#if not self.checking_file():
			#sys.exit(f'Clustering already performed for: {self.row["ISIN"]} {self.row["data"]}')
		
		self.load_data(self.row['ISIN'], self.row['data'])
		
		model = HDBSCAN_model(self.row['Tick_step']).model_build()
		
		if self.row['data'] == 'LOB':
			self.data = self.resample_data(data=self.data)
			prepro_data = self.prepro.LOB_preprocessing(self.data, self.row)
			print(prepro_data)

			data = self.HDBSCAN_clustering(prepro_data)
			print(data)
			
			postpro_data = self.postpro.LOB_postprocessing(data, self.row)
			print(postpro_data)
			
			write(os.path.join(self.results_path_LOB, self.filename_results), postpro_data, compression='GZIP', append=False)
			
		if self.row['data'] == 'FO':
			pass
			
		
		
		
#convert str to bool for argparse
def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')
	

if __name__ == "__main__":
	#retrieving arguments if any, specify processing way (slurm, parallelism, classic)
	parser = argparse.ArgumentParser()
	parser.add_argument('--job_id', type=int, default=0)
	parser.add_argument('--slurm_array', '-sa', type=str2bool, default=False)
	args = parser.parse_args()
	
	clust = clustering(args.job_id)    

	if args.slurm_array:
		clust.array_process()