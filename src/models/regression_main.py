# import packages
import os, sys
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
import tensorflow as tf
tf.random.set_seed(42)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from regression_CNN import CNN_model as CNNmodel
from regression_preprocessing import RegressionPreprocess as rprepro
#from regression_postprocessing import RegressionPostprocess as rpostpro
from base_log import Base, log_execution


class regression(Base):
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
		self.data_path = os.path.join(self.root_path,'data')
		self.ohlcv_path = os.path.join(self.data_path,'raw','OHLCV')
		self.processed_path = os.path.join(self.data_path,'processed','FOB')
		self.processed_path_LOB = os.path.join(self.processed_path,'LOB')
		self.processed_path_FO = os.path.join(self.processed_path,'FO')
		self.results_path = os.path.join(self.root_path,'results','clustering')
		self.results_path_LOB = os.path.join(self.results_path,'LOB')
		self.results_path_FO = os.path.join(self.results_path,'FO')
		self.LOB = None
		self.FO = None
		self.filename_results = None
		self.files = {}
		self.df_assets = pd.read_csv(os.path.join(self.data_path, 'assets.csv'), index_col=0)[['ISIN','RIC']]
		self.isin = ''
		self.resampling_unit = resampling_unit
		self.data = None
		self.row = ''
		self.prepro = rprepro()
		#self.postpro = cpostpro()
		self.files_input = pd.DataFrame(columns=['ISIN','data'])
		self.to_process = None

	def get_files(self):
		"""
		Retrieving clustering files and OHLCV files.
		
		
		
		"""
		for path, f_type in zip([self.results_path_LOB, self.results_path_FO, self.ohlcv_path], ['LOB', 'FO', 'OHLCV']):
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
		t = self.df_assets.loc[:,['LOB', 'FO']]
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
		
		self.df_data = pd.read_parquet(self.to_process['path'])
		self.df_data = self.df_data.set_index('index').between_time('9:00', '17:00').reset_index()
		
		
	def array_process(self):
		"""
		Running script with slurm array jobs
		"""
		self.get_files()
		self.to_process = self.df_assets.loc[self.job_id]
		print(self.to_process)
		self.load_data()
		data, ohlcv, n_interval, scaler_p, scaler_v = self.prepro.preprocessing(self.df_data, self.df_ohlcv)
		x_train, x_test, y_train, y_test = train_test_split(ohlcv, data, test_size=0.3, shuffle=False)
		_, _, n_train, n_test = train_test_split(ohlcv, n_interval, test_size=0.3, shuffle=False)
		
		model = CNNmodel().model_build(input_shape = np.expand_dims(x_train[0], axis=2).shape, timesteps = self.prepro.n_pred)
		print("X_train dtype:", x_train.dtype, "shape:", x_train.shape)
		print("N_train dtype:", n_train.dtype, "shape:", n_train.shape)
		print("bounds_train dtype:", y_train.dtype, "shape:", y_train.shape)

		model.fit(x_train, {"n_intervals": n_train, "interval_bounds": y_train}, epochs=200, batch_size=16, verbose=2)

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
	
	reg = regression(args.job_id)

	if args.slurm_array:
		reg.array_process()