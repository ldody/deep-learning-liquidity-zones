# import packages
import os, sys
import warnings
import pandas as pd
import numpy as np
import argparse
from joblib import Parallel, delayed


warnings.simplefilter(action='ignore', category=Warning)
warnings.simplefilter(action='ignore', category=FutureWarning)


class PivotPoints():
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
		self.results_path = os.path.join(self.root_path,'results','comp_models')
		self.results_path_LOB = os.path.join(self.root_path,'results','clustering_evaluation')
		self.results_path_FO = os.path.join(self.root_path,'results')
		self.LOB = None
		self.FO = None
		self.filename_results = None
		self.files = {}
		self.df_assets = pd.read_csv(os.path.join(self.data_path, 'assets.csv'), index_col=0)#[['ISIN','RIC']]
		self.isin = ''
		self.resampling_unit = resampling_unit
		self.data = None
		self.row = ''
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
		#self.df_ohlcv = self.df_ohlcv['Volume' not in self.df_ohlcv.columns]
		
		self.df_data = pd.read_parquet(self.to_process['path'])
		self.df_data = self.df_data.set_index('index').between_time('9:00', '17:00').reset_index()
		self.df_data = self.df_data[self.df_data['err'].isin([0,1])]
		self.df_data['rank_size'] = self.df_data.groupby('index', as_index=False)['size'].rank(ascending=False, method='first')
		self.df_data = self.df_data.loc[self.df_data['rank_size'] <= 10, ['index', 'price_min', 'price_max', 'rank_size']]
		
	def func_pivots(self, row):
		
		res = pd.DataFrame()
		
		self.to_process = row
		self.load_data()
		
		def pivots(df, window=5):
			df['Pivot'] = df['Close'].rolling(window).mean()  # Pivot simple
			
			df['R1'] = (2 * df['Pivot']) - df['Low'].rolling(window).min()
			df['S1'] = (2 * df['Pivot']) - df['High'].rolling(window).max()

			df['R2'] = df['Pivot'] + (df['High'].rolling(window).max() - df['Low'].rolling(window).min())
			df['S2'] = df['Pivot'] - (df['High'].rolling(window).max() - df['Low'].rolling(window).min())

			df['R3'] = df['High'].rolling(window).max() + 2 * (df['Pivot'] - df['Low'].rolling(window).min())
			df['S3'] = df['Low'].rolling(window).min() - 2 * (df['High'].rolling(window).max() - df['Pivot'])
			
			return df
		
		def launch():
			test = self.df_ohlcv.copy()
			test['Accuracy'] = 0

			for i, row in test.iterrows():
				tmp = self.df_data.loc[self.df_data['index'] == row['Local Time']]
				a = 0
				d = 0
				for p in ['Pivot','R1','S1','R2','S2','R3','S3']:

					for j, rowdata in tmp.iterrows():

						if (row[p] > rowdata['price_min']) & (row[p] < rowdata['price_max']):
							a += 1
							d += min(abs(row[p] - rowdata['price_min']), abs(row[p] - rowdata['price_max']))

				r = min(7, len(tmp))

				test.loc[test['Local Time'] == row['Local Time'], 'Accuracy'] = a/r
				test.loc[test['Local Time'] == row['Local Time'], 'MAE'] = d/a if a != 0 else np.nan
				
			return test
		
		for w in [1, 3, 5, 10, 15, 30]:
			self.df_ohlcv = pivots(self.df_ohlcv, w)
			t = launch()
			
			t['windows'] = w
			t['RIC'] = row['RIC']
			
			res = pd.concat([res, t])
			
		return res
		
	def array_process(self):
		"""
		Running script with slurm array jobs
		"""
		self.get_files()
		
		results = Parallel(n_jobs=-1)(delayed(self.func_pivots)(row) for _, row in self.df_assets.iterrows())
		
		pd.concat(results).to_csv(os.path.join(self.results_path, 'PivotPoints.csv'))
		print(pd.concat(results))

		
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
	
	reg = PivotPoints(args.job_id)

	if args.slurm_array:
		reg.array_process()