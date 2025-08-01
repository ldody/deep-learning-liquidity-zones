# import packages
import os, sys
import warnings
import pandas as pd
import numpy as np
import argparse
from joblib import Parallel, delayed
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

warnings.simplefilter(action='ignore', category=Warning)
warnings.simplefilter(action='ignore', category=FutureWarning)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from regression_preprocessing import RegressionPreprocess as rprepro


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
		self.prepro = rprepro()

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
		
	def preprocessing_data(self, ohlcv, data):
		"""
		Preprocessing the data.
		"""
		n_points = 50

		# 40 valeurs entre 0 et 1
		lin_vals = np.linspace(0, 1, n_points)  # (40,)

		# Étendre lin_vals à (1, 1, 40) pour le broadcast
		lin_vals_expanded = lin_vals[None, None, :]  # (1, 1, 40)

		# Étendre les bornes des intervalles à (300, 10, 1)
		starts = data[:, :, 0][..., None]  # (300, 10, 1)
		ends   = data[:, :, 1][..., None]  # (300, 10, 1)

		# Comparaison vectorisée : (300, 10, 40) booléen
		mask = (starts <= lin_vals_expanded) & (lin_vals_expanded <= ends)

		# Pour chaque échantillon et chaque point, on regarde si au moins un intervalle le contient
		y = np.any(mask, axis=1).astype(np.uint8)
		X = ohlcv
		X_flat = X.reshape(X.shape[0], -1)
		
		X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)
		
		return X_train, X_test, y_train, y_test

	def func_reg(self, row):
		"""
		Launching logistic regression.
		"""
		self.to_process = row
		self.load_data()
		data, ohlcv, n_interval, scaler_p, scaler_v, scaler_r, scaler_nb = self.prepro.preprocessing(self.df_data, self.df_ohlcv, self.to_process['1min'])
		X_train, X_test, y_train, y_test = self.preprocessing_data(ohlcv, data)
		
		base_model = LogisticRegression(max_iter=1000)
		model = MultiOutputClassifier(base_model)

		model.fit(X_train, y_train)
		
		y_pred = model.predict(X_test)

		report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
		
		return pd.DataFrame([report['micro avg']])
		
	def func_svc(self, row):
		"""
		Launching SVC.
		"""
		self.to_process = row
		self.load_data()
		data, ohlcv, n_interval, scaler_p, scaler_v, scaler_r, scaler_nb = self.prepro.preprocessing(self.df_data, self.df_ohlcv, self.to_process['1min'])
		X_train, X_test, y_train, y_test = self.preprocessing_data(ohlcv, data)
		
		svm_base = SVC(probability=True, kernel='poly')  # ou 'linear' si plus rapide
		model = OneVsRestClassifier(svm_base)

		# Entraînement
		model.fit(X_train, y_train)

		# Prédictions
		y_pred = model.predict(X_test)

		report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
		
		return pd.DataFrame([report['micro avg']])

	def array_process(self):
		"""
		Running script with slurm array jobs
		"""
		self.get_files()
		
		dict_model = {'reg':self.func_reg(),
					  'svc':self.SVC()}
		
		for i in dict_model:
			results = Parallel(n_jobs=-1)(delayed(dict_model[i])(row) for _, row in self.df_assets.iterrows())
			
			results = pd.concat(results)
			
			results.to_csv(os.path.join(self.results_path, f'{i}.csv'))
			
			print(results)
	

		
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
	parser.add_argument('--oat', '-oat', type=str2bool, default=False)
	args = parser.parse_args()
	
	reg = PivotPoints(args.job_id)

	if args.slurm_array:
		reg.array_process()
		
	if args.oat:
		reg.OAT()