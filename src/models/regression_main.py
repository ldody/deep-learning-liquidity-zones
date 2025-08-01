# import packages
import os, sys
import logging
import warnings
import pandas as pd
import numpy as np
import argparse
import pickle
import optuna
import time
from joblib import Parallel, delayed
import multiprocessing
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
tf.config.threading.set_intra_op_parallelism_threads(60)
tf.config.threading.set_inter_op_parallelism_threads(60)
tf.random.set_seed(42)

warnings.simplefilter(action='ignore', category=Warning)
warnings.simplefilter(action='ignore', category=FutureWarning)

		
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from regression_ANN import ANN_model as ANNmodel
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
		self.path_model = os.path.join(self.root_path,'model')
		self.data_path = os.path.join(self.root_path,'data')
		self.ohlcv_path = os.path.join(self.data_path,'raw','OHLCV')
		self.processed_path = os.path.join(self.data_path,'processed','FOB')
		self.processed_path_LOB = os.path.join(self.processed_path,'LOB')
		self.processed_path_FO = os.path.join(self.processed_path,'FO')
		self.results_path = os.path.join(self.root_path,'results','clustering')
		self.results_path_comp = os.path.join(self.root_path,'results','comp_models')
		self.results_path_LOB = os.path.join(self.root_path,'results','clustering_evaluation')
		self.results_path_FO = os.path.join(self.results_path,'FO')
		self.LOB = None
		self.FO = None
		self.filename_results = None
		self.files = {}
		self.df_assets = pd.read_csv(os.path.join(self.data_path, 'assets.csv'), index_col=0)#[['ISIN','RIC']]
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
		#self.df_ohlcv = self.df_ohlcv['Volume' not in self.df_ohlcv.columns]
		
		self.df_data = pd.read_parquet(self.to_process['path'])
		self.df_data = self.df_data.set_index('index').between_time('9:00', '17:00').reset_index()
		self.df_data = self.df_data[self.df_data['err'].isin([0,1])]
		self.df_data['rank_size'] = self.df_data.groupby('index', as_index=False)['size'].rank(ascending=False, method='first')
		
		
	def array_process(self):
		"""
		Running script with slurm array jobs
		"""
		self.get_files()
		self.to_process = self.df_assets.loc[self.job_id]
		print(self.to_process)
		self.load_data()
		data, ohlcv, n_interval, scaler_p, scaler_v, scaler_r, scaler_nb = self.prepro.preprocessing(self.df_data, self.df_ohlcv, self.to_process['1min'])

		x_train, x_test, y_train, y_test = train_test_split(ohlcv, data, test_size=0.3, shuffle=False)
		_, _, n_train, n_test = train_test_split(ohlcv, n_interval, test_size=0.3, shuffle=False)
		
		train_dataset = tf.data.Dataset.from_tensor_slices((x_train, {#"num_clusters": n_train, 
																	  "bounds": y_train[:,:,:2], 
																	  #"ranks": y_train[:,:,-1]
																	  }))
		train_dataset = train_dataset.batch(64).prefetch(tf.data.AUTOTUNE)
		
		print("X_train dtype:", x_train.dtype, "shape:", x_train.shape)
		print("N_train dtype:", n_train.dtype, "shape:", n_train.shape)
		print("bounds_train dtype:", y_train.dtype, "shape:", y_train.shape)
		print("rank dtype:", y_train[:,:,-1].dtype, "shape:", y_train[:,:,-1].shape)
		
		csv_logger = tf.keras.callbacks.CSVLogger('training_log.csv', append=True)
		checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('last_checkpoint.keras', 
																 save_weights_only=False,
																 save_best_only=False,
																 save_freq="epoch",
																 verbose=1)
																 
		if all(x in os.listdir(self.path) for x in ['training_combined_log.csv','last_checkpoint.keras']):
			last_epoch = pd.read_csv(os.path.join(self.path, 'training_log.csv'))['epoch'].iloc[-1] + 1
			
			model = tf.keras.models.load_model(os.path.join(self.path, 'last_checkpoint.keras'))
		
		else:
			model = ANNmodel().model_build(input_shape = x_train.shape[1:], timesteps = self.prepro.n_pred)
			last_epoch = 0

		model.fit(train_dataset, 
				  epochs=1000, 
				  verbose=2, 
				  initial_epoch=last_epoch,
				  callbacks=[csv_logger, checkpoint_callback])
				  
				  
	def combined_data_process(self):
		"""
		Running script with all the data combined.
		Only one model built.
		"""
		self.get_files()
		

		manager = multiprocessing.Manager()
		lock = manager.Lock()
		
		arrays_dict = manager.dict({'x_train': [],
					  'x_test': [],
					  'y_train': [],
					  'y_test': [],
					  'n_train': [],
					  'n_test': [],
					  'scaler_p': [],
					  'scaler_v': [],
					  'scaler_r': [],
					  'scaler_nb': []})
		
		def func_prepro(i, row, lock, arrays_dict):

			with lock:
				self.to_process = self.df_assets.loc[i]
				self.load_data()
				df_data = self.df_data.copy()
				df_ohlcv = self.df_ohlcv.copy()
				to_process = self.to_process.copy()
			
			data, ohlcv, n_interval, scaler_p, scaler_v, scaler_r, scaler_nb = rprepro().preprocessing(df_data=df_data, df_ohlcv=df_ohlcv, new_var=to_process['1min'])

			x_train, x_test, y_train, y_test = train_test_split(ohlcv, data, test_size=0.3, shuffle=False)
			_, _, n_train, n_test = train_test_split(ohlcv, n_interval, test_size=0.3, shuffle=False)
			
			with lock:
				for key in dict(arrays_dict):
						
					updated_list = arrays_dict[key]
					updated_list.append(locals()[key])
					arrays_dict[key] = updated_list

		try: 
			with open(os.path.join(self.path_model, 'prepro.json'), 'rb') as file:
				arrays_dict = pickle.load(file)
	
		except:
			Parallel(n_jobs=-1)(delayed(func_prepro)(i, row, lock, arrays_dict) for i, row in self.df_assets.iterrows())
			
			arrays_dict = dict(arrays_dict)

			for key, arrays in arrays_dict.items():
				try:
					arrays_dict[key] = np.concatenate(arrays, axis=0)
				except:
					pass
					
			with open(os.path.join(self.path_model, 'prepro.json'), 'wb') as file:
				pickle.dump(arrays_dict, file)
		
		optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
		STUDY_NAME = 'optuna_study'
		DB_PATH = os.path.join(self.path_model, 'ann_optimization.log')
		storage = optuna.storages.JournalStorage(
			optuna.storages.journal.JournalFileBackend(DB_PATH),
		)
		
		dict_params = optuna.load_study(storage=storage, study_name=STUDY_NAME).best_params
		print(optuna.load_study(storage=storage, study_name=STUDY_NAME))
		print(dict_params)
		
		dataset = tf.data.Dataset.from_tensor_slices((arrays_dict['x_train'], {"bounds": arrays_dict['y_train'][:,:,:2]})).shuffle(42)
		
		val_size = int(dataset.cardinality().numpy() * 0.7)
		
		train_dataset = dataset.take(val_size).batch(dict_params['batch_size']).prefetch(tf.data.AUTOTUNE)
		eval_dataset = dataset.skip(val_size).batch(dict_params['batch_size']).prefetch(tf.data.AUTOTUNE)
		
		test_dataset = tf.data.Dataset.from_tensor_slices((arrays_dict['x_test'], {"bounds": arrays_dict['y_test'][:,:,:2]}))
		test_dataset = test_dataset.batch(dict_params['batch_size']).prefetch(tf.data.AUTOTUNE)
		dataset = dataset.batch(dict_params['batch_size']).prefetch(tf.data.AUTOTUNE)
		
		csv_logger_train = tf.keras.callbacks.CSVLogger(os.path.join(self.path_model, 'training_combined_log.csv'), append=True)
		csv_logger_eval = tf.keras.callbacks.CSVLogger(os.path.join(self.path_model, 'eval_combined_log.csv'), append=True)
		csv_logger_test = tf.keras.callbacks.CSVLogger(os.path.join(self.path_model, 'test_combined_log.csv'), append=True)
		
		checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(self.path_model, 'last_checkpoint.keras'), 
																 save_weights_only=False,
																 save_best_only=False,
																 save_freq="epoch",
																 verbose=1)
		
		checkpoint_best_eval = tf.keras.callbacks.ModelCheckpoint(os.path.join(self.path_model, 'best_eval.keras'),          # Chemin du fichier où sauvegarder le modèle
																  monitor='val_loss',                # Ce qu'on veut surveiller
																  save_best_only=True,              # Sauvegarde uniquement le meilleur modèle
																  mode='min',                       # On veut minimiser la val_loss
																  verbose=1                         # Affiche un message à chaque sauvegarde
																  )
		
		model = ANNmodel().model_build(input_shape = arrays_dict['x_train'].shape[1:], timesteps = self.prepro.n_pred, **dict_params)
		
		print(train_dataset)
		print(eval_dataset)
		print(model)
		
		if all(x in os.listdir(self.path_model) for x in ['training_combined_log.csv','last_checkpoint.keras']):
			last_epoch = pd.read_csv(os.path.join(self.path_model, 'training_combined_log.csv'))['epoch'].iloc[-1] + 1
			
			#latest_checkpoint = tf.train.latest_checkpoint(os.path.join(self.path_model, 'last_checkpoint.keras'))
			model.load_weights(os.path.join(self.path_model, 'last_checkpoint.keras'))
		
		else:
			last_epoch = 0
		
		model.fit(train_dataset, 
				  epochs=dict_params['num_epochs'],  
				  verbose=2,
				  initial_epoch=last_epoch,
				  validation_data=test_dataset,
				  callbacks=[csv_logger_train, checkpoint_callback, checkpoint_best_eval])
				  
		model.save(os.path.join(self.path_model, 'ANN_model.keras'))
		
		model.predict(eval_dataset, callbacks=[csv_logger_eval])
		model.predict(test_dataset, callbacks=[csv_logger_test])
		
	def optimization(self):
		"""
		Running hyperparameters optimization process.
		"""
		self.get_files()
		
		# preparing bayesian optimization
		optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
		STUDY_NAME = 'optuna_study'
		DB_PATH = os.path.join(self.path_model, 'ann_optimization.log')
		storage = optuna.storages.JournalStorage(
			optuna.storages.journal.JournalFileBackend(DB_PATH),
		)
		

		#N_TRIALS = 240
		
		while True:
			try:
				print('Loading study')
				study = optuna.load_study(storage=storage, study_name=STUDY_NAME)
				break
				
			except:
				if self.job_id == 0:
					print('Creatind DB')
					study = optuna.create_study(storage=storage, study_name=STUDY_NAME, direction='minimize')
				
				else:
					print('Waiting for DB creation')
					time.sleep(60)
		
		manager = multiprocessing.Manager()
		lock = manager.Lock()
		
		arrays_dict = manager.dict({'x_train': [],
					  'x_test': [],
					  'y_train': [],
					  'y_test': [],
					  'n_train': [],
					  'n_test': [],
					  'scaler_p': [],
					  'scaler_v': [],
					  'scaler_r': [],
					  'scaler_nb': []})
					  
		def func_prepro(i, row, lock, arrays_dict):

			with lock:
				self.to_process = self.df_assets.loc[i]
				self.load_data()
				df_data = self.df_data.copy()
				df_ohlcv = self.df_ohlcv.copy()
				to_process = self.to_process.copy()
			
			data, ohlcv, n_interval, scaler_p, scaler_v, scaler_r, scaler_nb = rprepro().preprocessing(df_data=df_data, df_ohlcv=df_ohlcv, new_var=to_process['1min'])

			x_train, x_test, y_train, y_test = train_test_split(ohlcv, data, test_size=0.3, shuffle=False)
			_, _, n_train, n_test = train_test_split(ohlcv, n_interval, test_size=0.3, shuffle=False)
			
			with lock:
				for key in dict(arrays_dict):
						
					updated_list = arrays_dict[key]
					updated_list.append(locals()[key])
					arrays_dict[key] = updated_list

		try: 
			with open(os.path.join(self.path_model, 'prepro.json'), 'rb') as file:
				pickle.load(file)
	
		except:
			Parallel(n_jobs=-1)(delayed(func_prepro)(i, row, lock, arrays_dict) for i, row in self.df_assets.iterrows())
			
			arrays_dict = dict(arrays_dict)

			for key, arrays in arrays_dict.items():
				try:
					arrays_dict[key] = np.concatenate(arrays, axis=0)
				except:
					pass
					
			with open(os.path.join(self.path_model, 'prepro.json'), 'wb') as file:
				pickle.dump(arrays_dict, file)
		
		print('Starting BA')
		
		def objective(trial):
			num_units_CNN = trial.suggest_categorical('num_units_CNN', [int(2**x) for x in range(5,9)])
			dim_kernel_CNN = trial.suggest_categorical('dim_kernel_CNN', [(int(1+2*x), int(1+2*x)) for x in range(1,6)])
			num_units_GRU = trial.suggest_categorical('num_units_GRU', [int(2**x) for x in range(4,7)])
			num_units_concat = trial.suggest_categorical('num_units_concat', [int(2**x) for x in range(5,10)])
			num_units_output_bloc = trial.suggest_categorical('num_units_output_bloc', [int(2**x) for x in range(5,11)])
			batch_size = trial.suggest_categorical('batch_size', [int(2**x) for x in range(6,11)])
			num_epochs = trial.suggest_categorical('num_epochs', [100, 500, 1000, 1500, 2000])
			
			dict_params = {'num_units_CNN':num_units_CNN, 
						   'dim_kernel_CNN':dim_kernel_CNN,
						   'num_units_GRU':num_units_GRU,
						   'num_units_concat':num_units_concat,
						   'num_units_output_bloc':num_units_output_bloc
						   }
			
			# preparing datasets
			with open(os.path.join(self.path_model, 'prepro.json'), 'rb') as file:
				arrays_dict = pickle.load(file)
			
			dataset = tf.data.Dataset.from_tensor_slices((arrays_dict['x_train'], {"bounds": arrays_dict['y_train'][:,:,:2]})).shuffle(42)
			
			val_size = int(dataset.cardinality().numpy() * 0.05)

			#test_dataset = tf.data.Dataset.from_tensor_slices((arrays_dict['x_test'], {"bounds": arrays_dict['y_test'][:,:,:2]})).shuffle(42)
			
			print(f'len dataset before take {len(dataset)}')
			eval_dataset = dataset.skip(int(dataset.cardinality().numpy() * 0.9))
			dataset = dataset.take(val_size)
			print(f'len dataset before after {len(dataset)}')
			
			dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
			eval_dataset = eval_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
			
			#early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, start_from_epoch=200)
			
			model = ANNmodel().model_build(input_shape = arrays_dict['x_train'].shape[1:], timesteps = self.prepro.n_pred, **dict_params)
			print(model.output_names)

			print('Start fitting model')
			callback = LimitTrainingTime(43200)
			start_time = time.time()
			history = model.fit(dataset, 
								  epochs=num_epochs,  
								  verbose=2,
								  validation_data=eval_dataset,
								  callbacks=[callback]
								  )
			end_time = time.time()
			
			if end_time - start_time > 43200:
				raise optuna.exceptions.TrialPruned()
				
			else:
				return min(history.history['val_loss'][30:])
			
		#TRIALS_PER_JOB = N_TRIALS // int(os.getenv('SLURM_ARRAY_TASK_COUNT', 1))

		study.optimize(objective, n_trials=1, timeout=43200)
		print(study.best_trial)

	def prediction(self):
		"""
		Prediction on the test dataset.
		"""
		self.get_files()
		
		optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
		STUDY_NAME = 'optuna_study'
		DB_PATH = os.path.join(self.path_model, 'ann_optimization.log')
		storage = optuna.storages.JournalStorage(
			optuna.storages.journal.JournalFileBackend(DB_PATH),
		)
		
		with open(os.path.join(self.path_model, 'prepro.json'), 'rb') as file:
			arrays_dict = pickle.load(file)
			
		dict_params = optuna.load_study(storage=storage, study_name=STUDY_NAME).best_params
		
		test_dataset = tf.data.Dataset.from_tensor_slices((arrays_dict['x_test'], {"bounds": arrays_dict['y_test'][:,:,:2]}))
		test_dataset = test_dataset.batch(dict_params['batch_size']).prefetch(tf.data.AUTOTUNE)
		
		model = ANNmodel().model_build(input_shape = arrays_dict['x_train'].shape[1:], timesteps = self.prepro.n_pred, **dict_params)
		model.load_weights(os.path.join(self.path_model, 'last_checkpoint.keras'))
		
		# Prédiction manuelle
		y_preds = []
		y_trues = []

		for x_batch, y_batch in test_dataset:
			y_pred = model.predict(x_batch)
			y_preds.append(y_pred)
			y_trues.append(y_batch)

		# Concatène
		y_preds = tf.concat(y_preds, axis=0)
		y_trues = tf.concat(y_trues, axis=0)

		# Réutilise la métrique importée
		precision = self.prepro.precision_surface_metric(y_trues, y_preds)
		recall = self.prepro.recall_surface_metric(y_trues, y_preds)
		f1 = self.prepro.F1_score(y_trues, y_preds)
		
		pd.DataFrame({'precision':[precision],
					  'recall':[recall],
					  'f1':[f1]},
					  ).to_csv(os.path.join(self.results_path_comp, 'metrics.csv'))
		

class LimitTrainingTime(tf.keras.callbacks.Callback):
	def __init__(self, max_time_s):
		super().__init__()
		self.max_time_s = max_time_s
		self.start_time = None

	def on_train_begin(self, logs):
		self.start_time = time.time()

	def on_train_batch_end(self, batch, logs):
		now = time.time()
		if now - self.start_time >  self.max_time_s:
			self.model.stop_training = True

				  
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
	parser.add_argument('--combined', '-cmb', type=str2bool, default=False)
	parser.add_argument('--bayesian_opti', '-ba', type=str2bool, default=False)
	parser.add_argument('--pred', '-p', type=str2bool, default=False)
	args = parser.parse_args()
	
	reg = regression(args.job_id)

	if args.slurm_array:
		print('launch array')
		reg.array_process()
		
	elif args.combined:
		print('launch combined')
		reg.combined_data_process()
	
	elif args.bayesian_opti:
		print('launch opti')
		reg.optimization()
		
	elif args.pred:
		print('launch prediction')
		reg.prediction()