#packages
import os, sys
import zipfile
import pandas as pd
import numpy as np
import argparse
from fastparquet import write
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, LSTM, Bidirectional, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MultiHeadAttention, Dropout, Add, Reshape, Lambda
from tqdm import tqdm
#from base_log import Base, log_execution
tf.random.set_seed(42)


class ANN_model():
	"""
	Build and fitting CNN 2D model.

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
		self.job_id = job_id
		self.resampling_unit = resampling_unit
		
	def model_build(self, input_shape, timesteps: int = 10, num_units_concat=256, num_units_output=256, **kwargs):
		"""
		Building and compiling CNN 2D model.
		
		Args:
			input_shape (tuple): Dimensions de l'entrée (hauteur, largeur, canaux).

		Returns:
			model (tf.keras.Model): Compiled CNN 2D model.
		"""
		
		dict_params = kwargs
		
		# === 1. Transformer Block ===
		def transformer_block(inputs, num_heads=4, dim_ff=128, dropout_rate=0.1, **kwargs):
			"""Transformer Encoder Block"""
			attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
			attn_output = Dropout(dropout_rate)(attn_output)
			out1 = Add()([inputs, attn_output])  # Residual Connection
			out1 = LayerNormalization()(out1)
			
			ff_output = Dense(inputs.shape[-1], activation="relu")(out1)
			ff_output = Dropout(dropout_rate)(ff_output)
			out2 = Add()([out1, ff_output])  
			out2 = LayerNormalization()(out2)
			
			return out2

		# === 2. CNN Block ===
		def cnn_block(inputs, num_units_CNN=32, dim_kernel_CNN=(5,5), num_units_output_bloc=128, **kwargs):
			"""CNN for pattern recognition in candlestick data"""
			x = Conv2D(num_units_CNN, dim_kernel_CNN, activation='relu', padding='same')(inputs)
			x = Conv2D(int(num_units_CNN/2), (3, 3), activation='relu', padding='same')(x)
			x = Flatten()(x)
			return Dense(num_units_output_bloc, activation='relu')(x)

		# === 3. LSTM Block ===
		def lstm_block(inputs, num_units_LSTM=64,  num_units_output_bloc=128, **kwargs):
			"""LSTM to capture temporal dependencies"""
			x = Bidirectional(LSTM(num_units_LSTM, return_sequences=True))(inputs)
			x = Bidirectional(LSTM(int(num_units_LSTM/2)))(x)
			return Dense(num_units_output_bloc, activation='relu')(x)

		# === 4. Model Input ===
		input_lstm = Input(shape=input_shape, name='OHLCV')
		input_cnn = Reshape((input_shape[0], input_shape[1], 1))(input_lstm)  # Format (100,5,1)

		# === 5. Feature Extraction ===
		cnn_features = cnn_block(input_cnn, **dict_params)
		lstm_features = lstm_block(input_lstm, **dict_params)

		# === 6. Projection & Transformer ===
		merged_features = tf.keras.layers.Concatenate()([cnn_features, lstm_features])
		merged_features = Dense(num_units_concat, activation="relu")(merged_features)
		merged_features = Reshape((1, merged_features.shape[-1]))(merged_features)  # Add time dimension for Transformer
		transformed_features = transformer_block(merged_features, **dict_params)

		# === 7. Outputs ===
		#num_clusters_output = Dense(timesteps + 1, activation="softmax", name="num_clusters")(transformed_features[:, 0, :])  # Classification
		min_output = Dense(num_units_output, activation='sigmoid')(transformed_features[:, 0, :])
		range_output = Dense(num_units_output, activation="softplus")(transformed_features[:, 0, :])
		#bounds_output = Dense(2 * timesteps, activation="sigmoid")(bounds_output)  # Regression
		#bounds_output = Reshape((timesteps, 2), name="bounds")(bounds_output)
		bounds_output = Lambda(lambda x: tf.stack([x[0], x[0] + x[1]], axis=1), name="bounds")([min_output, range_output])
		#ranks_output = Dense(timesteps, activation="softmax", name="ranks")(transformed_features[:, 0, :])  # Classification
		
		# === 8. Build & Compile Model ===
		model = Model(inputs=input_lstm, 
					  outputs={#"num_clusters": num_clusters_output, 
							   "bounds": bounds_output, 
							   #"ranks": ranks_output
							   }
							   )
		
		
		model.compile(optimizer="adam", 
					  loss={#"num_clusters": "sparse_categorical_crossentropy", 
							"bounds": self.bounds_loss, 
							#"ranks": self.rank_loss
							},
					  metrics={#"num_clusters": ["mae",'accuracy'], 
							   "bounds": self.bounds_metric, 
							   #"ranks": ["mae",'accuracy']
							   }, 
					  #loss_weights={'num_clusters': 0.5, 'bounds': 1.5, 'ranks': 0.5}
					  )
					  
		return model

	def bounds_loss(self, y_true, y_pred):
		"""
		number of clusters output loss function of the model.
		"""
		mask = tf.greater(y_true, 0)
		
		# selecting only real clusters
		y_true_filtered = tf.boolean_mask(y_true, mask)
		y_pred_filtered = tf.boolean_mask(y_pred, mask)
		
		condition_1 = tf.greater(y_pred_filtered[0], y_true_filtered[0])  # min pred > min true
		condition_2 = tf.less(y_pred_filtered[1], y_true_filtered[1])  # max pred < max true
		
		condition_3 = tf.greater(y_pred_filtered[1], y_true_filtered[0]) # max pred > min true
		condition_4 = tf.less(y_pred_filtered[0], y_true_filtered[1]) # min pred < max true
		
		combined_condition_1 = tf.logical_and(condition_1, condition_4)
		combined_condition_2 = tf.logical_and(condition_2, condition_3)
	
		# cond 1
		exp_loss_1 = tf.exp(tf.abs(y_true_filtered[0] - y_pred_filtered[0]) + 1)  # exp loss
		mae_loss_1 = tf.abs(y_true_filtered[0] - y_pred_filtered[0])         # MAE
		
		# cond 2
		exp_loss_2 = tf.exp(tf.abs(y_true_filtered[1] - y_pred_filtered[1]) + 1)  # exp loss
		mae_loss_2 = tf.abs(y_true_filtered[1] - y_pred_filtered[1])         # MAE
		
		# applying loss according to condition
		diff_1 = tf.reduce_mean(tf.where(condition_1, mae_loss_1, exp_loss_1))
		diff_2 = tf.reduce_mean(tf.where(condition_2, mae_loss_2, exp_loss_2))
		
		return diff_1 + diff_2
		
	def bounds_metric(self, y_true, y_pred):
		"""
		number of clusters output metric function of the model.
		"""
		mask = tf.greater(y_true, 0)
		
		# selecting only real clusters
		y_true_filtered = tf.boolean_mask(y_true, mask)
		y_pred_filtered = tf.boolean_mask(y_pred, mask)
		
		return tf.abs(y_true - y_pred) 
		
		
	def rank_loss(self, y_true, y_pred):
		"""
		rank output loss function of the model.
		"""
		mask = tf.greater(y_true, 0.01)
		y_true_filtered = tf.boolean_mask(y_true, mask)
		y_pred_filtered = tf.boolean_mask(y_pred, mask)

		# Calculer la différence absolue entre les valeurs réelles et prédites, en évitant argsort
		rank_difference = tf.abs(tf.cast(y_true_filtered, tf.float32) - tf.cast(y_pred_filtered, tf.float32))

		# Retourner la moyenne de la perte
		return tf.reduce_mean(rank_difference)
		
		
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
		clust.get_files()
		clust.load_asset_characteristics()
		clust.array_process()