#packages
import os, sys
import zipfile
import pandas as pd
import numpy as np
import argparse
from fastparquet import write
import tensorflow as tf
from tensorflow.keras import layers, models
from tqdm import tqdm
#from base_log import Base, log_execution
tf.random.set_seed(42)


class CNN_model():
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
		
	def model_build(self, input_shape, latent_dim: int = 128, timesteps: int = 10):
		"""
		Building and compiling CNN 2D model.
		
		Args:
			input_shape (tuple): Dimensions de l'entrée (hauteur, largeur, canaux).

		Returns:
			model (tf.keras.Model): Compiled CNN 2D model.
		"""

		# Entrée de l'image
		inputs = layers.Input(shape=input_shape, name="image_input")

		# Extraction des caractéristiques (CNN)
		x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
		x = layers.MaxPooling2D((2, 2))(x)
		x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
		x = layers.MaxPooling2D((2, 2))(x)
		x = layers.Flatten()(x)
		x = layers.Dense(latent_dim, activation='relu')(x)
		x = layers.Dropout(0.2)(x)

		# Tête pour prédire le nombre d'intervalles
		dense_output = layers.Dense(32, activation='relu')(x)
		n_intervals = layers.Dense(1, activation='relu', name="n_intervals")(dense_output)

		# Répétition du vecteur latent pour le RNN
		latent_sequence = layers.RepeatVector(timesteps)(x)

		# RNN pour générer les bornes
		rnn_output = layers.LSTM(64, return_sequences=True)(latent_sequence)
		bounds = layers.TimeDistributed(layers.Dense(3, activation='linear'), name="interval_bounds")(rnn_output)

		# Création du modèle
		model = models.Model(inputs=inputs, outputs=[n_intervals, bounds])

		# Compilation
		model.compile(
			optimizer='adam',
			loss={"n_intervals": self.interval_loss, "interval_bounds": self.custom_loss},
			metrics={"n_intervals": self.interval_loss, "interval_bounds": [self.custom_loss, self.bounds_loss1, self.bounds_loss2, self.bounds_loss_inter]},
			#loss_weights={'n_intervals': 1.0, 'interval_bounds': 0.1}
		)

		return model
		
	def custom_loss(self, y_true, y_pred):
		"""
		Customizing loss of the model.
		"""
		bounds_true = tf.cast(y_true[...,:2], tf.float32)
		bounds_pred = tf.cast(y_pred[...,:2], tf.float32)
		
		max_intervals = tf.shape(bounds_true)[1]
		mask = tf.sequence_mask(self.n_inter, maxlen=max_intervals)
		mask = tf.cast(mask, tf.float32)[..., tf.newaxis]
		
		masked_bounds_true = bounds_true * mask
		masked_bounds_pred = bounds_pred * mask
		
		# Première contrainte : output[:, :, 1] <= output[:, :, 0]
		diff_1 = tf.maximum(0.0, (bounds_pred[:, :, 1] - bounds_pred[:, :, 0]))
		cond1 = tf.reduce_mean(diff_1) * 100
		
		# Deuxième contrainte : output[i, 0] < output[i-1, 1]
		diff_2 = tf.map_fn(lambda x: tf.maximum(0.0, (x[1:, 0] - x[:-1, 1])), bounds_pred)
		cond2 = tf.reduce_mean(diff_2) * 100
		
		
		mask_zero = tf.not_equal(masked_bounds_true, 0)
		
		y_true_masked = tf.boolean_mask(masked_bounds_true, mask_zero)
		y_pred_masked = tf.boolean_mask(masked_bounds_pred, mask_zero)
		
		# Calcul du MAPE
		total_loss = tf.reduce_mean(tf.abs(y_true_masked - y_pred_masked))
		
		return total_loss + cond1 + cond2
		
	def bounds_loss1(self, y_true, y_pred):
		"""
		bounds loss condition 1.
		"""
		bounds_true = tf.cast(y_true, tf.float32)
		bounds_pred = tf.cast(y_pred, tf.float32)
		
		# Première contrainte : output[:, :, 1] <= output[:, :, 0]
		diff_1 = tf.maximum(0.0, (bounds_pred[:, :, 1] - bounds_pred[:, :, 0]))
		cond1 = tf.reduce_mean(diff_1) * 100
		
		return cond1
		
	def bounds_loss2(self, y_true, y_pred):
		"""
		bounds loss condition 2.
		"""
		bounds_true = tf.cast(y_true, tf.float32)
		bounds_pred = tf.cast(y_pred, tf.float32)
		
		# Première contrainte : output[:, :, 1] <= output[:, :, 0]
		diff_2 = tf.map_fn(lambda x: tf.maximum(0.0, (x[1:, 0] - x[:-1, 1])), bounds_pred)
		cond2 = tf.reduce_mean(diff_2) * 100
		
		return cond2
		
	def bounds_loss_inter(self, y_true, y_pred):
		"""
		bounds loss mape without volume.
		"""
		bounds_true = tf.cast(y_true[...,:2], tf.float32)
		bounds_pred = tf.cast(y_pred[...,:2], tf.float32)
		
		max_intervals = tf.shape(bounds_true)[1]
		mask = tf.sequence_mask(self.n_inter, maxlen=max_intervals)
		mask = tf.cast(mask, tf.float32)[..., tf.newaxis]
		
		masked_bounds_true = bounds_true * mask
		masked_bounds_pred = bounds_pred * mask		
		
		mask_zero = tf.not_equal(masked_bounds_true, 0)
		
		y_true_masked = tf.boolean_mask(masked_bounds_true, mask_zero)
		y_pred_masked = tf.boolean_mask(masked_bounds_pred, mask_zero)
		
		# Calcul du MAE
		total_loss = tf.reduce_mean(tf.abs(y_true_masked - y_pred_masked))
		
		return total_loss
		
	def interval_loss(self, y_true, y_pred):
		"""
		MAE loss.
		"""
		n_pred = tf.cast(tf.round(y_pred), tf.int32)
		self.n_inter = tf.stop_gradient(n_pred)

		# Loss function for n intervals
		loss = tf.reduce_mean(tf.abs(y_true - y_pred))
		
		return loss
		
		
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