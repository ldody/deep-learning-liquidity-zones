#packages
import os, sys
import zipfile
import pandas as pd
import numpy as np
import argparse
from fastparquet import write
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Bidirectional, LayerNormalization, GRU, BatchNormalization
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
		
		def force_loc_9_2_to_one(x):
			# x: (batch_size, 10, 2)
			# Créer un tensor identique à x
			x_new = tf.identity(x)
			
			# Extraire x[:,9,2]
			values = x_new[:,9,1]
			
			# Condition: si > 1
			condition = values > 1.0
			
			# Remplacer par 1 si condition vraie, sinon garder la valeur originale
			updated_values = tf.where(condition, tf.ones_like(values), values)
			
			# Mettre à jour x_new[:,9,2] = 1
			# Pour cela on utilise tensor_scatter_nd_update
			indices = tf.stack([tf.range(tf.shape(x)[0]), 
								tf.fill([tf.shape(x)[0]], timesteps-1), 
								tf.fill([tf.shape(x)[0]], 1)], axis=1)
			
			x_new = tf.tensor_scatter_nd_update(x_new, indices, updated_values)
			return x_new
		
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

		# === 3. GRU Block ===            
		def gru_block(inputs, num_units_GRU=64, num_units_output_bloc=128, dropout_rate=0.2, **kwargs):
			"""GRU block with 2 layers, Dropout & BatchNorm"""
			x = Bidirectional(GRU(num_units_GRU, return_sequences=True))(inputs)
			x = Dropout(dropout_rate)(x)  # Régularisation
			x = Bidirectional(GRU(num_units_GRU // 2))(x)  # Réduction progressive
			x = BatchNormalization()(x)  # Stabilisation
			return Dense(num_units_output_bloc, activation='relu')(x)

		# === 4. Model Input ===
		input_gru = Input(shape=input_shape, name='OHLCV')
		print(input_gru)
		input_cnn = Reshape((input_shape[0], input_shape[1], 1))(input_gru)  # Format (100,5,1)

		# === 5. Feature Extraction ===
		cnn_features = cnn_block(input_cnn, **dict_params)
		gru_features = gru_block(input_gru, **dict_params)
		print(cnn_features)
		print(gru_features)

		# === 6. Projection & Transformer ===
		merged_features = tf.keras.layers.Concatenate()([cnn_features, gru_features])
		merged_features = Dense(num_units_concat, activation="relu")(merged_features)
		merged_features = Reshape((1, merged_features.shape[-1]))(merged_features)  # Add time dimension for Transformer
		transformed_features = transformer_block(merged_features, **dict_params)

		# === 7. Outputs ===
		min_output = Dense(timesteps, activation='sigmoid')(transformed_features[:, 0, :])
		range_output = Dense(timesteps, activation="sigmoid")(transformed_features[:, 0, :])        
		max_output = tf.keras.layers.Add()([min_output, range_output])

		bounds_output = tf.keras.layers.Lambda(lambda x: tf.stack(x, axis=-1))([min_output, max_output])

		bounds_output = tf.keras.layers.Lambda(force_loc_9_2_to_one)(bounds_output)
		
		outputs = {"bounds": bounds_output}
		
		# === 8. Build & Compile Model ===
		model = Model(inputs=input_gru, 
					  outputs={"bounds": bounds_output
					  }
							   )
		
		
		model.compile(optimizer="adam", 
					  loss={"bounds": bounds_loss
					  },
					  metrics=[recall_surface_metric, precision_surface_metric, F1_score, overlap_metric], 
					  #loss_weights={'num_clusters': 0.5, 'bounds': 1.5, 'ranks': 0.5}
					  )
					  
		return model
	
@tf.function	
def bounds_loss(y_true, y_pred):
	return (recall_surface_metric(y_true, y_pred) * 2  + 
			precision_surface_metric(y_true, y_pred) * 2 + 
			F1_score(y_true, y_pred) +
			overlap_metric(y_true, y_pred)
			)

@tf.function	
def surface_intersection(p_min, p_max, t_min, t_max):
	return tf.maximum(0.0, tf.minimum(p_max, t_max) - tf.maximum(p_min, t_min))

@tf.function
def recall_surface_metric(y_true, y_pred):
	p_min = y_pred[..., 0]
	p_max = y_pred[..., 1]

	t_min = y_true[..., 0]
	t_max = y_true[..., 1]

	true_mask = tf.cast(tf.reduce_sum(y_true, axis=-1) > 0, tf.float32)

	surface_true = tf.reduce_sum((t_max - t_min) * true_mask, axis=1) + 1e-6
	inter = surface_intersection(p_min, p_max, t_min, t_max) * true_mask 
	surface_inter = tf.reduce_sum(inter, axis=1)

	recall = surface_inter / surface_true
	return 1 - tf.reduce_mean(recall)

@tf.function
def precision_surface_metric(y_true, y_pred):
	p_min = y_pred[..., 0]
	p_max = y_pred[..., 1]

	t_min = y_true[..., 0]
	t_max = y_true[..., 1]

	true_mask = tf.cast(tf.reduce_sum(y_true, axis=-1) > 0, tf.float32)
	
	surface_pred = tf.reduce_sum((p_max - p_min), axis=1) + 1e-6
	inter = surface_intersection(p_min, p_max, t_min, t_max) * true_mask
	surface_inter = tf.reduce_sum(inter, axis=1)

	precision = surface_inter / surface_pred
	return 1 - tf.reduce_mean(precision)

@tf.function
def F1_score(y_true, y_pred):
	r = 1 - recall_surface_metric(y_true, y_pred)
	p = 1 - precision_surface_metric(y_true, y_pred)

	F1 = (2 * r * p) / (r + p)
	return 1 - F1

@tf.function
def overlap_metric(y_true, y_pred):
	# --- Pénalité chevauchement des intervalles prédits ---
	pred_n0 = y_pred[:,1:,0]    # shape (batch, 9)
	pred_n1_1 = y_pred[:,:-1,1] # shape (batch, 9)
	
	# Calculer la différence
	diff = pred_n0 - pred_n1_1  # shape (batch, 9)
	
	# Si diff < 0 => violation de la contrainte
	violations = tf.nn.relu(-diff)  # max(0, -diff)
	
	# Somme des violations sur l'axe des steps et batch
	overlap_loss = tf.reduce_mean(violations)*10
	
	return overlap_loss
		
		
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