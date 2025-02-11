#packages
import os, sys
import zipfile
import pandas as pd
import numpy as np
import argparse
from fastparquet import write
import hdbscan
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KernelDensity
import scipy.stats as stats
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from clustering_preprocessing import ClusteringPreprocess as cprepro
from clustering_postprocessing import ClusteringPostprocess as cpostpro

class HDBSCAN_model():
	"""
	Clustering LOB and FO data.

	Args:
		job_id (int, optionnal): Slurm job ID.
	"""
	def __init__(self, tick: float):
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
		self.tick = tick
	
	def model_build(self):
		"""
		Building and compiling HDBSCAN model.
		
		Args:
			input_shape (tuple): Dimensions de l'entrée (hauteur, largeur, canaux).

		Returns:
			model (tf.keras.Model): Compiled CNN 2D model.
		"""
		model = hdbscan.HDBSCAN(min_cluster_size=3, 
							min_samples=2, 
							gen_min_span_tree=True, 
							metric=self.custom_metric, 
							max_cluster_size=0, )
							
		return model
		
	def custom_metric(self, p, q):
		"""
		Customized distance metric.
		"""
		# Set weights
		weight_price = 1
		weight_side = 100000
		weight_size = 10
		
		# Distance components
		price_diff = 0 if abs(p[0] - q[0]) <= self.tick else abs(p[0] - q[0]) / self.tick
		side_penalty = weight_side if p[2] != q[2] else 0
		size_diff = abs(p[1] - q[1]) + 1
		
		# Total distance
		dist = np.sqrt(
			weight_price * price_diff**2 +
			side_penalty +
			weight_size * size_diff**2
		)
		
		return dist
		
	class evaluation:
		"""
		Evaluation of the clusters.
		"""
		def anova(X, labels):
			'''
			ANOVA test to ensure the data point distributions are well separated.
			'''
			data = X.copy()
			data['labels'] = labels
			data = data[data['labels'] != -1]
			
			groups = [data[data['labels'] == i]['price'] for i in data['labels'].unique()]
			try:
				return stats.f_oneway(*groups)
			except:
				return np.nan, np.nan
		
		
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