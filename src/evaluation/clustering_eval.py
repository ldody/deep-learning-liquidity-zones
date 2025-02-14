#packages
import os, sys
import zipfile
import pandas as pd
import numpy as np
from fastparquet import write

from base_log import Base, log_execution

class evaluation(Base):
	"""
	Clustering LOB and FO data.

	Args:
		job_id (int, optionnal): Slurm job ID.
	"""
	def __init__(self):
		"""
		Initializes the FOBPreprocessor instance.
		
		Attributes:
			path (str): Path of the current script.
			root_path (str): Root path of the project.
		"""
		self.path = os.path.dirname(os.path.abspath(__file__))
		self.root_path = self.path
		while os.path.basename(self.root_path) != 'PhD_article_2':
			self.root_path =  os.path.dirname(self.root_path)
		self.results_path = os.path.join(self.root_path,'results','clustering')
		self.results_path_LOB = os.path.join(self.results_path,'LOB')
		self.results_path_FO = os.path.join(self.results_path,'FO')
		self.results_evaluation = os.path.join(self.root_path,'results','clustering_evaluation')
		self.results_df = pd.DataFrame()

	def get_files(self):
		"""
		Retrieving cluster files.
		"""		
		self.files = [f for f in os.listdir(self.results_path_LOB) if 'score' not in f]
		
	def load_file(self, file):
		"""
		Loading cluster files.
		"""
		self.data = pd.read_parquet(os.path.join(self.results_path_LOB, file))
		
	def identifying_gap(self, f):
		"""
		Identifying overlaps.
		"""
		self.data['gap_min'] = 0
		self.data['gap_max'] = 0
		self.data['err'] = 0
		
		n_overlap = 0
		n_overlap_nested = 0
		n_overlap_tot = 0
		n_clusters = len(self.data)

		for _, chunk in self.data.groupby('index', as_index=False):
			chunk = chunk.sort_values('price_min', ascending=True)
			
			chunk['gap_min'] = chunk['price_max'].shift(1) - chunk['price_min']
			chunk['gap_max'] = chunk['price_max'].shift(1) - chunk['price_max']
			chunk['err'] = chunk[['gap_min','gap_max']].apply(lambda row: 1 if (row > 0).all() else (2 if (row > 0).any() else 0), axis=1)
			
			n_overlap_tot += len(chunk[chunk['err'] != 0])
			n_overlap += len(chunk[chunk['err'] == 2])
			n_overlap_nested += len(chunk[chunk['err'] == 1])
			
			if (chunk['err'] == 1).any():
				for i in range(len(chunk[chunk['err'] == 1])):
					chunk.loc[chunk['err'].shift(-1) == 1, 'err'] = 10

			self.data.loc[self.data.index, ['gap_min','gap_max','err']] = chunk.loc[chunk.index, ['gap_min','gap_max','err']]
			
		s = pd.Series([f.split('_')[0], n_overlap, n_overlap_nested, n_overlap_tot, n_clusters], index=['asset','overlap','overlap_nested','overlap_tot','clusters'])
		self.results_df = pd.concat([self.results_df, s.to_frame().T], ignore_index=True)
	
	
	def launch_evaluation(self):
		"""
		Launching evaluation process.
		"""
		for f in self.files:
			self.load_file(f)
			self.identifying_gap(f)
			write(os.path.join(self.results_evaluation, f), self.data, compression='GZIP', append=False)
		
	
	def main(self):
		"""
		executing evaluation class.
		"""
		self.get_files()
		self.launch_evaluation()
		self.results_df.to_csv(os.path.join(self.results_evaluation, 'evaluation_results.csv'))
		
		
if __name__ == "__main__":
	evaluation().main()