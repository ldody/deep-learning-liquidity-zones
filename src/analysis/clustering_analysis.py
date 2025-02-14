# import packages
import os, sys
import pandas as pd
import numpy as np
from fastparquet import write

class analysis():
	"""
	Launching analysis of the clustering.
	"""
	def __init__(self, path):
		self.path = path
		
	def load_data(self):
		
		DF = pd.DataFrame()
		for file in [f for f in os.listdir(self.path) if 'prepro' in f]:
			df = pd.read_parquet(os.path.join(self.path, file))
			df['asset'] = file.split('_')[0]
			DF = pd.concat([DF, df], ignore_index=True)
			
		write(os.path.join(self.path, 'results.parquet.gzip'), DF, compression='GZIP', append=False)