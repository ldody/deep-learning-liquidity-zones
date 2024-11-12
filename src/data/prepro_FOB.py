#packages
import os, sys
import zipfile
import pandas as pd
import numpy as np
import argparse
from fastparquet import write

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from FOBDBM import FOBDataBaseManagement as fobdm

class FOBPreprocessor:
	"""
	Preprocessing FOB to create LOB.

	Args:
		job_id (int, optionnal): Slurm job ID, Default=0.
	"""
	def __init__(self, job_id: int = 0):
		"""
		Initializes the FOBPreprocessor instance.
		
		Attributes:
			path (str): Path of the current script.
			root_path (str): Root path of the project.
			job_id (int): Slurm job ID.
			raw_path (str): Path of the repository with raw data of FOB /data/raw/FOB/.
			processed_path (str): Path of the repository with processed data of FOB /data/processed/FOB/.
			fobdm (Class): Class from the FOB database management.
			FOB (DataFrame): FOB DataFrame.
			LOB (DataFrame): LOB DataFrame.
			filename_tmp (str): Name of the temporary parquet file with LOB dataframe.
			filename_zip (str): Name of the gzip file with the final parquet file with LOB dataframe.
			file (str): Name of the FOB file in process.
			isin (str): Name of the ISIN in process.
			
		Args:
			job_id (int, optionnal): Slurm job ID, Default=0.
		"""
		self.path = os.path.dirname(os.path.abspath(__file__))
		self.root_path = self.path
		while os.path.basename(self.root_path) != 'PhD_article_2':
			self.root_path =  os.path.dirname(self.root_path)
		self.job_id = job_id
		self.raw_path = os.path.join(self.root_path,'data','raw','FOB')
		self.processed_path = os.path.join(self.root_path,'data','processed','FOB')
		self.fobdm = fobdm(self.job_id)
		self.FOB = None
		self.LOB = None
		self.filename_tmp = None
		self.filename = None
		self.filename_zip = None
		self.file = ''
		self.isin = ''
		
	def load_FOB(self):
		"""
		Load FOB file into a DataFrame.
		
		Attributes:
			raw_path (str): Path of the repository with raw data of FOB /data/raw/FOB/.
			file (str): Name of the FOB file in process.
			FOB (DataFrame): Store the updated FOB DataFrame.
			isin (str): Name of the ISIN in process.
			
		Args:
			None: This method does not require args.

		Returns:
			None: This method does not return anything.
		
		Raises:
			None: This method does not raise error.
		"""
		chunks = []
		for chunk in pd.read_csv(os.path.join(self.raw_path, os.path.splitext(self.file)[0]), 
								header=0, 
								low_memory=False, 
								chunksize=10000, 
								usecols=['isin',
										'event_date',
										'event_time_cet',
										'order_id',
										'order_event_type',
										'order_side',
										'order_price',
										'order_size',
										'order_type',
										'time_in_force', 
										'trade_size', 
										'trade_price']):
			chunk = chunk[(chunk['isin'] == self.isin)]

			chunk['event_time_cet'] = pd.to_datetime(chunk['event_date'] + ' ' + chunk['event_time_cet'])
			chunk = chunk.drop(columns=['event_date'])
			chunks.append(chunk)
			
		self.FOB = pd.concat(chunks)
		self.FOB[['previous_size', 'previous_price']] = self.FOB[['trade_size', 'order_price']]
	
	def shift_orders(self):
		"""
		Shift previous order price and size for the same order ID when 'Modify' or 'Cancel' event type.
		
		Attributes:
			FOB (DataFrame): Store the updated FOB DataFrame.
		
		Args:
			None: This method does not require args.

		Returns:
			None: This method does not return anything.
		
		Raises:
			None: This method does not raise error.
		"""
		ls_id = self.FOB[(self.FOB['order_event_type'] == 'Cancel') | (self.FOB['order_event_type'] == 'Modify')]['order_id'].unique().tolist()

		mask = self.FOB['order_id'].isin(ls_id)
		self.FOB.loc[mask, 'previous_price'] = self.FOB.loc[mask].groupby('order_id')['order_price'].shift(1)
		self.FOB.loc[mask, 'previous_size'] = self.FOB.loc[mask].groupby('order_id')['order_size'].shift(1)
		
		self.FOB.loc[self.FOB['order_event_type'] == 'Fill', 'previous_size'] = self.FOB.loc[self.FOB['order_event_type'] == 'Fill', 'trade_size']
	
	def construct_LOB(self):
		"""
		Contruct LOB dataframe.
		
		Attributes:
			FOB (DataFrame): FOB DataFrame.
			LOB (DataFrame): LOB DataFrame.
			filename_tmp (str): Name of the temporary csv file with LOB dataframe.
			processed_path (str): Path of the repository with processed data of FOB /data/processed/FOB/.
		
		Args:
			None: This method does not require args.

		Returns:
			None: This method does not return anything.
		
		Raises:
			None: This method does not raise error.
		"""
		time = self.FOB['event_time_cet'].sort_values().unique().tolist()
		lentime = len(time)

		self.LOB = pd.DataFrame(columns=['price', 'size', 'side'])
		
		if self.filename_tmp in os.listdir(self.processed_path):
			self.LOB = pd.read_parquet(os.path.join(self.processed_path, self.filename_tmp))
			self.LOB.index = pd.to_datetime(self.LOB.index)
			ls_t = pd.to_datetime(self.LOB.index.unique().tolist())
			
			for t in reversed(ls_t):
				if t in time:
					last_t = t
					break
					
			time = [x for x in time if x > last_t]
			
			if not time:
				print(f'{self.isin} already processed in {self.file}')
				return 0
				
			self.LOB = self.LOB.loc[last_t]
	
		for t, block in self.FOB.groupby('event_time_cet'):
			if t in ls_t:
				continue
			
			if time.index(pd.to_datetime(t)) % 500 == 0:
				print(f'{time.index(pd.to_datetime(t))} / {lentime}')
			
			self.LOB = self.LOB.reset_index(drop=True)
			ls = [self.LOB]
			
			# Reload and new order + Modify (add volume)
			tmp_add = block[['order_price', 'order_size', 'order_side']]
			
			if len(tmp_add) != 0:
				tmp_add.columns = ['price', 'size', 'side']
				tmp_add = tmp_add.groupby(['price', 'side'], as_index=False).sum()
				ls.append(tmp_add)
			
			# Cancel, fill and modify order (subtract volume)
			tmp_sub = block[['previous_price', 'previous_size', 'order_side']]
			
			if len(tmp_sub) != 0:
				tmp_sub.columns = ['price', 'size', 'side']
				tmp_sub = tmp_sub.groupby(['price', 'side'], as_index=False).sum()
				tmp_sub['size'] = tmp_sub['size'] * -1
				ls.append(tmp_sub)
			
			self.LOB = pd.concat(ls).groupby(['price', 'side'], as_index=False).sum()
			self.LOB = self.LOB[self.LOB['size'] != 0]
			self.LOB.index = pd.Index([t] * len(self.LOB))
			
			self.save_LOB(state='tmp')
			
			if (self.LOB['size'] < 0).any():
				sys.exit('Error: Negative values in LOB')
		
		self.save_LOB(state='def')
		
	def save_LOB(self, state: str = 'tmp'):
		"""
		Save LOB as tmp or final csv file.
		
		Attributes:
			LOB (DataFrame): LOB DataFrame.
			filename_tmp (str): Name of the temporary csv file with LOB dataframe.
			processed_path (str): Path of the repository with processed data of FOB /data/processed/FOB/ where the LOB is saved.
			
		Args:
			state (str, optionnal): tmp or final version of the LOB csv file, Default='tmp'.

		Returns:
			None: This method does not return anything.
		
		Raises:
			None: This method does not raise error.
		"""
		if state == 'tmp':
			if self.filename_tmp not in os.listdir(self.processed_path):
				write(os.path.join(self.processed_path, self.filename_tmp), self.LOB, compression='GZIP', append=False)
			else:
				write(os.path.join(self.processed_path, self.filename_tmp), self.LOB, compression='GZIP', append=True)
				
		if state == 'def':			
			os.rename(os.path.join(self.processed_path, self.filename_tmp), os.path.join(self.processed_path, self.filename_zip))
	
	def array_process(self):
		"""
		Lauch FOB preprocessing from slurm array jobs.
		
		Attributes:
			file (str): Name of the FOB file in process.
			isin (str): Name of the ISIN in process.
			filename_tmp (str): Name of the temporary csv file with LOB dataframe.
			filename_zip (str): Name of the zip file with the final csv file with LOB dataframe.
			processed_path (str): Path of the repository with processed data of FOB /data/processed/FOB/.
		
		Args:
			None: This method does not require args.

		Returns:
			None: This method does not return anything.
		
		Raises:
			None: This method does not raise error.
		"""
		self.file, self.isin = self.fobdm.main()
		date = os.path.splitext(os.path.splitext(self.file)[0])[0].split('_')[-1]
		self.filename_tmp = f'{self.isin}_{date}_tmp.parquet.gzip'
		self.filename_zip = f'{self.isin}_{date}.parquet.gzip'
		
		if self.filename_zip in os.listdir(self.processed_path):
			pass
			
		else:   
			self.load_FOB()
			self.shift_orders()
			self.construct_LOB()
		
		self.fobdm.terminate()
	
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
	parser.add_argument('--convert', '-c', type=str2bool, default=False)
	args = parser.parse_args()
	
	fobp = FOBPreprocessor(args.job_id)    
	
	if args.convert:
		fobp.csv_to_parquet()
		sys.exit('End of conversion')


	if args.slurm_array:
		fobp.array_process()