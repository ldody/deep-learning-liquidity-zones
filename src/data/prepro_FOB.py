#packages
import os, sys
import zipfile
import pandas as pd
import numpy as np
import argparse
import FOBDataBaseManagement as fobdm

class FOBPreprocessor:
	
	def __init__(self, job_id: int(0)):
		
		self.path = os.path.dirname(os.path.abspath(__file__))
		self.job_id = job_id
		self.raw_path = os.path.join(os.path.dirname(self.path),'data','raw','FOB')
		self.processed_path = os.path.join(os.path.dirname(self.path),'data','processed','FOB')
		self.fobdm = fobdm(self.job_id)
		self.FOB = None
		self.LOB = None
		self.filename = None
		
	def load_FOB(self, f, isin)
		
		chunk = []
		for chunk in pd.read_csv(os.path.join(raw_path, os.path.splitext(f)[0]), 
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
			chunk = chunk[chunk['isin'] == isin]

			chunk['event_time_cet'] = pd.to_datetime(chunk['event_date'] + ' ' + chunk['event_time_cet'])
			chunk = chunk.drop(columns=['event_date'])
			chunks.append(chunk)
			
		self.FOB = pd.concat(chunks)
	
	def shift_orders(self):
		
		ls_id = self.FOB[(self.FOB['order_event_type'] == 'Cancel') | (self.FOB['order_event_type'] == 'Modify')]['order_id'].unique().tolist()

		mask = self.FOB['order_id'].isin(ls_id)
		self.FOB.loc[mask, 'previous_price'] = self.FOB.loc[mask].groupby('order_id')['order_price'].shift(1)
		self.FOB.loc[mask, 'previous_size'] = self.FOB.loc[mask].groupby('order_id')['order_size'].shift(1)
	
	def construct_LOB(self):
		
		time = self.FOB['event_time_cet'].sort_values().unique().tolist()[:7]

		self.LOB = pd.DataFrame(columns=['price', 'size', 'side'])
		
		if 'test.csv' in os.listdir(self.processed_path):
			data = pd.read_csv('test.csv', header=0, index_col=0)
			data.index = pd.to_datetime(data.index)
			last_t = pd.to_datetime(data[-1:].index)
			time = [x for x in time if x > last_t]
			
			if not time:
				data.to_csv('test_def.csv', index=True)
				sys.exit(0)
				
			data = data.loc[last_t]
	
	
	def array_process(self):
		file, isin = self.fobdm.main()
		self.filename = 
		self.load_FOB(file, isin)
	
	
	
	
if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--job_id', type=int, default=0)
	parser.add_argument('--slurm_array', '-sa', type=bool, default=False)
	args = parser.parse_args([])
	
	fobp = FOBPreprocessor(args.job_id)
	
		if args.slurm_array:
			fobp.array_process()