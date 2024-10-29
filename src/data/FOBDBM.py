#packages
import os, sys
import zipfile
import pandas as pd
from filelock import FileLock

class FOBDataBaseManagement():
	
	def __init__(self, job_id: int(0)):
		
		self.path = os.path.dirname(os.path.abspath(__file__))
		self.raw_path = os.path.join(os.path.dirname(self.path),'data','raw','FOB')
		self.zip_files = self._get_zipfiles()
		self.DB_file = 'FOB_DB.csv'
		self.DB = None
		self.job_id = job_id
		self.file_toprocess = None
		self.isin_toprocess = None
		
	def _empty_DB_template(self):
		
		return pd.DataFrame(columns=['file', 'isin', 'state', 'allocate'])
	
	def _get_zipfiles(self):
		
		return [f for f in os.listdir(self.raw_path) if os.path.splitext(f)[1] == '.zip']
		
	def extract_zip(self, f):
		
		with zipfile.ZipFile(os.path.join(self.raw_path, s), 'r') as zip_ref:
			zip_ref.extractall(self.raw_path)
		
		
	def fill_empty_DB(self, DB_tmp, ls):
		
		DB_tmp['isin'] = ls
		DB_tmp[['file','state','allocate']] = zip_files[0],'Pending',None
		
		return DB_tmp
		
	def fill_DB(self):
		
		try:
			add_file = [f for f in self.zip_files if f not in self.DB['file'].unique().tolist()][0]
			
			if os.path.splitext(add_file)[0] not in os.listdir(self.raw_path):
				self.extract_zip(add_file)
				
			ls = pd.read_csv(os.path.join(self.raw_path, add_file), usecols=['isin'])['isin'].unique().tolist()
			DB_tmp = self.fill_empty_DB(self._empty_DB_template(), ls)
			self.DB = pd.concat([self.db, DB_tmp], ignore_index=True)
			
		except Exception as e:
			print(f'No more FOB file to process: {e}')
			
		finally:
		   print('All remaining FOB files in process or processed') 
		
	def fill_state_allocate(self, cond, st: str, alloc):
		
		self.DF.loc[self.DF[cond].index, ['state','allocate']] = st, alloc
		
	def terminate(self)
	
		self.DF.loc[self.DF[cond].index, ['state','allocate']] = 'Terminated', None
	
	def main(self):
		
		if self.DB_file not in os.listdir(self.path):
			self.empty_DB.to_csv(os.path.join(self.path, self.DB_file), header=True)
			
		with FileLock(os.path.join(self.path, self.DB_file)):
			self.DB = pd.read_csv(os.path.join(self.path, self.DB_file), index_col=0)
			
			cond = (self.DB['state'] != 'Processed') & (self.DB['allocate'] != None)
			if df.empty or self.DB[cond].empty:
				self.fill_DB()
				
			self.file_toprocess, self.isin_toprocess = self.DB[cond].reset_index(drop=True).loc[0, ['file', 'isin']].tolist()
			
			fill_cond = (self.DB['file'] == self.file_toprocess) & (self.DB['isin'] == self.isin_toprocess)
			self.fill_state_allocate(fill_cond, 'In progress', self.job_id)
			return self.file_toprocess, self.isin_toprocess