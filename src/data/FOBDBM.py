#packages
import os, sys
import zipfile
import pandas as pd
from filelock import FileLock

class FOBDataBaseManagement():
	"""
	FOB database management to allocate and follow the different preprocessing task.

	Args:
		job_id (int, optionnal): Slurm job ID, Default=0.
	"""
	def __init__(self, job_id: int = 0):
		"""
		Initializes the FOBDataBaseManagement instance.
		
		Attributes:
			path (str): Path of the current script.
			root_path (str): Root path of the project.
			job_id (int): Slurm job ID.
			raw_path (str): Path of the repository with raw data of FOB /data/raw/FOB/.
			zip_files (list): List of the zip files containing FOB.
			DB_file (str): Name of the csv file with the database.
			DB (DataFrame): Database DataFrame.
			file_toprocess (str): Name of the FOB file to process.
			isin_toprocess (str): Name of the ISIN to process.
			
		Args:
			job_id (int, optionnal): Slurm job ID, Default=0.
		"""
		self.path = os.path.dirname(os.path.abspath(__file__))
		self.root_path = self.path
		while os.path.basename(self.root_path) != 'PhD_article_2':
			self.root_path =  os.path.dirname(self.root_path)
		self.raw_path = os.path.join(self.root_path,'data','raw','FOB')
		self.zip_files = self._get_zipfiles()
		self.DB_file = 'FOB_DB.csv'
		self.DB = None
		self.job_id = job_id
		self.file_toprocess = None
		self.isin_toprocess = None
		
	def _empty_DB_template(self):
		"""
		Generate an empty dataframe template of the database.
		
		Attributes:
			None: This method does not require attributes.
			
		Args:
			None: This method does not require args.

		Returns:
			(DataFrame): Empty dataframe for the database with columns ['file', 'isin', 'state', 'allocate'].
		
		Raises:
			None: This method does not raise error.
		"""
		return pd.DataFrame(columns=['file', 'isin', 'state', 'allocate'])
	
	def _get_zipfiles(self):
		"""
		Retrieves the zip files with FOB.
		
		Attributes:
			None: This method does not require attributes.
			
		Args:
			None: This method does not require args.

		Returns:
			(list): List of zip files with FOB.
		
		Raises:
			None: This method does not raise error.
		"""
		return [f for f in os.listdir(self.raw_path) if os.path.splitext(f)[1] == '.zip']
		
	def extract_zip(self, f):
		"""
		Extract the csv file within a zip file.
		
		Attributes:
			None: This method does not require attributes.
			
		Args:
			f (str): Zip file to be extrated.            

		Returns:
			None: This method does not return anything.
		
		Raises:
			None: This method does not raise error.
		"""
		with zipfile.ZipFile(os.path.join(self.raw_path, s), 'r') as zip_ref:
			zip_ref.extractall(self.raw_path)
		
		
	def fill_empty_DB(self, DB_tmp, ls):
		"""
		Fill an empty database dataframe with default information.
		
		Attributes:
			None: This method does not require attributes.
			
		Args:
			DB_tmp (DataFrame): Empty dataframe of the database.

		Returns:
			DB_tmp (DataFrame): Temporary dataframe filled with default information.
		
		Raises:
			None: This method does not raise error.
		"""
		DB_tmp['isin'] = ls
		DB_tmp[['file','state','allocate']] = zip_files[0],'Pending',None
		
		return DB_tmp
		
	def fill_DB(self):
		"""
		Fill the database dataframe with default information for the FOB files added in.
		
		Attributes:
			DB (DataFrame): Updated database DataFrame. 
			
		Args:
			None: This method does not require args.

		Returns:
			None: This method does not return anything.
		
		Raises:
			IndexError: If no more FOB files have to be process.
		"""
		try:
			add_file = [f for f in self.zip_files if f not in self.DB['file'].unique().tolist()][0]
			
			if os.path.splitext(add_file)[0] not in os.listdir(self.raw_path):
				self.extract_zip(add_file)
				
			ls = pd.read_csv(os.path.join(self.raw_path, add_file), usecols=['isin'])['isin'].unique().tolist()
			DB_tmp = self.fill_empty_DB(self._empty_DB_template(), ls)
			self.DB = pd.concat([self.DB, DB_tmp], ignore_index=True)
			
		except Exception as e:
			print(f'No more FOB file to process: {e}')
			
		finally:
		   print('All remaining FOB files in process or processed') 
		
	def fill_state_allocate(self, cond, st: str, alloc):
		"""
		Fill the database dataframe with processing information.
		
		Attributes:
			DB (DataFrame): Updated database DataFrame. 
			
		Args:
			None: This method does not require args.

		Returns:
			None: This method does not return anything.
		
		Raises:
			None: This method does not raise error.
		"""
		self.DF.loc[self.DF[cond].index, ['state','allocate']] = st, alloc
		
	def terminate(self):
		"""
		Fill the database dataframe with processing information for termination.
		
		Attributes:
			DB (DataFrame): Updated database DataFrame. 
			
		Args:
			None: This method does not require args.

		Returns:
			None: This method does not return anything.
		
		Raises:
			None: This method does not raise error.
		"""
		self.DF.loc[self.DF[cond].index, ['state','allocate']] = 'Terminated', None
	
	def main(self):
		"""
		Launch the management of the dataframe and the allocation of files to process.
		
		Attributes:
			DB (DataFrame): Updated database DataFrame. 
			
		Args:
			None: This method does not require args.

		Returns:
			file_toprocess (str): Name of the FOB file to process.
			isin_toprocess (str): Name of the ISIN to process.
		
		Raises:
			None: This method does not raise error.
		"""
		if self.DB_file not in os.listdir(self.path):
			print(self._empty_DB_template())
			self._empty_DB_template().to_csv(os.path.join(self.path, self.DB_file), header=True)
			print('df =', pd.read_csv(os.path.join(self.path, self.DB_file)))
			sys.exit()
			
		with FileLock(os.path.join(self.path, self.DB_file)):
			self.DB = pd.read_csv(os.path.join(self.path, self.DB_file), index_col=0)
			
			cond = (self.DB['state'] != 'Processed') & (self.DB['allocate'] != None)
			if df.empty or self.DB[cond].empty:
				self.fill_DB()
				
			self.file_toprocess, self.isin_toprocess = self.DB[cond].reset_index(drop=True).loc[0, ['file', 'isin']].tolist()
			
			fill_cond = (self.DB['file'] == self.file_toprocess) & (self.DB['isin'] == self.isin_toprocess)
			self.fill_state_allocate(fill_cond, 'In progress', self.job_id)
			return self.file_toprocess, self.isin_toprocess