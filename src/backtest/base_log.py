# log method execution
def log_execution(func):
	"""
	Printing log.
	"""
	def wrapper(self, *args, **kwargs):
		
		print(f"Execution {func.__name__} : {func.__doc__.splitlines()[1]}")

		result = func(self, *args, **kwargs)

		print(f"{func.__name__} : {func.__doc__.splitlines()[1].split('.')[0]} Done.")

		return result
	return wrapper

# Applying log to all method in class based on Base
class Base:
	"""
	Base class to apply log on method.
	"""
	def __init_subclass__(cls):
		"""
		Initializing the subclass with log printing.
		"""
		for attr_name in dir(cls):
			attr = getattr(cls, attr_name)
			if callable(attr) and not attr_name.startswith('__'):
				setattr(cls, attr_name, log_execution(attr))
		super().__init_subclass__()