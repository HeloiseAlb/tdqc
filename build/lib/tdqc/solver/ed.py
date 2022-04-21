from tdqc.interfaces.solver import Solver

class EDSolver(Solver):
	"""
	Class implementing ED solver
	"""

	def __init__(self,):
		super().__init__()
		self.__check_validity__()

	def load_settings(self, settings):
		"""
		Initialize settings stored in local variable self.__settings
		"""
		if not "model" in settings:
			raise ValueError("Error loading ed-solver settings, 'model' parameter not found")
		self.__model = settings["model"]
		if not "state" in settings:
			raise ValueError("Error loading ed-solver settings, 'state' parameter not found")
		self.__state = settings["state"]
		if not "interval" in settings:
			raise ValueError("Error loading ed-solver settings, 'interval' parameter not found")
		self.__interval = settings["interval"]
		if not "steps" in settings:
			raise ValueError("Error loading ed-solver settings, 'step' parameter not found")
		self.__step = settings["steps"]

	def solve(self):
		"""
		"""