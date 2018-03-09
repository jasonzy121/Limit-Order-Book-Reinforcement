class Config:
	def __init__(self):
		self.eps_begin = 1.0
		self.eps_end = 0.1
		self.nsteps = 1000

		self.order_path = '../datasets/LOBSTER_SampleFile_GOOG_2012-06-21_10/GOOG_2012-06-21_34200000_57600000_orderbook_10.csv'
		self.message_path = '../datasets/LOBSTER_SampleFile_GOOG_2012-06-21_10/GOOG_2012-06-21_34200000_57600000_message_10.csv'
		self.depth = 3

		self.H = 10
		self.T = 3
		self.L = 2
		self.direction = -1
		self.base_point = 100
		self.start_time = 34200
		self.I = 4000

		self.state_shape = [self.depth, 4]
		self.state_history = 2