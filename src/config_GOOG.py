class Config:
	def __init__(self):
		self.mode = 'train'

		self.nsteps_train = 1000000
		self.print_freq = 50
		self.target_update_freq = 1000
		self.saving_freq = 25000
		self.simulation_freq = 1000
		self.model_output = '../output_GOOG'

		self.eps_begin = 1.0
		self.eps_end = 0.1
		self.nsteps = 1000
		self.dropout= 0.9

		self.lr_begin = 0.00025
		self.lr_end = 0.00005
		self.lr_nsteps = self.nsteps_train / 2

		self.gamma = 0.99
		self.grad_clip = True
		self.clip_val = 10
		self.batch_size = 32

		self.order_path = '../datasets/LOBSTER_SampleFile_GOOG_2012-06-21_10/GOOG_2012-06-21_34200000_57600000_orderbook_10.csv'
		self.message_path = '../datasets/LOBSTER_SampleFile_GOOG_2012-06-21_10/GOOG_2012-06-21_34200000_57600000_message_10.csv'
		self.depth = 3

		self.H = 600
		self.T = 20
		self.L = 10
		self.direction = -1
		self.base_point = 100
		self.train_start = 34200
		self.train_end = 46800
		self.test_start= 46800
		self.test_end= 57600
		self.I = 8000
		self.hidden_size= 10

		self.state_shape = [self.depth, 4]
		self.state_history = 2
