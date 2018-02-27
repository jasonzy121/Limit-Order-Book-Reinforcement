import numpy as np
import pandas as pd

class Message_Queue(object):
	def __init__(self, path):
		self._df = pd.read_csv(path, header=None)
		self._time = 34200.0
		self._idx2header = ['Time', 'Type', 'OrderID', 'Size', 'Price', 'Direction']
		self._header2idx = {tmp:idx for idx, tmp in enumerate(self._idx2header)}

	def iterate_queue(self):
		for idx, row in self._df.iterrows():
			order_type = int(row[1])
			order_size = int(row[3])
			order_price = int(row[4])
			order_direction = int(row[5])
			message = {'type':order_type, 'size':order_size, 'price':order_price, 'direction':order_direction}
			self._time = row[0]
			yield (idx, message)