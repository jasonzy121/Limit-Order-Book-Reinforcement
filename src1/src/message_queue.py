import numpy as np
import pandas as pd

class Message_Queue(object):
	def __init__(self, path):
		self._df = pd.read_csv(path, header=None)
		self._time = 34200.0
		self._row_idx = -1
		self._message_count = self._df.shape[0]
		self._idx2header = ['Time', 'Type', 'OrderID', 'Size', 'Price', 'Direction']
		self._header2idx = {tmp:idx for idx, tmp in enumerate(self._idx2header)}

	def iterate_queue(self):
		for idx, row in self._df.iloc[(self._row_idx+1):].iterrows():
			message = self._create_message(row)
			self._row_idx += 1
			self._time = row[0]
			yield (idx, message)

	def pop_to_next_time(self, time):
		while self._row_idx + 1 < self._message_count:
			row = self._df.iloc[self._row_idx+1]
			if row[0] <= time:
				self._row_idx += 1
				message = self._create_message(row)
				yield (self._row_idx, message)
			else:
				break
		self._time = time

	def finished(self):
		return (self._row_idx+1==self._message_count)

	def _create_message(self, row):
		order_type = int(row[1])
		order_size = int(row[3])
		order_price = int(row[4])
		order_direction = int(row[5])
		message = {'type':order_type, 'size':order_size, 'price':order_price, 'direction':order_direction}
		return message