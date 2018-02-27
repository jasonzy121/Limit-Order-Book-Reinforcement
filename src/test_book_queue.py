import numpy as np
import pandas as pd

from limit_order_book import Limit_Order_book
from message_queue import Message_Queue

message_path = '../datasets/LOBSTER_SampleFile_GOOG_2012-06-21_10/GOOG_2012-06-21_34200000_57600000_message_10.csv'
mq = Message_Queue(message_path)

book_path = '../datasets/LOBSTER_SampleFile_GOOG_2012-06-21_10/GOOG_2012-06-21_34200000_57600000_orderbook_10.csv'
df_book = pd.read_csv(book_path, header=None)
ask_book = df_book[np.arange(10)*4].values
ask_size_book = df_book[1+np.arange(10)*4].values
bid_book = df_book[2+np.arange(10)*4].values
bid_size_book = df_book[3+np.arange(10)*4].values
book = np.concatenate([tmp[:,:,np.newaxis] for tmp in [bid_book, bid_size_book, ask_book, ask_size_book]], axis=2)

for idx, message in mq.iterate_queue():
	if idx == 0:
		lob = Limit_Order_book(bid_book[0], bid_size_book[0], ask_book[0], ask_size_book[0], depth=10)
		snap_shot_book = book[0]
		assert(np.sum(snap_shot_book != lob.display_book())==0)
		print('Initialize LOB [SUCCESS]')
	else:
		lob.process(**message)
		snap_shot_book = book[1]
		print(lob.display_book())
		print(snap_shot_book)
		break