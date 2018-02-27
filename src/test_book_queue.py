import numpy as np
import pandas as pd

from limit_order_book import Limit_Order_book
from message_queue import Message_Queue

def print_info(idx, msg=None, status='[FAIL]'):
	if idx in [1]:
		print('Execute Buy Order %s' %status)
	elif idx in [48]:
		print('Execute Sell Order %s' %status)
	elif idx in [9]:
		print('Add Buy Order %s' %status)
	elif idx in [41]:
		print('Add Sell Order %s' %status)
	elif idx in [5]:
		print('Execute Hidden Order %s' %status)
	elif idx in [46]:
		print('Delete Buy Order %s' %status)
	elif idx in [47]:
		print('Delete Sell Order %s' %status)

	if status == '[FAIL]':
		print('ERROR! idx %d msg %s' %(idx, str(msg)))


message_path = '../datasets/LOBSTER_SampleFile_GOOG_2012-06-21_10/GOOG_2012-06-21_34200000_57600000_message_10.csv'
mq = Message_Queue(message_path)

book_path = '../datasets/LOBSTER_SampleFile_GOOG_2012-06-21_10/GOOG_2012-06-21_34200000_57600000_orderbook_10.csv'
df_book = pd.read_csv(book_path, header=None)
level = 10
ask_book = df_book[np.arange(level)*4].values
ask_size_book = df_book[1+np.arange(level)*4].values
bid_book = df_book[2+np.arange(level)*4].values
bid_size_book = df_book[3+np.arange(level)*4].values
book = np.concatenate([tmp[:,:,np.newaxis] for tmp in [bid_book, bid_size_book, ask_book, ask_size_book]], axis=2)

for idx, message in mq.iterate_queue():
	if idx == 0:
		ask_book_init = ask_book[0]
		ask_size_book_init = ask_size_book[0]
		bid_book_init = bid_book[0]
		bid_size_book_init = bid_size_book[0]
		lob = Limit_Order_book(bid_book_init, bid_size_book_init, ask_book_init, ask_size_book_init)
		snap_shot_book = book[idx]
		assert(np.sum(snap_shot_book != lob.display_book(level))==0)
		print('Initialize LOB [SUCCESS]')
	else:
		lob.process(**message)
		snap_shot_book = book[idx]
		try:
			assert(np.sum(snap_shot_book != lob.display_book(level))==0)
			print_info(idx, msg=message, status='[SUCCESS]')
		except:
			print_info(idx, msg=message, status='[FAIL]')
		

		if idx == 64:
			print('\nFinished! Unable to compare due to invisible order in the book!')
			print('Current LOB: ')
			print(lob.display_book(16))
			break