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

### test own order
assert(lob.own_earlier_orders == np.sum(lob.ask_size[:-1]))
print('Initial Own Order [SUCCESS]')

lob.update_own_order(5800100)
assert(lob.own_earlier_orders == 3237)
assert(lob.own_amount_to_trade == 100)
print('Update Price To Price On LOB [SUCCESS]')

lob.update_own_order(5802000)
assert(lob.own_earlier_orders == 3247)
assert(lob.own_amount_to_trade == 100)
print('Update Price To Price Not On LOB [SUCCESS]')

lob.update_own_order(5797000)
assert(lob.own_earlier_orders == 0)
assert(lob.own_amount_to_trade == 100)
print('Update Price To Best Ask On LOB [SUCCESS]')

lob.process(1, 10, 5796900, -1)
assert(lob.own_earlier_orders == 10)
assert(lob.own_amount_to_trade == 100)
print('Insert An Sell Order With Better Ask On LOB [SUCCESS]')

lob.process(1, 11, 5797000, 1)
assert(lob.own_earlier_orders == 0)
assert(lob.own_amount_to_trade == 99)
print('Execute An Buy Order With Same Ask [SUCCESS]')

lob.update_own_order(5791900)
assert(lob.own_earlier_orders == 0)
assert(lob.own_amount_to_trade == 24)
print('Update Price To Best Bid [SUCCESS]')

lob.process(1, 1, 5791600, 1)
lob.process(1, 1, 5791700, 1)
lob.process(1, 1, 5791800, 1)
assert(lob.own_earlier_orders == 0)
assert(lob.own_amount_to_trade == 24)
print('Insert Small Buy Orders With Better Bid On LOB [SUCCESS]')

lob.update_own_order(5791500)
assert(lob.own_earlier_orders == 0)
assert(lob.own_amount_to_trade == 21)
print('Update Price To Execute 3 Small Buy Orders [SUCCESS]')

lob.process(1, 30, 5791500, -1)
assert(lob.own_earlier_orders == 0)
assert(lob.own_amount_to_trade == 21)
print('Execute An Sell Order With Same Ask [SUCCESS]')

lob.process(3, 15, 5791500, -1)
assert(lob.own_earlier_orders == 0)
assert(lob.own_amount_to_trade == 21)
print('Cancel Half Of Previous Order [SUCCESS]')

lob.update_own_order(5797900)
assert(lob.own_earlier_orders == 225)
assert(lob.own_amount_to_trade == 21)
print('Update Price To Second Best Ask On LOB [SUCCESS]')


lob.process(1, 9, 5797900, -1)
assert(lob.own_earlier_orders == 225)
assert(lob.own_amount_to_trade == 21)
print('Insert An Sell Order With Second Best Ask On LOB [SUCCESS]')

lob.process(3, 215, 5797900, -1)
assert(lob.own_earlier_orders == 15)
assert(lob.own_amount_to_trade == 21)
print('Cancel Sell Order With Second Best Ask On LOB [SUCCESS]')

lob.process(1, 45, 5799500, 1)
assert(lob.own_earlier_orders == 0)
assert(lob.own_amount_to_trade == 0)
print('Insert Buy Orders [SUCCESS]')