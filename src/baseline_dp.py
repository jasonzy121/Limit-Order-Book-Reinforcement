import argparse
import numpy as np
import copy

from limit_order_book import Limit_Order_book
from message_queue import Message_Queue
from order_queue import Order_Queue
# from evaluate_policy import load_episodes, read_order_book

parser = argparse.ArgumentParser(description='Dynamic Programming Solution')
parser.add_argument('--tic', default= 'GOOG', help='Company Ticker')
parser.add_argument('--order_direction', default=-1, help='Buy 1, Sell -1', type=int)
parser.add_argument('--test_start', default=46800, help='Test End Time', type=float)
parser.add_argument('--test_end', default=57600, help='Test End Time', type=float)
parser.add_argument('--base_point', default=100, help='Base Point', type=int)
parser.add_argument('--num', default= 3, help= 'The number of base points to go', type= int)
parser.add_argument('--H', default=600, help='Horizon', type=int)
parser.add_argument('--T', default=3, help='Time steps', type=int)
parser.add_argument('--V', default=8000, help='Amount to trade', type=int)
args = parser.parse_args()

file_msg = '../datasets/LOBSTER_SampleFile_GOOG_2012-06-21_10/%s_2012-06-21_34200000_57600000_message_10.csv' % (args.tic)
file_order = '../datasets/LOBSTER_SampleFile_GOOG_2012-06-21_10/%s_2012-06-21_34200000_57600000_orderbook_10.csv' % (args.tic)

def optimal(time,start, H, lob, mq, T, current_mid_price, V):
	# step is how much we move at each time.
	# H is the number of inventory to sell. 
	# I is how many copies of inventory.
	if time==(start+H): # This code force that (args.end-args.start) is a multiple of args.tol
		if lob.own_amount_to_trade == 0:
			return lob.own_reward
		else:
			lob.update_own_order(args.order_direction*Limit_Order_book._DUMMY_VARIABLE)
			if lob.own_amount_to_trade > 0 and lob.own_trade_type == 1:
				return -Limit_Order_book._DUMMY_VARIABLE
			else:
				return lob.own_reward
	else:
		# current_mid_price = lob.bid[0] + (lob.ask[0] - lob.bid[0]) // 2
		init_price = np.arange(current_mid_price-args.num*args.base_point, current_mid_price+args.num*args.base_point, args.base_point)
		init_price = init_price[init_price > 0]
		max_reward= -1.0*float('inf')
		for i in range(len(init_price)):
			# print ('At least this works')
			lob_copy = copy.deepcopy(lob)
			# print (int(init_price[i]))
			lob_copy.update_own_order(int(init_price[i]), V)
			mq.reset()
			mq.jump_to_time(time)
			for idx, message in mq.pop_to_next_time(time+H/T):
				lob_copy.process(**message)
				# print (lob_copy.own_reward)
			if lob_copy.own_amount_to_trade == 0:
				max_reward= max(max_reward,lob_copy.own_reward)
				return max_reward
			else:
				max_reward= max(max_reward,lob_copy.own_reward+optimal(time+H//T,start,H,lob_copy,mq,T, current_mid_price, lob_copy.own_amount_to_trade))
		# for i in range(len(init_price)):
		# 	for j in range(I):
		# 	# print ('At least this works')
		# 		lob_copy = copy.deepcopy(lob)
		# 		lob_copy.update_own_order(init_price[i], V/I*j)
		# 		mq.reset()
		# 		mq.jump_to_time(time)
		# 		remaining= I-j+int(lob_copy.own_amount_to_trade/V*I)-1
		# 		for idx, message in mq.pop_to_next_time(time+H/T):
		# 			lob_copy.process(**message)
		# 			print (lob_copy.own_reward)
		# 		if lob_copy.own_amount_to_trade == 0:
		# 			max_reward= max(max_reward,lob_copy.own_reward+optimal(time+H//T,start,H,lob_copy,mq,T,remaining,I))
		# 			return max_reward
		# 		else:
		# 			max_reward= max(max_reward,lob_copy.own_reward+optimal(time+H//T,start,H,lob_copy,mq,T,remaining,I))
		return max_reward
oq = Order_Queue(file_order)
mq = Message_Queue(file_msg)
def load_episodes(test_start, test_end, order_direction, H, oq, mq):
	lob_data, time = read_order_book(test_start, test_end, H, oq, mq)
	lob = [Limit_Order_book(**lob_data, own_amount_to_trade = 0, 
					own_init_price=-order_direction*Limit_Order_book._DUMMY_VARIABLE,
					own_trade_type=order_direction) for lob_data in lob_data]
	return lob, time


def read_order_book(test_start, test_end, H, oq, mq):
	"""
	read the initial limit order book states from the file
	"""
	output = []
	time_output = []
	real_time = test_start
	while real_time < test_end:
		mq.reset()
		output.append(oq.create_orderbook_time(real_time, mq))
		time_output.append(real_time)
		real_time = real_time + H
	return output, time_output

episodes, real_times = load_episodes(args.test_start, args.test_end, args.order_direction, args.H, oq, mq)
rewards= []
for k in range(len(episodes)):
	episode = episodes[k] 
	# episode.own_amount_to_trade= args.V
	# print (episode.own_amount_to_trade)
	current_mid_price = episode.bid[0] + (episode.ask[0] - episode.bid[0]) // 2
	real_time = real_times[k]
	rewards.append(optimal(real_time,real_time, args.H,episode, mq, args.T, current_mid_price, args.V))
print (np.mean(rewards))