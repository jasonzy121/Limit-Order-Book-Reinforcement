import argparse
import numpy as np
import copy

from limit_order_book import Limit_Order_book
from message_queue import Message_Queue
from order_queue import Order_Queue
from evaluate_policy import evaluate_policy

parser = argparse.ArgumentParser(description='Dynamic Programming Algorithm')
parser.add_argument('--tic', default= 'GOOG', help='Company Ticker')
parser.add_argument('--base_point', default=100, help='Base Point', type=int)
parser.add_argument('--order_direction', default=-1, help='Buy 1, Sell -1', type=int)
parser.add_argument('--train_start', default=34200, help='Train Start Time', type=float)
parser.add_argument('--train_end', default=46800, help='Train End Time', type=float)
parser.add_argument('--test_start', default=46800, help='Test End Time', type=float)
parser.add_argument('--test_end', default=57600, help='Test End Time', type=float)
parser.add_argument('--H', default=600, help='Horizon', type=int)
parser.add_argument('--T', default=20, help='Time steps', type=int)
parser.add_argument('--V', default=100, help='Amount to trade', type=int)
parser.add_argument('--I', default=10, help='Inventory Length', type=int)
parser.add_argument('--mode', default='train', help='Mode: train or test')
parser.add_argument('--num', default= 10, help= 'The number of base points to go', type= int)
parser.add_argument('--diff', default= 0, help= 'The number of base points to go beyond midpoint', type= int)
args = parser.parse_args()

file_msg = '../datasets/%s_2012-06-21_34200000_57600000_message_10.csv' % (args.tic)
file_order = '../datasets/%s_2012-06-21_34200000_57600000_orderbook_10.csv' % (args.tic)

def action_wrapper(diff):
     def action(remaining_time, amount, lob_copy):
     	if remaining_time == args.H:
     		current_mid_price = lob_copy.bid[0] + (lob_copy.ask[0] - lob_copy.bid[0]) // 2
     		return max(current_mid_price + diff, 0)
     	else:
     		return lob_copy.own_price
     return action

def train(start, end, order_direction, V, k, H, oq, mq, action):	
	if time==(start+H): # This code force that (args.end-args.start) is a multiple of args.tol
		if lob.own_amount_to_trade == 0:
			return lob.own_reward
		else:
			lob.update_own_order(args.order_direction*Limit_Order_book._DUMMY_VARIABLE)
			# for idx, message in mq.pop_to_next_time(args.end):
			# 	lob.process(**message)
			# 	if lob.own_amount_to_trade == 0:
			# 		break
			return lob.own_reward
	else:
		current_mid_price = lob.bid[0] + (lob.ask[0] - lob.bid[0]) // 2
		init_price = np.arange(current_mid_price-args.num*args.base_point, current_mid_price+args.num*args.base_point, args.base_point)
		init_price = init_price[init_price > 0]
		max_reward= -99999999.0
		for i in range(len(init_price)):
			# print ('At least this works')
			lob_copy = copy.deepcopy(lob)
			lob_copy.update_own_order(init_price[i])
			mq_copy = copy.deepcopy(mq)
			for idx, message in mq_copy.pop_to_next_time(time+args.tol):
				lob_copy.process(**message)
				if lob_copy.own_amount_to_trade == 0:
					max_reward= max(max_reward, lob_copy.own_reward)
				else:
					max_reward= max(max_reward, optimal(time+args.tol, lob_copy, mq_copy))
		return max_reward
	


oq = Order_Queue(file_order)
mq = Message_Queue(file_msg)

if args.mode == 'train':
	train(args.train_start, args.train_end, args.order_direction, args.V, args.H, oq, mq)
elif args.mode == 'test':
	diff = args.diff * args.base_point
	action_func = action_wrapper(diff)
	rewards = evaluate_policy(args.test_start, args.test_end, args.order_direction, args.V, args.H, args.T, oq, mq, action_func)
	print(rewards)
	print(np.mean(rewards))