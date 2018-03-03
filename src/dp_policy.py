import argparse
import numpy as np
import copy

from limit_order_book import Limit_Order_book
from message_queue import Message_Queue
from order_queue import Order_Queue
from evaluate_policy import evaluate_policy

n_state = 2
states_len = [2, 3]

parser = argparse.ArgumentParser(description='Dynamic Programming Algorithm')
parser.add_argument('--tic', default= 'GOOG', help='Company Ticker')
parser.add_argument('--base_point', default=100, help='Base Point', type=int)
parser.add_argument('--order_direction', default=-1, help='Buy 1, Sell -1', type=int)
parser.add_argument('--spread_cutoff', default=10.0, help='Cutoff for low bid-ask spread/high spread', type=float)
parser.add_argument('--train_start', default=34200, help='Train Start Time', type=float)
parser.add_argument('--train_end', default=46800, help='Train End Time', type=float)
parser.add_argument('--test_start', default=46800, help='Test End Time', type=float)
parser.add_argument('--test_end', default=57600, help='Test End Time', type=float)
parser.add_argument('--H', default=600, help='Horizon', type=int)
parser.add_argument('--T', default=20, help='Time steps', type=int)
parser.add_argument('--V', default=100, help='Amount to trade', type=int)
parser.add_argument('--I', default=10, help='Inventory Length', type=int)
parser.add_argument('--L', default=10, help='Action Length', type=int)
parser.add_argument('--mode', default='train', help='Mode: train or test')
args = parser.parse_args()

file_msg = '../datasets/%s_2012-06-21_34200000_57600000_message_10.csv' % (args.tic)
file_order = '../datasets/%s_2012-06-21_34200000_57600000_orderbook_10.csv' % (args.tic)

def Calculate_Q(V, H, T, I, L, oq, mq):
	"""
	Q is indexed by states and actions, where states include time_step T 
	(need to calculate 0 to T, T+1 is left with 0s), inventory I, and 
	limit order book states. Actions has dimension L
	V is the total number of shares, I is the number of inventory units.
	H is the total time left to sell all of the inventory. One period is H/T.
	"""
	Q = np.zeros((T + 2, I, states_len[0], states_len[1], L))
	Q_counter = np.zeros((T + 2, I, states_len[0], states_len[1], L))
	for t in np.arange(T, -1, -1):
		time = H*(t/T)
		next_time = time + H/T
		"""
		load_episodes will load the current orderbook at time H*(t/T)
		and the orderbook at next time step H*(t+1)/T
		"""
		episodes, real_times = load_episodes(time, next_time, H, V, oq, mq)
		for k in range(len(episodes)):
			episode = episodes[k] 
			real_time = real_times[k]
			print(real_time)
			episode_states = get_state(episode[0])
			prices = generate_prices(episode[0], L)
			for i in range(I):
				for a in range(L):
					a_price = prices[a]
					if t == T:
						episode_next_state = get_state(episode[0])
						episode_next_i, im_reward = simulate(episode[0], int((i+1)*V/I) , a_price, real_time[0], real_time[0], mq)
					else:
						episode_next_state = get_state(episode[1])
						episode_next_i, im_reward = simulate(episode[0], int((i+1)*V/I) , a_price, real_time[0], real_time[1], mq)

					episode_next_i = int(episode_next_i/V*I)-1 # Have to change new order_size into inventory units.
					max_Q = np.amax(Q[t+1, episode_next_i, episode_next_state[0], episode_next_state[1], :])
					n = Q_counter[t, i, episode_states[0], episode_states[1], a]
					Q_counter[t, i, episode_states[0], episode_states[1], a] += 1
					Q[t, i, episode_states[0], episode_states[1], a] = n/(n+1) * Q[t, i, episode_states[0], episode_states[1], a] + 1/(n+1)*(im_reward+max_Q)
	return Q


def Optimal_strategy(Q):
	"""
	return argmax of each Q along the last axis (action)
	"""
	return np.argmax(Q, axis=len(Q.shape)-1)

def Optimal_action(remaining_time, amount, lob_copy):
	t = int((args.H - remaining_time) / args.H * args.T)
	i = int(amount / args.V * args.I)-1
	states = get_state(lob_copy)
	action = Optimal_Q[t,i,states[0],states[1]]
	prices = generate_prices(lob_copy, args.L)
	return prices[action]


def load_episodes(time, next_time, H, V, oq, mq):
	lob1_data, time_1 = read_order_book(time, H, oq, mq)
	lob1 = [Limit_Order_book(**lob_data, own_amount_to_trade = 0, 
					own_init_price=-args.order_direction*Limit_Order_book._DUMMY_VARIABLE,
					own_trade_type=args.order_direction) for lob_data in lob1_data]

	lob2_data, time_2 = read_order_book(next_time, H, oq, mq)
	lob2 = [Limit_Order_book(**lob_data, own_amount_to_trade = 0,
					own_init_price=-args.order_direction*Limit_Order_book._DUMMY_VARIABLE,
					own_trade_type=args.order_direction) for lob_data in lob2_data]
	return list(zip(lob1, lob2)), list(zip(time_1, time_2))


def read_order_book(time, H, oq, mq):
	"""
	read the initial limit order book states from the file
	"""
	output = []
	time_output = []
	real_time = args.train_start + time
	while real_time < args.train_end:
		mq.reset()
		output.append(oq.create_orderbook_time(real_time, mq))
		time_output.append(real_time)
		real_time= real_time + H
	return output, time_output


def generate_prices(lob, L):
	"""
	generate a list of action prices based on current lob info
	"""
	current_mid_price = lob.bid[0] + (lob.ask[0] - lob.bid[0]) // 2
	return np.arange(current_mid_price-(L//2)*args.base_point, current_mid_price+(L-L//2)*args.base_point, args.base_point)

def get_state(lob):
	"""
	calculate states based on the limit order book
	State 1: bid-ask spread
	State 2: bid-ask volume misbalance
	"""
	spread = (lob.ask[0] - lob.bid[0])/100.0
	state1 = 0 if spread < args.spread_cutoff else 1
	state2 = np.sign(lob.ask_size[0] - lob.bid_size[0])
	return [state1, state2]

def simulate(lob, amount, a_price, time, next_time, mq):
	"""
	simulate to next state, we need to calculate the remaining inventory given the current i and price a, and the immediate reward
	(revenue from the executed orders)
	"""
	mq.reset()
	mq.jump_to_time(time)

	lob_copy = copy.deepcopy(lob)
	lob_copy.update_own_order(a_price, amount)

	for idx, message in mq.pop_to_next_time(next_time):
		lob_copy.process(**message)
		if lob_copy.own_amount_to_trade == 0:
			break

	return [lob_copy.own_amount_to_trade, lob_copy.own_reward]

path_target = '../data/%s_Q_dp_%s.npy' % (args.tic,args.V)
oq = Order_Queue(file_order)
mq = Message_Queue(file_msg)

if args.mode == 'train':
	np.save(path_target, Calculate_Q(args.V, args.H, args.T, args.I, args.L,oq,mq))
elif args.mode == 'test':
	Q = np.load(path_target)
	Optimal_Q = Optimal_strategy(Q)
	rewards = evaluate_policy(args.test_start, args.test_end, args.order_direction, args.V, args.H, args.T, oq, mq, Optimal_action)
	print(rewards)
	print(np.mean(rewards))





