import argparse
import numpy as np
import copy

from limit_order_book import Limit_Order_book
from message_queue import Message_Queue

n_state = 2
states_len = [2, 3]

parser = argparse.ArgumentParser(description='Dynamic Programming Algorithm')
parser.add_argument('--file_msg', help='Message File Path')
parser.add_argument('--base_point', default=100, help='Base Point', type=int)
parser.add_argument('--order_direction', default=1, help='Buy 1, Sell -1', type=int)
parser.add_argument('--spread_cutoff', default=1.0, help='Cutoff for low bid-ask spread/high spread', type=float)

def Calculate_Q(V, H, T, I, L):
    """
    Q is indexed by states and actions, where states include time_step T 
    (need to calculate 0 to T, T+1 is left with 0s), inventory I, and 
    limit order book states. Actions has dimension L
    """
    Q = np.ones((T + 2, I, states_len[0], states_len[1], L))
    Q_counter = np.zeros((T + 2, I, states_len[0], states_len[1], L))
    for t in np.arange(T, 0, -1):
        time = H*(t/T)
        next_time = time + H/T
        """
        load_episodes will load the current orderbook at time H*(t/T)
        and the orderbook at next time step H*(t+1)/T
        """
        episodes = load_episodes(time, next_time, own_trade_type)
        for episode in episodes: 
            episode_states = get_state(episode[0])
            prices = generate_prices(episode[0], L)
            for i in range(I):
                for a in range(L):
                    a_price = prices[a]
                    episode_next_state = get_state(episode[1])
                    episode_next_i, im_reward = simulate(episode[0], i*V , a_price, time, next_time)
                    episode_next_i = episode_next_i/V

                    max_Q = np.amax(Q[t+1, episode_next_i, episode_next_state[0], episode_next_state[1], :])
                    n = Q_counter[t, i, episode_states[0], episode_states[1], a]
                    Q_counter[t, i, episode_states[0], episode_states[1], a] += 1
                    Q[t, i, episode_states[0], episode_states[1], a] = n/(n+1) * Q[t, i, episode_states[0], episode_states[1], a] + 1/(n+1)*(im_reward+max_Q)
    return Q


def Optimal_strategy(Q):
    """
    return argmax of each Q along the last axis (action)
    """
    return np.argmax(Q, axis=Q.shape[-1])


def load_episodes(time, next_time):
	lob1_data = read_order_book(time)
    lob1 = Limit_Order_book(*lob1_data, own_amount_to_trade=0,
					own_init_price=-own_trade_type*Limit_Order_book._DUMMY_VARIABLE,
					own_trade_type=own_trade_typen)

    lob2_data = read_order_book(next_time)
    lob2 = Limit_Order_book(*lob2_data, own_amount_to_trade=0,
					own_init_price=-own_trade_type*Limit_Order_book._DUMMY_VARIABLE,
					own_trade_type=own_trade_type)
    return [lob1, lob2]


def read_order_book(time):
	"""
	read the initial limit order book states from the file
	"""
	pass

def generate_prices(lob, L):
    """
    generate a list of action prices based on current lob info
    """
    current_mid_price = lob.bid[0] + (lob.ask[0] - lob.bid[0]) // 2
	return np.arange(current_mid_price-(L//2)*arg.base_point, current_mid_price+(L-L//2)*arg.base_point, arg.base_point)

def get_state(lob):
    """
    calculate states based on the limit order book
    State 1: bid-ask spread
    State 2: bid-ask volume misbalance
    """
    spread = (lob.ask[0] - lob.bid[0])/100.0
    state1 = 0 if spread < args.spread_cutoff else 1
    state2 = np.sign(lob_ask_size - lob.bid_size)
    return [state1, state2]

def simulate(lob, amount, a_price, time, next_time):
    """
    simulate to next state, we need to calculate the remaining inventory given the current i and price a, and the immediate reward
    (revenue from the executed orders)
    """
    mq = Message_Queue(args.file_msg)
    mq.pop_to_next_time(time)

    lob_copy = copy.deepcopy(lob)
	lob_copy.update_own_order(amount, a_price)
	mq_copy = copy.deepcopy(mq)

	for idx, message in mq_copy.pop_to_next_time(next_time):
		lob_copy.process(**message)
		if lob_copy.own_amount_to_trade == 0:
			break

	return [lob_copy.own_amount_to_trade, lob_copy.own_reward]






