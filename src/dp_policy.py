import numpy as np

from limit_order_book import Limit_Order_book
from message_queue import Message_Queue

n_state = 2
states_len = [2, 2]

def Calculate_Q(V, H, T, I, L):
    """
    Q is indexed by states and actions, where states include time_step T 
    (need to calculate 0 to T, T+1 is left with 0s), inventory I, and 
    limit order book states. Actions has dimension L
    """
    Q = np.ones((T + 2, I, states_len[0], states_len[1], L))
    Q_counter = np.ones((T + 2, I, states_len[0], states_len[1], L))
    for t in np.arange(T, 0, -1):
        time = H*(t/T)
        """
        load_episodes will ideally load the current orderbook at time H*(t/T),
        the orderbook at next time step H*(t+1)/T, and any new messages in between
        """
        episodes = load_episodes(time)
        for episode in episodes: 
            episode_states = get_state(episode[0])
            prices = generate_prices(episode[0], L)
            for i in range(I):
                for a in range(L):
                    a_price = prices[a]
                    episode_next_state = get_state(episode[1])
                    episode_next_i, im_reward = simulate(episode, i, a_price)
                    max_Q = max(Q[t+1, episode_next_i, episode_next_state[0], episode_next_state[1], :])
                    n = Q_counter[t, i, episode_states[0], episode_states[1], a]
                    Q_counter[t, i, episode_states[0], episode_states[1], a] += 1
                    Q[t, i, episode_states[0], episode_states[1], a] = n/(n+1) * Q[t, i, episode_states[0], episode_states[1], a] + 1/(n+1)*(im_reward+max_Q)

    return Q


def Optimal_strategy(Q):
    """
    return argmax of each Q along the last axis (action)
    """
    pass

def load_episodes(time):
    pass

def generate_prices(lob, L):
    """
    generate a list of action prices based on current lob info
    """
    pass

def get_state(lob):
    """
    calculate states based on the limit order book
    """
    pass

def simulate(episode, i, a_price):
    """
    simulate to next state, we need to calculate the remaining inventory given the current i and price a, and the immediate reward
    (revenue from the executed orders)
    """
    pass






