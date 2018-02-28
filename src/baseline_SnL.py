import argparse
import copy
import numpy as np

from limit_order_book import Limit_Order_book
from message_queue import Message_Queue

parser = argparse.ArgumentParser(description='Dynamic Programming Solution')
parser.add_argument('--file_msg', help='Message File Path')
parser.add_argument('--base_size', default=1, help='Base Order Size', type=int)
parser.add_argument('--order_size', default=12, help='Order Size', type=int)
parser.add_argument('--order_direction', default=1, help='Buy 1, Sell -1', type=int)
parser.add_argument('--start', default=34200, help='Start Time', type=float)
parser.add_argument('--end', default=34500, help='End Time', type=float)
parser.add_argument('--base_point', default=100, help='Base Point', type=int)
parser.add_argument('--adj_freq', default=100, help='Adjustment Frequency', type=float)
parser.add_argument('--tol', default=1e-8, help='Remaining Time To Submit Market Order', type=float)
args = parser.parse_args()

mq = Message_Queue(args.file_msg)
lob = Limit_Order_book(own_amount_to_trade=args.order_size,
					own_init_price=-args.order_direction*Limit_Order_book._DUMMY_VARIABLE,
					own_trade_type=args.order_direction)
for idx, message in mq.pop_to_next_time(args.start):
	lob.process(**message)

current_mid_price = lob.bid[0] + (lob.ask[0] - lob.bid[0]) // 2
init_price = np.arange(current_mid_price-10*args.base_point, current_mid_price+10*args.base_point, args.base_point)
init_price = init_price[init_price > 0]

reward = np.zeros(init_price.shape)

for i in range(len(init_price)):
	lob_copy = copy.deepcopy(lob)
	lob_copy.update_own_order(init_price[i])
	mq_copy = copy.deepcopy(mq)

	for idx, message in mq_copy.pop_to_next_time(args.end-args.tol):
		lob_copy.process(**message)
		if lob_copy.own_amount_to_trade == 0:
			break

	if lob_copy.own_amount_to_trade == 0:
		reward[i] = lob_copy.own_reward
	else:
		lob_copy.update_own_order(args.order_direction*Limit_Order_book._DUMMY_VARIABLE)
		for idx, message in mq_copy.pop_to_next_time(args.end):
			lob_copy.process(**message)
			if lob_copy.own_amount_to_trade == 0:
				break
		reward[i] = lob_copy.own_reward

print(init_price)
print(reward)