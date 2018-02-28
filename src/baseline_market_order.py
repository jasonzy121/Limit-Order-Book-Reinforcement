import argparse

from limit_order_book import Limit_Order_book
from message_queue import Message_Queue

parser = argparse.ArgumentParser(description='Dynamic Programming Solution')
parser.add_argument('--file_msg', help='Message File Path')
parser.add_argument('--base_size', default=1, help='Base Order Size', type=int)
parser.add_argument('--order_size', default=12, help='Order Size', type=int)
parser.add_argument('--order_direction', default=1, help='Buy 1, Sell -1', type=int)
parser.add_argument('--start', default=34200, help='Start Time', type=float)
parser.add_argument('--end', default=34500, help='End Time', type=float)
parser.add_argument('--adj_freq', default=100, help='Adjustment Frequency', type=float)
parser.add_argument('--tol', default=1e-8, help='Remaining Time To Submit Market Order', type=float)
args = parser.parse_args()

mq = Message_Queue(args.file_msg)
lob = Limit_Order_book(own_amount_to_trade=args.order_size,
					own_init_price=-args.order_direction*Limit_Order_book._DUMMY_VARIABLE,
					own_trade_type=args.order_direction)
for idx, message in mq.pop_to_next_time(args.start):
	lob.process(**message)

lob.update_own_order(args.order_direction*Limit_Order_book._DUMMY_VARIABLE)

current_time = args.start
while lob.own_amount_to_trade > 0 and not mq.finished():
	current_time += args.adj_freq
	for idx, message in mq.pop_to_next_time(current_time):
		lob.process(**message)
		if lob.own_amount_to_trade == 0:
			break

print(lob.display_book(16))
print(lob.own_reward)