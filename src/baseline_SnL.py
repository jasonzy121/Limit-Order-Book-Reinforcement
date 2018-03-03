import argparse
import copy
import numpy as np

from limit_order_book import Limit_Order_book
from message_queue import Message_Queue

parser = argparse.ArgumentParser(description='Dynamic Programming Solution')
parser.add_argument('--tic', default= 'GOOG', help='Company Ticker')
parser.add_argument('--order_size', default=1200, help='Order Size', type=int)
parser.add_argument('--order_direction', default=1, help='Buy 1, Sell -1', type=int)
parser.add_argument('--train_start', default=34201, help='Train Start Time', type=float)
parser.add_argument('--train_end', default=46800, help='Train End Time', type=float)
parser.add_argument('--test_start', default=46800, help='Test End Time', type=float)
parser.add_argument('--test_end', default=57600, help='Test End Time', type=float)
parser.add_argument('--H', default=600, help='Horizon', type=float)
parser.add_argument('--base_point', default=1000, help='Base Point', type=int)
parser.add_argument('--adj_freq', default=100, help='Adjustment Frequency', type=float)
parser.add_argument('--tol', default=100, help='Remaining Time To Submit Market Order', type=float)
parser.add_argument('--num', default= 10, help= 'The number of base points to go', type= int)
args = parser.parse_args()
# Use the train_start and train_end to find the best num. H: the total amount of time to execute the orders.
file_msg = '../datasets/%s_2012-06-21_34200000_57600000_message_10.csv' % (args.tic)

mq = Message_Queue(file_msg)
lob = Limit_Order_book(own_amount_to_trade=args.order_size,
					own_init_price=-args.order_direction*Limit_Order_book._DUMMY_VARIABLE,
					own_trade_type=args.order_direction)
for idx, message in mq.pop_to_next_time(args.train_start):
	lob.process(**message)

current_mid_price = lob.bid[0] + (lob.ask[0] - lob.bid[0]) // 2
init_price = np.arange(current_mid_price-args.num*args.base_point, current_mid_price+args.num*args.base_point, args.base_point)
init_price = init_price[init_price > 0]

reward = np.zeros(init_price.shape)

for i in range(len(init_price)):
	real_time= args.train_start
	# print ('I am at %d now'%(i))
	lob_copy = copy.deepcopy(lob)
	lob_copy.update_own_order(init_price[i])
	mq_copy = copy.deepcopy(mq)
	num_count= 0
	while real_time+args.H<args.train_end:
		# print ('I have been here once')
		real_time= real_time+args.H
		#-args.tol: before ending, see how much we can sell.
		for idx, message in mq_copy.pop_to_next_time(real_time):
			lob_copy.process(**message)
			# print ('I have been here once')
			if lob_copy.own_amount_to_trade == 0:
				break
		current_reward= 0
		# print (current_reward)
		if lob_copy.own_amount_to_trade==0:
			current_reward= lob_copy.own_reward
		else:
			lob_copy.update_own_order(args.order_direction*Limit_Order_book._DUMMY_VARIABLE)
			# for idx, message in mq_copy.pop_to_next_time(real_time):
			# 	lob_copy.process(**message)
			# 	if lob_copy.own_amount_to_trade == 0:
			# 		break
			if lob_copy.own_amount_to_trade > 0 and args.order_direction==1:
				current_reward = Limit_Order_book._DUMMY_VARIABLE * (-1)
				print ('I have ever been here')
			else:
				current_reward = lob_copy.own_reward
		# print (current_reward)
		reward[i]= num_count/(num_count+1)*reward[i]+ 1/(num_count+1)*current_reward
		num_count= num_count+1
if args.order_direction==1:
	best_index= np.argmax(reward)
else:
	best_index= np.argmax(reward)
print (reward)
# Now go to evaluate the test mode
mq = Message_Queue(file_msg)
lob = Limit_Order_book(own_amount_to_trade=args.order_size,
					own_init_price=-args.order_direction*Limit_Order_book._DUMMY_VARIABLE,
					own_trade_type=args.order_direction)
for idx, message in mq.pop_to_next_time(args.test_start):
	lob.process(**message)

current_mid_price = lob.bid[0] + (lob.ask[0] - lob.bid[0]) // 2
init_price= current_mid_price- args.num*args.base_point+ best_index*args.base_point
if init_price<0:
	print ('This does not make sense at all')

reward_test= 0
real_time= args.test_start
num_count= 0
while real_time+args.H<args.test_end:
	# print ('Hi, I have ever been in the for loop')
	real_time= real_time+ args.H
	for idx, message in mq_copy.pop_to_next_time(real_time):
		lob_copy.process(**message)
		if lob_copy.own_amount_to_trade == 0:
			break

	if lob_copy.own_amount_to_trade==0:
		current_reward= lob_copy.own_reward
	else:
		lob_copy.update_own_order(args.order_direction*Limit_Order_book._DUMMY_VARIABLE)
		# for idx, message in mq_copy.pop_to_next_time(real_time):
		# 	lob_copy.process(**message)
		# 	if lob_copy.own_amount_to_trade == 0:
		# 		break
		if lob_copy.own_amount_to_trade > 0 and args.order_direction== 1:
			current_reward = Limit_Order_book._DUMMY_VARIABLE * (-1)
			print ('Have I ever been here')
		else:
			current_reward = lob_copy.own_reward
	print (current_reward)
	reward_test= num_count/(num_count+1)*reward_test+ 1/(num_count+1)*current_reward
	num_count= num_count+1
print (reward_test)
print (best_index)
exit()




# for i in range(len(init_price)):
# 	lob_copy = copy.deepcopy(lob)
# 	lob_copy.update_own_order(init_price[i])
# 	mq_copy = copy.deepcopy(mq)

# 	for idx, message in mq_copy.pop_to_next_time(args.end-args.tol):
# 		lob_copy.process(**message)
# 		if lob_copy.own_amount_to_trade == 0:
# 			break

# 	if lob_copy.own_amount_to_trade == 0:
# 		reward[i] = lob_copy.own_reward
# 	else:
# 		lob_copy.update_own_order(args.order_direction*Limit_Order_book._DUMMY_VARIABLE)
# 		for idx, message in mq_copy.pop_to_next_time(args.end):
# 			lob_copy.process(**message)
# 			if lob_copy.own_amount_to_trade == 0:
# 				break
# 		if lob_copy.own_amount_to_trade > 0:
# 			reward[i] = Limit_Order_book._DUMMY_VARIABLE * args.order_direction
# 		else:
# 			reward[i] = lob_copy.own_reward

# print(init_price)
# print(max(reward))