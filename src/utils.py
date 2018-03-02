import numpy as np
import matplotlib.pyplot as plt

def plot(reward, 
	ticker=['AMZN','AAPL','GOOG','INTC','MSFT'],
	algo=['Market Order','S&L','Tree Search','Nevmyvaka']):
	### reward is a N * 4 array
	N_ticker, N_algo = reward.shape
	ind = np.arange(N_ticker)
	width = 1.0 / (N_algo + 1)
	fig, ax = plt.subplots()
	
	rects = dict()
	for i in range(N_algo):
		c = (i + 0.5) / N_algo
		rects[i] = ax.bar(ind+width*i, reward[:,i], width, color=(c,c,1-c))
	ax.set_ylabel('Reward')
	ax.set_xticks(ind+width*(N_algo-1)/2)
	ax.set_xticklabels(ticker)
	ax.legend((rects[i] for i in range(N_algo)), algo)
	plt.show()