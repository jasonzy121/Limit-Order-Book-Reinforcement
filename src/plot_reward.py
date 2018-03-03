import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Plot Reward')
parser.add_argument('--file', default= '../reward.csv', help='File Path', type=str)
parser.add_argument('--ylabel', default= 'Raw Reward', help='Raw Reward/Relative Reward', type=str)
args = parser.parse_args()

def plot(reward, 
	ticker=['AMZN','AAPL','GOOG','INTC','MSFT'],
	algo=['Market Order','SnL','Tree Search','Nevmyvaka'],
	ylabel='Raw Reward'):
	### reward is a N * 4 array
	N_ticker, N_algo = reward.shape
	ind = np.arange(N_ticker)
	width = 1.0 / (N_algo + 1)
	fig, ax = plt.subplots()
	
	rects = dict()
	for i in range(N_algo):
		c = (i + 0.5) / N_algo
		rects[i] = ax.bar(ind+width*i, reward[:,i], width, color=(c,c,1-c))
	ax.set_ylabel(ylabel)
	ax.set_xticks(ind+width*(N_algo-1)/2)
	ax.set_xticklabels(ticker)
	ax.legend((rects[i] for i in range(N_algo)), algo)
	plt.show()

def plot_file(path, ylabel):
	df = pd.read_csv(path)
	header = df.axes[1]
	algo = list(header[1:])
	ticker = list(df[header[0]])
	reward = df[header[1:]].values
	plot(reward, ticker, algo, ylabel)

if __name__ == '__main__':
	plot_file(args.file, args.ylabel)