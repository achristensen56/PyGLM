import numpy as np
from scipy.stats import norm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

#make data generation code for poisson and gaussian GLM's
#add deconvolution code (fancy deconvolution)
#add train test split code s

def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x)) + 1e-4


def sig_gam_model(weights, stim, scale, c): 
	'''
	full prediction (including draws) for weights, stimulus (design matrix), scale, 
	and c (offset) provided. 
	'''   
	h = stim.dot(weights)
	f = scale*sigmoid(h - c) 
	y = np.random.gamma(1, f)
	return y

def exp_gam_model(weights, stim, scale, c):
	h = stim.dot(weights)
	f = scale*np.exp(h-c)
	y = np.random.gamma(1, f)
	return y

def sig_cond_int(weights, stim, scale, c):
	h = stim.dot(weights)
	f = scale*sigmoid(h-c)
	return f

def poisson_model(weights, stim, scale, c):
	h = stim.dot(weights)
	f = scale*sigmoid(h - c) 
	y = np.random.poisson(f)
	return y

def gaussian_model(weights, stim, scale, c):
	h = stim.dot(weights)
	f = scale*sigmoid(h - c) 
	y = np.random.normal(f)
	return y

def exp_cond_int(weights, stim, scale, c):
	h = stim.dot(weights)
	f = scale*np.exp(h -c)
	return f

def generate_sig_data(T = 1000, n = 30, eps = 1e-1, c = 3, scale = 5, filt_amp = 10):
	'''simulate data, with normally distributed stimulus, and a gaussian filter. 
	Generates data from a exponential GLM, with sigmoid non-linearity
	'''
	stim = np.random.normal(0, scale = 2, size = [T, n])
	weights = filt_amp*norm.pdf(range(0, n), loc = n/2, scale = n/10)
	y = sig_gam_model(weights, stim, scale, c)
	return stim, weights, y

def generate_exp_data(T = 1000, n = 30, eps = 1e-1, c = 3, scale = 5, filt_amp = 10):
	stim = np.random.normal(0, scale = 2, size = [T, n])
	weights = filt_amp*norm.pdf(range(0, n), loc = n/2, scale = n/10)
	y = exp_gam_model(weights, stim, scale, c)
	return stim, weights, y

def generate_poisson_data(T = 1000, n = 30, eps = 1e-1, c = 3, scale = 5, filt_amp = 10):
	stim = np.random.normal(0, scale = 2, size = [T, n])
	weights = filt_amp*norm.pdf(range(0, n), loc = n/2, scale = n/10)
	y = poisson_model(weights, stim, scale, c)
	return stim, weights, y

def generate_gaussian_data(T = 1000, n = 30, eps = 1e-1, c = 3, scale = 5, filt_amp = 10):
	stim = np.random.normal(0, scale = 2, size = [T, n])
	weights = filt_amp*norm.pdf(range(0, n), loc = n/2, scale = n/10)
	y = gaussian_model(weights, stim, scale, c)
	return stim, weights, y

def gridplot(num_rows, num_cols):
	'''get axis and gridspec objects for grid plotting 
	returns gs, ax
	'''
	gs = gridspec.GridSpec(num_rows, num_cols, wspace=0.0)
	ax = [plt.subplot(gs[i]) for i in range(num_rows*num_cols)]

	return gs, ax

def sig_likelihood(X, y, scale, offset, weights):
	'''Evaluate the likelihood for an exponential noise model, and sigmoidal
	non-linearity, at scale, offset, and weights provided. 
	'''
	fx = X.dot(weights) - offset
	lam = sigmoid(fx)
	lam_ = scale*lam + 1e-3
	coef = np.log(lam_)
	distrib = y / lam_

	return np.sum(coef + distrib)

def exp_likelihood(X, y, scale, offset, weights): 
	'''Evaluate the likelihood for an exponential noise model, and exponential 
	non-linearity, at scale, offset, and weights provided. 
	'''

	fx = X.dot(weights) - offset
	lam = np.exp(fx)
	lam_ = scale*lam + 1e-3
	coef = np.log(lam_)
	distrib = y/lam_

	return np.sum(coef + distrib)

def simpleaxis(ax):
	'''
	remove the top and right spines and ticks from the axis. 
	'''
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()



