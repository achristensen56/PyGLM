import numpy as np
from scipy.stats import norm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

#make data generation code for poisson and gaussian GLM's
#add deconvolution code (fancy deconvolution)
#add train test split code 
#make the non-linearities generic for the data generation code. 


def generate_data(T = 1000, n = 30, eps = 1e-4, 
				noise_model = 'exponential', non_lin = np.exp, 
				c = 3, scale = 5, filt_amp = 10):
	'''
	currently supports noise_model = {'exponential', 'gaussian', 'poisson'}
	non_lin should be any function that applies elementwise to it's single 
	argument. 
	'''

	if noise_model == 'exponential':
		stim, weights, y = generate_gamma_data(non_lin, T = 1000, n = 30, eps = 1e-1,
											c = 3, scale = 5, filt_amp = 10)

	elif noise_model == 'gaussian':
		stim, weights, y = generate_gaussian_data(non_lin, T = 1000, n = 30, eps = 1e-1,
		 									c = 3, scale = 5, filt_amp = 10)

	elif noise_model == 'poisson':
		stim, weights, y = generate_poisson_data(non_lin, T = 1000, n = 30, eps = 1e-1,
		 									c = 3, scale = 5, filt_amp = 10)
	else:
		print "Noise model not recognized. Aborting..." 
		return None

	return stim, weights, y



def sigmoid(x):
	return 1.0/(1.0 + np.exp(-x)) + 1e-4


def cond_int(non_lin, weights, stim, scale, c):
	'''
	returns the conditional intensity \lambda = f(w^TX)
	'''
	h = stim.dot(weights)
	f = scale*non_lin(h-c)
	return f

def gamma_model(cond_int, p = 1):
	'''
	draws from a gamma distribution with shape parameter p. 
	and mean 'cond_int'
	'''
	y = np.random.gamma(p, cond_int)
	return y


def poisson_model(cond_int):
	y = np.random.poisson(cond_int)
	return y

def gaussian_model(cond_int):
	y = np.random.normal(cond_int)
	return y

def generate_gamma_data(non_lin, T = 1000, n = 30, eps = 1e-1, c = 3, scale = 5, filt_amp = 10):
	stim = np.random.normal(0, scale = 2, size = [T, n])
	weights = filt_amp*norm.pdf(range(0, n), loc = n/2, scale = n/10)
	y = gamma_model(cond_int(non_lin, weights, stim, scale, c))
	return stim, weights, y

def generate_poisson_data(non_lin, T = 1000, n = 30, eps = 1e-1, c = 3, scale = 5, filt_amp = 10):
	'''
	poisson data with any non-linearity
	'''
	stim = np.random.normal(0, scale = 2, size = [T, n])
	weights = filt_amp*norm.pdf(range(0, n), loc = n/2, scale = n/10)
	y = poisson_model(cond_int(non_lin, weights, stim, scale, c))
	return stim, weights, y

def generate_gaussian_data(non_lin, T = 1000, n = 30, eps = 1e-1, c = 3, scale = 5, filt_amp = 10):
	'''
	sigmoidal non-linearity
	'''
	stim = np.random.normal(0, scale = 2, size = [T, n])
	weights = filt_amp*norm.pdf(range(0, n), loc = n/2, scale = n/10)
	y = gaussian_model(cond_int(non_lin, weights, stim, scale, c))
	return stim, weights, y

def gridplot(num_rows, num_cols):
	'''get axis and gridspec objects for grid plotting 
	returns gs, ax
	'''
	gs = gridspec.GridSpec(num_rows, num_cols, wspace=0.0)
	ax = [plt.subplot(gs[i]) for i in range(num_rows*num_cols)]

	return gs, ax

def simpleaxis(ax):
	'''
	remove the top and right spines and ticks from the axis. 
	'''
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.get_xaxis().tick_bottom()
	ax.get_yaxis().tick_left()



