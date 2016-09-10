import numpy as np
from scipy.stats import norm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import tensorflow as tf
import allensdk.brain_observatory.stimulus_info as stim_info
from allensdk.core.brain_observatory_cache import BrainObservatoryCache
from scipy import asarray as ar,exp
from scipy import stats
from scipy.stats import spearmanr
from scipy.stats import levene
import pandas as pd
import sys
import logging
from sklearn.decomposition import PCA
logging.basicConfig()

def generate_data(T = 10000, n = 30, eps = 1e-4, 
				noise_model = 'exponential', non_lin = np.exp, 
				c = 3, scale = 5, filt_amp = 10, stim = None):
	'''
	currently supports noise_model = {'exponential', 'gaussian', 'poisson'}
	non_lin should be any function that applies elementwise to it's single 
	argument. 

	returns design, weights, observations
	'''

	if noise_model == 'exponential':
		stim, weights, y = generate_gamma_data(non_lin, T = T, n = n, eps = eps,
											c = c, scale = scale, filt_amp = filt_amp, stim = stim)

	elif noise_model == 'gaussian':
		stim, weights, y = generate_gaussian_data(non_lin, T = T, n = n, eps = eps,
		 									c = c, scale = scale, filt_amp = filt_amp, stim = stim)

	elif noise_model == 'poisson':
		stim, weights, y = generate_poisson_data(non_lin, T = T, n = n, eps = eps,
		 									c = c, scale = scale, filt_amp = filt_amp, stim = stim)
	elif noise_model == None:
		stim, weights, y = generate_gaussian_data(non_lin, T = T, n = n, eps = eps,
									c = c, scale = scale, filt_amp = filt_amp, stim = stim)
		
		y = non_lin(stim.dot(weights))

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

def generate_gamma_data(non_lin, T = 1000, n = 30, eps = 1e-1, c = 3, scale = 5, filt_amp = 10, stim = None):
	if stim == None:
		stim = np.random.normal(0, scale = 2, size = [T, n])
	weights = filt_amp*norm.pdf(range(0, n), loc = n/2, scale = n/10)
	y = gamma_model(cond_int(non_lin, weights, stim, scale, c) + eps)
	return stim, weights, y

def generate_poisson_data(non_lin, T = 1000, n = 30, eps = 1e-1, c = 3, scale = 5, filt_amp = 10, stim = None):
	'''
	poisson data with any non-linearity
	'''
	if stim == None:
		stim = np.random.normal(0, scale = 2, size = [T, n])
	weights = filt_amp*norm.pdf(range(0, n), loc = n/2, scale = n/10)
	y = poisson_model(cond_int(non_lin, weights, stim, scale, c))
	return stim, weights, y

def generate_gaussian_data(non_lin, T = 1000, n = 30, eps = 1e-1, c = 3, scale = 5, filt_amp = 10, stim = None):
	'''
	sigmoidal non-linearity
	'''
	if stim == None:
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
	ax.spines['bottom'].set_visible(False)
	ax.get_xaxis().tick_bottom()
	ax.get_xaxis().set_ticks([])
	ax.get_yaxis().tick_left()

def relu(X):
	return X*(X > 0)


def download_data(region, cre_line, stimulus = None):
	'''
	region = [reg1, reg2, ...]
	cre_line = [line1, line2, ...]
	'''
	boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
	ecs = boc.get_experiment_containers(targeted_structures=region, cre_lines=cre_line)

	ec_ids = [ ec['id'] for ec in ecs ]

	exp = boc.get_ophys_experiments(experiment_container_ids=ec_ids)

	if stimulus == None:
		exp = boc.get_ophys_experiments(experiment_container_ids=ec_ids)

	else:
		exp = boc.get_ophys_experiments(experiment_container_ids=ec_ids, stimuli = stimulus)


	exp_id_list = [ec['id'] for ec in exp]

	data_set = {exp_id:boc.get_ophys_experiment_data(exp_id) for exp_id in exp_id_list}

	return data_set 

def pca_features(images):
	model = PCA()
	scenes_r = model.fit_transform(images.reshape([len(images), -1])) 

	return scenes_r

def get_data(data_set, stimulus):
	time, dff_traces = data_set.get_dff_traces()
	images = data_set.get_stimulus_template(stimulus)
	stim_table = data_set.get_stimulus_table(stimulus)

	return dff_traces, images, stim_table

def arrange_data_glm(dff_traces, images, stim_table):
	#declare a dictionary of empty lists for each cell trace, 
	#and a list for the stimulus
	data = []
	stim_array = []

	#average each trace over the presentation of each stimulus
	for index, row in stim_table.iterrows():
	    stim_array.append(images[row.frame])
	    data.append(np.mean(dff_traces[:, row['start']:row['end'] ], axis = 1) )
	    
	stim_array = np.array(stim_array)
	#stim_array = stim_array[:, 0:10]

	data = np.array(data)

	n_timepoints, n_features = stim_array.shape
	n_timepoints, n_neurons = data.shape


	return data, stim_array



def arrange_data_rs(data_set, bin = True):
	'''
	arranges the data for running speed tuning curve purposes
	dataset = {experiment_id: boc dataset}
	'''
	ds_data = {}

	#collect the stimulus tables, and running speed for each dataset
	for ds in data_set.keys():
	    _, dff = data_set[ds].get_dff_traces()
	    cells = data_set[ds].get_cell_specimen_ids()
	    
	    data = {'cell_ids':cells, 'raw_dff':dff }
	    for stimulus in data_set[ds].list_stimuli():
	        
	        if stimulus == 'spontaneous':      
	            table = data_set[ds].get_spontaneous_activity_stimulus_table()
	        else:
	            table = data_set[ds].get_stimulus_table(stimulus)
	            
	        data[stimulus] = table

	    dxcm, dxtime = data_set[ds].get_running_speed()
	    data['running_speed'] = dxcm
	    
	    ds_data[ds] = data

	#arrange the data for each separate stimuli in a dictionary. Not averaging over
	#presentation of a given image, just concatenating all cell traces, and corresponing
	#running speed. 
	arranged_data = {}
	for ds in data_set.keys():
	    dff_data = ds_data[ds]
	    
	    data = {}
	    for stimulus in data_set[ds].list_stimuli():
	        rs = np.zeros([1])
	        dfof = np.zeros([len(dff_data['cell_ids']), 1])
	        for index, row in dff_data[stimulus].iterrows():
	            dfof = np.concatenate((dfof, dff_data['raw_dff'][:, row['start']: row['end']]), axis = 1)
	            rs = np.concatenate((rs, dff_data['running_speed'][row['start']: row['end']]), axis = 0)     

	        data[stimulus + '_rs'] = np.array(np.squeeze(rs))
	        data[stimulus + '_dff'] = np.array(np.squeeze(dfof))
	        
	    arranged_data[ds] = data  

	#groups the data into 'natural', 'spontaneous', or 'artificial'
	#TODO: subsample

	tb_data = {}
	for ds_id in arranged_data.keys():
	    
	    data  = arranged_data[ds_id]
	    
	    #binning into synthetic, natural, and stimulus
	    #_data = {'synthetic_rs': None, 'natural_rs': None, 'spontaneous_rs': None,'synthetic_dff': None, 'natural_dff': None, 'spontaneous_dff':None}
	    
	    #just binning into stimulus and spontaneous
	    _data = {'stimulus_rs': None, 'spontaneous_rs': None,'stimulus_dff': None, 'spontaneous_dff':None}
	    
	    for stimulus in data_set[ds_id].list_stimuli():
	        
	        if (stimulus == 'locally_sparse_noise') or ('gratings' in stimulus):
	            stim_key = 'stimulus'
	            #stim_key = 'synthetic'
	        elif ('natural' in stimulus):
	            stim_key = 'stimulus'
	            #stim_key = 'natural'
	        elif ('spontaneous' == stimulus):
	            stim_key = 'spontaneous'
	            
	        run_speed =  np.array(data[stimulus + '_rs'])
	        dff = np.array(data[stimulus + '_dff'])
	        
	        if _data[stim_key + '_rs'] == None:
	            _data[stim_key+ '_rs'] = run_speed
	        else:
	            _data[stim_key + '_rs'] = np.concatenate((_data[stim_key + '_rs'], run_speed), axis = 0)

	           
	        if _data[stim_key + '_dff'] == None:
	            _data[stim_key+ '_dff'] = dff
	        else:
	            _data[stim_key + '_dff'] = np.concatenate((_data[stim_key + '_dff'], dff), axis = 1)		    
	    
	    tb_data[ds_id] = _data  
	
	return tb_data

def make_tuning_curves(tb_data, data_set):
	'''
	returns a nested dictionary with key experiment id, key stimulus name, with 
	a (tuning curve, (rho, spearmansp, levensp)) tuple. Tuning curve is a dictionary with key 
	cell specimen ids, which contains a (19, 4) numpy array. The 0th column is the average response, 
	the 1st column is the standard error, the 2nd column is the average shuffled response, and the 
	3rd column is the shuffled standard error. 
	'''

	rs_results = {}
	bin_hist = np.zeros([19, 2])
	shuf_hist = np.zeros([19, 2])

	for ds in tb_data.keys():
	    stim_results = {}
	    for stimulus in data_set[ds].list_stimuli():
	        
	        if ('gratings' in stimulus) or (stimulus == 'locally_sparse_noise'):
	            stim_key = 'stimulus'
	            #stim_key = 'synthetic'
	        if ('natural' in stimulus):
	            stim_key = 'stimulus'
	            #stim_key = 'natural'
	        if ('spontaneous' == stimulus):
	            stim_key = 'spontaneous'
	        
	        
	        neural_responses = {k: np.ones([19,4]) for k in data_set[ds].get_cell_specimen_ids()}            
	        results = {}
	        run_speed = np.array(tb_data[ds][stim_key + '_rs']).flatten()
	        
	        
	        if max(run_speed) < 20:
	            pass
	        else:
	        
	            run_speed_shuffled = np.random.permutation(run_speed)
	        
	            bins = stats.mstats.mquantiles(run_speed, np.linspace(0, 1, 20), limit = (0, 50))
	            shuf_bins = stats.mstats.mquantiles(run_speed_shuffled, np.linspace(0, 1, 20), limit = (0, 50))
	            
	            for ind, k in enumerate(data_set[ds].get_cell_specimen_ids()):

	                temp = np.array(tb_data[ds][stim_key + '_dff'][ind])  
	                
	                for i in range(1, len(bins)):
	                    bin_hist[i- 1] = [bins[i -1], bins[i]]
	                    shuf_hist[i-1] = [bins[i-1], bins[i]]
	                    
	                    idx = np.where((run_speed > bins[i-1]) & (run_speed < bins[i]))
	                    shuf_idx = np.where((run_speed_shuffled > shuf_bins[i-1]) & (run_speed_shuffled < shuf_bins[i]))
	                    
	                    #this control shouldn't be necessary
	                    if len(idx[0] != 0):
	                        av = np.mean(temp[idx[0]])
	                        std = np.std(temp[idx[0]])
	                        std_shuf = np.std(temp[shuf_idx[0]])
	                        av_shuf = np.mean(temp[shuf_idx[0]])
	                    else:
	                        av = 0
	                        std = 0

	                    neural_responses[k][i-1, 0] = av
	                    neural_responses[k][i-1, 1] = std / np.sqrt(len(temp[idx[0]]))
	                    neural_responses[k][i-1, 2] = av_shuf
	                    neural_responses[k][i-1, 3] = std / np.sqrt(len(temp[shuf_idx[0]]))

	                
	                x = np.log(np.mean(bin_hist, axis = 1))
	                y = np.array(neural_responses[k][:, 0])

	                shuf_x = np.log(np.mean(bin_hist, axis = 1))
	                shuf_y = np.array(neural_responses[k][:, 2])
	                
	                stat, pvalue = levene(y, shuf_y)
	                
	                n = len(x)
	                
	                ymax = max(y)
	                xmax = x[np.where(y == ymax)[0]]
	                
	                rho, p = spearmanr(x, y)
	                    
	                results[k] = rho, p, pvalue

	        stim_results[stim_key] = (neural_responses, results)    
	    rs_results[ds] = stim_results	

	return rs_results
