import numpy as np
from scipy.stats import norm
import matplotlib.gridspec as gridspec
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
from statsmodels.robust.scale import mad

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

def tf_soft_rec(x):
	return tf.log(1 + tf.exp(x))
def np_soft_rec(x):
	return np.log(1 + np.exp(x))

def cond_int(non_lin, weights, stim, scale, c, nls = 0):
	'''
	returns the conditional intensity \lambda = f(w^TX)
	'''
	h = stim.dot(weights)
	f = scale*(non_lin(h-c) + nls)
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
	y = np.random.normal(cond_int, 1)
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

def simpleaxis(ax, bottom = False):
	'''
	remove the top and right spines and ticks from the axis. 
	'''
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	if bottom:
		ax.spines['bottom'].set_visible(False)
		ax.get_xaxis().set_ticks([])
	else:
		ax.get_xaxis().tick_bottom()
	

	ax.get_yaxis().tick_left()

def relu(X):
	return X*(X > 0)

def get_data_stats(all_tensors):
	'''
	returns mean, std 
	'''
	mean = [0]
	std = [0]

	for key in all_tensors.keys():
	    
	    #this computes the mean across the axis of numTrials.
	    t_mean = np.mean(all_tensors[key], 2).flatten()
	    t_std = np.std(all_tensors[key], 2).flatten()
	    

	    
	    mean = np.concatenate((mean, t_mean))
	    std = np.concatenate((std, t_std))
	
	return mean, std

def sort_scores(scores_dict):

	t_per_exp = []
	t_per_gaus = []

	t_per_l_exp = []
	t_per_l_sig = []
	t_per_l_sr = []

	le_dict = {}
	lg_dict = {}


	for key in scores_dict.keys():
	    
	    scores, features = scores_dict[key]
	    
	    n_cells, n_nl, n_nm = scores.shape

	    best_noise_model = []
	    best_non_linearity = []

	    likelihood_exponential = []
	    likelihood_gaussian = [] 

	    for i in range(n_cells):
	        idx = np.argmin(scores[i])

	        nl_ind, nm_ind = np.unravel_index(idx, (n_nl, n_nm))

	        if scores[i, nl_ind, nm_ind] != np.nan:
	            best_noise_model.append(nm_ind)
	            best_non_linearity.append(nl_ind)


	            if nm_ind == 0:
	                likelihood_exponential.append(scores[i, nl_ind, nm_ind])

	            if nm_ind == 1:
	                likelihood_gaussian.append(scores[i, nl_ind, nm_ind])
	                
	    best_noise_model = np.array(best_noise_model)
	    best_non_linearity = np.array(best_non_linearity)
	    
	    per_exp = sum(best_noise_model == 0) / float(len(best_noise_model))
	    per_gaus = sum(best_noise_model == 1) / float(len(best_noise_model))
	    
	    per_l_exp = sum(best_non_linearity == 0) / float(len(best_non_linearity))
	    per_l_sig = sum(best_non_linearity == 1) / float(len(best_non_linearity))
	    per_l_sr = sum(best_non_linearity == 2) / float(len(best_non_linearity))
	    
	    t_per_exp.append(per_exp)
	    t_per_gaus.append(per_gaus)
	    
	    t_per_l_exp.append(per_l_exp)
	    t_per_l_sig.append(per_l_sig)
	    t_per_l_sr.append(per_l_sr)

	    
	    le = -np.array(scores[:, :, 0].flatten())
	    le = le[~np.isnan(le)] 
	    lg = -np.array(scores[:, :, 1].flatten())
	    lg = lg[~np.isnan(lg)]

	    le_dict[key] = le
	    lg_dict[key] = lg


	return t_per_exp, t_per_gaus, t_per_l_exp, t_per_l_sig, t_per_l_sr, lg_dict, le_dict

def get_explainable_variance(data_tensor):
	'''
	data_tensor should be the output of arrange_data_trialTensor
	
	computes explainable variance by randomly splitting the trials in half, and regressing 1/2 the trials against
	the other half of the trials. 
	'''
	from scipy.stats import linregress

	n_neurons, n_conditions, n_trials, trialLength = data_tensor.shape

	data_tensor = np.mean(data_tensor, axis = 3)

	trial_idx = np.arange(0, n_trials)
	trial_idx = np.random.permutation(trial_idx)

	tr_1 = trial_idx[0:n_trials/2]
	tr_2 = trial_idx[n_trials/2:]

	av_1 = np.mean(data_tensor[:, :, tr_1], axis = 2)
	av_2 = np.mean(data_tensor[:, :, tr_2], axis = 2)

	ra_variance = np.zeros([n_neurons])

	for i in range(n_neurons):
		slope, intercept, r_value, p_value, std_err = linregress(av_1[i], av_2[i])
		ra_variance[i] = r_value**2

	return ra_variance

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

def pca_features(images, scale = True):

	if scale == True:

		av = np.mean(images, axis = 0)
		std = np.std(images, axis = 0)

		images -= av
		images /= std

	model = PCA()
	scenes_r = model.fit_transform(images.reshape([len(images), -1])) 

	return scenes_r

def get_data(data_set, stimulus):
	'''
	returns dff, images, stim_table
	'''

	time, dff_traces = data_set.get_dff_traces()

	try:
		images = data_set.get_stimulus_template(stimulus)
	except:
		print "No stimulus template..."
		images = None

	stim_table = data_set.get_stimulus_table(stimulus)

	return dff_traces, images, stim_table


def arrange_data_tuning(dff, dxcm, stim_table, ratio = True, fps = 30):
	'''
	Calculate average responses to different stimulus conditions, with the stim_table serving
	as a lookup table. 

	If ratio is True, response will be calculated in the AIBS way, average(stim_onset + .5 sec)
	/average(stim_onset - .5 secs)

	returns responses: n_conditions x n_neurons
	'''

	responses = []

	for i, row in stim_table.iterrows():

		if ratio:
			baseline = np.average(dff[:, row['start'] - fps : row['start']], axis = 1)
			response = np.average(dff[:, row['start'] : row['start'] + fps], axis = 1)

			ev_resp = (response - baseline) / baseline
		else:
			ev_resp = np.average(dff[:, row['start'] + fps/10: row['start'] + fps], axis = 1)

		responses.append(ev_resp)

	return np.array(responses)



def arrange_data_glm(dff_traces, images, stim_table):
	#declare a dictionary of empty lists for each cell trace, 
	#and a list for the stimulus
	data = []
	stim_array = []

	#average each trace over the presentation of each stimulus
	for index, row in stim_table.iterrows():
	    stim_array.append(images[row['frame']])
	    data.append(np.mean(dff_traces[:, row['start']:row['end'] ], axis = 1) )
	    
	stim_array = np.array(stim_array)
	#stim_array = stim_array[:, 0:10]

	data = np.array(data)

	return data, stim_array


def arrange_rs_glm(rs, stim_table):

	running_speed = []

	for index, row in stim_table.iterrows():
		running_speed.append(np.average(rs[row['start']:row['end']]))

	return running_speed

def get_index_array(stim_table):
	index_array = []

	for index, row in stim_table.iterrows():
		index_array.append(row['frame'])
	return np.array(index_array)

def trial_average(data, index_array):

	l, n_neurons = data.shape

	n_conditions = len(set(index_array))
	trial_average = np.zeros([n_conditions, n_neurons])
	for i in range(l):
		trial_average[index_array[i]] += data[i]

	trial_average /= n_conditions

	return trial_average

def arrange_ns_data_trialTensor(dff_traces, stim_table):
	'''
	In this function we want to take dff traces (n_neurons x ntrials*ntimepointspertrial)
	and return data_Tensor = n_neurons x n_conditions x n_trials x trialLength

	This is helpful for computing statistics of the data, like mean vs. standard deviation, 
	and could be a useful multipurpose pipelining tool in the future.   
	
	at this point this is untested with stim_table objects other than that from 'Natural Scenes'

	'''

	n_neurons, _ = dff_traces.shape
	trialLength = np.min(stim_table['end'] - stim_table['start'])
	n_conditions = len(stim_table['frame'].unique())
	n_trials = np.floor(len(stim_table['frame']) / n_conditions)

	data_Tensor = np.zeros([n_neurons, n_conditions, n_trials, trialLength]);

	trialCount = np.zeros([n_conditions])
	for index, row in stim_table.iterrows():
		data_Tensor[:, row['frame'], trialCount[row['frame']], :] = dff_traces[:, row['start']: row['start'] + trialLength]
		trialCount[row['frame']] +=1

	return data_Tensor, trialCount



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


def gaussian_variance(y, x, w, o, nls, s, non_lin = sigmoid):
	T = len(y)
	c_int = cond_int(non_lin, w, x, s, o, nls)

	return (1./T * sum((y-c_int)**2))

def mad_scaling(data):

	L, N = data.shape

	offset = np.mean(data, axis = 0) - mad(data, axis = 0, c = .6);
	top = np.mean(data, axis = 0) + mad(data, axis = 0, c = .6);
	return offset, top

def tf_identity(data):
	return data