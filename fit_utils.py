from glm_utils import *
import group_glm as gm
from sklearn.cross_validation import train_test_split
import pandas as pd
import pyprind
import multiprocessing
import pickle
import allensdk.brain_observatory.stimulus_info as stim_info
import itertools as it

def fit_glm_CV(dff_array, design_matrix, learning_rate = [0.001], non_linearity = ['exp', 'sigmoid'], batch_size = 10000, offset_init = 0, scale_init = 1, bias_init = 0,
	noise_model = ['exponential', 'gaussian', 'lognormal'], num_pcs = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 118], debug = False, max_iters = 1000):
	'''
	dff_array (numpy array) = n_time_bins x n_cells
	design_matrx (numpy array) = n_time_bins n_features
	non_linearity = list of non-linearities to fit
	noise_models = list of noise models to fit

	returns:
		scores = n_cells x n_non_linearities x n_noise_models
			
			likelihood on the test set for each non_linearity, and noise model 

		models = n_cells x n_non_linearities x n_noise_models x n_features
			
			the learned features -- average of the models learned over the 5 draws
	'''



	n_time_bins, n_cells = dff_array.shape
	_, n_features = design_matrix.shape

	#print 'the shape of the dff array is: ', dff_array.shape, 'the shape of the design matrix is: ', design_matrix.shape

	n_nl, n_nm, n_pcs, n_lrs = len(non_linearity), len(noise_model), len(num_pcs), len(learning_rate)

	scores = np.zeros([n_cells, n_nl, n_nm, n_pcs, n_lrs])
	features = np.zeros([n_cells, n_nl, n_nm, n_pcs, n_lrs, n_features + 3]) 

	nm_dict = {'exponential': gm.exponential_GLM, 'gaussian': gm.gaussian_GLM, 'lognormal': gm.lognormal_GLM}
	b = pyprind.ProgBar(len(noise_model)* len(non_linearity)*len(num_pcs))


	for j, nm in enumerate(noise_model):
		for k, nl in enumerate(non_linearity):
			for m, npcs in enumerate(num_pcs):
				for n, lr in enumerate(learning_rate):

					b.update()
					
					tp = True
					#lr = 1e-3

					if nl == tf.exp or nl == tf_soft_rec:
						#the exponential non-linearity 
						tp = False
						scale_init = 1
						bias_init = 0
						#lr = 1e-3



					#here we fit 5 different versions of the model, on different test sets, and average them together
					test_l = []

					weights = []
					nonlin_offset = []
					offset = []
					scale = []
					variance = []

					print "Fitting: ", nm, " noise model with: ", nl, " non-linearity, and: " , npcs ,"features"

					for i in range(5):

						X_train, X_test, y_train, y_test = train_test_split(design_matrix[:, 0:npcs], dff_array)

						weight_init = np.linalg.pinv(X_train).dot(y_train) 

						model = nm_dict[nm](weight_init, lr = lr, train_params = tp, alpha = 0, reg = '', non_lin = nl, 
							verbose = False, offset_init = offset_init, scale_init= scale_init, bias_init = bias_init)

						L, l = model.fit(X_train, y_train, X_test, y_test, max_iters = max_iters, batch_size = batch_size)
						

						if debug == True:

							plt.plot(L)
							plt.show()

							plt.plot(l)
							plt.show()

							if (~np.isfinite(l)).any():
								print "The test likelihood is NAN"


						w, o, nls, s = model.get_params()	


						w = np.array(w)

						weights.append(w)
						offset.append(o)
						nonlin_offset.append(nls)
						scale.append(s)

						l = model.get_log_likelihood(y_test, X_test)

						l.shape
						test_l.append(l)


					#the parameters
					w = np.mean(np.array(weights), axis = 0).T
					o = np.mean(offset, axis = 0)
					s = np.mean(scale, axis = 0)
					nls = np.mean(nonlin_offset, axis = 0)


					l = np.mean(np.array(test_l), axis = 0)

					#the average of the final test likelihood is what we actually use for model comparison

					scores[:, k, j, m, n] = l
					features[:, k, j, m, n, 0:npcs] = w
					features[:, k, j, m, n, -3] = nls
					features[:, k, j, m, n,-2] = o
					features[:, k, j, m, n, -1] = s
		

	return scores, features	 

	
def save_results_GLM((key, data_set)):
	dff, images, stim_table = get_data(data_set, stim_info.NATURAL_SCENES)
	r_images = pca_features(images)
	data, stim_array = arrange_data_glm(dff, r_images, stim_table)
	scores, features = fit_glm_CV(data, stim_array[:, 0:50])
	output = open('./boc/formatted/' + str(key) + '50_CV_results.pkl', 'wb')
	pickle.dump((scores, features), output)
	output.close()



if __name__ == '__main__':
		boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
		regions = ['VISpm']
		lines = ['Cux2-CreERT2']#, 'Rbp4-Cre', 'Rorb-IRES2-Cre'] 

		jobs = []

		for reg, cre in it.product(regions, lines):
			data_set = download_data([reg], [cre], [stim_info.NATURAL_SCENES])
			for key in data_set.keys():
				#p = multiprocessing.Process(target=save_results_GLM, args=((key, data_set[key]),))
				#jobs.append(p)
				#p.start()

				save_results_GLM((key, data_set[key]))
