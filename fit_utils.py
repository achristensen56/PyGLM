from glm_utils import *
import group_glm as gm
from sklearn.cross_validation import train_test_split
import pandas as pd
import pyprind
import multiprocessing
import pickle
import allensdk.brain_observatory.stimulus_info as stim_info
import itertools as it

def fit_glm_CV(dff_array, design_matrix, non_linearity = [tf.exp, tf.nn.sigmoid, tf.nn.relu], 
	noise_model = ['exponential', 'gaussian']):
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

	n_nl, n_nm = len(non_linearity), len(noise_model)

	scores = np.zeros([n_cells, n_nl, n_nm])
	features = np.zeros([n_cells, n_nl, n_nm, n_features + 2]) 

	nm_dict = {'exponential': gm.exponential_GLM, 'gaussian': gm.gaussian_GLM}


	b = pyprind.ProgBar(len(noise_model)* len(non_linearity))

	for j, nm in enumerate(noise_model):
		for k, nl in enumerate(non_linearity):

			b.update()
			
			tp = True
			lr = 1e-4

			if nl == tf.exp or nl == tf.nn.relu:
				#the exponential non-linearity 
				tp = False
				lr = 1e-6


			#here we fit 5 different versions of the model, on different test sets, and average them together
			test_l = []

			weights = []
			offset = []
			scale = []

			for i in range(5):

				X_train, X_test, y_train, y_test = train_test_split(design_matrix, dff_array)

				weight_init = np.linalg.pinv(X_train).dot(y_train) 

				model = nm_dict[nm](weight_init,
					lr = lr, train_params = tp, eps = 1e-4, bias_init = 0, alpha = 0, reg = '', non_lin = nl, verbose = False)

				L, l = model.fit(X_train, y_train, X_test, y_test, max_iters = 200, batch_size = 5000)
				

				w, o, s = model.get_params()	

				l = np.array(l)
				w = np.array(w)

				weights.append(w)
				offset.append(o)
				scale.append(s)
				test_l.append(np.squeeze(l[-1]))


			#the parameters
			w = np.mean(np.array(weights).reshape(5, n_cells, n_features), axis = 0)
			o = np.mean(offset)
			s = np.mean(offset)
			l = np.mean(np.array(test_l), axis = 0)

			#the average of the final test likelihood is what we actually use for model comparison

			scores[:, k, j] = l
			features[:, k, j, 0:-2] = w
			features[:, k, j, -2] = o
			features[:, k, j, -1] = s
			

	return scores, features	 

	
def save_results_GLM((key, data_set)):
	dff, images, stim_table = get_data(data_set, stim_info.NATURAL_SCENES)
	r_images = pca_features(images)
	data, stim_array = arrange_data_glm(dff, r_images, stim_table)
	scores, features = fit_glm_CV(data, stim_array)
	output = open('./boc/formatted/' + str(key) + '_CV_results.pkl', 'wb')
	pickle.dump((scores, features), output)
	output.close()



if __name__ == '__main__':
		boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
		regions = ['VISpm']

		#['VISl', 'VISp', 'VISpm', 'VISal']
		lines = ['Cux2-CreERT2']

		#['Cux2-CreERT2', 'Rbp4-Cre', 'Rorb-IRES2-Cre'] 

		jobs = []

		for reg, cre in it.product(regions, lines):
			data_set = download_data([reg], [cre], [stim_info.NATURAL_SCENES])
			for key in data_set.keys():
				#p = multiprocessing.Process(target=save_results_GLM, args=((key, data_set[key]),))
				#jobs.append(p)
				#p.start()

				save_results_GLM((key, data_set[key]))