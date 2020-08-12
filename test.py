import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from models.autoencoder import AutoEncoder
from sklearn.svm import OneClassSVM
from argparse import ArgumentParser
from utils import *

import numpy as np

def parse_args():
	parser = ArgumentParser(description='''Script to train and test autoencoders on different dataset''')
	parser.add_argument('-d', '--dataset', default='pepper', help='Dataset: pepper or boat')
	parser.add_argument('-t', '--type', default='img', help='Dataset type: img or csv')
	parser.add_argument('-m', '--model', default='autoencoder', help='Model: autoencoder, deep_autoencoder, convolutional_autoencoder or convolutional_autoencoder_raw')
	parser.add_argument('-e', '--epochs', default=30, type=int, help='Number of epochs')
	parser.add_argument('-o', '--optimizer', default='adam', help='Optimizer')
	parser.add_argument('-l', '--loss', default='binary_crossentropy', help='Loss function')
	parser.add_argument('-b', '--batch', default=64, type=int, help='Batch size')
	parser.add_argument('-k', '--kernel', nargs='+', default=[2, 2], type=int, help='Kernel size (only for CAE)')
	parser.add_argument('-n', '--nvalues', nargs='+', default=[1, 3, 5, 10], type=int, help='n values')

	return parser.parse_args()

def test(dset,
		dset_type,
		model_name,
		epochs=30,
		optimizer='adam',
		loss_fun='binary_crossentropy',
		batch_size=64,
		kernel_size=(5, 5),
		n_values=[1, 3, 5, 10],
		net_parameters=None):

	ext = 'csv' if dset_type == 'csv' else 'npz'

	norm_dset_path = f'dset/{dset}/{dset_type}/normal.{ext}'
	atk_dset_path = f'dset/{dset}/{dset_type}/attack.{ext}'

	is_img = dset_type != 'csv'
	normalize = dset_type != 'bin'

	print_with_time(f'[INFO] Loading {dset_type} dataset {dset}...')
	normal_dataset, attack_dataset, input_shape = load_dataset(norm_dset_path, atk_dset_path, is_img, normalize=normalize)

	x_train, x_test, x_test_treshold, x_val = split_normal_dataset(normal_dataset)
	x_abnormal = attack_dataset#[:x_test.shape[0]]

	print_with_time(f'[INFO] Dataset {dset} loaded.')
 

	x_train, x_test, x_test_treshold, x_val, x_abnormal, input_shape = parse_datasets(x_train,
																					x_test,
																					x_test_treshold,
																					x_val,
																					x_abnormal,
																					input_shape,
																					dset,
																					dset_type,
																					model_name)

	n_results = {}

	for n in n_values:

		print_with_time(f'[INFO] Training {n} network(s)')

		z_values = [0, 1, 2, 3]
		y_true = np.array([False]*x_test.shape[0] + [True]*x_abnormal.shape[0])
		y_pred_matrix = np.zeros((n, len(z_values), y_true.shape[0]))

		cols = x_test.shape[0] + x_abnormal.shape[0]
		cols_tresh = x_test_treshold.shape[0]

		final_losses = np.zeros(n)
		pred_losses = np.zeros((n, cols))
		pred_losses_tresh = np.zeros((n, cols_tresh))

		for i in range(n):

			print_with_time(f'[INFO] Training {i}th {model_name}')

			layers_size = net_parameters[model_name]

			model = AutoEncoder(model_name, input_shape, layers_size, kernel_size).model
			model.compile(optimizer=optimizer, loss=loss_fun)

			#print(model.summary())

			# train on only normal training data
			history = model.fit(
				x=x_train,
				y=x_train,
				epochs=epochs,
				batch_size=batch_size,
				validation_data = (x_val, x_val),
				verbose=0
			)

			print_with_time(f'[INFO] {i}th {model_name} trained for {epochs}')

			# [-1] takes the last value of the loss given all the iterations
			final_losses[i] = np.nanmean(history.history['val_loss'])

			# test  
			losses = []
			x_concat = np.concatenate([x_test, x_abnormal], axis=0)

			for x in x_concat:
				#compute loss for each test sample
				x = np.expand_dims(x, axis=0)
				loss = model.test_on_batch(x, x)
				losses.append(loss)

			pred_losses[i] = losses
			print_with_time(f'[INFO] {i}th {model_name} tested on {x_concat.shape[0]} samples')

			losses  = []

			for x in x_test_treshold:
				x = np.expand_dims(x, axis=0)
				loss = model.test_on_batch(x, x)
				losses.append(loss)

			pred_losses_tresh[i] = losses
			print_with_time(f'[INFO] Tested {i}th {model_name}')


		mean_tresh_losses = np.nanmean(pred_losses_tresh,axis=0) if n > 1 else pred_losses_tresh[0]
		var_tresh_losses = np.nanvar(pred_losses_tresh,axis=0) if n > 1 else pred_losses_tresh[0]

		mean_pred_losses = np.nanmean(pred_losses,axis=0) if n > 1 else pred_losses[0]
		var_pred_losses = np.nanvar(pred_losses,axis=0) if n > 1 else pred_losses[0]

		filename = f'{n}_{model_name}_{epochs}_{dset}_{dset_type}'
		plot_predictions(var_tresh_losses, var_pred_losses, filename, n)
		print_with_time('[INFO] Plotted results')

		z_results = {}

		for z in z_values:
		  
			treshold_upperbound = np.nanmean(var_tresh_losses) + np.nanstd(var_tresh_losses) * z
			treshold_lowerbound = np.nanmean(var_tresh_losses) - np.nanstd(var_tresh_losses) * z

			if n > 1:
				y_pred = np.array([x > treshold_upperbound for x in var_pred_losses])

			else:
				y_pred = np.array([x < treshold_lowerbound or x > treshold_upperbound for x in var_pred_losses])

			print(f'Z: {z}', end=', ')
			acc, f1 = compute_and_save_metrics(y_true, y_pred, False)

			z_results[z] = (acc, f1)

			n_results[n] = z_results

	return n_results


if __name__ == '__main__':
	args = parse_args()

	net_parameters = {
	    'autoencoder': [64],
	    'deep_autoencoder': [64, 32, 16, 32, 64],
	    'convolutional_autoencoder': [], # it's a placeholder
	    'convolutional_autoencoder_raw': [] # it's a placeholder
	}

	test(args.dataset,
		args.type,
		args.model,
		args.epochs,
		args.optimizer,
		args.loss,
		args.batch,
		tuple(args.kernel),
		args.nvalues,
		net_parameters)