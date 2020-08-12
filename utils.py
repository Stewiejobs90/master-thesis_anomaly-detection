import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
from matplotlib import pyplot as plt

from sklearn.metrics import confusion_matrix, accuracy_score, fbeta_score, precision_score, recall_score, roc_curve, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from time import asctime, localtime, time
from math import sqrt, ceil

import pandas as pd
import numpy as np
import random

def pad_for_conv(x,shape_out):
	shape = (x.shape[0],) + shape_out
	padding = np.zeros(shape)

	padding[:, :x.shape[1], :x.shape[2]] = x.reshape((-1, x.shape[1], x.shape[2]))
	return padding

def print_with_time(in_str=''):
	curr = asctime(localtime(time()))
	print(curr + ' |\t| ' + in_str)

def standardize_raw_data(normal_dataset, attack_dataset):

	sc = MinMaxScaler()
	norm_dset = sc.fit_transform(normal_dataset)
	atk_dset = sc.transform(attack_dataset)

	return norm_dset, atk_dset

def compute_accuracy(treshold_classification_upper, treshold_classification_lower, predictions, labels, n):

	y_true = np.array(labels)

	if n > 1:
		y_pred = np.array([x > treshold_classification_upper for x in predictions])
	else:
		y_pred = np.array([x < treshold_classification_lower or  x > treshold_classification_upper for x in predictions])

	accuracy = accuracy_score(y_true, y_pred)

	return accuracy

def compute_and_save_metrics(y_true, y_pred, print_all=True):

	tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

	accuracy = accuracy_score(y_true, y_pred)
	precision = precision_score(y_true, y_pred)
	recall = recall_score(y_true, y_pred)
	f2 = fbeta_score(y_true, y_pred, beta=1)
	f1 = f1_score(y_true, y_pred)

	if print_all:
		print('Total accuracy: %f' % accuracy)
		print('Total precision: %f' % precision)
		print('Total recall: %f' % recall)
		print('Total f2: %f' % f2)
		print('Total true negative: %f' % tn)
		print('Total false positive: %f' % fp)
		print('Total false negative: %f' % fn)
		print('Total true positive: %f' % tp)
	else:
		print(f'A: {accuracy:.6f}, P: {precision:.6f}, R: {recall:.6f}, F1 :{f1:.6f}')

	return accuracy, f1

def load_dataset(normal_dataset_path, attack_dataset_path, is_img=True, normalize=True, shuffle=True, drop_str=True):

	if is_img:
		normal_dset = np.load(normal_dataset_path)['x']
		attack_dset = np.load(attack_dataset_path)['x']

		if shuffle:
			norm = normal_dset.tolist()
			random.shuffle(norm)
			normal_dset = np.array(norm).astype(np.uint8)

			atk = attack_dset.tolist()
			random.shuffle(atk)
			attack_dset = np.array(atk).astype(np.uint8)

		if normalize:
			normal_dset = normal_dset/255
			attack_dset = attack_dset/255

		input_shape = normal_dset.shape[1:]


	else:
		normal_dset = pd.read_csv(normal_dataset_path, low_memory=False, skiprows=[1])
		attack_dset = pd.read_csv(attack_dataset_path, low_memory=False, skiprows=[1])

		if drop_str:

			normal_dset.drop(normal_dset.select_dtypes(['object']), inplace=True, axis=1)
			attack_dset.drop(attack_dset.select_dtypes(['object']), inplace=True, axis=1)

			normal_dset.fillna(0, inplace=True)
			attack_dset.fillna(0, inplace=True)

		if normalize:
			normal_dset, attack_dset = standardize_raw_data(normal_dset, attack_dset)

		input_shape = normal_dset.shape

	return normal_dset, attack_dset, input_shape

def split_normal_dataset(dataset, test_size=0.3, seed=1):
	x_train, x_test = train_test_split(dataset, test_size=test_size, random_state=seed)
	x_train, x_val = train_test_split(x_train, test_size=test_size, random_state=seed)

	x_treshold_samples = int(x_test.shape[0] * test_size)
	x_test_treshold = x_test[:x_treshold_samples]
	x_test = x_test[x_treshold_samples:]

	return x_train, x_test, x_test_treshold, x_val

def parse_datasets(x_train, x_test, x_test_treshold, x_val, x_abnormal, input_shape, dataset, dset_type, model):

	if 'convolutional' in model:

		if dset_type in ['img', 'bin']:

			shape_out = (132,132) if dataset == 'pepper' else (116,116)

			x_train = pad_for_conv(x_train, shape_out)
			x_test = pad_for_conv(x_test, shape_out)
			x_abnormal = pad_for_conv(x_abnormal, shape_out)
			x_val = pad_for_conv(x_val,shape_out)
			x_test_treshold = pad_for_conv(x_test_treshold, shape_out)

		else:
			shape_out = (16, 16) if dataset == 'pepper' else (7, 5)
		
		reshape_size = (-1,) + shape_out + (1,)

	elif dset_type in ['img', 'bin']:
		reshape_size = (-1, input_shape[0] * input_shape[1])

	else:
		return x_train, x_test, x_test_treshold, x_val, x_abnormal, (1, input_shape[1])

	out_train = x_train.reshape(reshape_size)
	out_test = x_test.reshape(reshape_size)
	out_treshold = x_test_treshold.reshape(reshape_size)
	out_val = x_val.reshape(reshape_size)
	out_abn = x_abnormal.reshape(reshape_size)
	reshape_size = reshape_size[1:3] if 'convolutional' in model else input_shape

	if (dataset, dset_type) == ('boat', 'csv'):

		reshape_size = (8, 8)

		x_train = pad_for_conv(out_train, reshape_size)
		x_test = pad_for_conv(out_test, reshape_size)
		x_abnormal = pad_for_conv(out_abn, reshape_size)
		x_val = pad_for_conv(out_val,reshape_size)
		x_test_treshold = pad_for_conv(out_treshold, reshape_size)

		shape_out = (-1,) + reshape_size + (1,)

		out_train = x_train.reshape(shape_out)
		out_test = x_test.reshape(shape_out)
		out_treshold = x_test_treshold.reshape(shape_out)
		out_val = x_val.reshape(shape_out)
		out_abn = x_abnormal.reshape(shape_out)

	return out_train, out_test, out_treshold, out_val, out_abn, reshape_size

def plot_predictions(var_tresh_losses, var_pred_losses, filename, n):
	z = int(n == 0)

	treshold_classification = np.max(var_tresh_losses)
	treshold_mean_classification = np.nanmean(var_tresh_losses)

	treshold_upperbound = np.nanmean(var_tresh_losses) + np.nanstd(var_tresh_losses) * z
	treshold_lowerbound = np.nanmean(var_tresh_losses) - np.nanstd(var_tresh_losses) * z

	plt.figure(figsize=(18,10))
	plt.plot(range(len(var_tresh_losses)),var_tresh_losses)
	plt.plot(range(len(var_tresh_losses)), [treshold_upperbound] * len(var_tresh_losses), color='k', ls='dashed')
	plt.plot(range(len(var_tresh_losses)), [treshold_lowerbound] * len(var_tresh_losses), color='k', ls='dashed')
	plt.savefig(f'images/{filename}_normal.png')
	plt.close()


	fig, ax = plt.subplots(figsize=(18,10))
	ax.plot(range(len(var_pred_losses)), var_pred_losses)

	if n == 1:
		ax.plot(range(len(var_pred_losses)), [treshold_mean_classification] * len(var_pred_losses), color='k')
		ax.plot(range(len(var_pred_losses)), [treshold_upperbound] * len(var_pred_losses), color='k', ls='dashed')
		ax.plot(range(len(var_pred_losses)), [treshold_lowerbound] * len(var_pred_losses), color='k', ls='dashed')
	else:
		ax.plot(range(len(var_pred_losses)), [treshold_mean_classification] * len(var_pred_losses), color='k')
		ax.plot(range(len(var_pred_losses)), [treshold_upperbound] * len(var_pred_losses), color='k', ls='dashed')

	ax.set_title('Anomaly', fontdict={'fontsize': 20})
	ax.legend(('Predictions', 'Treshold'), loc='best', fontsize='x-large', frameon=True)
	ax.set_xlabel('Test Samples', fontdict={'fontsize': 20})
	ax.set_ylabel('Losses Variance', fontdict={'fontsize': 20})
	ax.tick_params(labelsize=20)
	ax.set_facecolor((1., 1., 1.))
	ax.spines['right'].set_edgecolor('black')
	ax.spines['left'].set_edgecolor('black')
	ax.spines['top'].set_edgecolor('black')
	ax.spines['bottom'].set_edgecolor('black')
	ax.grid(True, color='grey')
	fig.savefig(f'images/{filename}_attack.png')
	plt.close(fig)