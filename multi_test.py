from test import test

net_parameters = {
	    'autoencoder': [128],
	    'deep_autoencoder': [128, 64, 32, 64, 128],
	    'convolutional_autoencoder': [], # it's a placeholder
	    'convolutional_autoencoder_raw': [] # it's a placeholder
	}

models = ['autoencoder', 'deep_autoencoder','convolutional_autoencoder'] #  
datasets = ['pepper'] # 'boat', 
types = ['bin', 'csv'] #  

log_path = 'multi_training.log'

for dataset in datasets:

	with open(log_path, 'a') as log_fd:
		log_fd.write(f'**********{dataset}**********\n')

	for d_type in types:

		with open(log_path, 'a') as log_fd:
			log_fd.write(f'\n{d_type}\n')

		for model in models:

			with open(log_path, 'a') as log_fd:
				log_fd.write(f'******{model}\n')

			res_dict = test(dataset, d_type, model, net_parameters=net_parameters)

			with open(log_path, 'a') as log_fd:
				log_fd.write(f'n\tz\tacc\t\tf1\n')

			for n, vals in res_dict.items():

				with open(log_path, 'a') as log_fd:
					log_fd.write('--------------------\n')

				for z, v in vals.items():

					with open(log_path, 'a') as log_fd:
						log_fd.write(f'{n}\t{z}\t{v[0]:.2f}\t{v[1]:.2f}\n')

			with open(log_path, 'a') as log_fd:
				log_fd.write(f'\n')
