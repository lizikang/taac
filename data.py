import pandas as pd
import numpy as np
import gensim
import os

from util import *


class Data:
	def __init__(self, args):
		# initialize variables
		self.input_path = args.input_path
		self.w2v_path = args.w2v_path

		self.window_size = args.window_size
		self.num_folds = args.num_folds
		self.validation_fold = args.validation_fold

		self.features = ['week', 'creative_id', 'click_times', 'product_id', 'product_category', 'advertiser_id', 'industry']
		self.pretrain_fea = ['creative_id', 'product_id', 'product_category', 'advertiser_id']

		# read train/test file
		train_all_data = pd.read_csv(os.path.join(self.input_path, 'train_all_data.csv'))
		test_all_data = pd.read_csv(os.path.join(self.input_path, 'test_all_data.csv'))
		train_all_data['week'] = train_all_data['time'] % 7 + 1
		test_all_data['week'] = test_all_data['time'] % 7 + 1

		# get the pretrained embedding matrix
		var = self.__dict__
		for fea in self.features:
			var[fea+'_size'] = pd.concat([train_all_data[fea], test_all_data[fea]]).nunique() + 1
			print('size of feature {}: {}'.format(fea, var[fea+'_size']))

			if fea in self.pretrain_fea:
				w2v_path = os.path.join(self.w2v_path, '{}_w2v.bin'.format(fea))
				w2vModel = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
				fea_word2idx, var[fea+'_embedMatrix'] = build_word2idx_embedMatrix(w2vModel)

				train_all_data[fea] = train_all_data[fea].astype('str').map(lambda x: fea_word2idx[x])
				test_all_data[fea] = test_all_data[fea].astype('str').map(lambda x: fea_word2idx[x])
			else:
				var[fea+'_embedMatrix'] = None

		# use k-flod cross validation to generate validation users
		users = train_all_data['user_id'].unique()
		user_size = len(users)
		user_size = 900000
		np.random.seed(11)
		shuffled_indices = np.random.permutation(user_size)
		shuffled_users = users[shuffled_indices]

		fold_size = user_size // self.num_folds
		valid_start_index, valid_end_index = fold_size*(self.validation_fold-1), fold_size*self.validation_fold
		valid_users = shuffled_users[valid_start_index: valid_end_index]

		# generate inputs of train/validation/test set
		self.train_data = train_all_data[~train_all_data['user_id'].isin(valid_users)]
		self.valid_data = train_all_data[train_all_data['user_id'].isin(valid_users)]
		self.test_data = test_all_data

		for mode in ['train', 'valid', 'test']:
			tail_data = var[mode+'_data'].groupby('user_id').tail(self.window_size)
			length_list = tail_data['user_id'].value_counts().sort_index().tolist()
			fea_list = []

			for fea in self.features:
				fea_list.append(generate_array(tail_data[fea], length_list))

			var[mode+'_inputs'] = np.stack(fea_list, -1)
			print('shape of {}_inputs: {}'.format(mode, var[mode+'_inputs'].shape))

		# generate labels of train/valid set
		self.train_labels = np.array(self.train_data.groupby('user_id')['agender'].tail(1))
		print('shape of train_labels: {}'.format(self.train_labels.shape))
		self.valid_labels = np.array(self.valid_data.groupby('user_id')['agender'].tail(1))
		print('shape of valid_labels: {}'.format(self.valid_labels.shape))
