import tensorflow as tf
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import time
import os

from data import Data
from model import Model
from util import *


def main():
	# define agruments
	parser = argparse.ArgumentParser()
	parser.add_argument('--input_path', type=str, default='files/data/')
	parser.add_argument('--w2v_path', type=str, default='files/w2v/size200_win15')
	parser.add_argument('--output_path', type=str, default='files/results')

	parser.add_argument('--window_size', type=int, default=150)
	parser.add_argument('--dimension', type=int, default=16)
	parser.add_argument('--hidden_units', type=int, default=600)
	parser.add_argument('--age_weight', type=float, default=0.5)

	parser.add_argument('--num_epochs', type=int, default=20)
	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--learn_rate', type=float, default=1e-3)
	parser.add_argument('--dropout_prob', type=float, default=0.2)

	parser.add_argument('--logid', type=int, default=0)
	parser.add_argument('--num_folds', type=int, default=10)
	parser.add_argument('--validation_fold', type=int, default=10)
	args = parser.parse_args()
	print(str(args))

	# define data
	print('data process begin...')
	t1 = time.time()
	data = Data(args)
	used_time = (time.time() - t1) / 60
	print('data process end, used time: {:.2f} min'.format(used_time))

	# define and train model
	model = Model(args, data)
	train(args, data, model)


def train(args, data, model):
	# calculate batch_size and iterations for train/validation/test set
	train_iterations = data.train_inputs.shape[0] // args.batch_size + 1
	valid_batch_size = args.batch_size * 50
	valid_iterations = data.valid_inputs.shape[0] // valid_batch_size + 1
	test_batch_size = args.batch_size * 50
	test_iterations = data.test_inputs.shape[0] // test_batch_size + 1

	# define evaluation metric for early stopping, b means best
	bvalid_acc, bvalid_age_acc, bvalid_gender_acc, btest_age_logits, btest_gender_logits = 0, 0, 0, 0, 0
	bepoch, times = 0, []

	# begin to train model
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for i in range(1, args.num_epochs+1):
			# shuffle train data at each epoch
			shuffled_indices = np.random.permutation(data.train_inputs.shape[0])
			data.train_inputs = data.train_inputs[shuffled_indices]
			data.train_labels = data.train_labels[shuffled_indices]

			# compute and display train loss
			start_time = time.time()
			for j in range(1, train_iterations+1):
				start, end = args.batch_size * (j-1), args.batch_size * j
				train_feed_dict = {model.batch_inputs: data.train_inputs[start:end], 
						   model.batch_labels: data.train_labels[start:end],
						   model.is_training: True}

				age_loss_, gender_loss_, loss_, train_op_ = sess.run([model.age_loss, 
										      model.gender_loss, 
										      model.loss, 
										      model.train_op], 
										      feed_dict = train_feed_dict)
				if j % (train_iterations//20) == 0: 
					print('epoch: {}, iteration: {}, age_loss: {:.6f}, gender_loss: {:.6f}, total_loss: {:.6f}'.format
					     (i, j, age_loss_, gender_loss_, loss_))
			times.append(time.time() - start_time)

			if i % 1 == 0:
				# generate validation predictions
				valid_age_list, valid_gender_list = [], []
				for j in range(1, valid_iterations+1):
					start, end = valid_batch_size * (j-1), valid_batch_size * j
					valid_feed_dict = {model.batch_inputs: data.valid_inputs[start:end],
							   model.is_training: False}

					valid_age_logits, valid_gender_logits = sess.run([model.age_logits, 
											  model.gender_logits], 
											  feed_dict=valid_feed_dict)
					valid_age_list.append(valid_age_logits)
					valid_gender_list.append(valid_gender_logits)

				# compute validation performance
				valid_pred_age = np.argmax(np.concatenate(valid_age_list, 0), 1) + 1
				valid_true_age = data.valid_labels % 10 + 1
				valid_age_acc = np.mean(np.equal(valid_pred_age, valid_true_age))

				valid_pred_gender = np.argmax(np.concatenate(valid_gender_list, 0), 1) + 1
				valid_true_gender = data.valid_labels // 10 + 1
				valid_gender_acc = np.mean(np.equal(valid_pred_gender, valid_true_gender))

				valid_acc = valid_age_acc + valid_gender_acc	
				print('epoch: {}, (valid) age acc: {:.8f}, gender acc: {:.8f}, total acc: {:.8f}\n'.format
				     (i, valid_age_acc, valid_gender_acc, valid_acc))

				# generate test predictions
				test_age_list, test_gender_list = [], []
				for j in range(1, test_iterations+1):
					start, end = test_batch_size * (j-1), test_batch_size * j
					test_feed_dict = {model.batch_inputs: data.test_inputs[start:end],
							  model.is_training: False}

					test_age_logits, test_gender_logits = sess.run([model.age_logits, 
											model.gender_logits], 
											feed_dict=test_feed_dict)
					test_age_list.append(test_age_logits)
					test_gender_list.append(test_gender_logits)
				test_age_logits = np.concatenate(test_age_list, 0)
				test_gender_logits = np.concatenate(test_gender_list, 0)

				# record the best validation performance 
				if valid_acc >= bvalid_acc:
					bvalid_acc, bvalid_age_acc, bvalid_gender_acc, btest_age_logits, btest_gender_logits, bepoch = \
					valid_acc, valid_age_acc, valid_gender_acc, test_age_logits, test_gender_logits, i

				# early stopping
				if i - bepoch >= 2:
					# display the model arguments
					print(args.logid, ':', args.window_size, args.dimension, args.hidden_units, args.age_weight, '\t', 
							       args.num_epochs, args.batch_size, args.learn_rate, args.dropout_prob, '\t', 
							       args.num_folds, args.validation_fold)

					# display the best validation performance
					total_time, avg_time = np.sum(times) / 60, np.mean(times) / 60
					print('best epoch: {}, time: {:.2f} min, {:.2f} min, (valid) age acc: {:.8f}, gender acc: {:.8f}, total acc: {:.8f}'.format
					     (bepoch, total_time, avg_time, bvalid_age_acc, bvalid_gender_acc, bvalid_acc))

					# compute and save the corresponding test result
					output_dir = os.path.join(args.output_path, str(args.logid))
					if not os.path.exists(output_dir):
						os.makedirs(output_dir)

					age_file = 'age_{}_{}_{}.npy'.format(args.logid, args.num_folds, args.validation_fold)
					age_path = os.path.join(output_dir, age_file)
					np.save(age_path, softmax(btest_age_logits))

					gender_file = 'gender_{}_{}_{}.npy'.format(args.logid, args.num_folds, args.validation_fold)
					gender_path = os.path.join(output_dir, gender_file)
					np.save(gender_path, softmax(btest_gender_logits))
					
					test_pred_age = np.argmax(btest_age_logits, 1) + 1
					test_pred_gender = np.argmax(btest_gender_logits, 1) + 1
					submission_df = pd.DataFrame({'user_id': np.arange(3000001, 4000001), 
								      'predicted_age': test_pred_age, 
								      'predicted_gender': test_pred_gender})

					submission_file = 'submission_{}_{}_{}.csv'.format(args.logid, args.num_folds, args.validation_fold)
					submission_path = os.path.join(output_dir, submission_file)
					submission_df.to_csv(submission_path, index=False)

					break


if __name__ == '__main__':
	main()
