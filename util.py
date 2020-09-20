from scipy.sparse import csr_matrix
import numpy as np


def generate_array(data, length_list):
	indices, indptr, tmp = [], [0], 0
	max_length = max(length_list)

	for n in length_list:
		indices.extend(list(range(max_length-n, max_length)))
		tmp += n
		indptr.append(tmp)

	csr = csr_matrix((data, indices, indptr), shape=(len(length_list), max_length))
	return csr.toarray()


def build_word2idx_embedMatrix(w2vModel):
	vocab_size, embedding_size = len(w2vModel.wv.vocab), w2vModel.vector_size
	embedMatrix = np.zeros((vocab_size+1, embedding_size))
	word2idx, index = {'0': 0}, 0

	for w in w2vModel.wv.vocab.keys():
		index += 1
		embedMatrix[index] = w2vModel.wv[w]
		word2idx[w] = index

	return word2idx, embedMatrix


def softmax(x):
	x_exp = np.exp(x)
	x_sum = np.sum(x_exp, axis=1, keepdims=True)
	return x_exp/x_sum


def average_gradients(tower_grads):
	"""Calculate the average gradient for each shared variable across all towers.
	Note that this function provides a synchronization point across all towers.
	Args:
	  tower_grads: List of lists of (gradient, variable) tuples. The outer list
	    is over individual gradients. The inner list is over the gradient
	    calculation for each tower.
	Returns:
	  List of pairs of (gradient, variable) where the gradient has been averaged
	  across all towers.
	"""
	average_grads = []
	for grad_and_vars in zip(*tower_grads):
		grads = []
		for g, _ in grad_and_vars:
			expend_g = tf.expand_dims(g, 0)
			grads.append(expend_g)
		grad = tf.concat(grads, 0)
		grad = tf.reduce_mean(grad, 0)
		v = grad_and_vars[0][1]
		grad_and_var = (grad, v)
		average_grads.append(grad_and_var)
	return average_grads
