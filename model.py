import tensorflow as tf
from layers import *


class Model():
	def __init__(self, args, data, x, y):
		# initialize parameters
		self.inputs = tf.placeholder(tf.int32, shape=[None, args.window_size, len(data.features)])
		self.labels = tf.placeholder(tf.int32, shape=[None, ])
		self.is_training = tf.placeholder(tf.bool, shape=[])
		self.dropout_prob = tf.cond(self.is_training, lambda: args.dropout_prob, lambda: 0.0)

		# embedding layer
		with tf.variable_scope("embedding_layer"):
			emb_list = []
			for i, fea in enumerate(data.features):
				emb_list.append(embedding(self.batch_inputs[:,:,i],
							  pretrain_vector=data.__dict__[fea+'_embedMatrix'],
							  vocab_size=data.__dict__[fea+'_size'],
							  dimension=args.dimension,
							  scope=fea+'_embs'))
			sequences = tf.concat(emb_list, -1)

		# lstm layer
		cell_fw = tf.nn.rnn_cell.LSTMCell(args.hidden_units)
		cell_bw = tf.nn.rnn_cell.LSTMCell(args.hidden_units)
		outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, sequences, dtype=tf.float32)

		outputs_fw, outputs_bw = outputs
		outputs = outputs_fw + outputs_bw
		outputs = lstm(outputs)

		# prediction layer
		with tf.variable_scope("prediction_layer"):
			pred_in = tf.reduce_max(outputs, 1)
			pred_in = tf.nn.dropout(pred_in, 1.0-self.dropout_prob)

			self.age_logits = tf.layers.dense(pred_in, 10)
			self.age_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.batch_labels%10, logits=self.age_logits)
			self.age_loss = tf.reduce_mean(self.age_loss)

			self.gender_logits = tf.layers.dense(pred_in, 2)
			self.gender_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.batch_labels//10, logits=self.gender_logits)
			self.gender_loss = tf.reduce_mean(self.gender_loss)

			decay_rate = 0.95
			global_step = tf.Variable(0, trainable=False)
			decay_steps = data.train_inputs.shape[0] // args.batch_size + 1
			learn_rate = tf.train.exponential_decay(args.learn_rate, global_step, decay_steps, decay_rate, staircase=True)

			self.loss = self.age_loss * args.age_weight + self.gender_loss * (1-args.age_weight)
			self.train_op = tf.train.AdamOptimizer(learn_rate).minimize(self.loss, global_step=global_step)
