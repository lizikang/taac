import tensorflow as tf


def embedding(inputs,
	      pretrain_vector=None,
	      vocab_size=None,
	      dimension=None,
	      zero_pad=False,
	      scope='embedding'):

	with tf.variable_scope(scope):
		if pretrain_vector is not None:
			with tf.device('/cpu:0'):
				lookup_table = tf.get_variable(name='lookup_table',
							       shape=pretrain_vector.shape,
							       initializer=tf.constant_initializer(pretrain_vector),
							       trainable=False)
				outputs = tf.nn.embedding_lookup(lookup_table, inputs)
		else:
			lookup_table = tf.get_variable(name='lookup_table',
						       shape=[vocab_size, dimension],
						       initializer=tf.truncated_normal_initializer(stddev=0.01))
			if zero_pad:
				lookup_table = tf.concat([tf.zeros(shape=[1, dimension]), lookup_table[1:, :]], 0)
			outputs = tf.nn.embedding_lookup(lookup_table, inputs)
	return outputs
						

def lstm(sequences, hidden_units=None, scope='lstm'):
	with tf.variable_scope(scope):
		batch_size = tf.shape(sequences)[0]
		time_steps = sequences.get_shape()[1]
		input_dimension = sequences.get_shape()[2]
		if not hidden_units: hidden_units = input_dimension

		cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_units)
		initial_state = cell.zero_state(batch_size, tf.float32)
		state = initial_state
		outputs = []

		for t in range(time_steps):
			if t>0: tf.get_variable_scope().reuse_variables()
			output, state = cell(sequences[:,t,:], state)
			outputs.append(output)
		outputs = tf.stack(outputs, 1)

	return outputs


def layer_normalization(inputs, epsilon=1e-8, scope='ln'):
	with tf.variable_scope(scope):
		mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
		normalized = (inputs-mean) / ((variance+epsilon) ** 0.5)
		
		inputs_shape = inputs.get_shape()
		params_shape = inputs_shape[-1:]
		gamma = tf.get_variable(name='gamma', shape=params_shape, initializer=tf.ones_initializer())
		beta= tf.get_variable(name='beta', shape=params_shape, initializer=tf.zeros_initializer())
		
		outputs =  normalized * gamma + beta
	return outputs
