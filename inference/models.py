import tensorflow as tf


def normalize(specs):
    with tf.variable_scope('normalize'):
        MAX_VAL = 32767
        specs = specs / MAX_VAL
        specs = (1 + specs) / 2
        specs = tf.transpose(specs, [0, 2, 1, 3])
    return specs


l2_reg1 = tf.contrib.layers.l2_regularizer(5e-3)
l2_reg2 = tf.contrib.layers.l2_regularizer(1e-2)

kernel_initializer = tf.initializers.variance_scaling()
bias_initializer = tf.initializers.constant(value=0.1)


def onset_and_frames(inputs, mode):
    inputs = normalize(inputs)

    with tf.variable_scope('pitch'):
        net1 = acoustic_model(inputs, mode)
        net1 = lstm_layer(net1, 512, mode)
        logits1 = fc_layer(net1, 88)

    with tf.variable_scope('frame'):
        net2 = acoustic_model(inputs, mode)
        logits2 = fc_layer(net2, 88)
        logits = tf.concat([logits1, logits2], axis=2)  #
        net2 = lstm_layer(logits, 128, mode)
        logits2 = fc_layer(net2, 88)

    return logits1, logits2


def acoustic_model(inputs, mode):

    with tf.variable_scope('conv_net'):

        net = tf.layers.conv2d(inputs,
                               8, [3, 5],
                               padding='valid',
                               kernel_initializer=kernel_initializer,
                               bias_initializer=bias_initializer,
                               kernel_regularizer=l2_reg1)
        net = tf.layers.batch_normalization(
            net,
            momentum=0.9,
            epsilon=1e-7,
            training=(mode == tf.estimator.ModeKeys.TRAIN))
        net = tf.nn.relu(net)

        net = tf.layers.conv2d(net,
                               16, [3, 5],
                               padding='same',
                               kernel_initializer=kernel_initializer,
                               bias_initializer=bias_initializer,
                               kernel_regularizer=l2_reg1)
        net = tf.layers.batch_normalization(
            net,
            momentum=0.9,
            epsilon=1e-7,
            training=(mode == tf.estimator.ModeKeys.TRAIN))
        net = tf.nn.relu(net)
        net = tf.layers.max_pooling2d(net, [2, 2], [1, 2], padding='same')

        net = tf.layers.conv2d(net,
                               32, [3, 5],
                               padding='valid',
                               kernel_initializer=kernel_initializer,
                               bias_initializer=bias_initializer,
                               kernel_regularizer=l2_reg1)
        net = tf.layers.batch_normalization(
            net,
            momentum=0.9,
            epsilon=1e-7,
            training=(mode == tf.estimator.ModeKeys.TRAIN))
        net = tf.nn.relu(net)

        net = tf.layers.conv2d(net,
                               64, [3, 5],
                               padding='valid',
                               kernel_initializer=kernel_initializer,
                               bias_initializer=bias_initializer,
                               kernel_regularizer=l2_reg1)
        net = tf.layers.batch_normalization(
            net,
            momentum=0.9,
            epsilon=1e-7,
            training=(mode == tf.estimator.ModeKeys.TRAIN))
        net = tf.nn.relu(net)
        net = tf.layers.max_pooling2d(net, [2, 2], [1, 2], padding='same')

        net = tf.layers.conv2d(net,
                               128, [3, 5],
                               padding='valid',
                               kernel_initializer=kernel_initializer,
                               bias_initializer=bias_initializer,
                               kernel_regularizer=l2_reg1)
        net = tf.layers.batch_normalization(
            net,
            momentum=0.9,
            epsilon=1e-7,
            training=(mode == tf.estimator.ModeKeys.TRAIN))
        net = tf.nn.relu(net)
        net = tf.layers.max_pooling2d(net, [2, 2], [1, 2], padding='same')

        # [B, time_step, spec_bins, channels]
        shapes = net.get_shape().as_list()

        net = tf.reshape(net, [-1, shapes[1], shapes[2] * shapes[3]])
        net = tf.layers.dropout(net,
                                0.5,
                                training=(mode == tf.estimator.ModeKeys.TRAIN))
        net = tf.layers.dense(net,
                              1024,
                              activation=tf.nn.relu,
                              kernel_initializer=kernel_initializer,
                              bias_initializer=bias_initializer,
                              kernel_regularizer=l2_reg2)
        # [batch_size, time_len, 1024]
        net = tf.layers.dropout(net,
                                0.5,
                                training=(mode == tf.estimator.ModeKeys.TRAIN))

    return net


def lstm_layer(net, units, mode):
    with tf.variable_scope('lstm'):
        cells_fw = [
            tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(units)
            for _ in range(1)
        ]
        cells_bw = [
            tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(units)
            for _ in range(1)
        ]
        (net, unused_state_f,
         unused_state_b) = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
             cells_fw,
             cells_bw,
             net,
             dtype=tf.float32,
             sequence_length=None,
             parallel_iterations=1)
        net = tf.layers.dropout(net,
                                0.5,
                                training=(mode == tf.estimator.ModeKeys.TRAIN))
    return net


def fc_layer(net, hiddens):
    return tf.layers.dense(net,
                           hiddens,
                           kernel_initializer=kernel_initializer,
                           bias_initializer=bias_initializer,
                           kernel_regularizer=l2_reg2)


def flatten(labels, logits):
    with tf.variable_scope('flatten_logits'):
        pitch_labels, frame_labels = labels
        pitch_logits, frame_logits = logits

        pitch_labels = tf.cast(tf.reshape(pitch_labels, [-1, 88]), tf.float32)
        frame_labels = tf.cast(tf.reshape(frame_labels, [-1, 88]), tf.float32)

        pitch_logits = tf.cast(tf.reshape(pitch_logits, [-1, 88]), tf.float32)
        frame_logits = tf.cast(tf.reshape(frame_logits, [-1, 88]), tf.float32)
    return (pitch_labels, frame_labels), (pitch_logits, frame_logits)


def onset_model(inputs, mode):

    l2_reg1 = tf.contrib.layers.l2_regularizer(0.0)
    l2_reg2 = tf.contrib.layers.l2_regularizer(1e-4)
    kernel_initializer = tf.initializers.variance_scaling()
    bias_initializer = tf.initializers.constant(value=0.1)

    inputs = normalize(inputs)
    inputs = tf.transpose(inputs, [0, 2, 1, 3])

    with tf.variable_scope('conv_block'):

        net = tf.layers.conv2d(inputs,
                               10, [15, 3],
                               padding='valid',
                               kernel_initializer=kernel_initializer,
                               bias_initializer=bias_initializer,
                               kernel_regularizer=l2_reg1)
        net = tf.layers.batch_normalization(
            net,
            momentum=0.9,
            epsilon=1e-7,
            training=(mode == tf.estimator.ModeKeys.TRAIN))
        net = tf.nn.relu(net)
        net = tf.layers.max_pooling2d(net, [2, 2], [2, 1], padding='same')

        net = tf.layers.conv2d(net,
                               20, [13, 3],
                               padding='valid',
                               kernel_initializer=kernel_initializer,
                               bias_initializer=bias_initializer,
                               kernel_regularizer=l2_reg1)
        net = tf.layers.batch_normalization(
            net,
            momentum=0.9,
            epsilon=1e-7,
            training=(mode == tf.estimator.ModeKeys.TRAIN))
        net = tf.nn.relu(net)
        net = tf.layers.max_pooling2d(net, [2, 2], [2, 1], padding='same')

    with tf.variable_scope('fc'):
        net = tf.transpose(net, [0, 2, 1, 3])
        net = tf.reshape(net, [-1, 7, 80 * 20])
        net = tf.layers.dense(net,
                              256,
                              activation=tf.nn.relu,
                              kernel_initializer=kernel_initializer,
                              bias_initializer=bias_initializer,
                              kernel_regularizer=l2_reg2)

    with tf.variable_scope('lstm'):
        cells_fw = [
            tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(128) for _ in range(1)
        ]
        cells_bw = [
            tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(128) for _ in range(1)
        ]
        (net, unused_state_f,
         unused_state_b) = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
             cells_fw,
             cells_bw,
             net,
             dtype=tf.float32,
             sequence_length=None,
             parallel_iterations=1)

    logits = tf.layers.dense(net,
                             1,
                             kernel_initializer=kernel_initializer,
                             bias_initializer=bias_initializer,
                             kernel_regularizer=l2_reg2)

    return logits
