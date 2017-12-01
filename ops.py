import tensorflow as tf
import numpy as np

def new_weights(shape):
    """
    Returns new TensorFlow weights in the given shape with normally distributed random initializations
    """
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
    #return tf.get_variable('W', shape, initializer=tf.contrib.layers.xavier_initializer(uniform=False,dtype=tf.float32))


def new_biases(length):
    """
    Returns new TensorFlow biases with the given length with initialization 0.05
    """
    return tf.Variable(tf.constant(0.0, shape=[length]),dtype=tf.float32)
    #return tf.get_variable('b', shape, initializer=tf.constant_initializer(0.))


def new_conv_layer(input,              # The previous layer
                   num_input_channels, # Num. channels in prev. layer
                   filter_size,        # Width and height of each filter
                   num_filters,        # Number of filters
                   use_pooling=True):  # Use 2x2 max-pooling
    """
    Helper function for creating a new convolutional layer.
    """

    # Shape of the filter-weights for the convolution.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights (filters) with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    layer += biases   # Adding biases to the results of the convolution.

    layer = tf.nn.relu(layer)  # Rectified Linear Unit (ReLU).

    if use_pooling:  # Use pooling to down-sample the image resolution?
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')  # This is 2x2 max-pooling

    #layer = tf.nn.lrn(layer, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    
    return layer, weights


def flatten_layer(layer):
    """
    Reduces a 4d tensor to a 2d tensor to be able to be fed into a fully connected layer.
    """

    layer_shape = layer.get_shape()  # Shape of the input layer

    num_features = layer_shape[1:4].num_elements()  # img_height * img_width * num_channels
    
    # Reshape the layer to [num_images, num_features]
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now: [num_images, img_height * img_width * num_channels]
    return layer_flat, num_features


def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?
    """
    Helper function for creating a new fully connected layer.
    """

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)        

    return layer


def weight_variable(shape):
    #return tf.get_variable('W', shape, initializer=tf.random_normal_initializer(0., 0.02))
   	return tf.get_variable('W', shape, initializer=tf.contrib.layers.xavier_initializer(uniform=False,dtype=tf.float32))

def bias_variable(shape,value=0.0):
    return tf.get_variable('b', shape, initializer=tf.constant_initializer(value))

def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def conv2d(x, shape, name, bias=0.0, stride=1):
    with tf.variable_scope(name):
        W = weight_variable(shape)
        h = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
        b = bias_variable([shape[-1]],bias)
        h = h + b
        return lrelu(h),W

def pool_layer(inputs, size, stride, name):
	with tf.variable_scope(name) as scope:
		outputs = tf.nn.max_pool(inputs,
			ksize=[1,size,size,1],
			strides=[1,stride,stride,1],
			padding='SAME',
			name=name)
		return outputs

def lrn_layer(inputs, name):
        depth_radius=4
        with tf.variable_scope(name) as scope:
            outputs = tf.nn.lrn(inputs,
                                depth_radius=depth_radius,
                                bias=1.0,
                                alpha=0.001/9.0,
                                beta=0.75,
                                name=scope.name)

        input_size = np.prod(inputs.get_shape().as_list()[1:])

        # Evaluate layer operations
        # First, cost to calculate normalizer (using local input squares sum)
        # norm = (1 + alpha/n*sum[n](local-input*local_input)
        local_flops = 1 + 1 + 1 + 2*depth_radius*depth_radius
        # Then cost to divide each input by the normalizer
        num_flops = (local_flops + 1)*input_size
        return outputs

def linear(x, shape, name, relu=True, bias=False):
	with tf.variable_scope(name):
		W = weight_variable(shape)
		h = tf.matmul(x, W)
		if bias:
			b = bias_variable([shape[-1]])
			h = h + b
		if relu:
			h = tf.nn.relu(h)
		return h

def _get_weights_var(name, shape, decay=False):
        # Declare an initializer for this variable
        initializer = tf.contrib.layers.xavier_initializer(uniform=False,dtype=tf.float32)
        # Declare variable (it is trainable by default)
        var = tf.get_variable(name=name,
                              shape=shape,
                              initializer=initializer,
                              dtype=tf.float32)
        if decay:
            # We apply a weight decay to this tensor var that is equal to the
            # model weight decay divided by the tensor size
            weight_decay = 1e2
            for x in shape:
                weight_decay /= x
            # Weight loss is L2 loss multiplied by weight decay
            weight_loss = tf.multiply(tf.nn.l2_loss(var),
                                      weight_decay,
                                      name='weight_loss')
            # Add weight loss for this variable to the global losses collection
            tf.add_to_collection('losses', weight_loss)
        return var

def fc_layer(inputs, neurons, decay, name, bias=0.0,relu=True, bn=False):
        with tf.variable_scope(name) as scope:
            if len(inputs.get_shape().as_list()) > 2:
                # We need to reshape inputs:
                #   [ batch size , w, h, c ] -> [ batch size, w x h x c ]
                # Batch size is a dynamic value, but w, h and c are static and
                # can be used to specifiy the reshape operation
                dim = np.prod(inputs.get_shape().as_list()[1:])
                reshaped = tf.reshape(inputs, shape=[-1, dim], name='reshaped')
            else:
                # No need to reshape inputs
                reshaped = inputs
            dim = reshaped.get_shape().as_list()[1]
            weights = _get_weights_var('weights',
                                            shape=[dim,neurons],
                                            decay=decay)

            biases = tf.get_variable('biases',
                                    shape=[neurons],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(bias))
            x = tf.add(tf.matmul(reshaped, weights), biases)
            if bn:
                x = tf.layers.batch_normalization(x, training=self.training)
            if relu:
                #outputs = tf.nn.relu(x)
                outputs=lrelu(x)
            else:
                outputs = x

        # Evaluate layer operations
        # Matrix multiplication plus bias
        num_flops = (2 * dim + 1) * neurons
        # ReLU
        if relu:
            num_flops += 2 * neurons
        return outputs