import tensorflow as tf

def slice_tensor_and_sum(x, way=20):
    sliced = tf.split(x, num_or_size_splits=way,axis=0)
    return tf.reduce_mean(sliced, axis=1)

def reduce_tensor(x):
    return x

def reshape_input(x):
    return tf.reshape(x, [tf.shape(x)[0],tf.shape(x)[1], tf.shape(x)[2]*tf.shape(x)[3]*tf.shape(x)[4]])

def reshape_input_q(x):
    return tf.reshape(x, [-1, tf.shape(x)[-1]*tf.shape(x)[-2]*tf.shape(x)[-3]])

def reshape_query(x):
    return tf.reshape(x, [-1, tf.shape(x)[-1]])
