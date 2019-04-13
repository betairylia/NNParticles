import tensorflow as tf
import numpy as np
from timeit import default_timer as timer
import os

bs = 16
size = 512
ch = 3

loops = 100

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

def custom_getter(getter, name, shape=None, dtype=tf.float32, *args, **kwargs):

    if dtype is tf.float16:

        var = getter(name, shape, tf.float32, *args, **kwargs)
        return tf.cast(var, dtype = dtype, name = name + '_cast')

    else:

        return getter(name, shape, dtype, *args, **kwargs)

with tf.variable_scope('net', custom_getter = custom_getter):
    Xs_f32 = tf.placeholder(tf.float32, [bs, size, size, ch])
    Kr_f32 = tf.get_variable('W_f32', dtype = tf.float32, shape = [5, 5, ch, 64], initializer = tf.initializers.random_normal())
    Ys_f32 = tf.nn.conv2d(Xs_f32, Kr_f32, [1, 1, 1, 1], 'SAME')

    Xs_f16 = tf.placeholder(tf.float16, [bs, size, size, ch])
    Kr_f16 = tf.get_variable('W_f16', dtype = tf.float16, shape = [5, 5, ch, 64], initializer = tf.initializers.random_normal())
    Ys_f16 = tf.nn.conv2d(Xs_f16, Kr_f16, [1, 1, 1, 1], 'SAME')

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)

sess.run(tf.global_variables_initializer())

print("Generating feed...")
feed = np.random.normal(size = [bs, size, size, ch])

print("warming up...")
for i in range(10):
    _ = sess.run([Ys_f32, Ys_f16], feed_dict = {Xs_f32: feed, Xs_f16: feed})

# FP32
print("Testing FP32 performance ...")
start = timer()

for i in range(loops):
    _ = sess.run(Ys_f32, feed_dict = {Xs_f32: feed})

end = timer()
print("FP32: %5.4f sec" % (end - start))

# FP16
print("Testing FP16 performance ...")
start = timer()

for i in range(loops):
    _ = sess.run(Ys_f16, feed_dict = {Xs_f16: feed})

end = timer()
print("FP16: %5.4f sec" % (end - start))

