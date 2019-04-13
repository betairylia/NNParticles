import tensorflow as tf

@tf.custom_gradient
def mod_assign(x):
    
