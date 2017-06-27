import numpy as np
from PIL import Image
import tensorflow as tf
WIDTH = 3
HEIGHT = 3
CHANNEL = 1
# Creates a graph.

a = tf.constant(np.array([9,9,3,4,4,4,7,8,9]).astype(np.float32), shape=[1,WIDTH, HEIGHT, CHANNEL], name='a')
b = tf.constant(np.array([1,1,-1,-1]).astype(np.float32), shape=[2, 2, 1, 1], name='b')
c = tf.nn.conv2d(a, b, strides=[1, 1, 1, 1], padding="VALID")
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))
# cpu
# real    0m27.890s
# user    0m16.432s
# sys     0m5.436s
# single gpu
# real    0m31.685s
# user    0m16.664s
# sys     0m4.440s