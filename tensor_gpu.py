import numpy as np
from PIL import Image
import tensorflow as tf
WIDTH = 1024
HEIGHT = 1024
CHANNEL = 128
# Creates a graph.

a = tf.constant(np.random.randn(WIDTH * HEIGHT * CHANNEL).astype(np.float32), shape=[CHANNEL, WIDTH, HEIGHT], name='a')
b = tf.constant(np.random.randn(WIDTH * HEIGHT * CHANNEL).astype(np.float32), shape=[CHANNEL, WIDTH, HEIGHT], name='b')
c = tf.matmul(a, b)
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