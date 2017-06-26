import numpy as np
from PIL import Image
import tensorflow as tf

x = np.array([[
    [[1, 1], [2, 2], [3, 3], [4, 4]],
    [[5, 5], [6, 6], [7, 7], [8, 8]],
    [[9, 9], [10, 10], [11, 11], [12, 12]]
]])
print(x.shape)
W = np.array([[
    [[1], [1]],
    [[1], [1]]
]])
print(W.shape)
x = tf.constant(x, dtype=tf.float32)
W = tf.constant(W, dtype=tf.float32)
conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

with tf.Session() as sess:
    result = sess.run(conv)
    print(result.shape)    
    print(result)