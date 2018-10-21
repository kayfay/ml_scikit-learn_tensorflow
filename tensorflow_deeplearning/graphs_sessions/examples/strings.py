"""
Strings in tensorflow
"""

import tensorflow as tf
import numpy as np

text = np.array("eggs and ham".split())
text_tensor = tf.constant(text)

with tf.Session() as sess:
    print(text_tensor.eval())
