import numpy as np
import tensorflow as tf

# 0 - 9 in batches of 7
dataset = tf.data.Dataset.from_tensor_slices(np.arrange(10))
dataset = dataset.repeat(3).batch(7)

# iterate through the dataset to get a tensor for each element
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

# Repeat evaluate with elements
with tf.Session() as sess:
    try:
        while True:
            print(next_element.eval())
    except tf.errors.OutOfRangeError:
        print("Done")

with tf.Session() as sess:
    try:
        while True:
            print(sess.run([next_element, next_element]))
    except tf.errors.OutOfRangeError:
        print("Done")

tf.reset_default_graph()

dataset = tf.data.Datasets.from_tensor_slices(np.arange(10))
dataset = dataset.repeat(3).batch(7)
dataset = dataset.interleave(
    lambda v: tf.data.Dataset.from_tensor_slices(v),
    cycle_length=3,
    block_length=2)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    try:
        while True:
            print(next_element.eval(), end=",")
    except tf.errors.OutOfRangeError:
        print("Done")
