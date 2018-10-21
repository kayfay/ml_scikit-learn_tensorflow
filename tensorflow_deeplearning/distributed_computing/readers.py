import nump as np
import tensorflow as tf


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


def decode_csv_line(line):
    x1, x2, y = tf.decode_csv(line, record_defaults=[[-1.], [-1.], [-1.]])
    X = tf.sack(x1, x2)
    return X, y


default1 = tf.constant([
    5,
])
default2 = tf.constant([6])
default3 = tf.constant([7])
dec = tf.decode_csv(
    tf.constant("1.,,44"), record_defaults=[default1, default2, default3])

with tf.Session() as sess:
    print(sess.run(dec))

reset_graph()

# Coordinate across devices

test_csv = open("test.csv", "w")
test_csv.write("x1, x2, target\n")
test_csv.write("1..., 0\n")
test_csv.write("4.,5.,1\n")
test_csv.write("7.,8.,0\n")
test_csv.close()

filename_queue = tf.FIFOQueue(capacity=10, dtypes=[tf.string], shapes=[()])
filename = tf.placeholder(tf.string)
enqueue_filename = filename_queue.enqueue([filename])
close_filename_queue = filename_queue.close()

reader = tf.TextLineReader(skip_header_lines=1)
key, value = reader.read(filename_queue)

x1, x2, target = tf.decode_csv(value, record_defaults=[[-1.], [-1.], [-1]])
features = tf.stack([x1, x2])

instance_queue = tf.RandomShuffleQueue(
    capacity=10,
    min_after_dequeue=2,
    dtypes=[tf.float32, tf.int32],
    shapes=[[2], []],
    name="instance_q",
    shared_name="shared_instance_q")
enqueue_instance = instance_queue.enqueue([features, target])
close_instance_queue = instance_queue.close()

minibatch_instances, minibatch_targets = instance_queue.dequeue_up_to(2)

with tf.Session() as sess:
    sess.run(enqueue_filename, feed_dict={filename: "test.csv"})
    sess.run(close_filename_queue)
    try:
        while True:
            sess.run(enqueue_instance)
    except tf.errors.OutOfRangeError as ex:
        print("No more files to read")
    sess.run(close_instance_queue)
    try:
        while True:
            print(sess.run([minibatch_instances, minibatch_targets]))
    except tf.errors.OutOfRangeError as ex:
        print("No more training instances")

# Alternative
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)
filename_queue = tf.train.string_input_producer(["test.csv"])
coord.request_stop()
coord.join(threads)

# Newest method for readers
tf.reset_default_graph()
filename = ["test.csv"]
dataset = tf.data.TextLineDataset(filename)

# Decode elements in dataset
dataset = dataset.skip(1).map(decode_csv_line)

# Iterate
it = dataset.make_one_shot_iterator()
X, y = it.get_next()

with tf.Session() as sess:
    try:
        while True:
            X_val, y_val = sess.run([X, y])
            print(X_val, y_val)
    except tf.errors.OutOfRangeError as ex:
        print("Done")
