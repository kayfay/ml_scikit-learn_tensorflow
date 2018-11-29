"""

Quoted from  https://github.com/dstl/re3d

This dataset was the output of a project carried out by Aleph Insights and
Committed Software on behalf of the Defence Science and Technology Laboratory
(Dstl). The project aimed to create a 'gold standard' dataset that could be used
to train and validate machine learning approaches to natural language processing
(NLP); specifically focusing on entity and relationship extraction relevant to
somebody operating in the role of a defence and security intelligence analyst.
The dataset was therefore constructed using documents and structured schemas
that were relevant to the defence and security analysis domain. Information
about the dataset and the method used to build the dataset is summarised below.

"""

from collections import Counter

import numpy as np
import pandas as pd
from nltk import book
import tensorflow as tf

import random
from collections import deque

# Config


# Functions
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
            buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


# Gets and parses the words into a list.
words = book.text1[1:-1]

# Builds the dictionary
vocabulary_size = 50000

vocabulary = [("UNK", None)] + Counter(words).most_common(vocabulary_size - 1)
vocabulary = np.array([word for word, _ in vocabulary])
dictionary = {word: code for code, word in enumerate(vocabulary)}
data = np.array([dictionary.get(word, 0) for word in words])

# Test Dictionary, Examples
" ".join(words[:9]), data[:9]

#  ('Moby Dick by Herman Melville 1851 ] ETYMOLOGY .',
#   array([  305,   304,    28, 13153, 15464,  5610, 15755, 11063,     3]))

" ".join([
    vocabulary[word_index]
    for word_index in [305, 304, 28, 13153, 15464, 5610, 15755, 11063, 3]
])

# 'Moby Dick by Herman Melville 1851 ] ETYMOLOGY .'

words[34], data[34]
# ('brain', 630)

# Generate batches
data_index = 0
batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
batch, [vocabulary[word] for word in batch]

#  (array([  304,   304,    28,    28, 10739, 10739, 12768, 12768], dtype=int32),
#   ['Dick', 'Dick', 'by', 'by', 'Herman', 'Herman', 'Melville', 'Melville'])

pd.DataFrame(labels, [vocabulary[word] for word in labels[:, 0]])

#                0
#  by           28
#  Moby        300
#  Dick        304
#  Herman    10739
#  Melville  12768
#  by           28
#  Herman    10739
#  1851       7065

# Build model
batch_size = 128
embedding_size = 128  # embedding vecgtor dimension
skip_window = 1  # words to consider to the left and right
num_skips = 2  # num times to reuse an input to generate a label

# Random validation set to sample nearest neightbors
# Limit validation samples to words with low numeric ID
#   since they have high frequency
valid_size = 16  # Random set of words to evaluate similarity on
valid_window = 100  # Pick dev samples in the head of the distribution
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64  # Number of negative words to sample

learning_rate = 0.01

# Input data
train_labels = tf.placeholder(dtype=tf.int32, shape=([batch_size, 1]))
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

vocabulary_size = 50000
embedding_size = 150

init_embeds = tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0)
embeddings = tf.Variable(init_embeds)
train_inputs = tf.placeholder(tf.int32, shape=[None])
embed = tf.nn.embedding_lookup(embeddings, train_inputs)

# Construct the variables for the NCE loss
nce_weights = tf.Variable(
    tf.truncated_normal(
        [vocabulary_size, embedding_size],
        stddev=1.0 / np.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

# Compute the average NCE loss for the batch.
loss = tf.reduce_mean(
    tf.nn.nce_loss(nce_weights, nce_biases, train_labels, embed, num_sampled,
                   vocabulary_size))

# Construct the Adam Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

# Compute the cosine similarity between minibatch examples and all embeddings.
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), axis=1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embeddings_lookup(normalized_embeddings,
                                           valid_dataset)
similarity = tf.matmul(
    valid_embeddings, normalized_embeddings, transpose_b=True)

# Add variable initializer
init = tf.global_variables_initializer()

# Train the model
num_steps = 10001

with tf.Session() as session:
    init.run()

    average_loss = 0
    for step in range(num_steps):
        print("\rIteration: {}".format(step), end="\t")
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips,
                                                    skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        _, loss_val = session.run([training_op, loss], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 2000 == 0:
            if step > 0:
                average_loss /= 2000

            print("Average loss at step ", step, ": ", average_loss)
            average_loss = 0

    if step % 10000 == 0:
        sim = similarity.eval()
        for i in range(valid_size):
            valid_word = vocabulary[valid_examples[i]]
            top_k = 8
            nearest = (-sim[i, :]).argsort()[1:top_k + 1]
            log_str = "Nearest to %s" % valid_word
            for k in range(top_k):
                close_word = vocabulary[nearest[k]]
                log_str = "%s %s" % (log_str, close_word)
            print(log_str)

    final_embeddings = normalized_embeddings.eval()
