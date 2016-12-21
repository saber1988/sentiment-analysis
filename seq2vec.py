import sys
import math

import numpy as np
import pandas as pd
from six.moves import xrange
import tensorflow as tf

from util import *

train = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
test = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)
unlabeled_train = pd.read_csv("unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

train_size = train["review"].size
test_size = test["review"].size
unlabeled_train_size = unlabeled_train["review"].size

print "Read %d labeled train reviews, %d labeled test reviews, and %d unlabeled reviews\n" % \
      (train_size, test_size, unlabeled_train_size)

sentiments = train["sentiment"]

sentences = []
sentences_to_plot = []

para_plot_only = 30
word_plot_only = 500

print "Parsing sentences from training set"
for index, row in train.iterrows():
    sentences.append(paragraph_to_words(row, stem=True))
    if index < para_plot_only:
        sentences_to_plot.append(paragraph_to_words(row, stem=False, lemmatize=False))

print "Parsing sentences from unlabeled set"
for index, row in unlabeled_train.iterrows():
    sentences.append(paragraph_to_words(row, stem=True))

print "Parsing sentences from test set"
for index, row in test.iterrows():
    sentences.append(paragraph_to_words(row, stem=True))

paragraph_size = len(sentences)
print('paragraph_size: ', paragraph_size)

del train, test, unlabeled_train

# vocabulary_size = 50000
min_freq = 4

# data, count, dictionary, reverse_dictionary = build_fixed_size_dataset(sentences, vocabulary_size)
data, count, dictionary, reverse_dictionary = build_dataset_with_frequent_words(sentences, min_freq)
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:1])
print('vocabulary size', len(dictionary))

sum_len = 0
for i in data:
    sum_len += len(i.words)
print('words size', sum_len)

vocabulary_size = len(reverse_dictionary)
word_index = 0
sentence_index = 0


def generate_batch(batch_size, window):
    global word_index
    global sentence_index
    assert window % 2 == 1
    half_window = (window - 1) / 2

    batch = np.ndarray(shape=(batch_size, window - 1), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    para_labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    for i in range(batch_size):
        while word_index + window > len(data[sentence_index].words):
            word_index = 0
            sentence_index = (sentence_index + 1) % len(data)
            if sentence_index == 0:
                print ('iterate over all data, start from beginning')

        k = 0
        for j in range (window - 1):
            batch[i][j] = data[sentence_index].words[word_index + k]
            if k == half_window - 1:
                k += 2
            else:
                k += 1

        labels[i,0] = data[sentence_index].words[word_index + half_window]
        para_labels[i, 0] = sentence_index

        word_index += 1

    return batch, labels, para_labels


window = 3
batch_size = 128
embedding_size = 128  # Dimension of the embedding vector.

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 16     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.

print valid_examples

graph = tf.Graph()

with graph.as_default():

    # Input data.
    train_inputs = tf.placeholder(tf.int32,shape=[batch_size, window - 1])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    #paragraph vector place holder
    train_para_labels = tf.placeholder(tf.int32,shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        # Embedding size is calculated as shape(train_inputs) + shape(embeddings)[1:]: [200, 4] + [vocab_size - 1, embedding_size]
        embed_word = tf.nn.embedding_lookup(embeddings, train_inputs)

        para_embeddings = tf.Variable(
            tf.random_uniform([paragraph_size, embedding_size], -1.0, 1.0))
        embed_para = tf.nn.embedding_lookup(para_embeddings, train_para_labels)

        embed = tf.concat(1, [embed_word, embed_para])

        reduced_embed = tf.div(tf.reduce_sum(embed, 1), window)

        # Construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss.
    loss = tf.reduce_mean(
        tf.nn.nce_loss(nce_weights, nce_biases, reduced_embed, train_labels, num_sampled, vocabulary_size))

    # Construct the SGD optimizer using a learning rate of 1.0.
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 1.0
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               10000, 0.99, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    para_norm = tf.sqrt(tf.reduce_sum(tf.square(para_embeddings), 1, keep_dims=True))
    normalized_para_embeddings = para_embeddings / para_norm
    valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)

    # Add variable initializer.
    init = tf.initialize_all_variables()

# num_steps = 100000
num_steps = 5000000

with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    init.run()
    print("Initialized")

    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_labels, batch_para_labels = generate_batch(
            batch_size, window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels, train_para_labels: batch_para_labels}

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run())
        _, loss_val, current_learning_rate = session.run([optimizer, loss, learning_rate], feed_dict=feed_dict)
        average_loss += loss_val

        if step % 5000 == 0:
            if step > 0:
                average_loss /= 5000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print("Average loss at step ", step, ": ", average_loss, ", learning rate: ", current_learning_rate)
            sys.stdout.flush()
            average_loss = 0

        # Note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 1000000 == 0:
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1: top_k + 1]
                log_str = "Nearest to %s:" % valid_word
                for k in xrange(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log_str = "%s %s," % (log_str, close_word)
                print(log_str)
    final_embeddings = normalized_embeddings.eval()
    final_para_embeddings = normalized_para_embeddings.eval()

random_forest_classify(final_para_embeddings[0: train_size], sentiments, final_para_embeddings[-test_size:], 100)

gradient_boosting_classify(final_para_embeddings[0: train_size], sentiments, final_para_embeddings[-test_size:], 100)

svc_classify(final_para_embeddings[0: train_size], sentiments, final_para_embeddings[-test_size:], svc_c=10.0)

try:
    from sklearn.manifold import TSNE

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    low_dim_embs = tsne.fit_transform(final_para_embeddings[:para_plot_only, :])
    para_labels = [' '.join(sentences_to_plot[i].words) for i in xrange(para_plot_only)]
    plot_with_para_labels(low_dim_embs, para_labels)

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    low_dim_embs = tsne.fit_transform(final_embeddings[:word_plot_only, :])
    labels = [reverse_dictionary[i] for i in xrange(word_plot_only)]
    plot_with_word_labels(low_dim_embs, labels)

except ImportError:
    print("Please install sklearn and matplotlib to visualize embeddings.")
