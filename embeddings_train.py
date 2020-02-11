"""
Train SGNS embeddings in TF2.

Based on the code from:
    https://github.com/tensorflow/tensorflow/blob/r0.9/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
    https://github.com/udacity/deep-learning/blob/master/embeddings/Skip-Grams-Solution.ipynb
    https://blog.csdn.net/fendouaini/article/details/102766508
    https://stackoverflow.com/a/59993811
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf
print("Tensorflow", tf.__version__)
print("Num GPUs available:", len(tf.config.experimental.list_physical_devices('GPU')))

from collections import Counter, deque
import os
import random
import time
import numpy as np

from embeddings_train_model import EmbeddingsTrainModel

# Training parameters
batch_size = 1000
num_steps = 5_000_000

# Word2Vec parameters
embedding_size = 200 # Dimension vector of Ythe embedding vector.
skip_window = 10 # How many words to consider before and after the current word
# Number of times the input is used to generate a label (the number of different output words we will pick within the span for a single word)
num_skips = 10 # for all words use 2*skip_window
num_sampled=100 # number of negative samples

# Monitor parameters
display_step = 10_000
log_step = 10_000
eval_step = 100_000

# Evaluation parameters
eval_words = ['five', 'of', 'going', 'hardware', 'american', 'britain']

# Download a small collection of Wikipedia articles
import urllib.request
import zipfile
url = 'http://mattmahoney.net/dc/text8.zip'
zip_path = 'text8.zip'
if not os.path.exists (zip_path):
    print ("Downloading the dataset... (It may take some time)")
    urllib.request.urlretrieve (url, zip_path)
    print ("Downloading done.")

# Decompress the dataset file
with zipfile.ZipFile (zip_path, "r") as f:
    text = f.read(f.namelist () [0]).decode("utf-8")

# Preprocessing the dataset
# Replace punctuation with tokens so we can use them in our model
text = text.lower()
text = text.replace('.', ' <PERIOD> ')
text = text.replace(',', ' <COMMA> ')
text = text.replace('"', ' <QUOTATION_MARK> ')
text = text.replace(';', ' <SEMICOLON> ')
text = text.replace('!', ' <EXCLAMATION_MARK> ')
text = text.replace('?', ' <QUESTION_MARK> ')
text = text.replace('(', ' <LEFT_PAREN> ')
text = text.replace(')', ' <RIGHT_PAREN> ')
text = text.replace('--', ' <HYPHENS> ')
text = text.replace('?', ' <QUESTION_MARK> ')
# text = text.replace('\n', ' <NEW_LINE> ')
text = text.replace(':', ' <COLON> ')
text_words = text.split()
# Remove all words with 5 or fewer occurences
word_counts = Counter(text_words)
words = [word for word in text_words if word_counts[word] > 5]

# Create word<->id mappings
word_counts = Counter(words)
sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)
id2word = {i: word for i, word in enumerate(sorted_words)}
word2id = {word: i for i, word in id2word.items()}

train_words = [word2id[word] for word in words]

words_count = len (train_words)
vocabulary_size = len (set (train_words))

print ("Words count: ", words_count)
print ("Unique words: ", vocabulary_size)
print ("Most common words: ", word_counts.most_common()[:10])
print ("Least common words: ", word_counts.most_common()[-10:])

data_index = 0
# Get window length (left and right of current word current word)
span = 2 * skip_window + 1  # [ skip_window target skip_window ]
context_words = [w for w in range (span) if w != skip_window]

# Generate training batch for skip-gram model
# For a different implementation of next_batch() see get_batches() in:
#   https://github.com/udacity/deep-learning/blob/master/embeddings/Skip-Grams-Solution.ipynb
def next_batch (batch_size, num_skips, skip_window):
    global data_index
    assert batch_size%num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray (shape = (batch_size), dtype = np.int32)
    labels = np.ndarray (shape = (batch_size, 1), dtype = np.int32)
    buffer = deque (maxlen = span)
    if data_index + span > words_count:
        data_index = 0
    buffer.extend (train_words [data_index: data_index + span])
    data_index += span
    for i in range (batch_size // num_skips):
        words_to_use = random.sample (context_words, num_skips) if num_skips<2*skip_window else context_words
        for j, context_word in enumerate (words_to_use):
            batch [i * num_skips + j] = buffer [skip_window]
            labels [i * num_skips + j, 0] = buffer [context_word]
        if data_index == words_count:
            buffer.extend (train_words [0:span])
            data_index = span
        else:
            buffer.append (train_words [data_index])
            data_index += 1
    # Backtrack a bit to avoid skipping words at the end of the batch
    data_index = (data_index + words_count - span) % words_count
    return batch, labels

# Words to test
x_test = np.array ([word2id [w] for w in eval_words])

# Create checkpoints regularly
checkpoint_step = 100_000
checkpoint_file = './checkpoints/embeddings-{epoch:07d}.ckpt'

import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Initialize logs for tensorboard
tensorboard_dir = './tensorboard'
train_log_dir = tensorboard_dir + '/gradient-tape/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
train_summary_writer.set_as_default()

# Create model
model = EmbeddingsTrainModel (vocabulary_size, embedding_size, num_sampled)

total_loss = 0
step0 = 0
start = time.time()

# Train for a given number of steps
for step in range (1, num_steps + 1):
    batch_x, batch_y = next_batch (batch_size, num_skips, skip_window)
    loss=model.train (batch_x, batch_y)
    total_loss += loss
    
    if step%checkpoint_step == 0 or step == num_steps:
        # Create checkpoint
        model.save_weights(checkpoint_file.format(epoch=step))
    
    if step%log_step == 0  or step == 1 or step == num_steps:
        # Save tensorboard log
        tf.summary.scalar('Loss', loss, step=step)
    
    if step%display_step == 0 or step == 1 or step == num_steps:
        end = time.time()
        avg_loss = total_loss / (step - step0)
        tf.summary.scalar('Avg loss', loss, step=step)
        print("Step {} / {}, ".format(step, num_steps),
              "Loss: {:.2f}, ".format(loss),
              "Avg loss: {:.2f}, ".format(avg_loss),
              "{:.1f} sec/batch".format(end-start))
        start = time.time()
        total_loss = 0
        step0 = step
    
    # Evaluate
    if step%eval_step == 0 or step == 1 or step == num_steps:
        print ("Evaluating...")
        sim = model.evaluate (x_test).numpy ()
        for i in range (len (eval_words)):
            top_k = 8 # Number of most similar words
            nearest = (-sim [i,:]).argsort () [1:top_k + 1]
            log_str = '"%s" nearest neighbors:' % eval_words [i]
            for k in range (top_k):
                log_str = '%s %s,' % (log_str, id2word [nearest [k]])
            print (log_str)

train_summary_writer.flush()


# Save the embeddings for tensorboard visualization
import embeddings_tensorboard as etb
embeddings=model.get_embeddings()
etb.save(embeddings, id2word, log_dir=tensorboard_dir)


# Create a t-SNE plot
from matplotlib import pylab
from sklearn.manifold import TSNE

num_points = 200
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
normalized_embeddings=model.get_normalized_embeddings().numpy()
two_d_embeddings = tsne.fit_transform(normalized_embeddings[1:num_points+1, :])

def plot(embeddings, labels):
  assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
  pylab.figure(figsize=(18, 18))  # in inches
  for i, label in enumerate(labels):
    x, y = embeddings[i,:]
    pylab.scatter(x, y)
    pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
  pylab.show()

words = [id2word[i] for i in range(1, num_points+1)]
plot(two_d_embeddings, words)
