"""

For details see https://github.com/tensorflow/tensorboard/issues/2471#issuecomment-580423961

"""

import io
import os
import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector

LOG_DIR = './tensorboard'  # Tensorboard log dir
META_DATA_FNAME = 'meta.tsv'  # Labels will be stored here
EMBEDDINGS_TENSOR_NAME = 'embeddings'

def register_embeddings(embeddings_name=EMBEDDINGS_TENSOR_NAME, meta_data_fname=META_DATA_FNAME, log_dir=LOG_DIR):
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embeddings_name
    embedding.metadata_path = meta_data_fname
    projector.visualize_embeddings(log_dir, config)

def save_embeddings(embeddings_tensor, id2word, embeddings_name=EMBEDDINGS_TENSOR_NAME, log_dir=LOG_DIR, meta_data_fname=META_DATA_FNAME):
    embeddings_path = os.path.join(log_dir, embeddings_name + '.ckpt')
    saver = tf.compat.v1.train.Saver([embeddings_tensor])  # Must pass list or dict
    saver.save(sess=None, save_path=embeddings_path)
    out_meta = io.open(os.path.join(log_dir, meta_data_fname), 'w', encoding='utf-8')
    for id, embedding in enumerate(embeddings_tensor.numpy()):
        word = id2word[id]
        out_meta.write(word + "\n")
    out_meta.close()

def save(embeddings_tensor, id2word, embeddings_name=EMBEDDINGS_TENSOR_NAME, log_dir=LOG_DIR, meta_data_fname=META_DATA_FNAME):
    register_embeddings(embeddings_name, meta_data_fname, log_dir)
    save_embeddings(embeddings_tensor, id2word, embeddings_name, log_dir, meta_data_fname)
