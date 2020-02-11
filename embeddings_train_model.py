from __future__ import division, print_function, absolute_import

import tensorflow as tf

class EmbeddingsTrainModel(tf.keras.Model):
    
    def __init__(self, vocabulary_size, embedding_size, num_sampled=64, learning_rate=0.1):
        """Initialize parameters and build model.
        Params
        ======
            vocabulary_size (int): Vocabulary size
            embedding_size (int): Dimension of embeddings
            num_sampled (int): Number of negative samples
            learning_rate (float): Learning rate
        """
        super(EmbeddingsTrainModel, self).__init__()
        
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.num_sampled = num_sampled
        self.learning_rate = learning_rate
        
        # Create embedding variables (each line represents a word embedding vector).
        self.embedding = tf.Variable (tf.random.uniform ([vocabulary_size, embedding_size], -1.0, 1.0, dtype=tf.float32))
        # Construct NCE loss variables
        self.nce_weights = tf.Variable (tf.random.uniform ([vocabulary_size, embedding_size], -1.0, 1.0, dtype=tf.float32))
        self.nce_biases = tf.Variable (tf.zeros ([vocabulary_size], dtype=tf.float32))
        
        self.optimizer = tf.optimizers.SGD (learning_rate=self.learning_rate, momentum=0.9, nesterov=True)
    
    # Run the model
    @tf.function
    def call(self, X):
        output = tf.nn.embedding_lookup (self.embedding, X)
        return output
    
    # Evaluate the model
    @tf.function
    def evaluate (self, X):
        # Calculate the cosine similarity between the input data embedding and each embedding vector
        # X = tf.cast (X, tf.float32)
        x_embed = self.call(X)
        x_embed_norm = x_embed / tf.sqrt (tf.reduce_sum (tf.square (x_embed), 1, keepdims = True))
        embedding_norm = self.embedding / tf.sqrt (tf.reduce_sum (tf.square (self.embedding), 1, keepdims = True))
        cosine_similarity = tf.matmul (x_embed_norm, embedding_norm, transpose_b = True)
        return cosine_similarity
    
    # Custom loss
    @tf.function
    def get_loss(self, X, Y):
        x_embed = self.call(X)
        y = tf.cast (Y, tf.int64)
        loss = tf.reduce_mean (
            tf.nn.nce_loss (weights = self.nce_weights,
                            biases = self.nce_biases,
                            labels = y,
                            inputs = x_embed,
                            num_sampled = self.num_sampled,
                            num_classes = self.vocabulary_size))
        return loss
    
    # Train the model
    @tf.function
    def train(self, inputs, labels):
        # Define the GradientTape context
        with tf.GradientTape() as tape:
            # Calculate the loss
            loss = self.get_loss(inputs, labels)
        # Get the gradients
        gradients = tape.gradient (loss, [self.embedding, self.nce_weights, self.nce_biases])
        # Update the weights
        self.optimizer.apply_gradients (zip (gradients, [self.embedding, self.nce_weights, self.nce_biases]))
        return loss
    
    def get_embeddings(self):
        return self.embedding

    def get_normalized_embeddings(self):
        embedding_norm = self.embedding / tf.sqrt (tf.reduce_sum (tf.square (self.embedding), 1, keepdims = True))
        return embedding_norm
