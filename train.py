import os
import pickle
import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

# directiories and paths
DATA_DIR = './data'
CHECKPOINTS_DIR = './data/checkpoints'
SUMMARIES_DIR = './data/summaries'
PROCESSED_DIR = './data/processed'
LABELS_PATH = './data/processed/labels.tsv'
FINAL_EMBED_PATH = './data/processed/model.ckpt'

# hyper parameters
BATCH_SIZE = 128
EMBED_SIZE = 32 # dimension of the word embedding vectors
SKIP_WINDOW = 6 # the context window
NUM_SAMPLED = 64    # Number of negative examples to sample.
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 5000
SKIP_STEP = 5000

def get_vocabulary_size(path):
    """ Get the vocabulary size. """
    vocabulary_path = os.path.join(path, 'vocabulary')
    with open(vocabulary_path, 'r') as f:
        vocabulary_size = int(f.readline())
    return vocabulary_size

def get_vocabulary(path):
    """ Get all the information about the vocabulary from file. """
    vocabulary_path = os.path.join(path, 'vocabulary')
    with open(vocabulary_path, 'r') as f:
        vocabulary_size = int(f.readline())
        count = []
        dictionary = dict()
        reverse_dictionary = []
        for i, line in enumerate(f):
            word, n = line.split()
            count.append((word, n))
            dictionary[word] = i
            reverse_dictionary.append(word)
    return vocabulary_size, count, dictionary, reverse_dictionary

def get_data(path):
    """ Get the data in id form from file. """
    ids_path = os.path.join(path, 'data_as_ids')
    with open(ids_path) as f:
        data = [int(x) for x in f.read().split()]
    return data   

def generate_sample(index_words, context_window_size):
    """ Form training pairs according to the skip-gram model. """
    N = len(index_words)
    index = 1   
    while True:
        center = index_words[index]
        context = random.randint(1, context_window_size)
        
        # get a random target before the center word
        target = index_words[max(0, index - context)]
        yield center, target
        
        # get a random target after the center word
        target = index_words[min(index + context, N-1)]
        yield center, target
        
        index = (index % (N-2)) + 1

def get_batch(iterator, batch_size):
    """ Group a numerical stream into batches and yield them as Numpy arrays. """
    while True:
        center_batch = np.zeros(batch_size, dtype=np.int32)
        target_batch = np.zeros([batch_size, 1])
        for index in range(batch_size):
            center_batch[index], target_batch[index] = next(iterator)
        yield center_batch, target_batch

class SkipGramModel:
    """ Build the graph for word2vec model """
    def __init__(self, vocab_size, embed_size, batch_size, num_sampled, learning_rate):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.lr = learning_rate
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    def _create_placeholders(self):
        """ Step 1: define the placeholders for input and output """
        with tf.name_scope("data"):
            self.center_words = tf.placeholder(tf.int32, shape=[self.batch_size], name='center_words')
            self.target_words = tf.placeholder(tf.int32, shape=[self.batch_size, 1], name='target_words')

    def _create_embedding(self):
        """ Step 2: define weights. In word2vec, it's actually the weights that we care about """
        # Assemble this part of the graph on the CPU. You can change it to GPU if you have GPU
        with tf.device('/cpu:0'):
            with tf.name_scope("embed"):
                self.embed_matrix = tf.Variable(tf.random_uniform([self.vocab_size, 
                                                                    self.embed_size], -1.0, 1.0), 
                                                                    name='embed_matrix')

    def _create_loss(self):
        """ Step 3 + 4: define the model + the loss function """
        with tf.device('/cpu:0'):
            with tf.name_scope("loss"):
                # Step 3: define the inference
                embed = tf.nn.embedding_lookup(self.embed_matrix, self.center_words, name='embed')

                # Step 4: define loss function
                # construct variables for NCE loss
                nce_weight = tf.Variable(tf.truncated_normal([self.vocab_size, self.embed_size],
                                                            stddev=1.0 / (self.embed_size ** 0.5)), 
                                                            name='nce_weight')
                nce_bias = tf.Variable(tf.zeros([self.vocab_size]), name='nce_bias')

                # define loss function to be NCE loss function
                self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight, 
                                                    biases=nce_bias, 
                                                    labels=self.target_words, 
                                                    inputs=embed, 
                                                    num_sampled=self.num_sampled, 
                                                    num_classes=self.vocab_size), name='loss')
    def _create_optimizer(self):
        """ Step 5: define optimizer """
        with tf.device('/cpu:0'):
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss, 
                                                              global_step=self.global_step)

    def _create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("histogram_loss", self.loss)
            # because you have several summaries, we should merge them all
            # into one op to make it easier to manage
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        """ Build the graph for our model """
        self._create_placeholders()
        self._create_embedding()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()

def train_model(model, batch_gen, num_train_steps):
    saver = tf.train.Saver() # defaults to saving all variables - in this case embed_matrix, nce_weight, nce_bias

    initial_step = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(CHECKPOINTS_DIR)
        # if that checkpoint exists, restore from checkpoint
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        # training loop    
        total_loss = 0.0 # we use this to calculate late average loss in the last SKIP_STEP steps
        writer = tf.summary.FileWriter(SUMMARIES_DIR, sess.graph)
        initial_step = model.global_step.eval()
        for index in range(initial_step, initial_step + num_train_steps):
            centers, targets = next(batch_gen)
            feed_dict={model.center_words: centers, model.target_words: targets}
            loss_batch, _, summary = sess.run([model.loss, model.optimizer, model.summary_op], 
                                              feed_dict=feed_dict)
            writer.add_summary(summary, global_step=index)
            total_loss += loss_batch
            if (index + 1) % SKIP_STEP == 0:
                print('Average loss at step {}: {:5.1f}'.format(index, total_loss / SKIP_STEP))
                total_loss = 0.0
                saver.save(sess, os.path.join(CHECKPOINTS_DIR, 'checkpoint'), index)
        
        # code to visualize the embeddings
        final_embed_matrix = sess.run(model.embed_matrix)
        
        # it has to variable. constants don't work here. you can't reuse model.embed_matrix
        embedding_var = tf.Variable(final_embed_matrix[:model.vocab_size], name='embedding')
        sess.run(embedding_var.initializer)

        config = projector.ProjectorConfig()
        summary_writer = tf.summary.FileWriter(PROCESSED_DIR)

        # add embedding to the config file
        embedding = config.embeddings.add()
        embedding.tensor_name = embedding_var.name
        
        # link this tensor to its metadata file, in this case the first 500 words of vocab
        embedding.metadata_path = LABELS_PATH

        # saves a configuration file that TensorBoard will read during startup.
        projector.visualize_embeddings(summary_writer, config)
        saver_embed = tf.train.Saver([embedding_var])
        saver_embed.save(sess, FINAL_EMBED_PATH, 1)


def main():
    # get vocabulary info
    vocabulary_size = get_vocabulary_size(DATA_DIR)

    # data in numerical id form
    data = get_data(DATA_DIR)

    # create batch generator from data
    single_gen = generate_sample(data, SKIP_WINDOW)
    batch_gen  = get_batch(single_gen, BATCH_SIZE)

    # build the model
    model = SkipGramModel(vocabulary_size, EMBED_SIZE, BATCH_SIZE, NUM_SAMPLED, LEARNING_RATE)
    model.build_graph()

    # create directories if they do not exist
    for path in [CHECKPOINTS_DIR, PROCESSED_DIR, SUMMARIES_DIR]:
        if not os.path.exists(path):
            os.mkdir(path)
    # create labels for tensorboard visualization
    if not os.path.exists(LABELS_PATH):
        _, count, _, _ = get_vocabulary(DATA_DIR)
        with open(LABELS_PATH, 'w') as f:
            f.write('\n'.join([word for word, _ in count])) 

    # train the model
    train_model(model, batch_gen, NUM_TRAIN_STEPS)

if __name__ == '__main__':
    main()



