import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from preprocess import get_reachabilities

class Node2Vec(tf.keras.Model):
    def __init__(self, vocab_size):
        """
        The Model class predicts the next words in a sequence.

        :param vocab_size: The number of unique words in the data
        """

        super(Model, self).__init__()



    def call(self, inputs, initial_state):
        """
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
        - You must use an LSTM or GRU as the next layer.

        :param inputs: word ids of shape (batch_size, window_size)
        :param initial_state: 2-d array of shape (batch_size, rnn_size) as a tensor
        :return: the batch element probabilities as a tensor, a final_state (Note 1: If you use an LSTM, the final_state will be the last two RNN outputs,
        Note 2: We only need to use the initial state during generation)
        using LSTM and only the probabilites as a tensor and a final_state as a tensor when using GRU
        """
        #TODO: Fill in


        #embedding layer output has to be (batch_size, window_size, embedding_size)
        #embeddings = []
        '''
        embeddings =[]
        for i in range(self.window_size):
            embeddings.append(tf.nn.embedding_lookup(self.E,inputs[:,i]))
            #print(inputs.shape)
            #each entry is batch size, embedding size
            #print(tf.nn.embedding_lookup(self.E,inputs[:,i]).shape)

        embeddings = tf.concat(embeddings, axis = -1)
        embedding = tf.reshape(embeddings, (inputs.shape[0], self.window_size, self.embedding_size))

        '''

        return probs, final_state

    def loss(self, probs, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction

        NOTE: You have to use np.reduce_mean and not np.reduce_sum when calculating your loss

        :param logits: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """

        #TODO: Fill in
        #We recommend using tf.keras.losses.sparse_categorical_crossentropy
        #https://www.tensorflow.org/api_docs/python/tf/keras/losses/sparse_categorical_crossentropy
        losses = tf.keras.losses.sparse_categorical_crossentropy(labels, probs)
        return tf.reduce_mean(losses)


class Predictor(tf.keras.Model):
    def __init__(self, vocab_size):
        """
        The Model class predicts the next words in a sequence.

        :param vocab_size: The number of unique words in the data
        """

        super(Model, self).__init__()



    def call(self, inputs, initial_state):
        """
        - You must use an embedding layer as the first layer of your network (i.e. tf.nn.embedding_lookup)
        - You must use an LSTM or GRU as the next layer.

        :param inputs: word ids of shape (batch_size, window_size)
        :param initial_state: 2-d array of shape (batch_size, rnn_size) as a tensor
        :return: the batch element probabilities as a tensor, a final_state (Note 1: If you use an LSTM, the final_state will be the last two RNN outputs,
        Note 2: We only need to use the initial state during generation)
        using LSTM and only the probabilites as a tensor and a final_state as a tensor when using GRU
        """
        #TODO: Fill in


        #embedding layer output has to be (batch_size, window_size, embedding_size)
        #embeddings = []
        '''
        embeddings =[]
        for i in range(self.window_size):
            embeddings.append(tf.nn.embedding_lookup(self.E,inputs[:,i]))
            #print(inputs.shape)
            #each entry is batch size, embedding size
            #print(tf.nn.embedding_lookup(self.E,inputs[:,i]).shape)

        embeddings = tf.concat(embeddings, axis = -1)
        embedding = tf.reshape(embeddings, (inputs.shape[0], self.window_size, self.embedding_size))

        '''

        return probs, final_state

    def loss(self, probs, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction

        NOTE: You have to use np.reduce_mean and not np.reduce_sum when calculating your loss

        :param logits: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """

        #TODO: Fill in
        #We recommend using tf.keras.losses.sparse_categorical_crossentropy
        #https://www.tensorflow.org/api_docs/python/tf/keras/losses/sparse_categorical_crossentropy
        losses = tf.keras.losses.sparse_categorical_crossentropy(labels, probs)
        return tf.reduce_mean(losses)


def train(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """


    pass


def test(model, test_inputs, test_labels):
    """
    hello
    """
    return perplexity


def main():

    #get all 10 graphs and their random sequences
    data_path = '../../data/first100k.dat.xz'
    graphs = get_reachabilities(data_path)

    #for graphs 1-9,
        #train word2vec model to get embeddings
        #get random walks as inputs and labels (skipgram)


        #train a model on the graph to identify the correct edge weight
        #test prediction model on the next graph










    pass

if __name__ == '__main__':
    main()
