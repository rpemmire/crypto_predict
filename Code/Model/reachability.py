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
        '''
        Initialize embedding matrix, linear layer, optimizer, sizes, etc.
        '''


    def call(self, inputs, initial_state):
        """
        Basic embedding to linear layer
        """
        #TODO: Fill in


        #embedding layer output has to be (batch_size, window_size, embedding_size)
        #embeddings = []
        '''
        Keras embedding layer to linear output layer

        '''

        return probs, final_state

    def loss(self, probs, labels):
        """
        Follow loss in the paper
        """

        #TODO: Fill in
        #We recommend using tf.keras.losses.sparse_categorical_crossentropy
        #https://www.tensorflow.org/api_docs/python/tf/keras/losses/sparse_categorical_crossentropy
        losses = tf.keras.losses.sparse_categorical_crossentropy(labels, probs)
        return tf.reduce_mean(losses)


class Predictor(tf.keras.Model):
    def __init__(self, vocab_size):
        """
        Basic feed forward network taking in two embeddings and outputting label 1 or 0

        :param vocab_size: The number of unique words in the data
        """

        super(Model, self).__init__()



    def call(self, inputs, initial_state):
        """
        feedforward prediction
        """
        #TODO: Fill in


        #embedding layer output has to be (batch_size, window_size, embedding_size)
        #embeddings = []


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


def train_Node2Vec(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """


    pass

def train_Predict(model, train_inputs, train_labels):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """


    pass

def test_Predict(model, test_inputs, test_labels):
    """
    hello
    """
    return perplexity


def main():

    #get all 10 graphs and their random sequences
    data_path = '../../data/first100k.dat.xz'
    graphs = get_reachabilities(data_path, .125, .5)

    #initialize prediction (must be trained across graphs)
    for i in range(len(graphs)-1):
        #initialize word2vec (must be reinitialized for each graph)


        #get random walks and return as inputs and labels (skipgram)
        random_walks = get_randomWalks(graphs[i], .125, .5, 40, 10)
        inputs, labels, nodetoID_dict = Node2Vec_getData(random_walks)

        #train word2vec model to get embeddings (return in a dictionary)
        nodeIDtoEmbedding = train_Node2Vec(inputs, labels, nodetoID_dict)


        #train a model on the graph to identify the correct edge weight
            #inputs: (node combos, their respective embedding combos)
            #labels: (whether or not those tranactions happen (if edge weight>1))
        train_inputs, train_labels = Prediction_getData(nodeIDtoEmbedding, graphs[i])
        #test prediction model on the next graph
        test_inputs, test_labels = Prediction_getData(nodeIDtoEmbedding, graphs[i+1])










    pass

if __name__ == '__main__':
    main()
