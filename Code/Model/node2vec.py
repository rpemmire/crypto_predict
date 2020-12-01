import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from preprocess import get_reachabilities
from preprocess import get_randomWalks
from preprocess import Node2Vec_getData
import networkx as nx
import json

class Node2Vec(tf.keras.Model):
    def __init__(self, vocab_size):
        """
        The Model class predicts the next words in a sequence.

        :param vocab_size: The number of unique words in the data
        """

        super(Node2Vec, self).__init__()
        '''
        Initialize embedding matrix, linear layer, optimizer, sizes, etc.
        '''
        self.batch_size = 1000 #can change
        self.vocab_sz = vocab_size
        self.embedding_sz = 128
        self.window_size = 2

        self.learning_rate = 0.001
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        #THIS SHOULDN"T ACTUALLY BE 1
        self.epochs = 2

        #TODO: Fill in
        self.E = tf.Variable(tf.random.truncated_normal([self.vocab_sz,self.embedding_sz], stddev=.1))
        self.W = tf.Variable(tf.random.truncated_normal([self.embedding_sz, self.vocab_sz], stddev=.1))
        self.b = tf.Variable(tf.random.truncated_normal([self.vocab_sz], stddev=.1))



    def call(self, inputs):
        """
        Basic embedding to linear layer
        """
        embedding = tf.nn.embedding_lookup(self.E,inputs) #output of embedding is batch_size * embedding_size
        #embedding*W is batch_size*vocab_size
        logits = tf.matmul(embedding,self.W) + self.b
        return logits

    def loss(self, logits, labels):
        """
        Follow loss in the paper
        """
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
        return tf.reduce_mean(losses)

def train_Node2Vec(model, data):
    """
    Runs through one epoch - all training examples.

    :param model: the initilized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """
    for ep in range(model.epochs):
        curr_loss = 0
        step = 0
        # for start, end in zip(range(0, len(data) - model.batch_size, model.batch_size), range(model.batch_size, len(data), model.batch_size)):
        for i in range(0,len(data),model.batch_size):
        #for i in range(0,1,model.batch_size):
            batch_X = data[i:i+model.batch_size, 0]
            batch_Y = data[i:i+model.batch_size, 0]
            with tf.GradientTape() as tape:
              logits = model.call(batch_X)
              loss = model.loss(logits, batch_Y)
              print(data.shape)
              print("node2vec", i/len(data))
            curr_loss += loss
            step += 1
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        if ep % 10 == 0:
            print('Epoch %d\tLoss: %.3f' % (ep, curr_loss / step))

    pass

def main():
    prev_walk = None
    #get all 10 graphs and their random sequences
    data_path = '../../Data/first100k.dat.xz'
    #returns new added edges for all graphs and all graphs
    print('creating graphs')
    graphs, added_Edges = get_reachabilities(data_path, .125, .5)
    random_walks = None
    print('len graphs ', len(graphs))
    for i in range(len(graphs)):
        #initialize word2vec (must be reinitialized for each graph)
        numNodes = len(graphs[i].nodes())
        embed_model = Node2Vec(numNodes)

        #get random walks and return as inputs and labels (skipgram)
        #pass in the added_Edges from the previous graph
        print('getting random walks')
        if i ==0:
            new_edges = None
        else:
            new_edges = added_Edges[i]
        random_walks = get_randomWalks(graphs[i], random_walks, new_edges, .125, .5, 40, 10)

        #TODO, append walks
        purged_walks = random_walks[:,0:40]
        data, nodetoID_dict = Node2Vec_getData(purged_walks, numNodes, embed_model.window_size)

        #train word2vec model to get embeddings
        print('training node2vec')
        train_Node2Vec(embed_model, tf.convert_to_tensor(data))

        embeddings = embed_model.E.read_value()
        id2Node_dict = {nodetoID_dict[j]: j for j in nodetoID_dict}

        #save, graph, embeddings, and dict
        #as graph_i, embeddings_i, idDict_i
        savePath = '../../Data/Node2Vec_outputs/'
        nx.write_weighted_edgelist(graphs[i], savePath + 'graph_' + str(i) + '.txt')
        np.save(savePath + 'embeddings_' + str(i), embeddings)

        f = open(savePath + 'idDict_' + str(i) + '.json', 'w')
        f.write(json.dumps(id2Node_dict))
        f.close()


        print('saved files')

    pass

if __name__ == '__main__':
    main()
