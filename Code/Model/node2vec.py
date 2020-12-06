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
        The Model class trains embeddings for nodes in a graph
        """

        super(Node2Vec, self).__init__()
        '''
        Initialize embedding matrix, linear layer, optimizer, sizes, etc.
        '''
        self.batch_size = 1000 #can change
        self.vocab_sz = vocab_size
        self.embedding_sz = 128

        self.window_size = 1
        self.epochs = 1

        self.learning_rate = 0.001
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        #TODO: Fill in
        self.E = tf.Variable(tf.random.truncated_normal([self.vocab_sz,self.embedding_sz], stddev=.1))
        self.W = tf.Variable(tf.random.truncated_normal([self.embedding_sz, self.vocab_sz], stddev=.1))
        self.b = tf.Variable(tf.random.truncated_normal([self.vocab_sz], stddev=.1))

    def call(self, inputs):
        """
        This method calls the model. It includes a basic embedding to feedforward
        layer structure.
        """
        embedding = tf.nn.embedding_lookup(self.E,inputs)
        logits = tf.matmul(embedding,self.W) + self.b
        return logits

    def loss(self, logits, labels):
        """
        Loss for a classification problem uses cross entropy loss with logits
        """
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
        return tf.reduce_mean(losses)

def train_Node2Vec(model, data):
    """
    Runs through the training data of skipgrams.
    """
    for ep in range(model.epochs):
        curr_loss = 0
        step = 0
        for i in range(0,len(data),model.batch_size):
            batch_X = data[i:i+model.batch_size, 0]
            batch_Y = data[i:i+model.batch_size, 1]
            with tf.GradientTape() as tape:
              logits = model.call(batch_X)
              loss = model.loss(logits, batch_Y)
            curr_loss += loss
            step += 1
            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            print("node2vec", i/len(data), loss, 'Epoch %d\tLoss: %.3f' % (ep, curr_loss / step))

    pass


def main():

    #get all 10 graphs and their random sequences
    data_path = '../../Data/first100k.dat.xz'
    print('creating graphs')
    graphs, added_Edges = get_reachabilities(data_path, .125, .5)

    #variable to keep track of random walks which are decayed at next time step
    random_walks = None


    for i in range(len(graphs)):
        #get random walks and return as inputs and labels (skipgram)

        #pass in the added_Edges from the previous graph
        print('getting random walks')
        if i ==0:
            new_edges = None
        else:
            new_edges = added_Edges[i]
        random_walks = get_randomWalks(graphs[i], random_walks, new_edges, .125, .5, 40, 10)

        #this is to selectively generate embeddings we haven't done
        if i ==5:
            #walks with removed decay factor
            purged_walks = random_walks[:,0:40]
            data, nodetoID_dict = Node2Vec_getData(purged_walks, 1)
            embed_model = Node2Vec(len(nodetoID_dict))

            #train word2vec model to get embeddings
            print('training node2vec')
            train_Node2Vec(embed_model, tf.convert_to_tensor(data))

            embeddings = embed_model.E.read_value()
            id2Node_dict = {nodetoID_dict[j]: j for j in nodetoID_dict}

            #save, graph, embeddings, and dict
            #as graph_i, embeddings_i, idDict_i
            #files later used for predict
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
