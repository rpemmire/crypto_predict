import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from preprocess import get_reachabilities
from preprocess import get_randomWalks
from preprocess import Node2Vec_getData
from preprocess import Prediction_getData


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
        self.epochs = 1

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


class Predictor(tf.keras.Model):
    def __init__(self):

        super(Predictor, self).__init__()
        """
        Basic feed forward network taking in two embeddings and outputting label 1 or 0

        :param vocab_size: The number of unique words in the data
        """

        self.batch_size = 10000
        self.learning_rate = 0.001
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.hidden_layer_sz = 256 #might need to change
        self.num_classes = 2
        self.layer1 = tf.keras.layers.Dense(self.hidden_layer_sz, activation='relu')
        self.layer2 = tf.keras.layers.Dense(self.hidden_layer_sz, activation='relu')
        self.layer3 = tf.keras.layers.Dense(self.num_classes, activation='softmax')


    def call(self, inputs):
        """
        feedforward prediction
        """
        #TODO: Fill in
        output1 = self.layer1(inputs)
        output2 = self.layer2(output1)
        probs = self.layer3(output2)


        return probs

    def loss(self, probs, labels):
        """
        Calculates average cross entropy sequence to sequence loss of the prediction

        """

        #TODO: Fill in
        #We recommend using tf.keras.losses.sparse_categorical_crossentropy
        #https://www.tensorflow.org/api_docs/python/tf/keras/losses/sparse_categorical_crossentropy
        #paper has super complicated way of explaining this but we think it's the same thing
        losses = tf.keras.losses.sparse_categorical_crossentropy(labels, probs)
        return tf.reduce_mean(losses)

    def accuracy(probs, labels):
        """
        Calculates averages

        """
        predictions = np.equal(np.argmax(probs, axis=1), labels)
        accuracy = np.sum(predictions)/len(labels)
        return accuracy

    def f1(probs, labels):
        """
        Calculates f1

        """
        predictions = np.argmax(probs, axis=1)
        numerator = 2*(tf.compat.v1.metrics.recall(labels, predictions) * tf.compat.v1.metrics.precision(labels, predictions))
        denom = tf.compat.v1.metrics.recall(labels, predictions) + tf.compat.v1.metrics.precision(labels, predictions)
        return  numerator/denominator


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
        #for i in range(0,len(data),model.batch_size):
        for i in range(0,1,model.batch_size):
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

def train_Predict(model, train_inputs, train_labels):
    """

    """

    loss_list = []
    #iterating through dataset with batch size
    # for start, end in zip(range(0, len(train_inputs) - model.batch_size, model.batch_size), range(model.batch_size, len(train_inputs), model.batch_size)):
    #for i in range(0,len(train_inputs),model.batch_size):
    for i in range(0,1,model.batch_size):
        batch_X = train_inputs[i:i+model.batch_size, :]
        batch_Y = train_labels[i:i+model.batch_size]
        #updating gradients
        with tf.GradientTape() as tape:
            probs = model.call(batch_X)
            loss = model.loss(probs, batch_Y)
            loss_list.append(loss)
            print("train predict", i/model.batch_size)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss_list


def test_Predict(model, test_inputs, test_labels):
    """
    returns average accuracy, average f1 score
    """
    acc_list = []
    f1_list = []
    #iterating through dataset with batch size
    # for start, end in zip(range(0, len(train_inputs) - model.batch_size, model.batch_size), range(model.batch_size, len(test_inputs)-model.batch_size, model.batch_size)):
    #     batch_X = data[start:end, :]
    #     batch_Y = data[start:end, :]
    #for i in range(0,len(test_inputs),model.batch_size):
    for i in range(0,1,model.batch_size):
        batch_X = test_inputs[i:i+model.batch_size, :]
        batch_Y = test_labels[i:i+model.batch_size]
        #updating gradients
        probs = model.call(batch_X)
        print("test predict", i/model.batch_size)
        acc_list.append(model.accuracy(probs, labels))
        f1_list.append(model.f1(probs, labels))
    return sum(acc_list)/len(acc_list), sum(f1_list)/len(f1_list)


def main():
    prev_walk = None
    #get all 10 graphs and their random sequences
    data_path = '../../Data/first100k.dat.xz'
    #returns new added edges for all graphs and all graphs
    print('creating graphs')
    graphs, added_Edges = get_reachabilities(data_path, .125, .5)
    random_walks = None



    predict_model = Predictor()
    #initialize prediction (must be trained across graphs)
    for i in range(len(graphs)-1):
        #initialize word2vec (must be reinitialized for each graph)
        numNodes = len(graphs[i].nodes())
        embed_model = Node2Vec(numNodes)

        #get random walks and return as inputs and labels (skipgram)
        #pass in the added_Edges from the previous graph
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
        id2Node_dict = {i: w for w, i in nodetoID_dict.items()}

        #train a model on the graph to identify the correct edge weight
            #inputs: (node combos, their respective embedding combos)
            #labels: (whether or not those tranactions happen (if edge weight>1))
        print('getting train data for predict')
        train_inputs, train_labels = Prediction_getData(embeddings, id2Node_dict , graphs[i])
        print('training predict')
        loss_list = train_Predict(predict_model, train_inputs, train_labels)

        #test prediction model on the next graph
        print('getting test data for predict')
        test_inputs, test_labels = Prediction_getData(embeddings, id2Node_dict, graphs[i+1])
        print('testing predict')
        acc, f1 = test_Predict(predict_model, test_inputs, test_labels)

        print("reachabilities", i+1, acc, f1)


    pass

if __name__ == '__main__':
    main()
