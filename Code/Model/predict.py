import networkx as nx
import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from preprocess import Prediction_getData
import json

class Predictor(tf.keras.Model):
    def __init__(self):

        super(Predictor, self).__init__()
        """
        Basic feed forward network taking in two embeddings and outputting label 1 or 0

        :param vocab_size: The number of unique words in the data
        """

        self.batch_size = 10
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

    def accuracy(self, probs, labels):
        """
        Calculates averages

        """
        predictions = np.equal(np.argmax(probs, axis=1), labels)
        accuracy = np.sum(predictions)/len(labels)
        return accuracy

    def f1(self, probs, labels):
        """
        Calculates f1

        """
        predictions = np.argmax(probs, axis=1)

        tp = 0
        fp = 0
        fn = 0

        for i in range(len(predictions)):
            #calculate true positives
            if predictions[i] ==1 and labels[i] ==1:
                tp+=1

            #calculate false positives
            if predictions[i] ==1 and labels[i] ==0:
                fp+=1

            #calculate false negatives
            if predictions[i] ==0 and labels[i] ==1:
                fn+=1

        precision = tp /(tp+fp)
        recall = tp/(tp+fn)

        numerator = 2*recall*precision
        denom = precision + recall
        return  numerator/denom

def train_Predict(model, train_inputs, train_labels):
    """

    """

    loss_list = []
    #iterating through dataset with batch size
    # for start, end in zip(range(0, len(train_inputs) - model.batch_size, model.batch_size), range(model.batch_size, len(train_inputs), model.batch_size)):
    for i in range(0,len(train_inputs),model.batch_size):
    #for i in range(0,1,model.batch_size):
        batch_X = train_inputs[i:i+model.batch_size, :]
        batch_Y = train_labels[i:i+model.batch_size]
        #updating gradients
        with tf.GradientTape() as tape:
            probs = model.call(batch_X)
            loss = model.loss(probs, batch_Y)
            loss_list.append(loss)
            #print("train predict", i/len(train_inputs), loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss_list


def test_Predict(model, test_inputs, test_labels):
    """
    returns average accuracy, average f1 score
    """
    acc_sum = 0
    predicted_list = []
    for i in range(0,len(test_inputs),model.batch_size):
    #for i in range(0,1,model.batch_size):
        batch_X = test_inputs[i:i+model.batch_size, :]
        batch_Y = test_labels[i:i+model.batch_size]
        #updating gradients
        probs = model.call(batch_X)

        accuracy = model.accuracy(probs, batch_Y)

        #print("test predict", i/len(test_inputs))
        acc_sum += accuracy*len(batch_Y)
        predicted_list.append(probs)
    predicted_list = tf.concat(predicted_list, axis = 0)
    #print(predicted_list.shape)
    f1 = model.f1(predicted_list, test_labels)
    return acc_sum/len(test_labels), f1


def load_predict_data(t, folder_path):

    graph1 = nx.read_weighted_edgelist(folder_path + 'graph_' + str(t) + '.txt')
    graph2 =nx.read_weighted_edgelist(folder_path + 'graph_' + str(t+1) + '.txt')
    embeddings = np.load(folder_path + 'embeddings_' + str(t) + '.npy')

    with open(folder_path + 'idDict_' + str(t) + '.json', 'r') as jsonfile:
        id2Node_dict = json.load(jsonfile)
        #print(id2Node_dict)

    return graph1, graph2 , embeddings, id2Node_dict


def main():

    #take in 10 graphs files, 10 sets of embeddings files, 10 dictionary files


    predict_model = Predictor()
    #initialize prediction (must be trained across graphs)
    for i in range(9):

        graph1, graph2 , embeddings, id2Node_dict = load_predict_data(i, '../../Data/Node2Vec_outputs/')

        #train a model on the graph to identify the correct edge weight
            #inputs: (node combos, their respective embedding combos)
            #labels: (whether or not those tranactions happen (if edge weight>1))
        #print('getting train data for predict')
        train_inputs, train_labels = Prediction_getData(embeddings, id2Node_dict , graph1)
        print('training predict')
        loss_list = train_Predict(predict_model, train_inputs, train_labels)

        #test prediction model on the next graph
        #print('getting test data for predict')
        test_inputs, test_labels = Prediction_getData(embeddings, id2Node_dict, graph2)
        print('testing predict')
        acc, f1 = test_Predict(predict_model, test_inputs, test_labels)

        print("reachabilities", i+1, 'accuracy',acc, 'f1',f1)

    pass

if __name__ == '__main__':
    main()
