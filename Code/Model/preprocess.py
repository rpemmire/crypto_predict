import tensorflow as tf
import numpy as np
from functools import reduce
import lzma
import networkx as nx
import random

#hyperparameters
decay_rate = 1/2 #1/(2^delta_t)
opt_out_threshold = .125

def get_reachabilities(file_path, opt_out, decay_rate):
    '''
    PROCESS

    Constructs either type of graph for each 10k increments of unique pairs

    At next timestep,
        - update all weights of existing nodes and edges (decay previous weights and add new weight)
        - delete any nodes with

    '''


    #gets list of new edges to be updated for each of 10 timesteps
    file = lzma.open(file_path, mode='rt')
    edges = np.loadtxt(file, int)
    sender = edges[:,1]
    receiver = edges[:,2]

    transactionLists = []
    pairs = list(zip(sender, receiver))
    while len(transactionLists) < 10:
        graph_edges = []
        while len(set(graph_edges)) < 10000:
            graph_edges.append(pairs.pop(0))
        transactionLists.append(graph_edges)

    print('partitioned data')


    #create 10 independent graphs
    FinalGraphs = []

    #initialize graph
    G = nx.Graph()

    #for each graph update, make a new graph
    for edgelist in transactionLists:

        newEdgeList = []
        #make the edge list irrespective of direction
        #do this by sorting so that smaller id is in front and then taking set()
        for createEdge in set(edgelist):
            if createEdge[1]<createEdge[0]:
                sender = createEdge[1]
                receiver = createEdge[0]
            else:
                sender = createEdge[0]
                receiver = createEdge[1]
            newEdgeList.append(tuple([sender, receiver]))

        #1) decrement all edge weights by half
        for edge in G.edges(data = True):
            #print(edge[2]['weight'])
            edge[2]['weight'] = edge[2]['weight'] * decay_rate
            #print(edge[2]['weight'])



        #2) add in all new edges, but just a set of the given edgelist (irrespective of direction)
            #this is because we only care about the presence of a transaction, not how many or in which direction
        for createEdge in set(newEdgeList):
            #print(createEdge)
            src = createEdge[0]
            dst = createEdge[1]

            # if edge already exists, add 1 to attribute
            if G.has_edge(src, dst):
                G[src][dst]["weight"] +=1
            # else add in new edge with attribute 1
            else:
                G.add_edge(src,dst,weight=1)

        #3) update all nodes, if sum of edge weights < threshold, delete node
        to_remove = []
        for node in G.nodes():
            edges = G.edges(node, data = True)
            max = 0
            for edge in edges:
                if edge[2]['weight'] > max:
                    max = edge[2]['weight']
            if max <= opt_out:
                to_remove.append(node)
        G.remove_nodes_from(to_remove)
        print(len(to_remove), "nodes removed")

        print("graph created", len(G.nodes()))
        FinalGraphs.append(G)

    return FinalGraphs

def get_amounts(file_path):
    '''
    PROCESS

    Constructs either type of graph for each 10k increments of unique pairs

    At next timestep,
        - update all weights of existing nodes and edges (decay previous and add new weight)

    '''

    file = lzma.open('../../.dat.xz', mode='rt')

    return None


def get_randomWalks(G, opt_out, length, walks_per_node):
    #get random walks for a given graph
    #output all sequences in forms of np.array(walks by walk_length)
    #10 walks each node at each timestep, training on all previousy created, but recently updated, walks

    #for each node, get 10 random walks of length 40
    sequences = []
    #while len(sequences != 10)
    # i = 0
    # while true:
        #proudce 100 walks
        #node = nodes[i]
        #for walkj in walks:
            #if node in walkj
            #add walkj to walks of node
            #if node has 10 walks
                #i +=1
                #break
        # if i = final node index +1
            #False

    pass

def Node2Vec_getData(randomWalks, numNodes, window_sz):
    #return skipgram (labels, outputs) tuple np, with a dictionary, (all translated from 1 to n)


    #convert all sequences to ints from 0 to N-1
    #use tf.keras.preprocessing.sequence.skipgrams after
    vocab = set(np.array(randomWalks).flatten())
    indices = list(range(0, len(vocab)))
    #zip up words and indices
    vocabdict = dict(zip(list(vocab), indices))

    print(len(vocabdict))

    pairs =[]
    for walk in randomWalks:
        walk_ids = [vocabdict[i] for i in walk]
        pairs.append(tf.keras.preprocessing.sequence.skipgrams(walk_ids, numNodes, window_size = window_sz, negative_samples=0.0))
    data = np.reshape(np.array(pairs), (-1, 2))

    return data, vocabdict

def Prediction_getData(embeddings, idtoNodeid, G):
    #return pairs of embeddings(concatenated), and ground truths

    #get numWords
    num_words = embeddings.shape[0]
    indices = list(range(num_words))

    #get all possible pairs of words
    pairs = []
    for i in indices:
        for j in indices:
            pairs.append(zip(i, j))


    #translate all to embeddings
    inputs = []
    outputs = []
    for pair in pairs:
        src = idtoNodeid[pair[0]]
        dst = idtoNodeid[pair[1]]
        pair[0] = tf.nn.embedding_lookup(embeddings, pair[0], max_norm=None, name=None)
        pair[1] = tf.nn.embedding_lookup(embeddings, pair[1], max_norm=None, name=None)


        #turn those pairs of embeddings into inputs (concatenate each pair)
        inputs.append(tf.concat([pair[0], pair[1]], axis=-1))
        #check if those embeddings (in nodeID form) have connections in the graph
        if G.has_edge(src, dst) and G[src][dst]["weight"] >= 1:
            outputs.append(1)
        else:
            outputs.append(0)


    #negative sampling of outputs of 0 until outputs are equal
    indices = []
    for i in range(0, len(outputs)) :
        if outputs[i] == 0:
            indices.append(i)
    indices = random.shuffle(indices)

    #get number of outputs with 1
    num_out1 = outputs.count(1)
    num_out0 = len(outputs) - num_out1
    diff = num_out0 - num_out1

    for i in range(len(diff)):
        #delete output and input at indices[i]
        outputs.pop(i)
        inputs.pop(i)



    return inputs, outputs
