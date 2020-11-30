import tensorflow as tf
import numpy as np
from functools import reduce
import lzma
import networkx as nx
import random
import queue as q
import random
import itertools

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
    while len(transactionLists) < 2:
        graph_edges = []
        while len(set(graph_edges)) < 10000:
            graph_edges.append(pairs.pop(0))
        transactionLists.append(graph_edges)

    print('partitioned data')


    #create 10 independent graphs
    FinalGraphs = []

    #initialize graph
    G = nx.Graph()

    added_Edges = {}
    #setting the dictionary to keep track of all nodes that are added
    # to be used when updating random walks
    for i in range(10):
        added_Edges[i] = []

    #index to keep track of which graph we adding edges to
    graph = -1
    #for each graph update, make a new graph
    for edgelist in transactionLists:
        newEdgeList = []
        graph += 1
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
                added_Edges[graph].append(src)

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
        graphAtT = G.copy()
        FinalGraphs.append(graphAtT)

    for key in added_Edges:
        added_Edges[key] = set(added_Edges[key])

    return FinalGraphs, added_Edges

def get_amounts(file_path):
    '''
    PROCESS

    Constructs either type of graph for each 10k increments of unique pairs

    At next timestep,
        - update all weights of existing nodes and edges (decay previous and add new weight)

    '''

    #gets list of new edges to be updated for each of 10 timesteps
    file = lzma.open(file_path, mode='rt')
    edges = np.loadtxt(file, int)
    sender = edges[:,1]
    receiver = edges[:,2]
    weight = edges[:,3]


    print('partitioned data')

    return None



def get_randomWalks(G, prev_walks, new_Edges, opt_out, d_factor, length, walks_per_node):
    #get random walks for a given graph
    #output all sequences in forms of np.array(walks by walk_length)
    #10 walks each node at each timestep, training on all previousy created, but recently updated, walks
    #https://networkx.org/documentation/stable/tutorial.html
    nodes = G.nodes()
    #creating a dictionary of walks to keep track of each nodes walk #
    node_dict = {}
    #keeping track of which row in the output matrix to addd the walk to
    walk_index = 0
    #creating a matrix of walks
    size = len(nodes)*walks_per_node
    walk_mat = np.zeros((size,41))
    # for node in nodes:
    #     queue1.put(node)
    #     node_dict[node] = 0
    #     print('added')
    # print(queue1.qsize())
    # while not queue1.empty():
    #     walk = generate_walk(G, length)
    #     matched = False
    #     while matched == False:
    #         node = queue1.get()
    #         if node in walk:
    #             node_dict[node] +=1
    #             walk.append(1)
    #             walk_mat[i] = walk
    #             i+=1
    #             if node_dict[node] < walks_per_node:
    #                 queue1.put(node)
    #             matched = True
    #         else:
    #             queue1.put(node)
    #         print(queue1.qsize())
    for node in nodes:
        node_dict[node] = 0
    num_finished = 0
    #print('first')
    while not num_finished == len(nodes):
        walk = generate_walk(G, length)
        matched = False
        ind = random.randint(0,length-1)
        num_times = 0
        while matched == False:
            # print(num_finished)
            node = walk[ind]
            num_times += 1
            if node_dict[node] < 10:
                walk.append(1)
                walk_mat[walk_index] = walk
                walk_index+=1

                #print(walk_index)

                matched = True
            node_dict[node] +=1
            if node_dict[node] == walks_per_node:
                num_finished += 1
            if ind < length - 1:
                ind += 1
            else:
                ind = 0
            if num_times == 40:
                break



    #print('second')
    if prev_walks is not None:

        #print("updating prevwalks")
        for row in range(np.shape(prev_walks)[0]-1,-1,-1):
            #print(row)
            if prev_walks[row,39] in new_Edges:
                prev_walks[row,40] = prev_walks[row,length] *d_factor


        indices = []
        for row in range(np.shape(prev_walks)[0]-1,-1,-1):
            if prev_walks[row,40] <= opt_out:
                indices.append(row)
        prev_walks = np.delete(prev_walks,indices,0)

        #print('before stiack')
        final = np.concatenate([walk_mat,prev_walks], axis = 0)
    else:
        final = walk_mat
    return final




def generate_walk(G,length):
    '''
    Creates one random walk of given length

    :param G: the graphs
    :param length: the length of the random walk - number of steps

    :return: walk - list of nodes visited
    '''
    # list of 40 nodes
    # get number of edges and plug output into random number gnenerator
    nodes = list(G.nodes())
    walk = []
    # choose a random start node
    node_ind = random.randint(0, len(nodes) - 1)
    start_node = nodes[node_ind]
    walk.append(start_node)
    node = start_node
    #getting all walks
    while(len(walk) < length):
        #all neighbors for node
        neighbors = list(G.neighbors(node))
        num_neigh = len(neighbors)
        neigh_ind = random.randint(0,num_neigh-1)
        #get next node in walk
        node = neighbors[neigh_ind]
        walk.append(node)
    return walk

def Node2Vec_getData(randomWalks, numNodes, window_sz):
    #return skipgram (labels, outputs) tuple np, with a dictionary, (all translated from 1 to n)


    #convert all sequences to ints from 0 to N-1
    #use tf.keras.preprocessing.sequence.skipgrams after
    vocab = set(np.array(randomWalks).flatten())
    indices = list(range(0, len(vocab)))
    #zip up words and indices
    vocabdict = dict(zip(list(vocab), indices))



    pairs =[]
    for walk in randomWalks:
        walk_ids = [vocabdict[i] for i in walk]
        pairs.append(np.array(tf.keras.preprocessing.sequence.skipgrams(walk_ids, numNodes, window_size = window_sz, negative_samples=0.0)[0]))

    data = np.concatenate(pairs, 0)



    return data, vocabdict

def Prediction_getData(embeddings, idtoNodeid, G):
    #return pairs of embeddings(concatenated), and ground truths

    #get numWords
    num_words = embeddings.shape[0]
    indices = list(range(num_words))

    #get all possible pairs of words
    #use combinatorics using itertools itertools.combinations('abcd',2)
    pairs = list(itertools.combinations(indices, 2))
    pairs = [list(pair) for pair in pairs]


    print('translating to embeddings')
    #translate all to embeddings

    #pairs_ids = map(idtoNodeid, pairs)
    pairs_ids = np.vectorize(idtoNodeid.get)(pairs)
    pairs_embeddings = tf.nn.embedding_lookup(embeddings, pairs, max_norm=None, name=None)

    #turn those pairs of embeddings into inputs (concatenate each pair)
    print(pairs_embeddings.shape)
    inputs = tf.reshape(pairs_embeddings, (len(pairs_ids), -1))
    print(inputs.shape)
    '''
    outputs = []
    for i in range(len(pairs_ids)):
        print(i/len(pairs_ids))
        #check if those embeddings (in nodeID form) have connections in the graph

        try:
            if G[pairs_ids[i,0]][pairs_ids[i,1]]["weight"] >= 1:
                outputs.append(1)
            else:
                outputs.append(0)
        except:
            outputs.append(0)
    '''
    outputs = []
    for i in range(len(pairs_ids)):
        #print(i/len(pairs_ids))
        outputs.append(random.randint(0,1))

    print('negative sampling')
    #negative sampling of outputs of 0 until outputs are equal
    print('get indices')
    indices = []
    for i in range(0, len(outputs)):
        if outputs[i] == 0:
            indices.append(i)
    random.shuffle(indices)

    #get number of outputs with 1
    num_out1 = outputs.count(1)
    num_out0 = len(outputs) - num_out1
    diff = num_out0 - num_out1

    print('delete inputs')
    inputs = np.delete(inputs, indices[:diff], axis = 0)
    print('delete outputs')
    outputs = np.delete(outputs, indices[:diff])

    return inputs, outputs
