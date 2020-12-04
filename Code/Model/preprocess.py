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
                G.add_edge(src,dst)
                G[src][dst]["weight"] +=1
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

def Node2Vec_getData(randomWalks, window_sz):
    #return skipgram (labels, outputs) tuple np, with a dictionary, (all translated from 1 to n)


    #convert all sequences to ints from 0 to N-1
    #use tf.keras.preprocessing.sequence.skipgrams after
    vocab = list(set(np.array(randomWalks).flatten()))
    indices = list(range(0, len(vocab)))
    #zip up words and indices
    vocabdict = dict(zip(vocab, indices))

    #print(vocabdict)


    pairs =[]
    for walk in randomWalks:
        walk_ids = [vocabdict[i] for i in walk]
        pairs.append(np.array(tf.keras.preprocessing.sequence.skipgrams(walk_ids, len(vocabdict), window_size = window_sz, negative_samples=0.0)[0]))

    data = np.concatenate(pairs, 0)

    np.random.shuffle(data)

    return data, vocabdict

def Prediction_getData(embeddings, idtoNodeid, G):
    #return pairs of embeddings(concatenated), and ground truths
    nodetoID_dict = {idtoNodeid[j]: j for j in idtoNodeid}

    outputs = []
    inputs = []
    negs = 0
    transition = np.zeros((embeddings.shape[0], embeddings.shape[0]))
    i = 0
    #(store where there were transactions) transition matrix (nxn for len(graph1))
    edgeList = list(G.edges(data='weight', default=0))
    for edge in edgeList:
        i+=1
        #print(i/len(edgeList))
        src = edge[0]
        dst = edge[1]
        weight = edge[2]
        try:
            IDsrc = int(nodetoID_dict[int(src)])
            IDdst = int(nodetoID_dict[int(dst)])
            if weight >=1:
                transition[IDsrc,IDdst] = 1
                #transition[IDdst,IDsrc] = 1
                inputs.append([IDsrc, IDdst])
                #inputs.append([IDdst, IDsrc])
                outputs.append(1)
                #outputs.append(1)
        except:
            pass

    positives = len(inputs)
    while negs < positives*30:
        x = random.randint(0,embeddings.shape[0]-1)
        y = random.randint(0,embeddings.shape[0]-1)
        if transition[x,y] != 1 and transition[y,x] != 1:
            transition[x,y] = 1
            #transition[y,x] = 1
            negs += 1
            #print('random sampling', negs/len(inputs), len(inputs))
            inputs.append([x,y])
            outputs.append(0)
            #inputs.append([y,x])
            #outputs.append(0)




    #shuffle inputs and outputs
    shuffleindices = list(range(len(outputs)))
    random.shuffle(shuffleindices)
    inputs = tf.gather(inputs, shuffleindices, axis = 0)
    outputs = tf.gather(outputs, shuffleindices)

    inputs = tf.nn.embedding_lookup(embeddings, np.array(inputs, dtype = int), max_norm=None, name=None)
    inputs = tf.reshape(inputs, (len(outputs), -1))
    print('length of data', inputs.shape)





    return inputs, outputs
