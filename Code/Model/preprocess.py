import tensorflow as tf
import numpy as np
from functools import reduce
import lzma
import networkx as nx
import random

def get_reachabilities(file_path, opt_out, decay_rate):
    '''
    Constructs each reachability graph for each 10k increments of unique pairs
    '''

    #gets list of new edges to be updated for each of 10 timesteps
    file = lzma.open(file_path, mode='rt')
    edges = np.loadtxt(file, int)
    sender = edges[:,1]
    receiver = edges[:,2]
    transactionLists = []
    pairs = list(zip(sender, receiver))

    #load in each increment of 10k unique sender receiver pairs
    while len(transactionLists) < 10:
        graph_edges = []
        while len(set(graph_edges)) < 10000:
            graph_edges.append(pairs.pop(0))
        transactionLists.append(graph_edges)

    #create 10 independent graphs
    FinalGraphs = []

    #initialize graph to edit
    G = nx.Graph()

    #setting the dictionary to keep track of all nodes that are added
    # to be used when updating random walks
    added_Edges = {}
    for i in range(10):
        added_Edges[i] = []

    #index to keep track of which graph we are adding edges to
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

        #1) decrement all edge weights by decay factor
        for edge in G.edges(data = True):
            edge[2]['weight'] = edge[2]['weight'] * decay_rate

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
                G[src][dst]["weight"] =1
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

        print("graph created, edges, nodes", len(list(G.edges())),  len(G.nodes()))
        print(len(to_remove), "nodes removed")

        graphAtT = G.copy()
        FinalGraphs.append(graphAtT)

    for key in added_Edges:
        added_Edges[key] = set(added_Edges[key])

    return FinalGraphs, added_Edges

def get_randomWalks(G, prev_walks, new_Edges, opt_out, d_factor, length, walks_per_node):
    '''
    get random walks for a given graph
    output all sequences in forms of np.array(walks by walk_length)
    '''

    #getting all nodes in the graph at the respective timestep
    nodes = G.nodes()
    '''
    creating a dictionary of walks to keep track of the number of walks created
    for each node at eachtimestep.
    '''
    node_dict = {}
    #keeping track of which row in the output matrix to addd the walk to
    walk_index = 0

    #creating a matrix of walks
    size = len(nodes)*walks_per_node
    walk_mat = np.zeros((size,41))

    #setting numbers of generated walks for each node at this timestep equal to 0
    for node in nodes:
        node_dict[node] = 0
    #counter for how many nodes have had 10 walks produced at this timestep
    num_finished = 0
    '''
    while the number of nodes with 10 walks generated is less than the number
    of nodes in the graph.
    '''
    while not num_finished == len(nodes):
        #generate a walk
        walk = generate_walk(G, length)
        #variable tracking whether the walk has been matched to a node
        matched = False
        #generating a random integer to select a node from at random from the walk
        ind = random.randint(0,length-1)
        #number of nodes in the walk we have tested
        num_times = 0
        #run while the walk has not been matcheed to a node in need of a walk
        while matched == False:
            #get a node from the walk
            node = walk[ind]
            #add one to nodes checked from the walk
            num_times += 1
            #if this node has less than 10 walks representing it in the matrix
            if node_dict[node] < walks_per_node:
                #add one to the end of the walk representing it's decay factor
                walk.append(1)
                #add the walk to the walk matrix
                walk_mat[walk_index] = walk
                #add one to the index at which we will add random walks to the walk matrix
                walk_index+=1
                # change Matched to true since walk has been designated to a node
                matched = True
            '''
            add one to the node's dictionary, up until 10 represents that a walk
            has been added to the matrix corresponding to this node.  After that,
            continue adding so that node_dict is greater than 10 = walks_per_node,
            and it no longer adds to the number of finished nodes.
            '''
            node_dict[node] +=1
            '''
            adding one to the number of finished nodes, i.e. the nodes that have
            10 walks represnting them in the walk matrix
            '''

            if node_dict[node] == walks_per_node:
                num_finished += 1
            #move to the next index of the walk, looping around at end of walk
            if ind < length - 1:
                ind += 1
            else:
                ind = 0

            '''
            if all nodes have been checked in the walk and all already have 10
            walks represtning them in the walk matrix, break adn produce a new walk
            '''
            if num_times == 40:
                break



    #decaying the previous walks
    if prev_walks is not None:
        '''
        starting at the end of the matrix of previous walks.  Decaying a walks
        if it's last node is the src node of a new edge (this is what is store
        in new_edges)
        '''
        for row in range(np.shape(prev_walks)[0]-1,-1,-1):
            #print(row)
            if prev_walks[row,39] in new_Edges:
                prev_walks[row,40] = prev_walks[row,length] *d_factor

        '''
        deleting all rows that have a decay factor <= the opt_out hyperparamter
        '''
        indices = []
        for row in range(np.shape(prev_walks)[0]-1,-1,-1):
            if prev_walks[row,40] <= opt_out:
                indices.append(row)
        prev_walks = np.delete(prev_walks,indices,0)

        #appending previous walks and current walks into the matrix to be returned
        final = np.concatenate([walk_mat,prev_walks], axis = 0)
    #implemented for the first timestep where the previous walks are none
    else:
        final = walk_mat

    '''
    returning a matrix of all walks at a given timestep that is comprised of
    10 new walks for each node for the graph of the respective timestep, and all
    walks from prevoius timesteps whose decay factors are above the deleting threshold.
    Each row (walk) is a vecotr comprised of the sequential nodes in a walk
    with a decay factor appended to the end of a walk.
    '''
    return final

def generate_walk(G,length):
    '''
    Creates one random walk of given length from a graph
    '''

    nodes = list(G.nodes())
    walk = []

    # choose a random start node
    node_ind = random.randint(0, len(nodes) - 1)
    start_node = nodes[node_ind]
    walk.append(start_node)
    node = start_node

    #random walk until length is achieved
    while(len(walk) < length):
        #pick random neighbor
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
    vocab = list(set(np.array(randomWalks).flatten()))
    indices = list(range(0, len(vocab)))
    #zip up words and indices
    vocabdict = dict(zip(vocab, indices))

    pairs =[]
    for walk in randomWalks:
        walk_ids = [vocabdict[i] for i in walk]
        pairs.append(np.array(tf.keras.preprocessing.sequence.skipgrams(walk_ids, len(vocabdict), window_size = window_sz, negative_samples=0.0)[0]))

    data = np.concatenate(pairs, 0)
    np.random.shuffle(data)

    return data, vocabdict

def Prediction_getData(embeddings, idtoNodeid, G):
    #return pairs of embeddings(concatenated), and ground truths

    #reverse dict so that we have account Ids mapped to embeddings
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
        src = edge[0]
        dst = edge[1]
        weight = edge[2]
        try:
            #check if they have embeddings (this means they have labels 1 and exist in dataset)
            IDsrc = int(nodetoID_dict[int(src)])
            IDdst = int(nodetoID_dict[int(dst)])
            if weight >=1:
                transition[IDsrc,IDdst] = 1
                inputs.append([IDsrc, IDdst])
                outputs.append(1)
        except:
            pass

    #negative sample so positives = negatives
    #pick random negative labels to be sampled (label 0)
    positives = len(inputs)
    while negs < positives:
        x = random.randint(0,embeddings.shape[0]-1)
        y = random.randint(0,embeddings.shape[0]-1)
        if transition[x,y] != 1 and transition[y,x] != 1:
            transition[x,y] = 1
            negs += 1
            inputs.append([x,y])
            outputs.append(0)


    #shuffle inputs and outputs
    shuffleindices = list(range(len(outputs)))
    random.shuffle(shuffleindices)
    inputs = tf.gather(inputs, shuffleindices, axis = 0)
    outputs = tf.gather(outputs, shuffleindices)

    #complete embedding lookups
    inputs = tf.nn.embedding_lookup(embeddings, np.array(inputs, dtype = int), max_norm=None, name=None)
    inputs = tf.reshape(inputs, (len(outputs), -1))

    return inputs, outputs
