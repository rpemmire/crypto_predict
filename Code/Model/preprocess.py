import tensorflow as tf
import numpy as np
from functools import reduce
import lzma
import networkx as nx

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

    for edgelist in transactionLists:

        newEdgeList = []
        #make the edge list irrespective of direction
        #do this by sorting so that smaller id is in front
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

            # if edge already exists, make its attribute 1
            if G.has_edge(src, dst):
                G[src][dst]["weight"] +=1
            # else add in new edge with attribute 1
            else:
                G.add_edge(src,dst,weight=1)

        #3) update all nodes, if < threshold, delete node
        to_remove = []
        for node in G.nodes():
            edges = G.edges(node, data = True)
            sum = 0
            for edge in edges:
                sum += edge[2]['weight']
            if sum <= opt_out:
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


def get_randomWalks(graph, opt_out, length, walks_per_node):
    #get random walks for a given graph
    #output all sequences in forms of np.array(walks by walk_length)

    #for each node, get 10 random walks of length 40
        sequences = []
        #while len(sequences != 10)



    pass
