import tensorflow as tf
import numpy as np
from functools import reduce
import lzma


#hyperparameters
decay_rate = 1/2 #1/(2^delta_t)
opt_out_threshold = .125

def get_reachabilities(file_path):
    '''
    PROCESS

    Constructs either type of graph for each 10k increments of unique pairs

    At next timestep,
        - update all weights of existing nodes and edges (decay previous weights and add new weight)
        - delete any nodes with

    '''

    #gets list of new edges to be updated for each of 10 timesteps
    file = lzma.open(file_path, mode='rt')
    edges = np.loadtxt(file, np.int)
    sender = edges[:,1]
    receiver = edges[:,2]

    graphs = []
    pairs = list(zip(sender, receiver))
    while len(graphs) < 10:
        graph_edges = []
        while len(set(map(tuple,graph_edges))) < 10000:
            graph_edges.append(list(pairs.pop(0)))
        graphs.append(graph_edges)

    graphs = np.array(graphs)
    print(np.shape(graphs))
    edges_data = np.reshape(graphs, (10,-1, 2))
    print(edges_data.shape)

    '''
    print(graphs[1])
    newgraphs = []
    for g in range(len(graphs)):
        newgraphs.append([np.array(list(x)) for x in graphs[g]])


        #zip(*x)

    #print(newgraphs)


    print(np.array(newgraphs).shape)

    newgraphs = np.array(newgraphs)#.flatten()


    print(np.array(newgraphs[0]).flatten())
    exit()



    print(np.array(newgraphs).shape)
    '''



    '''
    #create our first graph
    graph = dgl.DGLGraph()
    graph = dgl.add_nodes(graph, len(set(list(   ) + len(set(list(    ))))))


    for i in range(11):

        #if nodes and edges already there, decay all weights
        if graph.number_of_edges() != 0:
            pass
        #add in new nodes and edges, updating if there and creating if not

        #output the graph
        pass



    graph = dgl.add_nodes(graph, len(set(list(sender) + list(receiver))))

    src = []
    dst = []
    for i in molecule.edges:
        src.append(i[0])
        dst.append(i[1])

    graph = dgl.add_edges(graph, src, dst)
    graph = dgl.add_edges(graph, dst, src)
    '''

    return None

def get_amounts(file_path):
    '''
    PROCESS

    Constructs either type of graph for each 10k increments of unique pairs

    At next timestep,
        - update all weights of existing nodes and edges (decay previous and add new weight)

    '''

    file = lzma.open('../../.dat.xz', mode='rt')

    return None


def get_randomWalks():
    #get random walks for a given graph

    pass
