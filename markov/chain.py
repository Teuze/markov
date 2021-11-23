import pandas as pd
import numpy as np

def build_edges(selected:pd.DataFrame,labels:np.array,weights=None):

    if weights is None: weigths = pd.Series(1,index=selected.index)

    assert(len(selected) == len(weights))

    selected['weight'] = weights

    extracted = (selected
                .set_axis([0,1,2],axis=1)
                .groupby([0,1]).sum()
                .unstack()[2]
                .reindex(columns=labels)
                .reindex(index=labels)
                .fillna(0))

    return extracted

def build_nodes(selected:pd.DataFrame,labels:np.array,weights=None):

    if weights is None: weigths = pd.Series(1,index=selected.index)

    assert(len(selected) == len(weights))

    selected['weight'] = weights
    c = selected.columns[0]

    extracted = (selected
                .groupby(c).sum()
                .reindex(labels)
                .fillna(0))

    return extracted['weight']

def build_graph(sequence:pd.DataFrame, weights=None):

    labels = pd.unique(sequence.values.flatten())

    edges = pd.DataFrame(0,dtype=np.int32,index=labels,columns=labels)
    nodes = pd.Series(0,dtype=np.int32,index=labels)
    
    if weights is None: weigths = pd.Series(1,index=sequence.index)

    for c in sequence.columns[:-1]:

        nodes_temp = build_nodes(sequence[[c]],labels,weights)
        edges_temp = build_edges(sequence[[c,c+1]],labels,weights)
        nodes = nodes + nodes_temp
        edges = edges + edges_temp

    nodes = nodes + build_nodes(sequence[[c+1]],labels,weights)

    return (nodes,edges)

def build(sequence:pd.DataFrame, weights=None):

    return build_graph(sequence,weights)