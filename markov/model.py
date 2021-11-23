#!/usr/bin/python3

import pandas as pd
import numpy as np
import dill

from . import sequencer
from . import chain

class MarkovModel(object):

    def __init__(self,sequencer):

        self.sequencer = sequencer
        self.transitions = None
        self.priors = None

    def load(self,file_path:str):

        with np.load(file_path,allow_pickle=True) as data:
            s = data['sequencer'][0]
            t = data['transitions']
            l = data['labels']
            p = data['priors']

        self.sequencer = dill.loads(s)
        self.priors = pd.Series(p,index=l)
        self.transitions = pd.DataFrame(t,index=l,columns=l)

    def save(self,file_path:str):

        s = np.array([dill.dumps(self.sequencer)])
        t = self.transitions.values
        l = self.priors.index.values
        p = self.priors.values

        np.savez(file_path,
            sequencer = s,
            transitions = t,
            labels = l,
            priors = p)
            
    def train(self,data:pd.Series,weights=None):

        if weights is None:
            weights = pd.Series(data.value_counts().values)
            data = pd.Series(data.value_counts().index)

        # fetching sequence tokens
        sequence = self.sequencer(data)
        
        # sequence to graph (nodes,edges)
        nodes,edges = chain.build(sequence,weights)
        
        # normalizing transitions with marginals
        marginals = edges.sum(axis='rows')
        transitions = edges.div(marginals,axis='columns').fillna(0)

        # computing priors
        priors = pd.concat([sequence[0],weights],axis=1)
        priors = priors.set_axis([0,1],axis=1)
        priors = (priors
                 .groupby(0)
                 .sum()[1]
                 .reindex_like(nodes)
                 .fillna(0))
  
        self.priors = priors / priors.sum()
        self.transitions = transitions

        return (nodes,edges)

    def update(self,data:pd.Series):

        # train another model with additional data
        other = MarkovModel(self.sequencer)
        graph = other.train(data)
        
        mean = lambda x,y: (x+y)/2

        # assign mean values for all priors and transitions
        self.priors = self.priors.combine(other.priors,mean,0)
        self.transitions = self.transitions.combine(other.transitions,mean,0)

        return graph

    def generate(self,max_size:int):

        result = []

        # draw first sample from prior distribution
        first = self.priors.sample(1,weights=self.priors.values)
        
        result += list(first.index)

        # for i in size, generate ith sample by using transitions
        for i in range(max_size-1):
            before = result[i]
            current = self.transitions.loc[before]
            if all(current == 0): return result
            else: after = current.sample(1,weights=current.values)
            result += list(after.index)

        return result
            
    def likelihood(self,sample:pd.Series):

        full_sequence = self.sequencer(sample)
        
        known_tokens = full_sequence.isin(self.priors.index).prod(axis=1).astype(np.bool)
        
        sequence = full_sequence[known_tokens]

        # warn if tokens from sequence are unknown by the model
        if not known_tokens.prod(): print("Warning: New (unknown) token(s) in sample")

        # compute initial probabilities as priors empirical distribution
        probas = pd.Series(0,dtype=np.float64,index=full_sequence.index)
        
        probas.loc[sequence.index] = self.priors[sequence[0]].values
        
        # multiply for each transition
        # by the corresponding probability
        for c in sequence.columns[:-1]:

            mask  = sequence[c+0].notnull()
            mask &= sequence[c+1].notnull()

            selected = sequence[mask][[c,c+1]].rename(columns={c:0,c+1:1})

            current_probability = self.transitions.loc[selected[0],selected[1]]
            current_probability = pd.Series(np.diag(current_probability))
            current_probability.index = selected.index

            next_probability = probas * current_probability
            notnull = next_probability.notnull()
            probas[notnull] = next_probability[notnull]

        # add sequence length and normalize probabilities
        length = full_sequence.notnull().sum(axis=1)
        normed = np.power(probas,1/length)

        result = pd.DataFrame(index=full_sequence.index)
        result['sample'] = sample
        result['length'] = length
        result['probas'] = probas
        result['normed'] = normed

        return result
