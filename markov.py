#!/usr/bin/python3

"""
Markov Modeling Tool.

Usage:
  markov train <model> using <data>
  markov generate <target> using <model>
  markov estimate <data> using <model> output <target>
  markov -h | --help
  markov --version

Options:
  -h --help     Show this screen.
  --version     Show version.
"""

__author__ = "Richard Jarry"
__version__ = "0.1"

import toml
from docopt import docopt

import pandas as pd
import markov.model as mm
import markov.sequencer as ms

def train(data,model,s_type,s_option):

    df = pd.read_csv(data)
    s = df[df.columns[0]]

    if s_type == 'ngrams':
        sequencer = lambda s: ms.ngrams(s,s_option)

    elif s_type == 'regex':
        sequencer = lambda s: ms.regex(s,s_option)

    else: raise NameError ("Unknown sequencer type.")

    m = mm.MarkovModel(sequencer)
    graph = m.train(s)
    m.save(model)
    return graph

def generate(model,target,sample_size,joined=True):

    m = mm.MarkovModel(None)
    m.load(model)

    width,length = sample_size
    
    if joined:
        seed = lambda: ''.join(m.generate(length))
        result = pd.Series([seed() for _ in range(width)])
    else:
        seed = lambda: m.generate(length)
        result = pd.DataFrame([seed() for _ in range(width)])
    
    result.to_csv(target)
    return result

def estimate(model,target,data):

    m = mm.MarkovModel(None)
    m.load(model)

    df = pd.read_csv(data)
    s = df[df.columns[0]]

    result = m.likelihood(s)
    result.to_csv(target)
    return result

if __name__ == "__main__":

    args = docopt(__doc__)
    opts = toml.load('config.toml')

    data = args['<data>']
    model = args['<model>']
    target = args['<target>']

    if args['train']:
        s_type = opts['sequencer']['type']
        s_opts = opts['sequencer']['option']
        train(data,model,s_type,s_opts)

    elif args['generate']:
        length = opts['generator']['max_size']
        width = opts['generator']['n_samples']
        generate(model,target,(width,length))

    elif args['estimate']:
        estimate(model,target,data)

    elif args['--version']:
        print(__version__)
