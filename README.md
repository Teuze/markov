# Markov

Generating and estimating likelihood of semi-structured character strings

## 1. Project prerequisites

This project uses the following libraries:

```
docopt
dill
toml
pandas
numpy
```

This list is subject to change, in doubt please check file `requirements.txt`.

## 2. Basic usage

### 2.1 Training a model

Training data should be CSV-formatted. As an example, training trigrams :

```
./markov.py train target-model.npz using path/to/training/data.csv
```

The model type can be changed using the parameters file `config.toml`

### 2.2 Sequence generation

Once the model is trained, you can use it for sequence generation.

```
./markov.py generate synthesized-data.csv using target-model.npz
```

You can choose between generating a pure sequence (formatted as a csv table)
or a concatenated (character string) sequence, by changing in `markov.py` the value
of parameter `joined` in the function called `generate`.

### 2.3 Statistical measures

You can also use a trained model to measure likelihood of a sequence.
The idea behind this measure is to compare the sequence probability observed in the training data, and the sequence probability observed in the wild. Unusual transitions should make a sequence pop up.

```
./markov.py estimate real-data.csv using target-model.npz output statistics.csv
```

## 3. Project overview

At the core of `markov` are three Python files : `model`,`chain` et `sequencer`.

 - The first one is about abstracting statistical computations and providing a high-level interface for sequence generation and likelihood measurement.

 - The second one turns token sequences in graphs by leveraging adjacency matrices.

 - The last one uses regular expressions to turn character strings into token sequences.

