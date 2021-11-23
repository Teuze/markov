import pandas as pd

def regex(s:pd.Series,pat:str) -> pd.DataFrame:

    sequence = s.str.extractall(pat).unstack()
    sequence.columns = sequence.columns.droplevel()

    return sequence

def ngrams(s:pd.Series,order:int=3) -> pd.DataFrame:

    return regex(s,f'(.{{{order}}})')
