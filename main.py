import pandas as pd
import numpy as np
import os
import re
import string
import pickle

#==============================================================#
# READ IN THE DATA
#==============================================================#

# The translation file
df = pd.read_table("Data/German.txt", sep="\t", header=None)
df = df.iloc[:,[0,1]]
df.columns = ['English', 'German']
df.head()

# The contractions file
with open("Data\contraction_expansion.txt", "rb") as f:
    contractions = pickle.load(f)


#==============================================================#
# CLEAN THE DATA
#==============================================================#

def replace_contractions(text):

    expanded = []
    for key in text.split():
        if key in contractions:
            value = contractions[key]
        else:
            value = key
        expanded.append(value)

    return ' '.join(expanded)

# Replace contractions
df['English'] = df['English'].apply(replace_contractions)

# Delete all punctuations
remove_punct = str.maketrans('','',string.punctuation)
df['English'] = df['English'].apply(lambda x : x.translate(remove_punct))
df['German'] = df['German'].apply(lambda x : x.translate(remove_punct))

# Delete all digits
df['English'] = df['English'].apply(lambda x: re.sub(r'[\d]+','', x))
df['German'] = df['German'].apply(lambda x: re.sub(r'[\d]+','', x))


df.head(10)