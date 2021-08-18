import pandas as pd
import numpy as np
import os
import re
import string
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==============================================================#
# READ IN THE DATA
# ==============================================================#

# The translation file
df = pd.read_table("Data/deu.txt", sep="\t", header=None)
df = df.iloc[:, [0, 1]]
df.columns = ['English', 'German']
df.head()

# The contractions file
with open("Data\contraction_expansion.txt", "rb") as f:
    contractions = pickle.load(f)


# ==============================================================#
# CLEAN THE DATA
# ==============================================================#

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
remove_punct = str.maketrans('', '', string.punctuation)
df['English'] = df['English'].apply(lambda x: x.translate(remove_punct))
df['German'] = df['German'].apply(lambda x: x.translate(remove_punct))

# Delete all digits
df['English'] = df['English'].apply(lambda x: re.sub(r'[\d]+', '', x))
df['German'] = df['German'].apply(lambda x: re.sub(r'[\d]+', '', x))

# ==============================================================#
# PREPARE THE DATA FOR MODELING
# ==============================================================#

# Add the SOS and EOS tokens to the target language
df['German'] = "SOS " + df['German'] + " EOS"


# Tokenize the sentences
def tokenize(text):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    return tokenizer, sequences

eng_tokenizer, eng_sequences = tokenize(df['English'])
ger_tokenizer, ger_sequences = tokenize(df['German'])


# Create word-index-word mappings
eng_word2idx = eng_tokenizer.word_index
eng_idx2word = eng_tokenizer.index_word

ger_word2idx = ger_tokenizer.word_index
ger_idx2word = ger_tokenizer.index_word


# Get the vocabulary size
ENG_VOCAB_SIZE = len(eng_tokenizer.word_index) + 1    # 1 is added for the padding token
GER_VOCAB_SIZE = len(ger_tokenizer.word_index) + 1    # 1 is added for the padding token
print(f"English Vocab size : {ENG_VOCAB_SIZE} \nGerman Vocab size  : {GER_VOCAB_SIZE}")


# Get the maximum length of the english and german sentences
ENG_MAX_LEN = max([len(i) for i in eng_sequences])
GER_MAX_LEN = max([len(i) for i in ger_sequences])
print(f"Longest English sentence : {ENG_MAX_LEN} \nLongest German sentence  : {GER_MAX_LEN}")


# Padding the sequences
eng_padded = pad_sequences(eng_sequences, maxlen=ENG_MAX_LEN, padding='post')
ger_padded = pad_sequences(ger_sequences, maxlen=GER_MAX_LEN, padding='post')


# Convert to an array
eng_padded = np.array(eng_padded)
ger_padded = np.array(ger_padded)


# ==============================================================#
# BUILD THE MODEL
# ==============================================================#

