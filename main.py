#=====================================================#
# IMPORTS
#=====================================================#
import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import os

#=====================================================#
# CONFIGURATIONS
#=====================================================#
BATCH_SIZE = 64
EPOCHS = 100
LATENT_DIM = 256
NUM_SAMPLES = 10000
MAX_SEQ_LENGTH = 100
VOCAB_SIZE = 20000
EMBEDDING_DIM = 100

#=====================================================#
# READ IN AND FORMAT THE DATA
#=====================================================#

# Read in the language file
df = pd.read_table('deu.txt', sep='\t', header=None)
df = df[[0,1]]
df.columns = ['English','German']
df = df.head(NUM_SAMPLES)

input_texts = df['English'].to_list()
target_texts = (df['German'] + " <EOS>").to_list()
target_texts_inputs = ("<SOS> " + df['German']).to_list()

print("Number of samples : ", len(input_texts))


# Tokenize the inputs
tokenizer_inputs = Tokenizer(num_words=VOCAB_SIZE)
tokenizer_inputs.fit_on_texts(input_texts)
input_sequences = tokenizer_inputs.texts_to_sequences(input_texts)

num_words_input = len(tokenizer_inputs.word_index) + 1
print("Size of input vocabulary :", num_words_input)

max_len_input = max((len(i) for i in input_sequences))
print("Length of maximum input sequence :", max_len_input)

# Tokenize the outputs
tokenizer_outputs = Tokenizer(num_words=VOCAB_SIZE, filters='')
tokenizer_outputs.fit_on_texts(target_texts + target_texts_inputs)
target_sequences = tokenizer_outputs.texts_to_sequences(target_texts)
target_sequences_inputs = tokenizer_outputs.texts_to_sequences(target_texts_inputs)

num_words_output = len(tokenizer_outputs.word_index) + 1
max_len_target = max((len(i) for i in target_sequences))
print("Length of maximum output sequence :", max_len_target)


# pad the sequences
encoder_inputs = pad_sequences(input_sequences, maxlen=max_len_input)
print("Encoder inputs shape :", encoder_inputs.shape)

decoder_inputs = pad_sequences(target_sequences_inputs, maxlen=max_len_target, padding='post')
decoder_targets = pad_sequences(target_sequences, maxlen=max_len_target, padding='post')
print("Decoder inputs shape :", encoder_inputs.shape)

#=====================================================#
# LOAD THE WORD VECTORS
#=====================================================#

glove = {}
with open(r'D:\Machine Learning\Glove\glove.6B.100d.txt', encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        glove[word] = vector

print("Loaded the Glove vectors")

#=====================================================#
# PREPARE THE EMBEDDING MATRIX
#=====================================================#

matrix_rows = min(VOCAB_SIZE, num_words_input)
embedding_matrix = np.zeros((matrix_rows, EMBEDDING_DIM))

for word, i in tokenizer_inputs.word_index.items():
    embedding_vector = glove.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

print("Created the embedding matrix")


#=====================================================#
# START BUILDING THE NEURAL NETWORK
#=====================================================#

# Create the embedding layer
embedding_layer = Embedding(input_dim = matrix_rows, output_dim=EMBEDDING_DIM,
                            weights=embedding_matrix,
                            input_length=max_len_input)

# Create decoded one-hot encoded tagrets
decoder_targets_one_hot = np.zeros((len(input_texts), max_len_target, num_words_output), dtype='float32')
print(decoder_targets_one_hot.shape)

for i, d in enumerate(decoder_targets):
    for t, word in enumerate(d):
        decoder_targets_one_hot[i, t, word] = 1

