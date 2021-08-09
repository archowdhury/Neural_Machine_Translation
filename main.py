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
import csv
import matplotlib.pyplot as plt

pd.set_option('display.width', 200)


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
# READ IN THE DATA
#=====================================================#

# Read in the language file
df = pd.read_table('deu.txt', sep='\t', header=None)
df = df[[0,1]]
df.columns = ['English','German']
df = df.head(NUM_SAMPLES)


input_text = df['English'].to_list()
output_text = (df['German'] + " <EOS>").to_list()
target_input = ("<SOS> " + df['German']).to_list()

print("Number of samples : ", len(input_text))

#=====================================================#
# PROCESS THE DATA FOR MODELING
#=====================================================#

# Tokenize the inputs
tokenizer_inputs = Tokenizer(num_words=VOCAB_SIZE)
tokenizer_inputs.fit_on_texts(input_text)
input_sequences = tokenizer_inputs.texts_to_sequences(input_text)

num_words_input = len(tokenizer_inputs.word_index) + 1
print("Size of input vocabulary :", num_words_input)

max_len_input = max((len(i) for i in input_sequences))
print("Length of maximum input sequence :", max_len_input)


# Tokenize the outputs
tokenizer_outputs = Tokenizer(num_words=VOCAB_SIZE, filters='')
tokenizer_outputs.fit_on_texts(output_text + target_input)
output_sequences = tokenizer_outputs.texts_to_sequences(output_text)
target_sequences = tokenizer_outputs.texts_to_sequences(target_input)

num_words_output = len(tokenizer_outputs.word_index) + 1
max_len_target = max((len(i) for i in target_sequences))
print("Length of maximum output sequence :", max_len_target)


# Pad the sequences
encoder_data = pad_sequences(input_sequences, maxlen=max_len_input)
print("Encoder inputs shape :", encoder_data.shape)

decoder_data = pad_sequences(output_sequences, maxlen=max_len_target, padding='post')
target_data = pad_sequences(target_sequences, maxlen=max_len_target, padding='post')
print("Decoder inputs shape :", decoder_data.shape, target_data.shape)


#=====================================================#
# LOAD THE WORD VECTORS
#=====================================================#

df_glove = pd.read_table(r'D:\Machine Learning\Glove\glove.6B.100d.txt',
                         sep=' ',
                         encoding="utf8",
                         quoting=csv.QUOTE_NONE,
                         header=None, index_col=[0])

glove = df_glove.T.to_dict('list')
glove = {key:np.asarray(val) for key, val in glove.items()}

print("Loaded the Glove vectors. Total words : ", len(glove))


#=====================================================#
# PREPARE THE EMBEDDING MATRIX
#=====================================================#

matrix_rows = min(VOCAB_SIZE, num_words_input)
embedding_matrix = np.zeros((matrix_rows, EMBEDDING_DIM))

for word, idx in tokenizer_inputs.word_index.items():
    embedding_vector = glove.get(word)
    if embedding_vector is not None:
        embedding_matrix[idx] = embedding_vector

print("Created the embedding matrix")


# the following code is just to re-verify that the embedding matrix has been correctly created
check_word = 'man'
index_in_matrix = tokenizer_inputs.word_index[check_word]

glove_vector = glove[check_word]
embedding_vector = embedding_matrix[index_in_matrix]

if np.array_equal(glove_vector, embedding_vector):
    print(f"The glove and embedding arrays for '{check_word}' match!")
else:
    print("Problem!!!!")



#=====================================================#
# START BUILDING THE NEURAL NETWORK
#=====================================================#

# Create decoded one-hot encoded targets
decoder_OHE = to_categorical(decoder_data)
print(decoder_OHE.shape)



#---------------------------------
# Create the Encoder
#---------------------------------

encoder_input_layer = Input(shape=(max_len_input,))

encoder_embedding_layer = Embedding(input_dim=matrix_rows,
                                    output_dim=EMBEDDING_DIM,
                                    weights=[embedding_matrix],
                                    input_length=max_len_input)

encoder_LSTM_layer = LSTM(LATENT_DIM, return_state=True, dropout=0.5)

x = encoder_input_layer
x = encoder_embedding_layer(x)
encoder_outputs, h, c = encoder_LSTM_layer(x)
encoder_states = [h, c]


#---------------------------------
# Create the Decoder
#---------------------------------

decoder_input_layer = Input(shape=(max_len_target,))

decoder_embedding_layer = Embedding(input_dim=num_words_output, output_dim=LATENT_DIM)

decoder_LSTM_layer = LSTM(LATENT_DIM, return_sequences=True, return_state=True, dropout=0.5)

decoder_dense_layer = Dense(num_words_output, activation='softmax')


x = decoder_input_layer
x = decoder_embedding_layer(x)
decoder_temp_outputs, _, _ = decoder_LSTM_layer(x, initial_state=encoder_states)
decoder_output_layer = decoder_dense_layer(decoder_temp_outputs)


# Create the final model
#-----------------------

model = Model([encoder_input_layer, decoder_input_layer], decoder_output_layer)
model.summary()

# Compile the model
#------------------

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


#=====================================================#
# TRAIN THE NEURAL NETWORK
#=====================================================#

r = model.fit([encoder_data, decoder_data],
              decoder_OHE,
              batch_size=BATCH_SIZE,
              epochs=20,
              validation_split=0.2)

#=====================================================#
# PLOT THE METRICS
#=====================================================#

plt.plot(r.history['loss'], label='training loss')
plt.plot(r.history['val_loss'], label='validation loss')
plt.legend()
plt.show()

plt.plot(r.history['accuracy'], label='training accuracy')
plt.plot(r.history['val_accuracy'], label='validation accuracy')
plt.legend()
plt.show()


#=====================================================#
# SAVE THE MODEL
#=====================================================#

model.save('Model\se2seq_base_model')