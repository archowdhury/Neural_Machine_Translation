import pandas as pd
import numpy as np
import os
import re
import string
import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Concatenate, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from Attention import AttentionLayer

from sklearn.model_selection import train_test_split

# ==============================================================#
# DEFINE THE PARAMETERS
# ==============================================================#

EMBEDDING_DIM = 300

# ==============================================================#
# READ IN THE DATA
# ==============================================================#

# The translation file
df = pd.read_table("Data/deu.txt", sep="\t", header=None)
df = df.iloc[:, [0, 1]]
df.columns = ['English', 'German']
df = df.head(2000)
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

# BUILD THE ENCODER
#----------------------------------------------------------------

# Encoder input
encoder_inputs = Input(shape=(ENG_MAX_LEN,))

# Embedding layer
encoder_embedding = Embedding(ENG_VOCAB_SIZE, EMBEDDING_DIM)(encoder_inputs)

# Bidirectional LSTM layer
encoder_LSTM = Bidirectional(LSTM(15, return_sequences=True, return_state=True))

# Encoder outputs
encoder_outputs1, fwd_state_h, fwd_state_c, bck_state_h, bck_state_c = encoder_LSTM(encoder_embedding)

# Concatenate both 'h' and 'c' states
final_encoder_h = Concatenate()([fwd_state_h, bck_state_h])
final_encoder_c = Concatenate()([fwd_state_c, bck_state_c])

# Get the context vector
encoder_states = [final_encoder_h, final_encoder_c]



# BUILD THE DECODER
#----------------------------------------------------------------

# Decoder inputs
decoder_inputs = Input(shape=(None,))

# Decoder embedding
decoder_embedding = Embedding(GER_VOCAB_SIZE, EMBEDDING_DIM)(decoder_inputs)

# Decoder LSTM
decoder_LSTM = LSTM(30, return_sequences=True, return_state=True)

# Decoder outputs
decoder_outputs, _, _ = decoder_LSTM(decoder_embedding, initial_state=encoder_states)

# Add the attention layer
attention_layer = AttentionLayer()
attention_results, attention_weights = attention_layer([encoder_outputs1, decoder_outputs])

# Concatenate the attention output and the decoder LSTM output
decoder_concat_input = Concatenate(axis=-1, name='decoder-concat-layer')([decoder_outputs, attention_results])

# Dense layer
decoder_dense = Dense(GER_VOCAB_SIZE, activation='softmax')

# decoder outputs
decoder_outputs = decoder_dense(decoder_concat_input)

# Define the final model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


# ==============================================================#
# TRAIN THE MODEL
# ==============================================================#

X_train, X_test, y_train, y_test = train_test_split(eng_padded, ger_padded, test_size=0.1, random_state=0)

# Define callbacks

checkpoint = ModelCheckpoint("give Your path to save check points", monitor='val_accuracy')
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5)
callbacks_list = [checkpoint, early_stopping]

# Training set
encoder_input_data = X_train
# To make same as target data skip last number which is just padding
decoder_input_data = y_train[:,:-1]
# Decoder target data has to be one step ahead so we are taking from 1 as told in keras docs
decoder_target_data = y_train[:,1:]

# devlopment set
encoder_input_test = X_test
decoder_input_test = y_test[:,:-1]
decoder_target_test= y_test[:,1:]

EPOCHS= 30 #@param {type:'slider',min:10,max:100, step:10 }
history = model.fit([encoder_input_data, decoder_input_data],decoder_target_data,
                    epochs=EPOCHS,
                    batch_size=32,
                    validation_data = ([encoder_input_test, decoder_input_test],decoder_target_test),
                    callbacks= callbacks_list)