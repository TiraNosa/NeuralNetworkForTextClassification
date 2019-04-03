from sklearn.model_selection import train_test_split

# np.random.seed(1337)

import os
import pandas as pd

import utility

# Preparing the text data

texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids

trainingData = "V1.4_Training_new.csv"
testData = "SubtaskB_EvaluationData_labeled.csv"
out_path = testData[:-4] + "_CNN_batch100_FreqThreshold=5_MaxEpoch=10_2.csv"

data = pd.read_csv(trainingData)
for index, row in data.iterrows():
    texts.append(row['sentence'])
    labels.append(row['label'])

labels_index['0'] = 0
labels_index['1'] = 1
print('Found %s texts.' % len(texts))

# format our text samples and labels into tensors that can be fed into a neural network
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np

VALIDATION_SPLIT = 0.2

MAX_NUM_WORDS = 20000
MAX_SEQUENCE_LENGTH = 1000


def preprocessing(texts):
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(texts)

    # removing low freq words
    count_thres = 5
    low_count_words = [w for w, c in tokenizer.word_counts.items() if c < count_thres]
    # print(tokenizer.texts_to_sequences(texts))
    for w in low_count_words:
        del tokenizer.word_index[w]
        del tokenizer.word_docs[w]
        del tokenizer.word_counts[w]
    # print(tokenizer.texts_to_sequences(texts))

    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    return sequences, word_index


sequences, word_index = preprocessing(texts)
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train, x_val, y_train, y_val = train_test_split(data, labels, test_size=0.33, random_state=0)

# Preparing the Embedding layer
# Next, we compute an index mapping words to known embeddings, by parsing the data dump of pre-trained embeddings:
GLOVE_DIR = os.path.join('Glove')

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# At this point we can leverage our embedding_index dictionary and our word_index to compute our embedding matrix:
EMBEDDING_DIM = 100
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# We load this embedding matrix into an Embedding layer.
# Note that we set trainable=False to prevent the weights from being updated during training.
from keras.layers import Embedding

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

# Training a 1D convnet
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.models import Model
import keras_metrics

BATCH_SIZE = 100
MAX_EPOCH = 10
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(BATCH_SIZE, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(BATCH_SIZE, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(BATCH_SIZE, 5, activation='relu')(x)
x = GlobalMaxPooling1D()(x)
x = Dense(BATCH_SIZE, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

from keras.callbacks import ModelCheckpoint

filepath = "weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

CUR_EPOCH = 0
acc = 0
# while acc <= 0.8:
#     model.fit(x_train, y_train, validation_data=(x_val, y_val),
#               epochs=1, batch_size=BATCH_SIZE, callbacks=[checkpoint])
#     score, acc = model.evaluate(x_val, y_val,
#                                 batch_size=BATCH_SIZE)
#     CUR_EPOCH += 1
#     print("Current Epoch=" + str(CUR_EPOCH))
#     if CUR_EPOCH == MAX_EPOCH:
#         break

model.fit(x_train, y_train, validation_data=(x_val, y_val),
          epochs=MAX_EPOCH, batch_size=BATCH_SIZE, callbacks=[checkpoint])

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            input_length=MAX_SEQUENCE_LENGTH)

from numpy import array

def readFile(file):
    df = pd.read_csv(file)
    df.head()

    # add a column encoding the label as an integer
    # because categorical variables are often better represented by integers than strings
    col = ['label', 'sentence']
    df = df[col]
    df = df[pd.notnull(df['sentence'])]
    df.columns = ['label', 'sentence']
    df['category_id'] = df['label']
    df['label_str'] = df['label'].replace({1: 'suggestion', 0: 'non-suggestion'})
    df.head()
    return df

def classify(sent_list):
    df = readFile(testData)
    label_list = []

    sequences, word_index = preprocessing(df.sentence)

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    # split the data into a training set and a validation set
    # indices = np.arange(data.shape[0])
    # np.random.shuffle(indices)
    # data = data[indices]

    Xnew = array(data)

    # load weights
    model.load_weights(filepath)
    # Compile model (required to make predictions)
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    # score, acc = model.evaluate(x_val, y_val, batch_size=BATCH_SIZE)
    # print(acc)
    result_list = model.predict(Xnew)
    for result in result_list:
        if result[0] > result[1]:
            label_list.append(0)
        else:
            label_list.append(1)


    category_id_df = df[['label_str', 'category_id']].drop_duplicates().sort_values('category_id')

    y_val = df.label
    from sklearn import metrics

    print(metrics.classification_report(y_val, label_list, target_names=df['label'].astype('str').unique()))
    return label_list


if __name__ == '__main__':
    sent_list = utility.read_csv(testData)
    label_list = classify(sent_list)
    utility.write_csv(sent_list, label_list, out_path)
