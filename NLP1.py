## Azizu Ahmad Rozaki Riyanto
import pandas as pd
## Dataset from https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp/data
train_df = pd.read_csv("/kaggle/input/sentimen/train.txt",delimiter=';', header=None, names=['sentence','label'])
val_df = pd.read_csv("/kaggle/input/sentimen/val.txt",
                 delimiter=';', header=None, names=['sentence','label'])

tets_df = pd.read_csv("/kaggle/input/sentimen/test.txt",
                 delimiter=';', header=None, names=['sentence','label'])

df = pd.concat([train_df, val_df, tets_df], axis=0)
df = df.reset_index()

df = df.drop(columns='index')
df.tail(10)

df.info()

df.isnull().sum()
df.iloc[19999,:]

temp = df.duplicated(keep = False)
df[temp]

df = df.drop_duplicates()

df.info()

unique_values = df.apply(lambda x: x.unique())
print(unique_values)

temp = pd.get_dummies(df.label)
df = pd.concat([df,temp],axis=1)
df = df.drop(columns='label')
df.head()

df.info()

df['sentence'] = df['sentence'].apply(str.lower)
df['sentence']

feature = df['sentence'].values
label = df[['anger','fear','joy','love','sadness','surprise']]

import nltk
from nltk.stem import PorterStemmer
import numpy as np

stemmer = PorterStemmer()
feature = np.vectorize(stemmer.stem)(feature)

feature

len(label)

from sklearn.model_selection import train_test_split
x_train,x_val,y_train,y_val = train_test_split(feature,label,test_size=0.2,random_state=42,shuffle=True,stratify = label)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

token = Tokenizer(num_words=10000)
token.fit_on_texts(x_train)
token.fit_on_texts(x_val)

print(len(token.word_index))

sekuens_train = token.texts_to_sequences(x_train)
sekuens_val = token.texts_to_sequences(x_val)

pad_train = pad_sequences(sekuens_train,maxlen =50)
pad_val = pad_sequences(sekuens_val,maxlen =50)

import tensorflow as tf
from tensorflow.keras.optimizers import Adamax
from keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
import os

learning_rate = 0.001

model = tf.keras.Sequential([
tf.keras.layers.Embedding(input_dim=10000, output_dim=64,input_length=50,trainable =True),
tf.keras.layers.LSTM(256),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Dense(512, activation='relu'),
tf.keras.layers.Dropout(0.5),
tf.keras.layers.Dense(1024, activation='relu'),
tf.keras.layers.Dense(512, activation='relu'),
tf.keras.layers.Dense(6, activation='softmax')
])

for layer in model.layers :
    layer.trainable = True

model.compile(loss='categorical_crossentropy',optimizer=Adamax(learning_rate),metrics=['accuracy'])

num_epochs = 50
batch_size = 8
folder_path = f"run/"

callbacks = [
            tf.keras.callbacks.ModelCheckpoint(os.path.join(folder_path, f"best_modelfold10.h5"), save_best_only=True, verbose=1),
            tf.keras.callbacks.EarlyStopping(patience=100, monitor='val_loss', verbose=1),
            tf.keras.callbacks.TensorBoard(log_dir='logs')
            ]

history = model.fit(pad_train, y_train, epochs=num_epochs,
                    validation_data=(pad_val, y_val), verbose=2,batch_size=batch_size,callbacks=callbacks,)


