import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
print(tf.__version__)

import matplotlib as mp

import pandas as pd
df=pd.read_csv('kddcup99_csv.csv')

print(df.columns)
print(df.shape) #(494020,42)

print(df.describe())
print(df.label.value_counts())

print(df.info())

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
df['label']=encoder.fit_transform(df['label'])
print(df['label'])
x=df[df.columns.difference(['label'])]
y=df['label']

#duration -->
x=pd.get_dummies(x)


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3,shuffle=False,random_state=1004)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

X_train=scaler.fit_transform(X_train)
X_test=scaler.fit_transform(X_test)

len(df['label'].unique())

learning_rate = 0.001
training_epochs = 3
batch_size = 30


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(23, activation='softmax')
])


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

import numpy as np
model.fit(X_train, np.array(y_train),epochs=training_epochs,validation_split=0.1,batch_size=batch_size)
model.evaluate(X_test,np.array(y_test), verbose=2)
#148206/148206 - 10s - loss: 1.7095 - accuracy: 0.8600
