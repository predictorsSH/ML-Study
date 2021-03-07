import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import tensorflow as tf
print(tf.__version__)

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

from tensorflow.keras.utils import to_categorical
y=to_categorical(y)
print(y.shape)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

x=pd.get_dummies(x)
x=scaler.fit_transform(x)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3,shuffle=False)

print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

type(X_test)

X_train=X_train.reshape((-1,118,1))
X_test=X_test.reshape((-1,118,1))

learning_rate = 0.001
training_epochs = 3
batch_size = 100
#
# model = tf.keras.Sequential([
#     tf.keras.layers.LSTM(128 ,activation='relu',input_shape=(118,1)),
#     tf.keras.layers.Dense(128),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(64),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(23,activation='softmax')
# ])
#
# model.compile(optimizer='nadam',loss='categorical_crossentropy', metrics=['accuracy'])
#
# #Epoch 1/3
# #345814/345814 [==============================] - 344s 993us/sample - loss: 861217.4126 - accuracy: 0.6462
#
# # from tensorflow.keras.utils import to_categorical
# # y_train=to_categorical(y_train)
#
# model.fit(X_train, y_train,epochs=training_epochs ,batch_size=100)

