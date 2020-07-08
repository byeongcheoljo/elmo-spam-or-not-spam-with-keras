import tensorflow_hub as hub
import tensorflow as tf
from keras import backend as K
import os
from tensorflow.keras.utils import multi_gpu_model

# gpu 0,1,2,3 SETTING
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

sess = tf.Session()
K.set_session(sess)

elmo = hub.Module("https://tfhub.dev/google/elmo/1", trainable=True)

sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())

import pandas as pd
data = pd.read_csv("/spam.csv", encoding='latin-1') ## data

data['v1'] = data['v1'].replace(['ham','spam'],[0,1])

y_data = list(data['v1'])
x_data = list(data['v2'])

print(x_data[:5])
print(y_data[:5])

print(len(x_data))
num_of_train = int(len(x_data)*0.8)
num_of_test = int(len(x_data) - num_of_train)
print(num_of_train)
print(num_of_test)

import numpy as np
X_train = np.asarray(x_data[:num_of_train])
y_train = np.asarray(y_data[:num_of_train])
X_test = np.asarray(x_data[num_of_train:])
y_test = np.asarray(y_data[num_of_train:])

def ELMoEmbedding(x):
    return elmo(tf.squeeze(tf.cast(x, tf.string)), as_dict=True, signature="default")["default"]


from keras.models import Model
from keras.layers import Dense, Lambda, Input

input_text = Input(shape=(1,), dtype=tf.string)
embedding_layer = Lambda(ELMoEmbedding, output_shape=(1024, ))(input_text)
hidden_layer = Dense(256, activation='relu')(embedding_layer)
output_layer = Dense(1, activation='sigmoid')(hidden_layer)
model = Model(inputs=[input_text], outputs=output_layer)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20, batch_size=100)
print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))
model.summary()
