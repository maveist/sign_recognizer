import operator
import os

import numpy as np
import tensorflow as tf

from sign_recognizer.model.utils import get_root_project_path, get_word_list


class SignDetector:

    def __init__(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(8, activation='softmax')
        ])

    def load(self, path):
        self.model = tf.keras.models.load_model(path)

    def compile(self):
        self.model.compile(optimizer='rmsprop',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, x_input, y_input):
        self.model.fit(x_input, y_input, epochs=70, callbacks=[self.cp_callback])

    def evaluate(self, x_input):
        probability_model = tf.keras.Sequential([
            self.model,
            tf.keras.layers.Softmax()
        ])
        res = probability_model(x_input)
        word_list = get_word_list()
        word_max_predict = {}
        for word in word_list:
            word_max_predict[word] = 0
        for predict in res:
            word_max_predict[word_list[np.argmax(predict)]] += 1
        print(word_max_predict)
        print(res)
        choosen_word = max(word_max_predict.items(), key=operator.itemgetter(1))[0]
        return choosen_word
