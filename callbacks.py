'''
# Keras Callbacks
This notebook contains a set of customized keras callbacks that 
should enable increased efficiency while testing your keras model, 
not only improving your metrics but also saving you valuable time 
and resources.
'''

import keras.callbacks

#1. Save loss values after each batch
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

