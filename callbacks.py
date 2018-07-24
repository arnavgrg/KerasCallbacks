'''
Keras Callbacks

This notebook contains a set of customized keras callbacks that 
should enable increased efficiency while testing your keras model, 
not only improving your metrics but also saving you valuable time 
and resources.

To add more than one callback:
model.fit(train_x,train_y,callbacks=[callback_1,callback_2,callback_3,...])
'''

from __future__ import print_function
from tensorflow import keras
import warnings

class increment_in_loss(keras.callbacks.Callback):
    
    '''
    Stop training if loss does not decrease over n consecutive epochs.
    The default and recommended value for n is 3.

    Usage: 
        history = increment_in_loss(2)
        model.fit(train_x,train_y,callbacks=[history])

    View epoch number at which loss starts to increase:
        print(history.epoch_count)

    View loss before increase over n consecutive epochs:
        print(history.losses[-1])
    '''
    
    def __init__(self, threshold=3):
        self.threshold = threshold
        self.epoch_count = 0
        self.error_count = 0
        self.losses = []

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        if (self.epoch_count):
            if self.losses[self.epoch_count - 1] < self.losses[self.epoch_count]:
                #Optionally raise a warning
                #warnings.warn("Detected increase in loss..")
                #warnings.simplefilter("always")
                self.error_count += 1
                if self.error_count == self.threshold:
                    self.model.stop_training = True
            else:
                self.error_count = 0
        self.epoch_count += 1

    def on_train_end(self, logs={}):
        self.epoch_count -= self.threshold
        for x in range(self.threshold):
            index_ = -x-1
            self.losses.pop(index_)

class loss_after_each_batch(keras.callbacks.Callback):
    
    '''
    Save loss values after each batch instead of after
    each epoch.

    Usage: 
        history = loss_after_each_batch()
        model.fit(train_x,train_y,callbacks=[history])

    View view complete list of losses after each batch:
        print(history.losses)

    To view loss on the last batch:
        print(history.losses[-1])
    '''

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

#3. Detect oscillation due to high learning rate