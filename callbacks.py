'''
--- Keras Callbacks ---

This notebook contains a set of customized keras callbacks that 
should enable increased efficiency while testing your keras model, 
not only improving your metrics but also saving you valuable time 
and resources.

To add more than one callback, just list add them to the list:
    model.fit(train_x,train_y,callbacks=[callback_1,callback_2,...])
'''

from __future__ import print_function
from tensorflow import keras
import warnings

def error_message(callback_name):
    print("Stopping training. "+str(callback_name)+" detected.")

class detect_oscillation_about_minima(keras.callbacks.Callback):

    '''
    Detect oscialltions about minima point caused by a high learning
    rate and stop training. (lack of convergence)

    Usage:
        history = detect_oscillation_about_minima()
        model.fit(train_x, train_y, callbacks=[history])

    To see mean loss/convergence value:
        print(history.loss_convergence)

    To see epoch number when model stops training:
        print(history.epoch_count)
    '''

    def __init__(self):
        self.epoch_count = 0
        self.losses = []
        self.loss_convergence = 0

    def on_train_begin(self, logs={}):
        self.losses = []
    
    def check_convergence(self):
        difference = self.losses[self.epoch_count] - self.losses[self.epoch_count - 1]
        change = abs((difference/self.losses[self.epoch_count-1])*100)
        if change < 0.001:
            self.loss_convergence = self.losses[self.epoch_count-1]
            return True

    def check_oscillation(self):
        if self.losses[self.epoch_count - 1] < self.losses[self.epoch_count]:
                if self.losses[self.epoch_count - 2] > self.losses[self.epoch_count - 1]:
                    if self.losses[self.epoch_count - 3] < self.losses[self.epoch_count - 2]:
                        sum_of_values = 0
                        for x in range(4):
                            index_ = -x-1
                            sum_of_values += self.losses[index_]
                        self.loss_convergence = sum_of_values/4
                        return True

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        if self.epoch_count:
            if self.losses[self.epoch_count - 1] == self.losses[self.epoch_count]:
                self.loss_convergence = self.losses[self.epoch_count]
                error_message("Oscillations/Lack of convergence")
                self.model.stop_training = True
            if self.epoch_count > 3 and self.check_oscillation():
                #Potentially oscillating about minima but overshooting because learning
                #rate is too high/not diminishing
                error_message("Oscillations/Lack of convergence")
                self.model.stop_training = True
            if self.check_convergence():
                error_message("Oscillations/Lack of convergence")
                self.model.stop_training = True
        self.epoch_count += 1

class increment_in_loss(keras.callbacks.Callback):
    
    '''
    Stop training if loss does not decrease over n consecutive epochs.
    The default and recommended value for n is 3.

    To optionally display a warning every time loss starts to increase, 
    uncomment the two lines inside on_epoch_end.

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
        if self.epoch_count:
            if self.losses[self.epoch_count - 1] < self.losses[self.epoch_count]:
                #warnings.warn("Detected increase in loss..")
                #warnings.simplefilter("always")
                self.error_count += 1
                if self.error_count == self.threshold:
                    error_message("Successive increments in loss")
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

class approximate_training_time(keras.callbacks.Callback):

    '''
    Return an approximation for the total training time.

    Usage:
    '''
    