'''
# Keras Callbacks
This notebook contains a set of customized keras callbacks that 
should enable increased efficiency while testing your keras model, 
not only improving your metrics but also saving you valuable time 
and resources.
'''

#1. Save loss values after each batch
class loss_after_each_bactch(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

#2. Stop training if loss does not decrease in 3 consecutive epochs
class no_decrement_in_loss(keras.callbacks.Callback):    
    def on_train_begin(self, logs={}):
        self.losses = []
        self.epoch_count = 0
        self.error_count = 0

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        if (self.epoch_count):
            if self.losses[self.epoch_count - 1] < self.losses[self.epoch_count]:
                self.error_count += 1
                if self.error_count == 3:
                    self.model.stop_training = True
            else:
                self.error_count = 0
        self.epoch_count += 1

#Stop 