from __future__ import absolute_import, division, print_function, unicode_literals
'''
------------------------------------------------------------------------------------------------------------------------
Know about over-learning and lack of learning.
------------------------------------------------------------------------------------------------------------------------
'''
import tensorflow as tf
from tensorflow import keras
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

'''
------------------------------------------------------------------------------------------------------------------------
Download IMDB Data Set
------------------------------------------------------------------------------------------------------------------------
'''
# An error occurs in the original code, so it is a corrected part.(start)
from functools import partial
import numpy as np
np.load = partial(np.load, allow_pickle=True)
# end of modify

imdb = keras.datasets.imdb

NUM_WORDS = 10000

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=NUM_WORDS)

print('train_data[0] = {0}'.format(train_data[0]))

'''
------------------------------------------------------------------------------------------------------------------------
Convert integers back to words.
------------------------------------------------------------------------------------------------------------------------
'''
# Dictionary mapping words to integers
word_index = imdb.get_word_index()

# The first one in the index is reserved.
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print('trin_data[0] transrated string = \n{0}'.format(decode_review(train_data[0])))

def multi_hot_sequences(sequences, dimension):
    # Create a matrix of all zeros with shape (len (sequences), dimension).
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0  # Set results [i] to 1 for a particular index.
    return results

train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)

plt.plot(train_data[0])
plt.show()

'''
------------------------------------------------------------------------------------------------------------------------
Demonstration of overlearning
------------------------------------------------------------------------------------------------------------------------
'''
baseline_model = keras.Sequential([
    # You need `input_shape` to see` .summary`.
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

baseline_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])

baseline_model.summary()

# learning
baseline_history = baseline_model.fit(train_data, train_labels, epochs=20, batch_size=512, validation_data=(test_data, test_labels), verbose=2)

'''
------------------------------------------------------------------------------------------------------------------------
Build a smaller model:
Create a model with fewer hidden units compared to the model you just created.
------------------------------------------------------------------------------------------------------------------------
'''
smaller_model = keras.Sequential([
    keras.layers.Dense(4, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(4, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

smaller_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])

smaller_model.summary()

# learning
smaller_history = smaller_model.fit(train_data, train_labels, epochs=20, batch_size=512, validation_data=(test_data, test_labels), verbose=2)

'''
------------------------------------------------------------------------------------------------------------------------
Build a larger model:
As a practice, you can create a larger model and see how quickly overlearning occurs.
------------------------------------------------------------------------------------------------------------------------
'''
bigger_model = keras.models.Sequential([
    keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

bigger_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','binary_crossentropy'])

bigger_model.summary()

# learning
bigger_history = bigger_model.fit(train_data, train_labels, epochs=20, batch_size=512, validation_data=(test_data, test_labels), verbose=2)

'''
------------------------------------------------------------------------------------------------------------------------
Graph loss during training and verification.
The solid line is the loss of the training data set, and the dashed line is the loss of the verification data set (the smaller the loss of the verification data, the better the model). 
If you look at this, you can see that the small network is slower to start overlearning than the comparison model (after 6 epochs, not 4 epochs).
------------------------------------------------------------------------------------------------------------------------
'''
def plot_history(histories, key='binary_crossentropy'):
  plt.figure(figsize=(16,10))

  for name, history in histories:
    val = plt.plot(history.epoch, history.history['val_'+key], '--', label=name.title()+' Val')

    plt.plot(history.epoch, history.history[key], color=val[0].get_color(), label=name.title()+' Train')

  plt.xlabel('Epochs')
  plt.ylabel(key.replace('_',' ').title())
  plt.legend()

  plt.xlim([0,max(history.epoch)])

plot_history([('baseline', baseline_history), ('smaller', smaller_history), ('bigger', bigger_history)])

plt.show()

'''
------------------------------------------------------------------------------------------------------------------------
strategy

Add weight regularization
Do you know the principle of "Occam's razor"? Given that there are two explanations, the most likely explanation is that it is the "simplest" explanation with the fewest assumptions. 
This principle also applies to models learned using neural networks. 
With some training data and a network structure, when there is more than one set of weights that can explain the data (that is, when there are multiple models), a simple model is harder to over-learn than a complex one. .

The "simple model" here is one with a small entropy of the distribution of parameter values ​​(or one with a small number of parameters, as we saw above). 
Therefore, a general method to reduce over-learning is to impose constraints such that the distribution of weight values ​​becomes more orderly (regular) by taking only small weight values. 
This is called "weight regularization" and is done by adding the cost associated with the magnitude of the weight to the loss function of the network. There are two types of costs.

* L1 regularization Add a cost that is proportional to the absolute value of the weighting factor (called the "L1 norm" of the weight).

* L2 regularization Add a cost that is proportional to the square of the weighting factor (called the square of the weighting factor "L2 norm").
  L2 regularization is called a weight decay in neural network terms. Don't get confused as they are called differently. 
  Weight attenuation is mathematically synonymous with L2 regularization.
------------------------------------------------------------------------------------------------------------------------
'''
l2_model = keras.models.Sequential([
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001), activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

l2_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'binary_crossentropy'])

l2_model_history = l2_model.fit(train_data, train_labels, epochs=20, batch_size=512, validation_data=(test_data, test_labels), verbose=2)

'''
------------------------------------------------------------------------------------------------------------------------
Let's look at the effects of L2 regularization.
------------------------------------------------------------------------------------------------------------------------
'''
plot_history([('baseline', baseline_history), ('l2', l2_model_history)])

plt.show()

'''
------------------------------------------------------------------------------------------------------------------------
Add a dropout:

Dropout is one of the most commonly used neural network regularization techniques. 
This method was developed by Hinton and his students at the University of Toronto. 
Dropouts apply to layers, and randomly drop out (ie, zeroize) the features output from the layer during training. 
For example, suppose that for a given input sample a layer is training, you would normally output the vector [0.2, 0.5, 1.3, 0.8, 1.1]. 
After applying the dropout, this vector will contain several zeros randomly scattered, eg [0, 0.5, 1.3, 0, 1.1]. 
"Dropout rate" is the percentage of features that are zeroed, usually between 0.2 and 0.5. During testing, no units are dropped out and instead the output value is scaled down at the same rate as the dropout rate. 
This is to balance against having more units active than at the time of training.
------------------------------------------------------------------------------------------------------------------------
'''
dpt_model = keras.models.Sequential([
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dropout(rate=0.5),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

dpt_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy','binary_crossentropy'])

dpt_model_history = dpt_model.fit(train_data, train_labels, epochs=20, batch_size=512, validation_data=(test_data, test_labels), verbose=2)

# Output result
plot_history([('baseline', baseline_history), ('dropout', dpt_model_history)])

plt.show()
