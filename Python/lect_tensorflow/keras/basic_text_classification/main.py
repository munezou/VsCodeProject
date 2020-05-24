from __future__ import absolute_import, division, print_function, unicode_literals

'''
------------------------------------------------------------------------------------------------------------------------
Movie review text classification
------------------------------------------------------------------------------------------------------------------------
'''
import tensorflow as tf
from tensorflow import keras

import numpy as np

print(tf.__version__)

print()

'''
------------------------------------------------------------------------------------------------------------------------
 DownLoad IMDB dataset
------------------------------------------------------------------------------------------------------------------------
'''
# An error occurs in the original code, so it is a corrected part.(start)
from functools import partial
import numpy as np
np.load = partial(np.load, allow_pickle=True)
# end of modify

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

'''
------------------------------------------------------------------------------------------------------------------------
Survey data
------------------------------------------------------------------------------------------------------------------------
'''
print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))
print("Test entries: {}, labels: {}".format(len(test_data), len(test_labels)))
print()

'''
------------------------------------------------------------------------------------------------------------------------
The text of the review has been converted to multiple integers, each of which represents a specific word in the dictionary.
Let's see what the first review looks like.
------------------------------------------------------------------------------------------------------------------------
'''
print('train_data[0] = \n{0}'.format(train_data[0]))
print ('len(train_data[0]) = {0}, len(train_data[1] = {1}'.format(len(train_data[0]), len(train_data[1])))
print()

'''
------------------------------------------------------------------------------------------------------------------------
Convert integers back to words.
------------------------------------------------------------------------------------------------------------------------
'''

'''
start of converting routine
'''
# Dictionary mapping words to integers
word_index = imdb.get_word_index()

# The first one in the index is reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

'''
end of converting routine
'''
print(decode_review(train_data[0]))
print()

'''
------------------------------------------------------------------------------------------------------------------------
Data preparation
The review (an array of integers) must be converted to a tensor before being put into the neural network. There are two ways to do this.

Converts an array to a vector of 0's and 1's representing the occurrence of a word, similar to one-hot encoding.
For example, the array [3, 5] is a 10,000-dimensional vector of all zeros except for indices 3 and 5.
And let this be the first layer of the network, that is, the Dense (all bound) layer that can handle floating point vector data.
However, this is a memory intensive method that requires a matrix of number of words x number of reviews.

窶「 Another way is to align the array to the same length by padding, making it an integer tensor in the form of a sample * maximum length.
Then we make the Embedding layer that can handle this form the first layer of the network.

In this tutorial, we will adopt the latter.

Since movie reviews must be the same length, we will use the pad_sequences function to standardize the length.
------------------------------------------------------------------------------------------------------------------------
'''

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding='post', maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding='post', maxlen=256)

# Let's look at the sample length.
print ('len(train_data[0]) = {0}, len(train_data[1] = {1}'.format(len(train_data[0]), len(train_data[1])))
print('train_data[0] = \n{0}'.format(train_data[0]))
print()

'''
------------------------------------------------------------------------------------------------------------------------
Build a model
A neural network consists of stacking layers.

These layers are stacked in a row to form a classifier.

1. The first layer is the Embedding layer. This layer takes an integer encoded vocabulary and searches for an embedded vector corresponding to each word index.
   Embedded vectors are learned in model training. 
   An additional dimension is added to the output matrix for vectorization. 
   As a result, the dimensions are (batch, sequence, embedding).
2. Next is the GlobalAveragePooling1D (one-dimensional global average pooling) tier. 
   This layer finds, for each sample, the mean value in the dimensional direction of the sequence and returns a vector of fixed length. 
   This results in the model being able to handle variable-length input in its simplest form.
3. This fixed-length output vector is passed to the all coupled (Dense) layer with 16 hidden units.
4. The last layer is fully coupled to one output node. 
   By using the sigmoid activation function, the value is a floating point number between 0 and 1 representing probability or confidence.

Hidden unit
 The model above has two middle or "hidden" layers between the input and output.
 The output (unit, node or neuron) is the number of dimensions of the internal representation of the layer. 
 In other words, this network is the degree of freedom when learning internal expression.

The network may learn more complex internal representations if there are more hidden units in the model (if the dimensionality of the internal representation space is larger) and or if there are more layers I can do it. 
 However, as a result, in addition to the computational complexity of the network being increased, it becomes possible to learn patterns that you do not want to learn. 
 Patterns that you do not want to learn are patterns that improve the performance of training data but do not improve the performance of test data.
 This problem is called overfitting. This issue will be examined later.
------------------------------------------------------------------------------------------------------------------------
'''
# Input format is the number of vocabulary used in movie review (10,000 words)
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

'''
------------------------------------------------------------------------------------------------------------------------
Loss function and optimizer

1. The first layer is the Embedding layer. 
   This layer takes an integer encoded vocabulary and searches for an embedded vector corresponding to each word index. 
   Embedded vectors are learned in model training. An additional dimension is added to the output matrix for vectorization. 
   As a result, the dimensions are (batch, sequence, embedding).
2. Next is the GlobalAveragePooling1D (one-dimensional global average pooling) tier.
   This layer finds, for each sample, the mean value in the dimensional direction of the sequence and returns a vector of fixed length.
   This results in the model being able to handle variable-length input in its simplest form.
3. This fixed-length output vector is passed to the all coupled (Dense) layer with 16 hidden units.
4. The last layer is fully coupled to one output node. 
   By using the sigmoid activation function, the value is a floating point number between 0 and 1 representing probability or confidence.
------------------------------------------------------------------------------------------------------------------------   
Hidden unit
The model above has two middle or "hidden" layers between the input and output.
The output (unit, node or neuron) is the number of dimensions of the internal representation of the layer.
In other words, this network is the degree of freedom when learning internal expression.

The network may learn more complex internal representations if there are more hidden units in the model (if the dimensionality of the internal representation space is larger) and / or if there are more layers I can do it.
However, as a result, in addition to the computational complexity of the network being increased, it becomes possible to learn patterns that you do not want to learn.
Patterns that you do not want to learn are patterns that improve the performance of training data but do not improve the performance of test data.
This problem is called overfitting. This issue will be examined later.
------------------------------------------------------------------------------------------------------------------------
Loss Function and Optimizer¶
To train the model, you need a loss function and an optimizer.
Since this problem is a binary classification problem, and the output of the model is probability (one unit of layer and sigmoid activation function), 
we will use the binary_crossentropy (binary cross entropy) function as the loss function.

This is not the only loss function candidate. 
For example, you can use mean_squared_error (mean squared error). 
However, in general, binary_crossentropy is better for dealing with probabilities. 
binary_crossentropy is a measure that measures the "distance" between probability distributions. 
In this case, it is the distance between the true distribution and the distribution of predicted values.

Later on, when examining regression problems (for example, estimating house prices), you will see the use of another loss function, mean_squared_error.

Now let's set up the model's optimizer and loss function.
------------------------------------------------------------------------------------------------------------------------
'''
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

'''
------------------------------------------------------------------------------------------------------------------------
Create data for verification
When training, I would like to verify the accuracy rate on data that the model does not see. 
Separate 10,000 samples from the original training data to create a validation set.

(Why not use test data here?
Our goal is to develop and tune models using training data only.
After that, test data is used only once to verify the accuracy rate. )
------------------------------------------------------------------------------------------------------------------------
'''
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

'''
------------------------------------------------------------------------------------------------------------------------
Model training
Train a 40 epoch model using a mini-batch of 512 samples. 
This results in 40 iterations of all the samples in x_train and y_train. During training, we will monitor model loss and accuracy rates using 10,000 samples of validation data.
------------------------------------------------------------------------------------------------------------------------
'''
history = model.fit(partial_x_train, partial_y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

'''
------------------------------------------------------------------------------------------------------------------------
Model evaluation
Well, let's look at the performance of the model. 
Two values are returned. 
It is a loss (it is a numerical value which shows an error, and a smaller one is better) and an accuracy rate.
------------------------------------------------------------------------------------------------------------------------
'''
results = model.evaluate(test_data, test_labels)

print(results)
print()

'''
------------------------------------------------------------------------------------------------------------------------
Draw a time series graph of accuracy rate and loss
model.fit () returns a History object containing a dictionary that records everything that occurred during training.
------------------------------------------------------------------------------------------------------------------------
'''
history_dict = history.history
print('history_dict.keys() = \n{0}'.format(history_dict.keys()))

'''
------------------------------------------------------------------------------------------------------------------------
There are 4 entries. 
Each indicates the indicator that was being monitored during training and validation. 
You can use this to create graphs that compare training and verification losses and graphs that compare training and verification accuracy rates.
------------------------------------------------------------------------------------------------------------------------
'''
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" は青いドット
plt.plot(epochs, loss, 'bo', label='Training loss')
# ”b" は青い実線
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear diagram
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()