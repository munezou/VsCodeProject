'''
------------------------------------------------------------------------------------------
text
    Transformer model for language understanding

This tutorial trains a Transformer model to translate Portuguese to English. 
This is an advanced example that assumes knowledge of text generation and attention.

The core idea behind the Transformer model is self-attention—the ability to 
attend to different positions of the input sequence to compute a representation of that sequence. 
Transformer creates stacks of self-attention layers 
and is explained below in the sections Scaled dot product attention and Multi-head attention.

A transformer model handles variable-sized input using stacks of self-attention layers instead of RNNs or CNNs. 
This general architecture has a number of advantages:

	* It make no assumptions about the temporal/spatial relationships across the data. 
		This is ideal for processing a set of objects (for example, StarCraft units).
	* Layer outputs can be calculated in parallel, instead of a series like an RNN.
	* Distant items can affect each other's output without passing through many RNN-steps, 
		or convolution layers (see Scene Memory Transformer for example).
	* It can learn long-range dependencies. This is a challenge in many sequence tasks.
	
The downsides of this architecture are:

	* For a time-series, 
		the output for a time-step is calculated from the entire history instead of only the inputs and current hidden-state. 
		This may be less efficient.
	* If the input does have a temporal/spatial relationship, like text, 
		some positional encoding must be added or the model will effectively see a bag of words.
	
After training the model in this notebook, you will be able to input a Portuguese sentence and return the English translation.
------------------------------------------------------------------------------------------
'''
# common library
from __future__ import absolute_import, division, print_function, unicode_literals

import re
import os
import json
import time
import pprint
import contextlib
from pathlib import Path
from PIL import Image

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import tensorflow as tf
import tensorflow_datasets as tfds

print(__doc__)

AUTOTUNE = tf.data.experimental.AUTOTUNE

# Display current path
basic_path = Path.cwd()
PROJECT_ROOT_DIR = basic_path.joinpath('Python/Normal/tensorflow')
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)

im = Image.open(PROJECT_ROOT_DIR.joinpath('images/attention_map_portuguese.png'))
im.show()

print   (
        '---------------------------------------------------------------------------------\n'
        '      Setup input pipeline                                                       \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
------------------------------------------------------------------------------------------
Use TFDS to load the Portugese-English translation dataset from the TED Talks Open Translation Project.

This dataset contains approximately 50000 training examples, 1100 validation examples, and 2000 test examples.
------------------------------------------------------------------------------------------
'''
examples, metadata = tfds.load(
                        name='ted_hrlr_translate/pt_to_en',
                        with_info=True,
                        as_supervised=True
                    )

train_examples, val_examples = examples['train'], examples['validation']

'''
-----------------------------------------------------------------------------------------
Create a custom subwords tokenizer from the training dataset.
-----------------------------------------------------------------------------------------
'''

tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
					(en.numpy() for pt, en in train_examples), 
					target_vocab_size=2**13
				)

tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
					(pt.numpy() for pt, 
					en in train_examples), 
					target_vocab_size=2**13
				)

sample_string = 'Transformer is awesome.'

tokenized_string = tokenizer_en.encode(sample_string)
print ('Tokenized string is {}\n'.format(tokenized_string))

original_string = tokenizer_en.decode(tokenized_string)
print ('The original string: {}\n\n'.format(original_string))

assert original_string == sample_string

'''
-----------------------------------------------------------------------------------------
The tokenizer encodes the string by breaking it into subwords 
if the word is not in its dictionary.
-----------------------------------------------------------------------------------------
'''
for ts in tokenized_string:
	print ('{} ----> {}'.format(ts, tokenizer_en.decode([ts])))

BUFFER_SIZE = 20000
BATCH_SIZE = 64

'''
-----------------------------------------------------------------------------------------
Add a start and end token to the input and target.
-----------------------------------------------------------------------------------------
'''
def encode(lang1, lang2):
	lang1 = [tokenizer_pt.vocab_size] + tokenizer_pt.encode(
	lang1.numpy()) + [tokenizer_pt.vocab_size+1]

	lang2 = [tokenizer_en.vocab_size] + tokenizer_en.encode(
		lang2.numpy()) + [tokenizer_en.vocab_size+1]

	return lang1, lang2

'''
----------------------------------------------------------------------------------------
Note: 
To keep this example small and relatively fast, drop examples with a length of over 40 tokens.
----------------------------------------------------------------------------------------
'''
MAX_LENGTH = 40

def filter_max_length(x, y, max_length=MAX_LENGTH):
	return tf.logical_and(
							tf.size(x) <= max_length,
							tf.size(y) <= max_length
						)

'''
-----------------------------------------------------------------------------------------
Operations inside .map() run in graph mode and receive a graph tensor that do not have a numpy attribute. 
The tokenizer expects a string or Unicode symbol to encode it into integers. 
Hence, you need to run the encoding inside a tf.py_function, 
which receives an eager tensor having a numpy attribute that contains the string value.
----------------------------------------------------------------------------------------
'''
def tf_encode(pt, en):
	return tf.py_function(encode, [pt, en], [tf.int64, tf.int64])

train_dataset = train_examples.map(tf_encode)
train_dataset = train_dataset.filter(filter_max_length)
# cache the dataset to memory to get a speedup while reading from it.
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(BUFFER_SIZE).padded_batch(
					BATCH_SIZE, 
					padded_shapes=([-1], [-1])
				)

train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)


val_dataset = val_examples.map(tf_encode)
val_dataset = val_dataset.filter(filter_max_length).padded_batch(
    BATCH_SIZE, padded_shapes=([-1], [-1]))

pt_batch, en_batch = next(iter(val_dataset))

print('pt_batch = \n{0},\n en_batch = \n{1}\n'.format(pt_batch, en_batch))

print   (
        '---------------------------------------------------------------------------------\n'
        '      Positional encoding                                                        \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
-------------------------------------------------------------------------------------------
Since this model doesn't contain any recurrence or convolution, positional encoding is added to give the model some information about the relative position of the words in the sentence.

The positional encoding vector is added to the embedding vector. Embeddings represent a token in a d-dimensional space where tokens with similar meaning will be closer to each other. 
But the embeddings do not encode the relative position of words in a sentence. 
So after adding the positional encoding, words will be closer to each other based on the similarity of their meaning and their position in the sentence, in the d-dimensional space.

See the notebook on positional encoding to learn more about it. The formula for calculating the positional encoding is as follows:
--------------------------------------------------------------------------------------------
'''
im = Image.open(PROJECT_ROOT_DIR.joinpath('images/positional_encording.jpg'))
im.show()

def get_angles(pos, i, d_model):
	angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
	return pos * angle_rates

def positional_encoding(position, d_model):
	angle_rads = get_angles(
					np.arange(position)[:, np.newaxis],
					np.arange(d_model)[np.newaxis, :],
					d_model
				)
	
	# apply sin to even indices in the array; 2i
	angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
	
	# apply cos to odd indices in the array; 2i+1
	angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
	
	pos_encoding = angle_rads[np.newaxis, ...]
	
	return tf.cast(pos_encoding, dtype=tf.float32)

pos_encoding = positional_encoding(50, 512)
print ('pos_encoding.shape = {0}\n'.format(pos_encoding.shape))

plt.pcolormesh(pos_encoding[0], cmap='RdBu')
plt.xlabel('Depth')
plt.xlim((0, 512))
plt.ylabel('Position')
plt.colorbar()
plt.show()

