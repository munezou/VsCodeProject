'''
************************************************
# Data gathering
#----------------------------------
#
# This function gives us the ways to access
# the various data sets we will need
************************************************
'''
# Data Gathering
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import io
import requests
import matplotlib.pyplot as plt
import tarfile
import codecs
from PIL import Image
from zipfile import ZipFile
import tensorflow as tf
from packaging import version

# Iris Data
from sklearn.datasets import load_iris

print(__doc__)

# Display current path
PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)
assert version.parse(tf.version.VERSION).release[0] >= 2, \
"This notebook requires TensorFlow 2.0 or above."

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       The Iris Dataset (R. Fisher / Scikit-Learn)                                                    \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
One of the most frequently used ML datasets is the iris flower dataset. 
We will use the easy import tool, datasets from scikit-learn. 
You can read more about it here: 
http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html#sklearn.datasets.load_iris
---------------------------------------------------------------------------------------------------------------
'''

iris = load_iris()
print('len(iris.data) = {0}'.format(len(iris.data)))
print('len(iris.target) = {0}'.format(len(iris.target)))
print('iris.data[0] = {0}'.format(iris.data[0]))
print('set(iris.target) = {0}'.format(set(iris.target)))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Low Birthrate Dataset (Hosted on Github)                                                       \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
The 'Low Birthrate Dataset' is a dataset from a famous study by Hosmer and Lemeshow in 1989 called, "Low Infant Birth Weight Risk Factor Study". 
It is a very commonly used academic dataset mostly for logistic regression. 
We will host this dataset on the public Github here: 
https://github.com/nfmcclure/tensorflow_cookbook/raw/master/01_Introduction/07_Working_with_Data_Sources/birthweight_data/birthweight.dat
---------------------------------------------------------------------------------------------------------------
'''
birthdata_url = 'https://github.com/nfmcclure/tensorflow_cookbook/raw/master/01_Introduction/07_Working_with_Data_Sources/birthweight_data/birthweight.dat'
birth_file = requests.get(birthdata_url)
birth_data = birth_file.text.split('\r\n')
birth_header = birth_data[0].split('\t')
birth_data = [[float(x) for x in y.split('\t') if len(x)>=1] for y in birth_data[1:] if len(y)>=1]
print('len(birth_data) = {0}'.format(len(birth_data)))
print('len(birth_data[0]) = {0}'.format(len(birth_data[0])))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Housing Price Dataset (UCI)                                                                    \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
We will also use a housing price dataset from the University of California at Irvine (UCI) Machine Learning Database Repository. 
It is a great regression dataset to use. 
You can read more about it here: https://archive.ics.uci.edu/ml/datasets/Housing
---------------------------------------------------------------------------------------------------------------
'''
housing_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
housing_header = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
housing_file = requests.get(housing_url)
housing_data = [[float(x) for x in y.split(' ') if len(x)>=1] for y in housing_file.text.split('\n') if len(y)>=1]
print('len(housing_data) = {0}'.format(len(housing_data)))
print('len(housing_data[0]) = {0}'.format(len(housing_data[0])))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       MNIST Handwriting Dataset (Yann LeCun)                                                         \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
The MNIST Handwritten digit picture dataset is the Hello World of image recognition. 
The famous scientist and researcher, Yann LeCun, hosts it on his webpage here, http://yann.lecun.com/exdb/mnist/ . 
But because it is so commonly used, many libraries, including TensorFlow, host it internally. 
We will use TensorFlow to access this data as follows.

If you haven't downloaded this before, please wait a bit while it downloads
---------------------------------------------------------------------------------------------------------------
'''
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

one_hot_y_train = tf.one_hot(
                        indices=y_train[:],
                        depth=10,
                        on_value=1.0,
                        off_value=0.0,
                        axis=-1,
                        dtype=tf.float32
                    )

one_hot_y_test = tf.one_hot(
                        indices=y_test[:],
                        depth=10,
                        on_value=1.0,
                        off_value=0.0,
                        axis=-1,
                        dtype=tf.float32
                    )

print('len(x_train) = {0}'.format(len(x_train)))
print('len(y_train) = {0}'.format(len(y_train)))
print('len(x_test) = {0}'.format(len(x_test)))
print('len(y_test) = {0}'.format(len(y_test)))
print('y_train[5] = {0}'.format(y_train[5]))
print('one_hot_y_train[5, :] = {0}\n'.format(one_hot_y_train[5, :]))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       CIFAR-10 Image Category Dataset                                                                \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
--------------------------------------------------------------------------------------------------------------
The CIFAR-10 data ( https://www.cs.toronto.edu/~kriz/cifar.html ) contains 60,000 32x32 color images of 10 classes collected by Alex Krizhevsky, 
Vinod Nair, and Geoffrey Hinton. Alex Krizhevsky maintains the page referenced here. 
This is such a common dataset, that there are built in functions in TensorFlow to access this data (the keras wrapper has these commands). 
Note that the keras wrapper for these functions automatically splits the images into a 50,000 training set and a 10,000 test set.
--------------------------------------------------------------------------------------------------------------
'''
# Running this command requires an internet connection and a few minutes to download all the images.
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

'''
-------------------------------------------------------------------------------------------------------------
Downloading data from http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
The ten categories are (in order):

    0.Airplane
    1.Automobile
    2.Bird
    3.Car
    4.Deer
    5.Dog
    6.Frog
    7.Horse
    8.Ship
    9.Truck
-------------------------------------------------------------------------------------------------------------
'''
print('X_train.shape = {0}'.format(X_train.shape))
print('y_train.shape = {0}'.format(y_train.shape))
print('y_train[0,] = {0}'.format(y_train[0,]))

# Plot the 0-th image (a frog)
img = Image.fromarray(X_train[0,:,:,:])
plt.imshow(img)
plt.show()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Ham/Spam Texts Dataset (UCI)                                                                   \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
We will use another UCI ML Repository dataset called the SMS Spam Collection. 
You can read about it here: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection . 
As a sidenote about common terms, when predicting if a data point represents 'spam' (or unwanted advertisement), 
the alternative is called 'ham' (or useful information).

This is a great dataset for predicting a binary outcome (spam/ham) from a textual input. 
This will be very useful for short text sequences for Natural Language Processing (Ch 7) and Recurrent Neural Networks (Ch 9).
---------------------------------------------------------------------------------------------------------------
'''
# Get/read zip file
zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
r = requests.get(zip_url)
z = ZipFile(io.BytesIO(r.content))
file = z.read('SMSSpamCollection')

# Format Data
text_data = file.decode()
text_data = text_data.encode('ascii',errors='ignore')
text_data = text_data.decode().split('\n')
text_data = [x.split('\t') for x in text_data if len(x)>=1]
[text_data_target, text_data_train] = [list(x) for x in zip(*text_data)]

print('len(text_data_train) = {0}'.format(len(text_data_train)))
print('set(text_data_target) = {0}'.format(set(text_data_target)))
print('text_data_train[1] = \n{0}\n'.format(text_data_train[1]))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Movie Review Data                                                                              \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
The Movie Review database, collected by Bo Pang and Lillian Lee (researchers at Cornell), 
serves as a great dataset to use for predicting a numerical number from textual inputs.

You can read more about the dataset and papers using it here: https://www.cs.cornell.edu/people/pabo/movie-review-data/
---------------------------------------------------------------------------------------------------------------
'''
movie_data_url = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
r = requests.get(movie_data_url)

# Stream data into temp object
stream_data = io.BytesIO(r.content)
tmp = io.BytesIO()
while True:
    s = stream_data.read(16384)
    if not s:  
        break
    tmp.write(s)
stream_data.close()
tmp.seek(0)

# Extract tar file
tar_file = tarfile.open(fileobj=tmp, mode="r:gz")

pos = tar_file.extractfile('rt-polaritydata/rt-polarity.pos')
neg = tar_file.extractfile('rt-polaritydata/rt-polarity.neg')

# Save pos/neg reviews
pos_data = []
for line in pos:
    pos_data.append(line.decode('ISO-8859-1').encode('ascii',errors='ignore').decode())
    
neg_data = []
for line in neg:
    neg_data.append(line.decode('ISO-8859-1').encode('ascii',errors='ignore').decode())
    
tar_file.close()

print('len(pos_data) = {0}'.format(len(pos_data)))
print('len(neg_data) = {0}'.format(len(neg_data)))
print('neg_data[0] = {0}\n'.format(neg_data[0]))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       The Complete Works of William Shakespeare (Gutenberg Project)                                  \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
For training a TensorFlow Model to create text, we will train it on the complete works of William Shakespeare. 
This can be accessed through the good work of the Gutenberg Project. 
The Gutenberg Project frees many non-copyright books by making them accessible for free from the hard work of volunteers.

You can read more about the Shakespeare works here: http://www.gutenberg.org/ebooks/100
---------------------------------------------------------------------------------------------------------------
'''
# The Works of Shakespeare Data
shakespeare_url = 'http://www.gutenberg.org/cache/epub/100/pg100.txt'
# Get Shakespeare text
response = requests.get(shakespeare_url)
shakespeare_file = response.content

# Decode binary into string
shakespeare_text = shakespeare_file.decode('utf-8')

# Drop first few descriptive paragraphs.
shakespeare_text = shakespeare_text[7675:]
print('len(shakespeare_text) = {0}\n'.format(len(shakespeare_text)))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       English-German Sentence Translation Database (Manythings/Tatoeba)                              \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
The Tatoeba Project is also run by volunteers 
and is set to make the most bilingual sentence translations available between many different languages. 
Manythings.org compiles the data and makes it accessible.

http://www.manythings.org/corpus/about.html#info

More bilingual sentence pairs: http://www.manythings.org/bilingual/
---------------------------------------------------------------------------------------------------------------
'''

'''
----------------------< no work >-------------------------
# sentence_url = 'http://www.manythings.org/anki/deu-eng.zip'
r = requests.get(sentence_url)
z = ZipFile(io.BytesIO(r.content))
file = z.read('deu.txt')


# Format Data
eng_ger_data = file.decode()
eng_ger_data = eng_ger_data.encode('ascii',errors='ignore')
eng_ger_data = eng_ger_data.decode().split('\n')
eng_ger_data = [x.split('\t') for x in eng_ger_data if len(x)>=1]
[english_sentence, german_sentence] = [list(x) for x in zip(*eng_ger_data)]
--------------------------------------------------------------
'''
file_path = str(os.path.join(PROJECT_ROOT_DIR, 'deu_eng', 'deu.txt'))

with open(file_path, encoding='utf-8') as f:
    eng_ger_data = f.readlines()
    english_sentence = []
    german_sentence = []
    for tmpStr in eng_ger_data:
        tmp = tmpStr.split('\t')
        english_sentence.append(tmp[0])
        german_sentence.append(tmp[1])
    

print('len(english_sentence) = {0}'.format(len(english_sentence)))
print('len(german_sentence) = {0}'.format(len(german_sentence)))
print('eng_ger_data[10] = {0}\n'.format(eng_ger_data[10]))