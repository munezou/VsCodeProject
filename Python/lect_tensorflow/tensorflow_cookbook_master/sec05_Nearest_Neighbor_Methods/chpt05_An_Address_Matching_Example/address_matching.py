'''
# Address Matching with k-Nearest Neighbors
#----------------------------------
#
# This function illustrates a way to perform
# address matching between two data sets.
#
# For each test address, we will return the
# closest reference address to it.
#
# We will consider two distance functions:
# 1) Edit distance for street number/name and
# 2) Euclidian distance (L2) for the zip codes
'''

# import required libraries
from __future__ import absolute_import, division, print_function, unicode_literals
import os
from cpuinfo import get_cpu_info
import datetime
from packaging import version
import random
import string
import numpy as np
import tensorflow as tf

print(__doc__)

'''
--------------------------------------------
In casee of windows, os name is 'nt'.
In case of linux, os name is 'posix'.
--------------------------------------------
'''

if os.name == 'nt':
    print(
        '--------------------------------------------------------------------------\n'
        '                      cpu information                                     \n'
        '--------------------------------------------------------------------------\n'
    )
    # display the using cpu information
    for key, value in get_cpu_info().items():
        print("{0}: {1}".format(key, value))

    print()
    print()

# Display current path
PROJECT_ROOT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)))
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: {0}\n".format(tf.version.VERSION))
assert version.parse(tf.version.VERSION).release[0] >= 2, "This notebook requires TensorFlow 2.0 or above."

# First we generate the data sets we will need
# n = Size of created data sets
n = 10
street_names = ['abbey', 'baker', 'canal', 'donner', 'elm']
street_types = ['rd', 'st', 'ln', 'pass', 'ave']

random.seed(31)  # make results reproducible
rand_zips = [random.randint(65000, 65999) for i in range(5)]


# Function to randomly create one typo in a string w/ a probability
def create_typo(s, prob=0.75):
    if random.uniform(0, 1) < prob:
        rand_ind = random.choice(range(len(s)))
        s_list = list(s)
        s_list[rand_ind] = random.choice(string.ascii_lowercase)
        s = ''.join(s_list)
    return s


# Generate the reference dataset
numbers = [random.randint(1, 9999) for _ in range(n)]
streets = [random.choice(street_names) for _ in range(n)]
street_suffs = [random.choice(street_types) for _ in range(n)]
zips = [random.choice(rand_zips) for _ in range(n)]
full_streets = [str(x) + ' ' + y + ' ' + z for x, y, z in zip(numbers, streets, street_suffs)]
reference_data = [list(x) for x in zip(full_streets, zips)]

# Generate test dataset with some typos
typo_streets = [create_typo(x) for x in streets]
typo_full_streets = [str(x) + ' ' + y + ' ' + z for x, y, z in zip(numbers, typo_streets, street_suffs)]
test_data = [list(x) for x in zip(typo_full_streets, zips)]

# Now we can perform address matching
def top_match_index(test_address, test_zip, ref_address, ref_zip):
    # Declare Zip code distance for a test zip and reference set
    zip_dist = tf.square(tf.subtract(ref_zip, test_zip))

    # Declare Edit distance for address
    address_dist = tf.edit_distance(test_address, ref_address, normalize=True)

    # Create similarity scores
    zip_max = tf.gather(tf.squeeze(zip_dist), tf.argmax(input=zip_dist, axis=1))
    zip_min = tf.gather(tf.squeeze(zip_dist), tf.argmin(input=zip_dist, axis=1))
    zip_sim = tf.cast(tf.math.divide(tf.subtract(zip_max, zip_dist), tf.subtract(zip_max, zip_min)), dtype=tf.float32)
    address_sim = tf.subtract(1., address_dist)

    # Combine distance functions
    address_weight = tf.cast(0.5, dtype=tf.float32)
    zip_weight = tf.cast((1. - address_weight), dtype=tf.float32)
    weighted_sim = tf.add(tf.transpose(a=tf.multiply(address_weight, address_sim)), tf.multiply(zip_weight, zip_sim))

    # Predict: Get max similarity entry
    return tf.cast(tf.argmax(input=weighted_sim, axis=1), dtype=tf.int32)


# Function to Create a character-sparse tensor from strings
def sparse_from_word_vec(word_vec):
    num_words = len(word_vec)
    indices = [[xi, 0, yi] for xi, x in enumerate(word_vec) for yi, y in enumerate(x)]
    chars = list(''.join(word_vec))
    return tf.sparse.SparseTensor(indices, chars, [num_words, 1, 1])


# Loop through test indices
reference_addresses = [x[0] for x in reference_data]
reference_zips = np.array([[x[1] for x in reference_data]])

# Create sparse address reference set
sparse_ref_set = sparse_from_word_vec(reference_addresses)

for i in range(n):
    test_address_entry = test_data[i][0]
    test_zip_entry = [[test_data[i][1]]]

    # Create sparse address vectors
    test_address_repeated = [test_address_entry] * n
    sparse_test_set = sparse_from_word_vec(test_address_repeated)

    best_match = top_match_index(sparse_test_set, test_zip_entry, sparse_ref_set, reference_zips)

    best_street = reference_addresses[best_match[0]]

    [best_zip] = reference_zips[0][best_match.numpy()]
    [[test_zip_]] = test_zip_entry

    print('Address: {0}, {1}'.format(test_address_entry, test_zip_))
    print('Match  : {0}, {1}\n'.format(best_street, best_zip))

date_today = datetime.date.today()

print(
    '------------------------------------------------------------------------------------------------------\n'
)

print(
    '       finished      address_matching.py                          ({0})             \n'.format(
        date_today)
)

print(
    '------------------------------------------------------------------------------------------------------\n'
)
print()
print()
print()