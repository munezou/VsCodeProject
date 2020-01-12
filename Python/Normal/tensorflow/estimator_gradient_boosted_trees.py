'''
------------------------------------------------------------------------------------------
estimator
    Gradient Boosted Trees: Model understanding

TFor an end-to-end walkthrough of training a Gradient Boosting model check out the boosted trees tutorial. 
In this tutorial you will:

* Learn how to interpret a Boosted Trees model both locally and globally
* Gain intution for how a Boosted Trees model fits a dataset
------------------------------------------------------------------------------------------
'''
# common library
from __future__ import absolute_import, division, print_function, unicode_literals

import time
import os
import sys
import platform
import shutil
import subprocess
import random
from pathlib import Path
from packaging import version
from PIL import Image

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow.compat.v2.feature_column as fc
from sklearn.metrics import roc_curve

import tensorflow as tf

print(__doc__)

AUTOTUNE = tf.data.experimental.AUTOTUNE

pd.options.display.max_rows = None

# Display current path
basic_path = Path.cwd()
PROJECT_ROOT_DIR = basic_path.joinpath('Python/Normal/tensorflow')
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
"This notebook requires TensorFlow 2.0 or above."

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       How to interpret Boosted Trees models both locally and globally                                \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Local interpretability refers to an understanding of a model’s predictions at the individual example level, 
while global interpretability refers to an understanding of the model as a whole. 
Such techniques can help machine learning (ML) practitioners detect bias and bugs during the model development stage.

For local interpretability, you will learn how to create and visualize per-instance contributions. 
To distinguish this from feature importances, we refer to these values as directional feature contributions (DFCs).

For global interpretability you will retrieve and visualize gain-based feature importances, 
permutation feature importances and also show aggregated DFCs.
---------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Load the titanic dataset                                                                       \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
---------------------------------------------------------------------------------------------------------------
You will be using the titanic dataset, where the (rather morbid) goal is to predict passenger survival, 
given characteristics such as gender, age, class, etc.
---------------------------------------------------------------------------------------------------------------
'''

# Load dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

tf.random.set_seed(123)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Create feature columns, input_fn, and the train the estimator                                  \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Preprocess the data                                                                            \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
----------------------------------------------------------------------------------------------------------------
Create the feature columns, using the original numeric columns as is and one-hot-encoding categorical variables.
----------------------------------------------------------------------------------------------------------------
'''
fc = tf.feature_column
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

def one_hot_cat_column(feature_name, vocab):
    return fc.indicator_column(
                fc.categorical_column_with_vocabulary_list(feature_name, vocab)
            )
    
feature_columns = []

for feature_name in CATEGORICAL_COLUMNS:
    # Need to one-hot encode categorical features.
    vocabulary = dftrain[feature_name].unique()
    feature_columns.append(one_hot_cat_column(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(fc.numeric_column(feature_name, dtype=tf.float32))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Build the input pipeline                                                                       \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
---------------------------------------------------------------------------------------------------------------
Create the input functions using the from_tensor_slices method in the tf.data API 
to read in data directly from Pandas.
--------------------------------------------------------------------------------------------------------------
'''

# Use entire batch since this is such a small dataset.
NUM_EXAMPLES = len(y_train)

def make_input_fn(X, y, n_epochs=None, shuffle=True):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((X.to_dict(orient='list'), y))
        if shuffle:
            dataset = dataset.shuffle(NUM_EXAMPLES)
        
        # For training, cycle thru dataset as many times as need (n_epochs=None).
        dataset = (dataset.repeat(n_epochs).batch(NUM_EXAMPLES))
        
        return dataset
    return input_fn

# Training and evaluation input functions.
train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, shuffle=False, n_epochs=1)

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Train the model                                                                                \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

params = {
            'n_trees': 50,
            'max_depth': 3,
            'n_batches_per_layer': 1,
            # You must enable center_bias = True to get DFCs. This will force the model to
            # make an initial prediction before using any features (e.g. use the mean of
            # the training labels for regression or log odds for classification when
            # using cross entropy loss).
            'center_bias': True
        }

est = tf.estimator.BoostedTreesClassifier(feature_columns, **params)
# Train model.
est.train(train_input_fn, max_steps=100)
print()

# Evaluation.
results = est.evaluate(eval_input_fn)
pd_series = pd.Series(results).to_frame()

print('pd_series = \n{0}\n'.format(pd_series))

'''
-------------------------------------------------------------------------------------------------------------------
For performance reasons, 
when your data fits in memory, we recommend use the boosted_trees_classifier_train_in_memory function. 
However if training time is not of a concern or if you have a very large dataset and want to do distributed training, 
use the tf.estimator.BoostedTrees API shown above.

When using this method, you should not batch your input data, as the method operates on the entire dataset.
-------------------------------------------------------------------------------------------------------------------
'''

in_memory_params = dict(params)
in_memory_params['n_batches_per_layer'] = 1
# In-memory input_fn does not use batching.
def make_inmemory_train_input_fn(X, y):
    y = np.expand_dims(y, axis=1)
    def input_fn():
        return dict(X), y
    return input_fn

train_input_fn = make_inmemory_train_input_fn(dftrain, y_train)

# Train the model.
est = tf.estimator.BoostedTreesClassifier(
            feature_columns, 
            train_in_memory=True, 
            **in_memory_params
        )

est.train(train_input_fn)
print(est.evaluate(eval_input_fn))

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Model interpretation and plotting                                                              \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

sns_colors = sns.color_palette('colorblind')

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Local interpretability                                                                         \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
--------------------------------------------------------------------------------------------------------------
Next you will output the directional feature contributions (DFCs) to explain individual predictions 
using the approach outlined in Palczewska et al and by Saabas in Interpreting Random Forests 
(this method is also available in scikit-learn for Random Forests in the treeinterpreter package). 
The DFCs are generated with:

pred_dicts = list(est.experimental_predict_with_explanations(pred_input_fn))

(Note: The method is named experimental as we may modify the API before dropping the experimental prefix.)
--------------------------------------------------------------------------------------------------------------
'''

pred_dicts = list(est.experimental_predict_with_explanations(eval_input_fn))

# Create DFC Pandas dataframe.
labels = y_eval.values
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])
df_dfc = pd.DataFrame([pred['dfc'] for pred in pred_dicts])
df_dfc_describe_T = df_dfc.describe().T

print('df_dfc_describe_T = \n{0}\n'.format(df_dfc_describe_T))

'''
---------------------------------------------------------------------------------------------------------------
A nice property of DFCs is that the sum of the contributions + the bias is equal to the prediction for a given example.
---------------------------------------------------------------------------------------------------------------
'''
# Sum of DFCs + bias == probabality.
bias = pred_dicts[0]['bias']
dfc_prob = df_dfc.sum(axis=1) + bias
print('dfc_prob = \n{0}\n'.format(dfc_prob))

print('probs = \n{0}\n'.format(probs))

try:
    np_testing_assert_almost_equal = np.testing.assert_almost_equal(dfc_prob.values, probs.values)
    if np_testing_assert_almost_equal == None:
        print('dfc_prob.values == probs.values')
    pass
except AssertionError as ex:
    print(ex)
    pass
finally:
    pass

'''
----------------------------------------------------------------------------------------------------------------
Plot DFCs for an individual passenger. 
Let's make the plot nice by color coding based on the contributions' directionality and add the feature values on figure.
-----------------------------------------------------------------------------------------------------------------
'''

# Boilerplate code for plotting :)
def _get_color(value):
    """
    ---------------------------------------------------------
    To make positive DFCs plot green, negative DFCs plot red.
    ---------------------------------------------------------
    """
    green, red = sns.color_palette()[2:4]
    if value >= 0: return green
    return red

def _add_feature_values(feature_values, ax):
    """
    --------------------------------------------------------
    Display feature's values on left of plot.
    --------------------------------------------------------
    """
    x_coord = ax.get_xlim()[0]
    OFFSET = 0.15
    for y_coord, (feat_name, feat_val) in enumerate(feature_values.items()):
        t = plt.text(x_coord, y_coord - OFFSET, '{}'.format(feat_val), size=12)
        t.set_bbox(dict(facecolor='white', alpha=0.5))
    from matplotlib.font_manager import FontProperties
    font = FontProperties()
    font.set_weight('bold')
    t = plt.text(x_coord, y_coord + 1 - OFFSET, 'feature\nvalue',
    fontproperties=font, size=12)

def plot_example(example):
    TOP_N = 8 # View top 8 features.
    sorted_ix = example.abs().sort_values()[-TOP_N:].index  # Sort by magnitude.
    example = example[sorted_ix]
    colors = example.map(_get_color).tolist()
    ax = example.to_frame().plot(kind='barh',
                                    color=[colors],
                                    legend=None,
                                    alpha=0.75,
                                    figsize=(10,6)
                                )
    
    ax.grid(False, axis='y')
    ax.set_yticklabels(ax.get_yticklabels(), size=14)

    # Add feature values.
    _add_feature_values(dfeval.iloc[ID][sorted_ix], ax)
    return ax
