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
from numpy.random import seed, uniform
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata

from sklearn.metrics import roc_curve

import tensorflow as tf

print(__doc__)

AUTOTUNE = tf.data.experimental.AUTOTUNE

pd.options.display.max_rows = None

# Display current path
basic_path = Path.cwd()
PROJECT_ROOT_DIR = basic_path.joinpath('Python', 'Normal', 'tensorflow')
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)
assert version.parse(tf.version.VERSION).release[0] >= 2, \
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

est = tf.estimator.BoostedTreesClassifier(
            n_batches_per_layer=1,
            n_trees=50,
            max_depth=3,
            center_bias=True
        )

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

print('est.evaluate(eval_input_fn) = {0}\n\n'.format(est.evaluate(eval_input_fn)))

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

# Plot results.
ID = 182
example = df_dfc.iloc[ID]  # Choose ith example from evaluation set.
TOP_N = 8  # View top 8 features.
sorted_ix = example.abs().sort_values()[-TOP_N:].index
ax = plot_example(example)
ax.set_title('Feature contributions for example {}\n pred: {:1.2f}; label: {}'.format(ID, probs[ID], labels[ID]))
ax.set_xlabel('Contribution to predicted probability', size=14)
plt.show()

'''
------------------------------------------------------------------------------------------------------------------
The larger magnitude contributions have a larger impact on the model's prediction. 
Negative contributions indicate the feature value for this given example reduced the model's prediction, 
while positive values contribute an increase in the prediction.
------------------------------------------------------------------------------------------------------------------
'''

# You can also plot the example's DFCs compare with the entire distribution using a voilin plot.
# Boilerplate plotting code.
def dist_violin_plot(df_dfc, ID):
    # Initialize plot.
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Create example dataframe.
    TOP_N = 8  # View top 8 features.
    example = df_dfc.iloc[ID]
    ix = example.abs().sort_values()[-TOP_N:].index
    example = example[ix]
    example_df = example.to_frame(name='dfc')

    # Add contributions of entire distribution.
    parts=ax.violinplot([df_dfc[w] for w in ix],
                    vert=False,
                    showextrema=False,
                    widths=0.7,
                    positions=np.arange(len(ix)))
    face_color = sns_colors[0]
    alpha = 0.15
    for pc in parts['bodies']:
        pc.set_facecolor(face_color)
        pc.set_alpha(alpha)

    # Add feature values.
    _add_feature_values(dfeval.iloc[ID][sorted_ix], ax)

    # Add local contributions.
    ax.scatter(example,
                np.arange(example.shape[0]),
                color=sns.color_palette()[2],
                s=100,
                marker="s",
                label='contributions for example')

    # Legend
    # Proxy plot, to show violinplot dist on legend.
    ax.plot([0,0], [1,1], label='eval set contributions\ndistributions',
            color=face_color, alpha=alpha, linewidth=10)
    legend = ax.legend(loc='lower right', shadow=True, fontsize='x-large',
                        frameon=True)
    legend.get_frame().set_facecolor('white')

    # Format plot.
    ax.set_yticks(np.arange(example.shape[0]))
    ax.set_yticklabels(example.index)
    ax.grid(False, axis='y')
    ax.set_xlabel('Contribution to predicted probability', size=14)

# plot this example
dist_violin_plot(df_dfc, ID)
plt.title('Feature contributions for example {}\n pred: {:1.2f}; label: {}'.format(ID, probs[ID], labels[ID]))
plt.show()

'''
---------------------------------------------------------------------------------------------------------------
Finally, third-party tools, such as LIME and shap, can also help understand individual predictions for a model.
---------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Global feature importances                                                                     \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
---------------------------------------------------------------------------------------------------------------
Additionally, you might want to understand the model as a whole, rather than studying individual predictions. 
Below, you will compute and use:

     * Gain-based feature importances using est.experimental_feature_importances
     * Permutation importances
     * Aggregate DFCs using est.experimental_predict_with_explanations 

Gain-based feature importances measure the loss change when splitting on a particular feature, 
while permutation feature importances are computed by evaluating model performance on the evaluation 
set by shuffling each feature one-by-one and attributing the change in model performance to the shuffled feature.

In general, 
permutation feature importance are preferred to gain-based feature importance, 
though both methods can be unreliable in situations where potential predictor variables vary 
in their scale of measurement or their number of categories and when features are correlated (source). 
Check out this article for an in-depth overview and great discussion on different feature importance types.
------------------------------------------------------------------------------------------------------------------
'''

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Gain-based feature importances                                                                 \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

# Gain-based feature importances are built into the TensorFlow Boosted Trees estimators using est.experimental_feature_importances.
importances = est.experimental_feature_importances(normalize=True)
df_imp = pd.Series(importances)

# Visualize importances.
N = 8
ax = (
        df_imp.iloc[0:N][::-1].plot(
                                        kind='barh',
                                        color=sns_colors[0],
                                        title='Gain feature importances',
                                        figsize=(10, 6)
                                    )
    )

ax.grid(False, axis='y')
plt.show()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Average absolute DFCs                                                                          \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
'''
----------------------------------------------------------------------------------------------------------------
You can also average the absolute values of DFCs to understand impact at a global level.
----------------------------------------------------------------------------------------------------------------
'''
# Plot.
dfc_mean = df_dfc.abs().mean()

N = 8

sorted_ix = dfc_mean.abs().sort_values()[-N:].index  # Average and sort by absolute.

ax = dfc_mean[sorted_ix].plot(
                                kind='barh',
                                color=sns_colors[1],
                                title='Mean |directional feature contributions|',
                                figsize=(10, 6)
                            )
ax.grid(False, axis='y')

# You can also see how DFCs vary as a feature value varies.
FEATURE = 'fare'
feature = pd.Series(df_dfc[FEATURE].values, index=dfeval[FEATURE].values).sort_index()
ax = sns.regplot(feature.index.values, feature.values, lowess=True)
ax.set_ylabel('contribution')
ax.set_xlabel(FEATURE)
ax.set_xlim(0, 100)
ax.set_ylim(-0.4, 0.3)
plt.show()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Permutation feature importance                                                                          \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

def permutation_importances(est, X_eval, y_eval, metric, features):
    """Column by column, shuffle values and observe effect on eval set.

    source: http://explained.ai/rf-importance/index.html
    A similar approach can be done during training. See "Drop-column importance"
    in the above article."""
    baseline = metric(est, X_eval, y_eval)
    imp = []
    for col in features:
        save = X_eval[col].copy()
        X_eval[col] = np.random.permutation(X_eval[col])
        m = metric(est, X_eval, y_eval)
        X_eval[col] = save
        imp.append(baseline - m)
    return np.array(imp)

def accuracy_metric(est, X, y):
    """TensorFlow estimator accuracy."""
    eval_input_fn = make_input_fn(
                                    X,
                                    y=y,
                                    shuffle=False,
                                    n_epochs=1
                                )
    
    return est.evaluate(input_fn=eval_input_fn)['accuracy']

features = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
importances = permutation_importances(est, dfeval, y_eval, accuracy_metric, features)
df_imp = pd.Series(importances, index=features)

sorted_ix = df_imp.abs().sort_values().index
ax = df_imp[sorted_ix][-5:].plot(kind='barh', color=sns_colors[2], figsize=(10, 6))
ax.grid(False, axis='y')
ax.set_title('Permutation feature importance')
plt.show()

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       Visualizing model fitting                                                                      \n'
        '------------------------------------------------------------------------------------------------------\n'
        )

'''
---------------------------------------------------------------------------------------------------------------
Lets first simulate/create training data using the following formula:

𝑧=𝑥∗𝑒(−𝑥2−𝑦2)

Where (z) is the dependent variable you are trying to predict and (x) and (y) are the features.
--------------------------------------------------------------------------------------------------------------
'''

# Create fake data
seed(0)
npts = 5000
x = uniform(-2, 2, npts)
y = uniform(-2, 2, npts)
z = x*np.exp(-x**2 - y**2)

# Prep data for training.
df = pd.DataFrame({'x': x, 'y': y, 'z': z})

xi = np.linspace(-2.0, 2.0, 200),
yi = np.linspace(-2.1, 2.1, 210),
xi,yi = np.meshgrid(xi, yi)

df_predict = pd.DataFrame(
                            {
                                'x' : xi.flatten(),
                                'y' : yi.flatten(),
                            }
                        )

predict_shape = xi.shape

def plot_contour(x, y, z, **kwargs):
    # Grid the data.
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(111)
    # Contour the gridded data, plotting dots at the nonuniform data points.
    plt.contour(x, y, z, 15, linewidths=1.0, colors='k')
    CS = ax1.pcolormesh(x,y,z, cmap='RdBu_r')
    pp = fig.colorbar(CS, ax=ax1, orientation="vertical")

    pp.set_clim(-0.48, 0.48)
    # Plot data points.
    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)

# You can visualize the function. 
# Redder colors correspond to larger function values.

zi = griddata((x, y), z, (xi, yi), method='linear')
plot_contour(xi, yi, zi)
plt.scatter(df.x, df.y, color='b', marker='.')
plt.title('Contour on training data')
plt.show()

fc = [
        tf.feature_column.numeric_column('x'),
        tf.feature_column.numeric_column('y')
    ]

def predict(est):
    """Predictions from a given estimator."""
    predict_input_fn = lambda: tf.data.Dataset.from_tensors(dict(df_predict))
    preds = np.array([p['predictions'][0] for p in est.predict(predict_input_fn)])
    return preds.reshape(predict_shape)

# First let's try to fit a linear model to the data.
train_input_fn = make_input_fn(df, df.z)
est = tf.estimator.LinearRegressor(fc)
est.train(train_input_fn, max_steps=500);

plot_contour(xi, yi, predict(est))
plt.show()

'''
-------------------------------------------------------------------------------------------
It's not a very good fit. 
Next let's try to fit a GBDT model to it and try to understand how the model fits the function.
-------------------------------------------------------------------------------------------
'''
n_trees = 1 #@param {type: "slider", min: 1, max: 80, step: 1}

est = tf.estimator.BoostedTreesRegressor(fc, n_batches_per_layer=1, n_trees=n_trees)
est.train(train_input_fn, max_steps=500)
plot_contour(xi, yi, predict(est))
plt.text(-1.8, 2.1, '# trees: {}'.format(n_trees), color='w', backgroundcolor='black', size=20)
plt.show()

n_trees = 5 #@param {type: "slider", min: 1, max: 80, step: 1}

est = tf.estimator.BoostedTreesRegressor(fc, n_batches_per_layer=1, n_trees=n_trees)
est.train(train_input_fn, max_steps=500)
plot_contour(xi, yi, predict(est))
plt.text(-1.8, 2.1, '# trees: {}'.format(n_trees), color='w', backgroundcolor='black', size=20)
plt.show()

n_trees = 10 #@param {type: "slider", min: 1, max: 80, step: 1}

est = tf.estimator.BoostedTreesRegressor(fc, n_batches_per_layer=1, n_trees=n_trees)
est.train(train_input_fn, max_steps=500)
plot_contour(xi, yi, predict(est))
plt.text(-1.8, 2.1, '# trees: {}'.format(n_trees), color='w', backgroundcolor='black', size=20)
plt.show()

n_trees = 22 #@param {type: "slider", min: 1, max: 80, step: 1}

est = tf.estimator.BoostedTreesRegressor(fc, n_batches_per_layer=1, n_trees=n_trees)
est.train(train_input_fn, max_steps=500)
plot_contour(xi, yi, predict(est))
plt.text(-1.8, 2.1, '# trees: {}'.format(n_trees), color='w', backgroundcolor='black', size=20)
plt.show()

n_trees = 40 #@param {type: "slider", min: 1, max: 80, step: 1}

est = tf.estimator.BoostedTreesRegressor(fc, n_batches_per_layer=1, n_trees=n_trees)
est.train(train_input_fn, max_steps=500)
plot_contour(xi, yi, predict(est))
plt.text(-1.8, 2.1, '# trees: {}'.format(n_trees), color='w', backgroundcolor='black', size=20)
plt.show()

n_trees = 80 #@param {type: "slider", min: 1, max: 80, step: 1}

est = tf.estimator.BoostedTreesRegressor(fc, n_batches_per_layer=1, n_trees=n_trees)
est.train(train_input_fn, max_steps=500)
plot_contour(xi, yi, predict(est))
plt.text(-1.8, 2.1, '# trees: {}'.format(n_trees), color='w', backgroundcolor='black', size=20)
plt.show()

# As you increase the number of trees, the model's predictions better approximates the underlying function.

print   (
        '------------------------------------------------------------------------------------------------------\n'
        '       finished        estimator_gradient_boosted_trees.py       　          (2020/05/16)             \n'
        '------------------------------------------------------------------------------------------------------\n'
        )
print()
print()
print()