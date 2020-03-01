'''
------------------------------------------------------------------------------------------
structed data
    Classification on imbalanced data

This tutorial demonstrates how to classify a highly imbalanced dataset 
in which the number of examples in one class greatly outnumbers the examples in another. 
You will work with the Credit Card Fraud Detection dataset hosted on Kaggle. 
The aim is to detect a mere 492 fraudulent transactions from 284,807 transactions in total. 
You will use Keras to define the model and class weights to help the model learn from the imbalanced data. .

This tutorial contains complete code to:

	* Load a CSV file using Pandas.
	* Create train, validation, and test sets.
	* Define and train a model using Keras (including setting class weights).
	* Evaluate the model using various metrics (including precision and recall).
	* Try common tecniques for dealing with imbalanced data like:
		+ Class weighting
		+ Oversampling
------------------------------------------------------------------------------------------
'''
# common library
from __future__ import absolute_import, division, print_function, unicode_literals

import os
import pprint
import contextlib
import tempfile
from pathlib import Path
from PIL import Image

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf

print(__doc__)

AUTOTUNE = tf.data.experimental.AUTOTUNE
pd.options.display.max_rows = None
mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Display current path
basic_path = Path.cwd()
PROJECT_ROOT_DIR = basic_path.joinpath('Python/Normal/tensorflow')
print('PROJECT_ROOT_DIR = \n{0}\n'.format(PROJECT_ROOT_DIR))

# Display tensorflow version
print("TensorFlow version: ", tf.version.VERSION)

print   (
        '---------------------------------------------------------------------------------\n'
        '      Data processing and exploration                                            \n'
        '---------------------------------------------------------------------------------\n'
        )
print   (
        '---------------------------------------------------------------------------------\n'
        '      Download the Kaggle Credit Card Fraud data set                             \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
------------------------------------------------------------------------------------------
Pandas is a Python library with many helpful utilities for loading 
and working with structured data and can be used to download CSVs into a dataframe.

Note: 
This dataset has been collected and analysed during a research collaboration of Worldline 
and the Machine Learning Group of ULB (Université Libre de Bruxelles) on big data mining and fraud detection. 
More details on current 
and past projects on related topics are available here and the page of the DefeatFraud project
------------------------------------------------------------------------------------------
'''
file = tf.keras.utils
raw_df = pd.read_csv('https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv')
print('raw_df.head() = \n{0}\n'.format(raw_df.head()))

raw_df[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V26', 'V27', 'V28', 'Amount', 'Class']].describe()

print   (
        '---------------------------------------------------------------------------------\n'
        '      Examine the class label imbalance                                          \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
------------------------------------------------------------------------------------------
Let's look at the dataset imbalance:
------------------------------------------------------------------------------------------
'''
neg, pos = np.bincount(raw_df['Class'])
total = neg + pos
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'
      .format(total, pos, 100 * pos / total))

'''
-----------------------------------------------------------------------------------------
This shows the small fraction of positive samples.
-----------------------------------------------------------------------------------------
'''
print   (
        '---------------------------------------------------------------------------------\n'
        '      Clean, split and normalize the data                                        \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
----------------------------------------------------------------------------------------
The raw data has a few issues. 
First the Time and Amount columns are too variable to use directly. 
Drop the Time column (since it's not clear what it means) 
and take the log of the Amount column to reduce its range.
---------------------------------------------------------------------------------------
'''
cleaned_df = raw_df.copy()

# You don't want the `Time` column.
cleaned_df.pop('Time')

# The `Amount` column covers a huge range. Convert to log-space.
eps=0.001 # 0 => 0.1¢
cleaned_df['Log Ammount'] = np.log(cleaned_df.pop('Amount')+eps)

'''
---------------------------------------------------------------------------------------
Split the dataset into train, validation, and test sets. 
The validation set is used during the model fitting to evaluate the loss and any metrics, 
however the model is not fit with this data. 
The test set is completely unused during the training phase 
and is only used at the end to evaluate how well the model generalizes to new data. 
This is especially important with imbalanced datasets 
where overfitting is a significant concern from the lack of training data.
---------------------------------------------------------------------------------------
'''
# Use a utility from sklearn to split and shuffle our dataset.
train_df, test_df = train_test_split(cleaned_df, test_size=0.2)
train_df, val_df = train_test_split(train_df, test_size=0.2)

# Form np arrays of labels and features.
train_labels = np.array(train_df.pop('Class'))
bool_train_labels = train_labels != 0
val_labels = np.array(val_df.pop('Class'))
test_labels = np.array(test_df.pop('Class'))

train_features = np.array(train_df)
val_features = np.array(val_df)
test_features = np.array(test_df)

'''
-----------------------------------------------------------------
Normalize the input features using the sklearn StandardScaler. 
This will set the mean to 0 and standard deviation to 1.

Note: 
The StandardScaler is only fit using the train_features 
to be sure the model is not peeking at the validation or test sets.
-----------------------------------------------------------------
'''
scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)

val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

train_features = np.clip(train_features, -5, 5)
val_features = np.clip(val_features, -5, 5)
test_features = np.clip(test_features, -5, 5)


print('Training labels shape:', train_labels.shape)
print('Validation labels shape:', val_labels.shape)
print('Test labels shape:', test_labels.shape)

print('Training features shape:', train_features.shape)
print('Validation features shape:', val_features.shape)
print('Test features shape:', test_features.shape)

'''
-------------------------------------------------------------------------------
Caution: 
If you want to deploy a model, it's critical that you preserve the preprocessing calculations. 
The easiest way to implement them as layers, and attach them to your model before export.
-------------------------------------------------------------------------------
'''
print   (
        '---------------------------------------------------------------------------------\n'
        '      Look at the data distribution                                              \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
------------------------------------------------------------------------------------------
Next compare the distributions of the positive and negative examples over a few features. 
Good questions to ask yourself at this point are:

	* Do these distributions make sense?
		+ Yes. You've normalized the input and these are mostly concentrated in the +/- 2 range.
	* Can you see the difference between the ditributions?
		+ Yes the positive examples contain a much higher rate of extreme values.
-----------------------------------------------------------------------------------------
'''
pos_df = pd.DataFrame(train_features[ bool_train_labels], columns = train_df.columns)
neg_df = pd.DataFrame(train_features[~bool_train_labels], columns = train_df.columns)

sns.jointplot(
    pos_df['V5'], 
    pos_df['V6'],
    kind='hex', 
    xlim = (-5,5), 
    ylim = (-5,5)
)

plt.suptitle("Positive distribution")

sns.jointplot(
    neg_df['V5'], 
    neg_df['V6'],
    kind='hex', 
    xlim = (-5,5), 
    ylim = (-5,5)
)

_ = plt.suptitle("Negative distribution")

plt.show()

print   (
        '---------------------------------------------------------------------------------\n'
        '      Define the model and metrics                                               \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
------------------------------------------------------------------------------------------
Define a function that creates a simple neural network with a densly connected hidden layer, 
a dropout layer to reduce overfitting, 
and an output sigmoid layer that returns the probability of a transaction being fraudulent:
------------------------------------------------------------------------------------------
'''
METRICS = [
    tf.keras.metrics.TruePositives(name='tp'),
    tf.keras.metrics.FalsePositives(name='fp'),
    tf.keras.metrics.TrueNegatives(name='tn'),
    tf.keras.metrics.FalseNegatives(name='fn'), 
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
]

def make_model(metrics = METRICS, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    model = tf.keras.Sequential([
                tf.keras.layers.Dense(
                    16, activation='relu',
                    input_shape=(train_features.shape[-1],)),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(1, activation='sigmoid',
                                    bias_initializer=output_bias),
            ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=metrics
    )

    return model

im = Image.open(PROJECT_ROOT_DIR.joinpath('images/struct_imbalanced_data_00.jpg'))
im.show()

print   (
        '---------------------------------------------------------------------------------\n'
        '      Baseline model                                                             \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
-----------------------------------------------------------------------------------------
Build the model

Now create and train your model using the function that was defined earlier. 
Notice that the model is fit using a larger than default batch size of 2048, 
this is important to ensure that each batch has a decent chance of containing a few positive samples. 
If the batch size was too small, they would likely have no fraudulent transactions to learn from.

Note: this model will not handle the class imbalance well. You will improve it later in this tutorial.
-----------------------------------------------------------------------------------------
'''
EPOCHS = 100
BATCH_SIZE = 2048

early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_auc', 
                    verbose=1,
                    patience=10,
                    mode='max',
                    restore_best_weights=True
                )

model = make_model()
model.summary()

'''
----------------------------------------------------------------------------------------
Test run the model:
----------------------------------------------------------------------------------------
'''
model.predict(train_features[:10])

print   (
        '---------------------------------------------------------------------------------\n'
        '      Optional: Set the correct initial bias.                                    \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
------------------------------------------------------------------------------------------
These are initial guesses are not great. 
You know the dataset is imbalanced. 
Set the output layer's bias to reflect that 
(See: A Recipe for Training Neural Networks: "init well"). 
This can help with initial convergence.

With the default bias initialization the loss should be about math.log(2) = 0.69314
------------------------------------------------------------------------------------------
'''
results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
print("Loss: {:0.4f}".format(results[0]))

im = Image.open(PROJECT_ROOT_DIR.joinpath('images/struct_imbalanced_data_01.jpg'))
im.show()

initial_bias = np.log([pos/neg])
print('initial_bias = \n{0}\n'.format(initial_bias))

'''
-----------------------------------------------------------------------------------------
Set that as the initial bias, and the model will give much more reasonable initial guesses.

It should be near: pos/total = 0.0018
-----------------------------------------------------------------------------------------
'''
model = make_model(output_bias = initial_bias)
model.predict(train_features[:10])

im = Image.open(PROJECT_ROOT_DIR.joinpath('images/struct_imbalanced_data_02.jpg'))
im.show()

results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
print("Loss: {:0.4f}".format(results[0]))

'''
-----------------------------------------------------------------------------------------
This initial loss is about 50 times less than if would have been with naive initilization.

This way the model doesn't need to spend the first few epochs just learning that positive examples are unlikely. 
This also makes it easier to read plots of the loss during training.
----------------------------------------------------------------------------------------
'''
print   (
        '---------------------------------------------------------------------------------\n'
        '      Checkpoint the initial weights                                             \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
------------------------------------------------------------------------------------------
To make the various training runs more comparable, 
keep this initial model's weights in a checkpoint file, 
and load them into each model before training.
------------------------------------------------------------------------------------------
'''
initial_weights = os.path.join(tempfile.mkdtemp(),'initial_weights')
model.save_weights(initial_weights)

print   (
        '---------------------------------------------------------------------------------\n'
        '      Confirm that the bias fix helps                                            \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
------------------------------------------------------------------------------------------
Before moving on, confirm quick that the careful bias initialization actually helped.

Train the model for 20 epochs, with and without this careful initialization, and compare the losses:
-----------------------------------------------------------------------------------------
'''
model = make_model()
model.load_weights(initial_weights)
model.layers[-1].bias.assign([0.0])

zero_bias_history = model.fit(
                        train_features,
                        train_labels,
                        batch_size=BATCH_SIZE,
                        epochs=20,
                        validation_data=(val_features, val_labels), 
                        verbose=0
                    )

model = make_model()
model.load_weights(initial_weights)
careful_bias_history = model.fit(
                            train_features,
                            train_labels,
                            batch_size=BATCH_SIZE,
                            epochs=20,
                            validation_data=(val_features, val_labels), 
                            verbose=0
                        )

def plot_loss(history, label, n):
    # Use a log scale to show the wide range of values.
    plt.semilogy(
            history.epoch,  history.history['loss'],
            color=colors[n], label='Train '+label
        )
    
    plt.semilogy(
            history.epoch,  history.history['val_loss'],
            color=colors[n], label='Val '+label,
            linestyle="--"
        )
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.legend()
    plt.show()

plot_loss(zero_bias_history, "Zero Bias", 0)
plot_loss(careful_bias_history, "Careful Bias", 1)

'''
------------------------------------------------------------------------------------
The above figure makes it clear: 
In terms of validation loss, on this problem, 
this careful initialization gives a clear advantage.
------------------------------------------------------------------------------------
'''
print   (
        '---------------------------------------------------------------------------------\n'
        '      Train the model                                                            \n'
        '---------------------------------------------------------------------------------\n'
        )

model = make_model()
model.load_weights(initial_weights)
baseline_history = model.fit(
                        train_features,
                        train_labels,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        callbacks = [early_stopping],
                        validation_data=(val_features, val_labels)
                    )

print   (
        '---------------------------------------------------------------------------------\n'
        '      Check training history                                                     \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
------------------------------------------------------------------------------------------
In this section, 
you will produce plots of your model's accuracy and loss on the training and validation set. 
These are useful to check for overfitting, which you can learn more about in this tutorial.

Additionally, you can produce these plots for any of the metrics you created above. 
False negatives are included as an example.
-----------------------------------------------------------------------------------------
'''
def plot_metrics(history):
    metrics =  ['loss', 'auc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch,  history.history[metric], 
                color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])

        plt.legend()

plot_metrics(baseline_history)
plt.show()

'''
-----------------------------------------------------------------------------------------
Note: 
That the validation curve generally performs better than the training curve. 
This is mainly caused by the fact that the dropout layer is not active when evaluating the model.
-----------------------------------------------------------------------------------------
'''
print   (
        '---------------------------------------------------------------------------------\n'
        '      Evaluate metrics                                                           \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
-----------------------------------------------------------------------------------------
You can use a confusion matrix to summarize the actual vs. 
predicted labels where the X axis is the predicted label and the Y axis is the actual label.
-----------------------------------------------------------------------------------------
'''
train_predictions_baseline = model.predict(train_features, batch_size=BATCH_SIZE)
test_predictions_baseline = model.predict(test_features, batch_size=BATCH_SIZE)

def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
    print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
    print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
    print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
    print('Total Fraudulent Transactions: ', np.sum(cm[1]))

'''
---------------------------------------------------------------------------------------
Evaluate your model on the test dataset and display the results for the metrics you created above.
---------------------------------------------------------------------------------------
'''
baseline_results = model.evaluate(
                        test_features, test_labels,
                        batch_size=BATCH_SIZE, 
                        verbose=0
                    )

for name, value in zip(model.metrics_names, baseline_results):
    print(name, ': ', value)
print()

plot_cm(test_labels, test_predictions_baseline)

plt.show()

'''
----------------------------------------------------------------------------------------
If the model had predicted everything perfectly, 
this would be a diagonal matrix where values off the main diagonal, 
indicating incorrect predictions, would be zero. 
In this case the matrix shows that you have relatively few false positives, 
meaning that there were relatively few legitimate transactions that were incorrectly flagged. 
However, 
you would likely want to have even fewer false negatives despite the cost of increasing the number of false positives. 
This trade off may be preferable because false negatives would allow fraudulent transactions to go through, 
whereas false positives may cause an email to be sent to a customer to ask them to verify their card activity.
----------------------------------------------------------------------------------------
'''
print   (
        '---------------------------------------------------------------------------------\n'
        '      Plot the ROC                                                               \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
------------------------------------------------------------------------------------------
Now plot the ROC. 
This plot is useful because it shows, at a glance, 
the range of performance the model can reach just by tuning the output threshold.
-----------------------------------------------------------------------------------------
'''
def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5,20])
    plt.ylim([80,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')

plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])
plot_roc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')
plt.legend(loc='lower right')
plt.show()

'''
-------------------------------------------------------------------------------------------------
It looks like the precision is relatively high, 
but the recall and the area under the ROC curve (AUC) aren't as high as you might like. 
Classifiers often face challenges when trying to maximize both precision and recall, 
which is especially true when working with imbalanced datasets. 
It is important to consider the costs of different types of errors in the context of the problem you care about. 
In this example, 
a false negative (a fraudulent transaction is missed) may have a financial cost, 
while a false positive (a transaction is incorrectly flagged as fraudulent) may decrease user happiness.
------------------------------------------------------------------------------------------------
'''
print   (
        '---------------------------------------------------------------------------------\n'
        '      Class weights                                                              \n'
        '---------------------------------------------------------------------------------\n'
        )
print   (
        '---------------------------------------------------------------------------------\n'
        '      Calculate class weights                                                    \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
-----------------------------------------------------------------------------------------
The goal is to identify fradulent transactions, 
but you don't have very many of those positive samples to work with, 
so you would want to have the classifier heavily weight the few examples that are available. 
You can do this by passing Keras weights for each class through a parameter. 
These will cause the model to "pay more attention" to examples from an under-represented class.
-----------------------------------------------------------------------------------------
'''
# Scaling by total/2 helps keep the loss to a similar magnitude.
# The sum of the weights of all examples stays the same.
weight_for_0 = (1 / neg)*(total)/2.0 
weight_for_1 = (1 / pos)*(total)/2.0

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))

print   (
        '---------------------------------------------------------------------------------\n'
        '      Train a model with class weights                                           \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
-----------------------------------------------------------------------------------------
Now try re-training and evaluating the model with class weights to see how that affects the predictions.

Note: 
Using class_weights changes the range of the loss. 
This may affect the stability of the training depending on the optimizer. 
Optimizers whose step size is dependent on the magnitude of the gradient, like optimizers.SGD, may fail. 
The optimizer used here, optimizers.Adam, is unaffected by the scaling change. 
Also note that because of the weighting, the total losses are not comparable between the two models.
---------------------------------------------------------------------------------------
'''
weighted_model = make_model()
weighted_model.load_weights(initial_weights)

weighted_history = weighted_model.fit(
                        train_features,
                        train_labels,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        callbacks = [early_stopping],
                        validation_data=(val_features, val_labels),
                        # The class weights go here
                        class_weight=class_weight
                    ) 

print   (
        '---------------------------------------------------------------------------------\n'
        '      Check training history                                                     \n'
        '---------------------------------------------------------------------------------\n'
        )
plot_metrics(weighted_history)
plt.show()

print   (
        '---------------------------------------------------------------------------------\n'
        '      Evaluate metrics                                                           \n'
        '---------------------------------------------------------------------------------\n'
        )
train_predictions_weighted = weighted_model.predict(train_features, batch_size=BATCH_SIZE)
test_predictions_weighted = weighted_model.predict(test_features, batch_size=BATCH_SIZE)

weighted_results = weighted_model.evaluate(
                        test_features, test_labels,
                        batch_size=BATCH_SIZE, verbose=0
                    )

for name, value in zip(weighted_model.metrics_names, weighted_results):
    print(name, ': ', value)
print()

plot_cm(test_labels, test_predictions_weighted)
plt.show()

'''
-----------------------------------------------------------------------------------------
Here you can see that with class weights the accuracy and precision are lower 
because there are more false positives, 
but conversely the recall and AUC are higher because the model also found more true positives. 
Despite having lower accuracy, 
this model has higher recall (and identifies more fraudulent transactions). 
Of course, 
there is a cost to both types of error 
(you wouldn't want to bug users by flagging too many legitimate transactions as fraudulent, either). 
Carefully consider the trade offs between these different types of errors for your application.
-----------------------------------------------------------------------------------------
'''
print   (
        '---------------------------------------------------------------------------------\n'
        '      Plot the ROC                                                               \n'
        '---------------------------------------------------------------------------------\n'
        )
plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])
plot_roc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')

plot_roc("Train Weighted", train_labels, train_predictions_weighted, color=colors[1])
plot_roc("Test Weighted", test_labels, test_predictions_weighted, color=colors[1], linestyle='--')

plt.legend(loc='lower right')
plt.show()

print   (
        '---------------------------------------------------------------------------------\n'
        '      Oversampling                                                               \n'
        '---------------------------------------------------------------------------------\n'
        )
print   (
        '---------------------------------------------------------------------------------\n'
        '      Oversample the minority class                                              \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
------------------------------------------------------------------------------------------
A related approach would be to resample the dataset by oversampling the minority class.
------------------------------------------------------------------------------------------
'''
pos_features = train_features[bool_train_labels]
neg_features = train_features[~bool_train_labels]

pos_labels = train_labels[bool_train_labels]
neg_labels = train_labels[~bool_train_labels]

print   (
        '---------------------------------------------------------------------------------\n'
        '      Using NumPy                                                                \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
------------------------------------------------------------------------------------------
You can balance the dataset manually by choosing the right number of random indices 
from the positive examples:
------------------------------------------------------------------------------------------
'''
ids = np.arange(len(pos_features))
choices = np.random.choice(ids, len(neg_features))

res_pos_features = pos_features[choices]
res_pos_labels = pos_labels[choices]

res_pos_features.shape

resampled_features = np.concatenate([res_pos_features, neg_features], axis=0)
resampled_labels = np.concatenate([res_pos_labels, neg_labels], axis=0)

order = np.arange(len(resampled_labels))
np.random.shuffle(order)
resampled_features = resampled_features[order]
resampled_labels = resampled_labels[order]

resampled_features.shape

print   (
        '---------------------------------------------------------------------------------\n'
        '      Using tf.data                                                              \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
------------------------------------------------------------------------------------------
If you're using tf.data the easiest way to produce balanced examples is to start with a positive and a negative dataset, 
and merge them. See the tf.data guide for more examples.
------------------------------------------------------------------------------------------
'''
BUFFER_SIZE = 100000

def make_ds(features, labels):
    ds = tf.data.Dataset.from_tensor_slices((features, labels))#.cache()
    ds = ds.shuffle(BUFFER_SIZE).repeat()
    return ds

pos_ds = make_ds(pos_features, pos_labels)
neg_ds = make_ds(neg_features, neg_labels)

'''
------------------------------------------------------------------------------------------
Each dataset provides (feature, label) pairs:
------------------------------------------------------------------------------------------
'''
for features, label in pos_ds.take(1):
    print("Features:\n", features.numpy())
    print()
    print("Label: ", label.numpy())

'''
------------------------------------------------------------------------------------------
Merge the two together using experimental.sample_from_datasets:
------------------------------------------------------------------------------------------
'''
resampled_ds = tf.data.experimental.sample_from_datasets([pos_ds, neg_ds], weights=[0.5, 0.5])
resampled_ds = resampled_ds.batch(BATCH_SIZE).prefetch(2)

for features, label in resampled_ds.take(1):
    print(label.numpy().mean())

'''
-----------------------------------------------------------------------------------------
To use this dataset, you'll need the number of steps per epoch.

The definition of "epoch" in this case is less clear. 
Say it's the number of batches required to see each negative example once:
----------------------------------------------------------------------------------------
'''
resampled_steps_per_epoch = np.ceil(2.0*neg/BATCH_SIZE)

print('resampled_steps_per_epoch = {0}\n'.format(resampled_steps_per_epoch))

print   (
        '---------------------------------------------------------------------------------\n'
        '      Train on the oversampled data                                              \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
------------------------------------------------------------------------------------------
Now try training the model with the resampled data set instead of using class weights 
to see how these methods compare.

Note: 
Because the data was balanced by replicating the positive examples, 
the total dataset size is larger, and each epoch runs for more training steps.
-----------------------------------------------------------------------------------------
'''
resampled_model = make_model()
resampled_model.load_weights(initial_weights)

# Reset the bias to zero, since this dataset is balanced.
output_layer = resampled_model.layers[-1] 
output_layer.bias.assign([0])

val_ds = tf.data.Dataset.from_tensor_slices((val_features, val_labels)).cache()
val_ds = val_ds.batch(BATCH_SIZE).prefetch(2) 

resampled_history = resampled_model.fit(
                        resampled_ds,
                        epochs=EPOCHS,
                        steps_per_epoch=resampled_steps_per_epoch,
                        callbacks = [early_stopping],
                        validation_data=val_ds
                    )

'''
------------------------------------------------------------------------------------------
If the training process were considering the whole dataset on each gradient update, 
this oversampling would be basically identical to the class weighting.

But when training the model batch-wise, as you did here, 
the oversampled data provides a smoother gradient signal: 
Instead of each positive example being shown in one batch with a large weight, 
they're shown in many different batches each time with a small weight.

This smoother gradient signal makes it easier to train the model.
------------------------------------------------------------------------------------------
'''

print   (
        '---------------------------------------------------------------------------------\n'
        '      Check training history                                                     \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
------------------------------------------------------------------------------------------
Note that the distributions of metrics will be different here, 
because the training data has a totally different distribution from the validation and test data.
-----------------------------------------------------------------------------------------
'''
plot_metrics(resampled_history )
plt.show()

print   (
        '---------------------------------------------------------------------------------\n'
        '      Re-train                                                                   \n'
        '---------------------------------------------------------------------------------\n'
        )
'''
------------------------------------------------------------------------------------------
Because training is easier on the balanced data, the above training procedure may overfit quickly.

So break up the epochs to give the callbacks.EarlyStopping finer control over when to stop training.
-----------------------------------------------------------------------------------------
'''
resampled_model = make_model()
resampled_model.load_weights(initial_weights)

# Reset the bias to zero, since this dataset is balanced.
output_layer = resampled_model.layers[-1] 
output_layer.bias.assign([0])

resampled_history = resampled_model.fit(
                        resampled_ds,
                        # These are not real epochs
                        steps_per_epoch = 20,
                        epochs=10*EPOCHS,
                        callbacks = [early_stopping],
                        validation_data=(val_ds)
                    )

print   (
        '---------------------------------------------------------------------------------\n'
        '      Re-check training history                                                  \n'
        '---------------------------------------------------------------------------------\n'
        )
plot_metrics(resampled_history)
plt.show()

print   (
        '---------------------------------------------------------------------------------\n'
        '      Evaluate metrics                                                           \n'
        '---------------------------------------------------------------------------------\n'
        )
train_predictions_resampled = resampled_model.predict(train_features, batch_size=BATCH_SIZE)
test_predictions_resampled = resampled_model.predict(test_features, batch_size=BATCH_SIZE)

resampled_results = resampled_model.evaluate(
                        test_features, test_labels,
                        batch_size=BATCH_SIZE, verbose=0
                    )

for name, value in zip(resampled_model.metrics_names, resampled_results):
    print(name, ': ', value)
print()

plot_cm(test_labels, test_predictions_weighted)
plt.show()

print   (
        '---------------------------------------------------------------------------------\n'
        '      Plot the ROC                                                               \n'
        '---------------------------------------------------------------------------------\n'
        )

plot_roc("Train Baseline", train_labels, train_predictions_baseline, color=colors[0])
plot_roc("Test Baseline", test_labels, test_predictions_baseline, color=colors[0], linestyle='--')

plot_roc("Train Weighted", train_labels, train_predictions_weighted, color=colors[1])
plot_roc("Test Weighted", test_labels, test_predictions_weighted, color=colors[1], linestyle='--')

plot_roc("Train Resampled", train_labels, train_predictions_resampled,  color=colors[2])
plot_roc("Test Resampled", test_labels, test_predictions_resampled,  color=colors[2], linestyle='--')
plt.legend(loc='lower right')
plt.show()
