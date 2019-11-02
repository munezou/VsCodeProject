from __future__ import absolute_import, division, print_function, unicode_literals

'''
------------------------------------------------------------------------------------------------------------------------
Acquisition of sample data set
------------------------------------------------------------------------------------------------------------------------
'''
import os

import tensorflow as tf
from tensorflow import keras

print('tf.__version__ = {0}'.format(tf.__version__))

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

'''
------------------------------------------------------------------------------------------------------------------------
Declare model
------------------------------------------------------------------------------------------------------------------------
'''
# A function that returns a short sequential model
def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(10, activation=tf.keras.activations.softmax)
  ])

  model.compile(optimizer=tf.keras.optimizers.Adam(),
                loss=tf.keras.losses.sparse_categorical_crossentropy,
                metrics=['accuracy'])

  return model

# make an instance of basic model.
model = create_model()
model.summary()

'''
------------------------------------------------------------------------------------------------------------------------
How to use check point callback.
------------------------------------------------------------------------------------------------------------------------
'''
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# make check point callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1)

model = create_model()

# hand 'call back' to training.
model.fit(train_images, train_labels,  epochs = 10, validation_data = (test_images,test_labels), callbacks = [cp_callback])

model = create_model()

loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))

model.load_weights(checkpoint_path)
loss,acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

'''
------------------------------------------------------------------------------------------------------------------------
Checkpoint callback options
------------------------------------------------------------------------------------------------------------------------
'''
# Embed an epoch number (using `str.format`) in the file name.
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights every 5 epochs.
    period=5)

model = create_model()
model.fit(train_images, train_labels, epochs = 50, callbacks = [cp_callback], validation_data = (test_images,test_labels), verbose=0)

latest = tf.train.latest_checkpoint(checkpoint_dir)
print('latest = {0}'.format(latest))

'''
------------------------------------------------------------------------------------------------------------------------
Reset the model and load the last checkpoint for testing.
------------------------------------------------------------------------------------------------------------------------
'''
model = create_model()
model.load_weights(latest)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

'''
------------------------------------------------------------------------------------------------------------------------
Manually save the weights.
------------------------------------------------------------------------------------------------------------------------
'''
# Save weights.
model.save_weights('./checkpoints/my_checkpoint')

# Weight recovery
model = create_model()
model.load_weights('./checkpoints/my_checkpoint')

loss,acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

'''
------------------------------------------------------------------------------------------------------------------------
Save entire model
You can save the entire model, including the weight values, model settings, and (depending on the configuration) the optimizer settings, to a file. 
This allows you to save the state of the model at a point in time and resume training where it left off without having to access the original Python code.

(If you save the model using optimizer other than optimizer included in tf.train module to HDF5 file, you can save the optimizer settings.)
------------------------------------------------------------------------------------------------------------------------
'''
model = create_model()

model.fit(train_images, train_labels, epochs=5)

# Save the entire model into one HDF5 file.
model.save('my_model.h5')

'''
------------------------------------------------------------------------------------------------------------------------
Re-create the model using the saved file.
------------------------------------------------------------------------------------------------------------------------
'''
# Recreate exactly the same model, including weights and the optimizer.
new_model = keras.models.load_model('my_model.h5')
new_model.summary()

'''
------------------------------------------------------------------------------------------------------------------------
Check the accuracy rate.
------------------------------------------------------------------------------------------------------------------------
'''
loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

'''
------------------------------------------------------------------------------------------------------------------------
as saved_model
Note: Storage of the tf.keras model by this method is experimental and may change in future versions.

Make a new model.
------------------------------------------------------------------------------------------------------------------------
'''
model = create_model()

model.fit(train_images, train_labels, epochs=5)

# Create saved_model.
saved_model_path = tf.contrib.saved_model.save_keras_model(model, "saved_models")

'''
------------------------------------------------------------------------------------------------------------------------
Reload the new Keras model from the saved model (SavedModel).
------------------------------------------------------------------------------------------------------------------------
'''
new_model = tf.contrib.saved_model.load_keras_model(saved_model_path)
new_model

'''
------------------------------------------------------------------------------------------------------------------------
Run the restored model.
------------------------------------------------------------------------------------------------------------------------
'''
# You must compile the model before evaluating it.
# This step is not necessary if you just deploy the model.

new_model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

# モデルを評価します。
loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))







