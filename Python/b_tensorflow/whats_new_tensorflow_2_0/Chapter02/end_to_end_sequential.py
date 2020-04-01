import tensorflow as tf
#import tensorflow.keras as keras

# Load Data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()


# Convert data from numpy arrays to tensors
x_train = tf.convert_to_tensor(
    x_train,
    dtype='float'
)

y_train = tf.one_hot(
    y_train,
    depth=10
)


x_test = tf.convert_to_tensor(
    x_test,
    dtype='float'
)
y_test = tf.one_hot(
    y_test,
    depth=10
)


# Build neural network for classification
model = tf.keras.Sequential()


model.add(
    tf.keras.layers.Flatten()
)

model.add(
    tf.keras.layers.Dense(
        100,
        activation='relu',
    )
)
model.add(
    tf.keras.layers.Dense(
        100,
        activation='relu'
    )
)
model.add(
    tf.keras.layers.Dense(
        10,
        activation='softmax'
    )
)

# Compile model
model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.SGD(),
    metrics=['accuracy']
)

# Train model
model.fit(
    x_train,
    y_train,
    batch_size=32,
	steps_per_epoch=30,
    epochs=100
)

print()

# Run evaluation
score = model.evaluate(
    x_test,
    y_test
)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
