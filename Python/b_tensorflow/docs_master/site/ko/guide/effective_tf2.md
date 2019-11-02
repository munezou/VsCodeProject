#   2.0

Note:     .           

[  ](https://github.com/tensorflow/docs/blob/master/site/en/guide/effective_tf2.md)
    .     
[tensorflow/docs](https://github.com/tensorflow/docs)     
.    
[docs-ko@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-ko)
  .

 2.0       . [ API](https://github.com/tensorflow/community/blob/master/rfcs/20180827-api-names.md)  API  ([Unified RNNs](https://github.com/tensorflow/community/blob/master/rfcs/20180920-unify-rnn-interface.md),
[Unified Optimizers](https://github.com/tensorflow/community/blob/master/rfcs/20181016-optimizer-unification.md))  (runtime) [ ](https://www.tensorflow.org/guide/eager)(eager execution) .

 [RFC](https://github.com/tensorflow/community/pulls?utf8=%E2%9C%93&q=is%3Apr)   2.0     .    2.0    .   1.x  .

##    

### API 

 API TF 2.0 [  ](https://github.com/tensorflow/community/blob/master/rfcs/20180827-api-names.md).    . `tf.app`, `tf.flags`, `tf.logging`  [absl-py](https://github.com/abseil/abseil-py)   . `tf.contrib`    .     `tf.math`  (subpackage)   `tf.*` (namespace) . `tf.summary`, `tf.keras.metrics`, `tf.keras.optimizers`   API 2.0  .     [v2 upgrade script](upgrade.md)    .

###  

 1.x  `tf.*` API  [  ](https://ko.wikipedia.org/wiki/%EC%B6%94%EC%83%81_%EA%B5%AC%EB%AC%B8_%ED%8A%B8%EB%A6%AC)  .  `session.run()`            .  2.0 ( )  .  2.0    (implementation detail)  .

       `tf.control_dependencies()`   .     (`tf.function`       ).

###   

 1.x      (namespace)  . `tf.Variable()`    (node) .         .  `tf.Variable`           .      .             .  (variable scope),  (global collection), `tf.get_global_step()` `tf.global_variables_initializer()`    .  (optimizer)      (graident) .  2.0    ([Variables 2.0 RFC](https://github.com/tensorflow/community/pull/11)).       ! `tf.Variable`      (garbage collection) .

     (Keras)( )    .

###   

`session.run()`    .      .  2.0 `tf.function()` (decorator)     .          JIT ([Functions 2.0 RFC](https://github.com/tensorflow/community/pull/20)).     2.0       .

-   :    ( (pruning),  (kernel fusion) ).
-   (portability):      ([SavedModel 2.0 RFC](https://github.com/tensorflow/community/pull/34)).       .

```python
#  1.x
outputs = session.run(f(placeholder), feed_dict={placeholder: input})
#  2.0
outputs = f(input)
```

              .     , C++,    .       `@tf.function`  [](function.ipynb)(AutoGraph)      .

*   `for`/`while` -> `tf.while_loop` (`break` `continue`  .)
*   `if` -> `tf.cond`
*   `for _ in dataset` -> `dataset.reduce`

     . (sequence) ,  (reinforcement learning),              .

##  2.0  

###    .

 1.x    " (kitchen sink)" .       `session.run()`    .  2.0         (refactoring) .   `tf.function`    .    (step)  (forward pass)    `tf.function`  .

###      .

  (layer)      `variables` `trainable_variables`  .       .

 :

```python
def dense(x, W, b):
  return tf.nn.sigmoid(tf.matmul(x, W) + b)

@tf.function
def multilayer_perceptron(x, w0, b0, w1, b1, w2, b2 ...):
  x = dense(x, w0, b0)
  x = dense(x, w1, b1)
  x = dense(x, w2, b2)
  ...

#  w_i, b_i    .     .
```

 :

```python
#   linear(x)  .
layers = [tf.keras.layers.Dense(hidden_size, activation=tf.nn.sigmoid) for _ in range(n)]
perceptron = tf.keras.Sequential(layers)

# layers[3].trainable_variables => returns [w3, b3]
# perceptron.trainable_variables => returns [w0, b0, ...]
```

   `tf.train.Checkpointable`  `@tf.function`   .     SavedModels   .  `.fit()` API       .

 (transfer learning)          . (trunk)   (multi-headed)    .

```python
trunk = tf.keras.Sequential([...])
head1 = tf.keras.Sequential([...])
head2 = tf.keras.Sequential([...])

path1 = tf.keras.Sequential([trunk, head1])
path2 = tf.keras.Sequential([trunk, head2])

#   .
for x, y in main_dataset:
  with tf.GradientTape() as tape:
    prediction = path1(x)
    loss = loss_fn_head1(prediction, y)
  # trunk head1   .
  gradients = tape.gradient(loss, path1.trainable_variables)
  optimizer.apply_gradients(zip(gradients, path1.trainable_variables))

# trunk  head2  .
for x, y in small_dataset:
  with tf.GradientTape() as tape:
    prediction = path2(x)
    loss = loss_fn_head2(prediction, y)
  # trunk   head2  .
  gradients = tape.gradient(loss, head2.trainable_variables)
  optimizer.apply_gradients(zip(gradients, head2.trainable_variables))

# trunk      .
tf.saved_model.save(trunk, output_path)
```

### tf.data.Datasets @tf.function .

           .          `tf.data.Dataset`.  [ ](https://docs.python.org/ko/3/glossary.html#term-iterable)( )         . `tf.function()`    (prefetch)/(streaming)     . `tf.function()`         .

```python
@tf.function
def train(model, dataset, optimizer):
  for x, y in dataset:
    with tf.GradientTape() as tape:
      prediction = model(x)
      loss = loss_fn(prediction, y)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

 `.fit()` API        .

```python
model.compile(optimizer=optimizer, loss=loss_fn)
model.fit(dataset)
```

###      .

     (control flow) `tf.cond` `tf.while_loop`      .

       (sequence) . `tf.keras.layers.RNN` RNN (cell)  (recurrent)       .         .

```python
class DynamicRNN(tf.keras.Model):

  def __init__(self, rnn_cell):
    super(DynamicRNN, self).__init__(self)
    self.cell = rnn_cell

  def call(self, input_data):
    # [batch, time, features] -> [time, batch, features]
    input_data = tf.transpose(input_data, [1, 0, 2])
    outputs = tf.TensorArray(tf.float32, input_data.shape[0])
    state = self.cell.zero_state(input_data.shape[1], dtype=tf.float32)
    for i in tf.range(input_data.shape[0]):
      output, state = self.cell(input_data[i], state)
      outputs = outputs.write(i, output)
    return tf.transpose(outputs.stack(), [1, 0, 2]), state
```

       [](./function.ipynb) .

### tf.metrics   tf.summary .

(summary)   `tf.summary.(scalar|histogram|...)` .  (context manager)      . (       .) TF 1.x       .  "(merge)"  `add_summary()`  .     `step`    .

```python
summary_writer = tf.summary.create_file_writer('/tmp/summaries')
with summary_writer.as_default():
  tf.summary.scalar('loss', 0.1, step=42)
```

`summary`    `tf.metrics` .    .    `.result()`    . `.reset_stats()`      .

```python
def train(model, optimizer, dataset, log_freq=10):
  avg_loss = tf.keras.metrics.Mean(name='loss', dtype=tf.float32)
  for images, labels in dataset:
    loss = train_step(model, optimizer, images, labels)
    avg_loss.update_state(loss)
    if tf.equal(optimizer.iterations % log_freq, 0):
      tf.summary.scalar('loss', avg_loss.result(), step=optimizer.iterations)
      avg_loss.reset_states()

def test(model, test_x, test_y, step_num):
  loss = loss_fn(model(test_x), test_y)
  tf.summary.scalar('loss', loss, step=step_num)

train_summary_writer = tf.summary.create_file_writer('/tmp/summaries/train')
test_summary_writer = tf.summary.create_file_writer('/tmp/summaries/test')

with train_summary_writer.as_default():
  train(model, optimizer, dataset)

with test_summary_writer.as_default():
  test(model, test_x, test_y, optimizer.iterations)
```

(TensorBoard)        : `tensorboard --logdir /tmp/summaries`
