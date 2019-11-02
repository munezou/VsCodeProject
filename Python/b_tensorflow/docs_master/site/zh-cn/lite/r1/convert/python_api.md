# Python API 指南

本提供了一些示例来明如何通 Python API 用 TensorFlow Lite 器，以及解器。

注意 : 本文介的是 Tensorflow nightly 版本的器， 行 `pip install tf-nightly` 安装此版本。
旧版文档参考[“ TensorFlow 1.12 及之前版本的模型”](#pre_tensorflow_1.12)。

[TOC]

  
## 概述

然也可以在命令行中用 TensorFlow Lite 器，但用 Python 脚本用 API 的方式可以作模型流水 (model development pipeline) 的一，通常会更加便捷；可以更早的了解正在的模型是否移

## API

`tf.lite.TFLiteConverter`：用于将 TensorFlow 模型 TensorFlow Lite 的 API。 
`tf.lite.Interpreter`：用于用 Python 解器的 API。

不同的模型原始格式，`TFLiteConverter` 提供了多用于的方法。
`TFLiteConverter.from_session()` 用于 GraphDefs。
`TFLiteConverter.from_saved_model()` 用于 SavedModels。
`TFLiteConverter.from_keras_model_file()` 用于 `tf.Keras` 文件。
[基本示例](#basic) 展示浮点模型的用法。[示例](#complex) 展示更的模型用法。

## 基本示例 <a name="basic"></a>

以下部分示了如何把基本浮点模型从各原始数据格式成 TensorFlow Lite FlatBuffers。

### 使用 tf.Session 出 GraphDef <a name="basic_graphdef_sess"></a>

以下示例展示了如何从 `tf.Session` 象一个 TensorFlow GraphDef 成 TensorFlow Lite FlatBuffer。

```python
import tensorflow as tf

img = tf.placeholder(name="img", dtype=tf.float32, shape=(1, 64, 64, 3))
var = tf.get_variable("weights", dtype=tf.float32, shape=(1, 64, 64, 3))
val = img + var
out = tf.identity(val, name="out")

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  converter = tf.lite.TFLiteConverter.from_session(sess, [img], [out])
  tflite_model = converter.convert()
  open("converted_model.tflite", "wb").write(tflite_model)
```

### 使用文件出 GraphDef <a name="basic_graphdef_file"></a>

以下示例展示了当 GraphDef 被存成文件，是怎一个 TensorFlow GraphDef 到 TensorFlow Lite FlatBuffer。支持文件后 .pb 和 .pbtxt。

示例中用到的文件下包：[Mobilenet_1.0_224](https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_1.0_224_frozen.tgz)。
函数只支持用 [freeze_graph.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py) 的 GraphDef。

```python
import tensorflow as tf

graph_def_file = "/path/to/Downloads/mobilenet_v1_1.0_224/frozen_graph.pb"
input_arrays = ["input"]
output_arrays = ["MobilenetV1/Predictions/Softmax"]

converter = tf.lite.TFLiteConverter.from_frozen_graph(
  graph_def_file, input_arrays, output_arrays)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
```

### 出 SavedModel <a name="basic_savedmodel"></a>

以下示例展示了如何将 SavedModel 成 TensorFlow Lite FlatBuffer。

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
```

于更的 SavedModel, 可以 `TFLiteConverter.from_saved_model()` 函数可参数：
`input_arrays`，`input_shapes`，`output_arrays`，`tag_set`，`signature_key`。
行 `help(tf.lite.TFLiteConverter)` 看参数情。

### 出 tf.keras 文件 <a name="basic_keras_file"></a>

以下示例展示如何将 `tf.keras` 模型成 TensorFlow Lite FlatBuffer。示例需要先安装[`h5py`](http://docs.h5py.org/en/latest/build.html)

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_keras_model_file("keras_model.h5")
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
```

`tf.keras` 文件必包含模型和重。一个全面的包括模型造在内的示例如下所示：

```python
import numpy as np
import tensorflow as tf

# Generate tf.keras model.
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(2, input_shape=(3,)))
model.add(tf.keras.layers.RepeatVector(3))
model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(3)))
model.compile(loss=tf.keras.losses.MSE,
              optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
              metrics=[tf.keras.metrics.categorical_accuracy],
              sample_weight_mode='temporal')

x = np.random.random((1, 3))
y = np.random.random((1, 3, 3))
model.train_on_batch(x, y)
model.predict(x)

# Save tf.keras model in HDF5 format.
keras_file = "keras_model.h5"
tf.keras.models.save_model(model, keras_file)

# Convert to TensorFlow Lite model.
converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_file)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
```

## 示例 <a name="complex"></a>

于属性默不足的模型，在用 `convert()` 之前置属性。
例如置任何常量都需要使用 `tf.lite.constants.<CONSTANT_NAME>`，以下示例中使用了常量 `QUANTIZED_UINT8`。
可以在 Python 端中行 `help(tf.lite.TFLiteConverter)` 取有属性的文档。

尽管示例中只演示了包含常量的 GraphDefs，同的可以用于一入数据格式。

### 出量化 GraphDef <a name="complex_quant"></a>

以下示例展示了如何把量化模型成 TensorFlow Lite FlatBuffer。

```python
import tensorflow as tf

img = tf.placeholder(name="img", dtype=tf.float32, shape=(1, 64, 64, 3))
const = tf.constant([1., 2., 3.]) + tf.constant([1., 4., 4.])
val = img + const
out = tf.fake_quant_with_min_max_args(val, min=0., max=1., name="output")

with tf.Session() as sess:
  converter = tf.lite.TFLiteConverter.from_session(sess, [img], [out])
  converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
  input_arrays = converter.get_input_arrays()
  converter.quantized_input_stats = {input_arrays[0] : (0., 1.)}  # mean, std_dev
  tflite_model = converter.convert()
  open("converted_model.tflite", "wb").write(tflite_model)
```

## TensorFlow Lite Python 解器 <a name="interpreter"></a>

### 从模型文件用解器 <a name="interpreter_file"></a>

以下示例展示了得 TensorFlow Lite FlatBuffer 文件后，如何使用 TensorFlow Lite Python 解器。
此代演示了如何随机入数据行推理。可以在 Python 端中行 `help(tf.lite.Interpreter)` 取解器的文档。

```python
import numpy as np
import tensorflow as tf

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="converted_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
```

### 从模型数据用解器 <a name="interpreter_data"></a>

以下示例展示了如何从之前加好的 TensorFlow Lite Flatbuffer 模型，用 TensorFlow Lite Python 解器。
此代示了一个从建 TensorFlow 模型始的端到端案例。

```python
import numpy as np
import tensorflow as tf

img = tf.placeholder(name="img", dtype=tf.float32, shape=(1, 64, 64, 3))
const = tf.constant([1., 2., 3.]) + tf.constant([1., 4., 4.])
val = img + const
out = tf.identity(val, name="out")

with tf.Session() as sess:
  converter = tf.lite.TFLiteConverter.from_session(sess, [img], [out])
  tflite_model = converter.convert()

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
```

## 附加明

### 源建 <a name="latest_package"></a>

了行最新版本的 TensorFlow Lite Converter Python API，可以一方式安装 nightly 版本：
[pip](https://www.tensorflow.org/install/pip)（推荐），
[Docker](https://www.tensorflow.org/install/docker)，
[从源代建 pip 包](https://www.tensorflow.org/install/source)。

###  TensorFlow 1.12 及之前版本的模型 <a name="pre_tensorflow_1.12"></a>

参考下表在 TensorFlow 1.12 之前的版本中 TensorFlow 模型到 TensorFlow Lite
行 `help()` 取 API 的情。 

TensorFlow 版本 | Python API
------------------ | ---------------------------------
1.12               | `tf.contrib.lite.TFLiteConverter`
1.9-1.11           | `tf.contrib.lite.TocoConverter`
1.7-1.8            | `tf.contrib.lite.toco_convert`