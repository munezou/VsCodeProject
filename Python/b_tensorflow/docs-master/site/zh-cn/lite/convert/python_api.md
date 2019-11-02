# 模型器（Converter）的 Python API 指南

此面提供了一个于在 TensorFlow 2.0 中如何使用 
[TensorFlow Lite 器（TensorFlow Lite converter）](index.md) Python API 的示例。

[TOC]

## Python API

在 TensorFlow 2.0 中，用来将原始的 TensorFlow 模型格式 TensorFlow Lite 的 Python API 是 `tf.lite.TFLiteConverter`。在 `TFLiteConverter` 中有以下的方法（classmethod）：

*   `TFLiteConverter.from_saved_model()`：用来
    [SavedModel 格式模型](https://www.tensorflow.org/guide/saved_model)。
*   `TFLiteConverter.from_keras_model()`：用来
    [`tf.keras` 模型](https://www.tensorflow.org/guide/keras/overview)。
*   `TFLiteConverter.from_concrete_functions()`：用来
    [concrete functions](concrete_function.md)。

注意: 在 TensorFlow Lite 2.0 中有一个不同版本的
`TFLiteConverter` API，  API 只包含了
[`from_concrete_function`](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/lite/TFLiteConverter#from_concrete_function)。
本文中用到的的新版本 API 可以通 pip 安装
[`tf-nightly-2.0-preview`](#2.0-nightly)。

本文展示了 API 的 [示例用法](#examples)，不同 TensorFlow 版本的 API 列表看 [1.X 版本到 2.0 版本 API 的改](#differences)，和
[安装 TensorFlow](#versioning) 来安装和使用。

## 示例 <a name="examples"></a>

###  SavedModel 格式模型 <a name="saved_model"></a>

以下示例展示了如何将一个
[SavedModel](https://www.tensorflow.org/guide/saved_model) 
TensorFlow Lite 中的 [`FlatBuffer`](https://google.github.io/flatbuffers/)格式。

```python
import tensorflow as tf

# 建立一个的模型。
root = tf.train.Checkpoint()
root.v1 = tf.Variable(3.)
root.v2 = tf.Variable(2.)
root.f = tf.function(lambda x: root.v1 * root.v2 * x)

# 保存模型。
export_dir = "/tmp/test_saved_model"
input_data = tf.constant(1., shape=[1, 1])
to_save = root.f.get_concrete_function(input_data)
tf.saved_model.save(root, export_dir, to_save)

# 模型。
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
tflite_model = converter.convert()
```

此 API 不支持指定入向量的度。 如果的模型需要指定入向量的度，使用
[`from_concrete_functions`](#concrete_function) 来完成。 示例：

```python
model = tf.saved_model.load(export_dir)
concrete_func = model.signatures[
  tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
concrete_func.inputs[0].set_shape([1, 256, 256, 3])
converter = TFLiteConverter.from_concrete_functions([concrete_func])
```

###  Keras 模型 <a name="keras"></a>

以下示例展示了如何将一个
[tf.keras 模型](https://www.tensorflow.org/guide/keras/overview) 
TensorFlow Lite 中的 [`FlatBuffer`](https://google.github.io/flatbuffers/) 格式。

```python
import tensorflow as tf

# 建一个的 Keras 模型。
x = [-1, 0, 1, 2, 3, 4]
y = [-3, -1, 1, 3, 5, 7]

model = tf.keras.models.Sequential(
    [tf.keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
model.fit(x, y, epochs=50)

# 模型。
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
```

###  concrete function <a name="concrete_function"></a>

以下示例展示了如何将 TensorFlow 中的
[concrete function](concrete_function.md) TensorFlow Lite 中的
[`FlatBuffer`](https://google.github.io/flatbuffers/) 格式。

```python
import tensorflow as tf

# 建立一个模型。
root = tf.train.Checkpoint()
root.v1 = tf.Variable(3.)
root.v2 = tf.Variable(2.)
root.f = tf.function(lambda x: root.v1 * root.v2 * x)

# 生成 concrete function。
input_data = tf.constant(1., shape=[1, 1])
concrete_func = root.f.get_concrete_function(input_data)

# 模型。
#
# `from_concrete_function` 的入参数被一个一个 concrete function 的列表，然而
# 段支持次用接受一个concrete function。
# 同多个concrete function的功能正在中。
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()
```

### 端到端 MobileNet  <a name="mobilenet"></a>

以下示例展示了如何将将一个提前好的 
`tf.keras` MobileNet 模型 TensorFlow Lite 支持的型并行推断 （inference）。 随机数据分在
TensorFlow 和 TensorFlow Lite 模型中行的果将被比。如果是从文件加模型，使用 `model_path` 来代替 `model_content`。

```python
import numpy as np
import tensorflow as tf

# 加 MobileNet tf.keras 模型。
model = tf.keras.applications.MobileNetV2(
    weights="imagenet", input_shape=(224, 224, 3))

# 模型。
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 加 TFLite 模型并分配量（tensor）。
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# 取入和出量。
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# 使用随机数据作入 TensorFlow Lite 模型。
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# 函数 `get_tensor()` 会返回一量的拷。
# 使用 `tensor()` 取指向量的指。
tflite_results = interpreter.get_tensor(output_details[0]['index'])

# 使用随机数据作入 TensorFlow 模型。
tf_results = model(tf.constant(input_data))

# 比果。
for tf_result, tflite_result in zip(tf_results, tflite_results):
  np.testing.assert_almost_equal(tf_result, tflite_result, decimal=5)
```

##  1.X 版本到 2.0 版本 API 的改 <a name="differences"></a>

本了从 1.X 到 2.0 版本 Python API 的改。
如果某些改有，提交
[GitHub issue](https://github.com/tensorflow/tensorflow/issues)。

### `TFLite器` 支持的格式型

`TFLite器` 在 2.0 版本中支持由 1.X 版本和 2.0 版本生成的 SavedModels 和 Keras 模型。但是，程不再支持由
1.X 版本的 `GraphDefs`。 者可通用 `tf.compat.v1.lite.TFLiteConverter` 来把的
`GraphDefs` 到 TensorFlow Lite 版本。

### 量化感知（Quantization-aware training）

以下与
[量化感知（Quantization-aware training）](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/quantize)
有的属性和方法在 TensorFlow 2.0 中从`TFLiteConverter` 中被移除。

*   `inference_type`
*   `inference_input_type`
*   `quantized_input_stats`
*   `default_ranges_stats`
*   `reorder_across_fake_quant`
*   `change_concat_input_ranges`
*   `post_training_quantize` - 在 1.X API 中被弃用
*   `get_input_arrays()`

支持量化感知的重写器（rewriter）函数不支持由 TensorFlow 2.0 生成的模型。此外，TensorFlow Lite 的量化 API
已按支持 Keras 中量化感知 API 的思路重新和精。 在新的量化 API 部署前，些属性将不会出在 2.0 的 API 中。者可以使用
`tf.compat.v1.lite.TFLiteConverter` 来由重写器函数生成的模型。

### 于 `TFLiteConverter` 中属性的改

属性 `target_ops` 已成 `TargetSpec` 中的属性且作未来化框架的充被重命名 `supported_ops`。

此外，以下属性被移除:

*   `drop_control_dependency` (default: `True`) - TFLite 不支持控制流（control flow），所以此属性将恒 `True`。
*   _Graph visualization_ - 在 TensorFlow 2.0 中，推荐使用
    [visualize.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/visualize.py)  TensorFlow Lite （graph）的可化。
    不同于 GraphViz, 它支持者已行 post training 量化的（graph）可化。以下与可化的属性将被移除：
    *   `output_format`
    *   `dump_graphviz_dir`
    *   `dump_graphviz_video`

### 通用 API 的改

#### 方法

以下在 1.X 中被弃用的方法不会在 2.0 中出：

*   `lite.toco_convert`
*   `lite.TocoConverter`

#### `lite.constants`

在 2.0 中，了少 TensorFlow 和 TensorFlow Lite 之的重移除了 `lite.constants` API。以下的列表展示了
`lite.constant` 中的型在 TensorFlow 中的型：

*   `lite.constants.FLOAT`: `tf.float32`
*   `lite.constants.INT8`: `tf.int8`
*   `lite.constants.INT32`: `tf.int32`
*   `lite.constants.INT64`: `tf.int64`
*   `lite.constants.STRING`: `tf.string`
*   `lite.constants.QUANTIZED_UINT8`: `tf.uint8`

此外，`lite.constants.TFLITE` 和 `lite.constants.GRAPHVIZ_DOT` 被移除（由于 `TFLiteConverter` 中的 flage `output_format`被移除）。

#### `lite.OpHint`

由于 API `OpHint` 与 2.0 的 API 不兼容，故不可用。 此 API可用于基于 LSTM 的模型。 在 2.0 中
LSTMs 的支持正在被探究。所有与 `lite.experimental` 有的 API 都因此被移除。

## 安装 TensorFlow <a name="versioning"></a>

### 安装 TensorFlow 2.0 nightly <a name="2.0-nightly"></a>

可用以下命令安装 TensorFlow 2.0 nightly：

```
pip install tf-nightly-2.0-preview
```

### 在已安装的 1.X 中使用 TensorFlow 2.0 <a name="use-2.0-from-1.X"></a>

可通以下代片段从最近安装的 1.X 中使用 TensorFlow 2.0。

```python
import tensorflow.compat.v2 as tf

tf.enable_v2_behavior()
```

### 从源代安装 <a name="latest_package"></a>

使用最新版本的 TensorFlow Lite 器 Python API，
可通以下方式安装 nightly build：
[pip](https://www.tensorflow.org/install/pip) (推荐方式) 或
[Docker](https://www.tensorflow.org/install/docker), 或
[从源代建 pip 包](https://www.tensorflow.org/install/source).
