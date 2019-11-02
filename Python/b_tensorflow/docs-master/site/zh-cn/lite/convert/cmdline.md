# 模型器命令参考

本介如何使用 TensorFlow 2.0 命令行工具中的[TensorFlow Lite 模型器](index.md)。首的方法是使用 [Python API](python_api.md)。

## 述

TensorFlow Lite 器命令行工具 `tflite_convert` 它支持基模型。使用 `TFLiteConverter` [Python API](python_api.md) 支持任何及量化或其他参数的(例如：SavedModels 名，或者在 Keras 模型上自定象).

## 使用

下列命令参数用于入和出文件。

*   `--output_file`. 型: string. 指定出文件的路径。
*   --saved_model_dir. 型: string. 指定含有 TensorFlow 1.x 或者 2.0 使用 SavedModel 生成文件的路径目。
*   --keras_model_file. Type: string. 指定含有 TensorFlow 1.x 或者 2.0 使用 tf.keras model 生成 HDF5 文件的路径目。

例如：

```
tflite_convert \
  --saved_model_dir=/tmp/mobilenet_saved_model \
  --output_file=/tmp/mobilenet.tflite
```

## 附加明

### 从源代建

想要行最新版本的 TensorFlow Lite 模型器可以通 [pip](https://www.tensorflow.org/install/pip) 安装 TensorFlow 2.0 版或者[克隆 TensorFlow 代](https://www.tensorflow.org/install/source)然后使用 `bazel` 从源代 TensorFlow 。下面是一个从源代 TensorFlow 的例子。

```
bazel run //third_party/tensorflow/lite/python:tflite_convert -- \
  --saved_model_dir=/tmp/mobilenet_saved_model \
  --output_file=/tmp/mobilenet.tflite
```
