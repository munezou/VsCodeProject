# TensorFlow Lite converter
TensorFlow Lite converter是用于将TensorFlow模型化化的[FlatBuffer](https://google.github.io/flatbuffers/)格式，以便TensorFlow Lite解器用。

注意：此面包含TensorFlow 1.x的converter API文档，[TensorFlow 2.0的API点此接](https://www.tensorflow.org/lite/convert/)

## FlatBuffers
FlatBuffers是一个高效的源跨平台序列化。它似于[protocol buffers](https://developers.google.com/protocol-buffers)，区在于FlatBuffers在数据之前不需要其次要表行解析/解，从而避免个象行内存分配。FlatBuffers的代占用空比protocol buffers小一个数量。

## 从模型培到部署
TensorFlow Lite converter可以从TensorFlow模型中生成TensorFlow Lite [FlatBuffers](https://google.github.io/flatbuffers/)文件(.tflite)。

converter支持以下入格式：
- [SavedModels](https://www.tensorflow.org/guide/saved_model#using_savedmodel_with_estimators)
- 量固定常数(Frozen)的`GraphDef`:由[freeze_graph.py](https://www.tensorflow.org/code/tensorflow/python/tools/freeze_graph.py)生成的模型
- `tf.keras` HDF5模型
- 任何从 `tf.Session`取的模型（限Python API）

然后，将TensorFlow Lite FlatBuffer文件部署到客端，TensorFlow Lite 解器会使用模型在上行推断(inference)。会程如下所示：
![TFLite converter workflow](https://github.com/tensorflow/tensorflow/raw/master/tensorflow/lite/g3doc/images/convert/workflow.svg?sanitize=true)

## 

TensorFlow Lite Converter 可以通以下方式使用：
- [Python](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/convert/python_api.md)（**首方式**）：使用Python API可以更松地将模型模型流(model development pipeline)的一部分，并有助于在早期程中解[兼容性](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/tf_ops_compatibility.md)
- [命令行](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/convert/cmdline_examples.md)
