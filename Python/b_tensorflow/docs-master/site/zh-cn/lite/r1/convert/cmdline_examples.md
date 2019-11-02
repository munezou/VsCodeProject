# 器的命令行例

个面展示如何在命令行中使用 TensorFlow Lite 器

[TOC]

## 在命令行中使用的命令 <a name="tools"></a>

以下是在命令行中使用器的方法：

*   `tflite_convert`: 从 TensorFlow 1.9 起始支持
    `tflite_convert` 作 Python 包的一部分被安装。便起，以下所有示例使用 `tflite_convert` 指代。
    *   示例: `tflite_convert --output_file=...`
*   `bazel`: 了使用最新版本的 TensorFlow Lite Converter，可以使用
    [pip](https://www.tensorflow.org/install/pip) 或[克隆 TensorFlow ](https://www.tensorflow.org/install/source) 来安装并使用 nightly 版本的的 `bazel`。
    *   示例: `bazel run //tensorflow/lite/python:tflite_convert ----output_file=...`

### 在低于 1.9 版本的 TensorFlow 中模型  <a name="pre_tensorflow_1.9"></a>

如果安装有低于 1.9 版本的 Tensorflow，并想模型，我推荐使用
[Python API](python_api.md#pre_tensorflow_1.9)。 如果想要使用命令行模型, 在 Tensorflow 1.7 中，可使用 toco。

可以通在端中入`toco help`来取更多于命令行参数的信息。

在 TensorFlow 1.8 中没有可用的命令行工具。

## 基示例 <a name="basic"></a>

以下部分向展示怎将各数据从支持的型到 TensorFlow Lite FlatBuffers。

###  TensorFlow GraphDef <a name="graphdef"></a>

以下部分向展示如何将基本的 TensorFlow GraphDef (使用 [freeze_graph.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py) ) TensorFlow Lite FlatBuffer 来行浮点数推理。被的包含存在点文件中的量，些量被作 Const ops 保存。

```
curl https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_0.50_128_frozen.tgz \
  | tar xzv -C /tmp
tflite_convert \
  --output_file=/tmp/foo.tflite \
  --graph_def_file=/tmp/mobilenet_v1_0.50_128/frozen_graph.pb \
  --input_arrays=input \
  --output_arrays=MobilenetV1/Predictions/Reshape_1
```

`input_shapes` 的取在可能被自定。

###  TensorFlow SavedModel <a name="savedmodel"></a>

以下部分向展示如何将基本的 TensorFlow SavedModel  Tensorflow Lite FlatBuffer 来行浮点数推理。

```
tflite_convert \
  --output_file=/tmp/foo.tflite \
  --saved_model_dir=/tmp/saved_model
```

[SavedModel](https://www.tensorflow.org/guide/saved_model#using_savedmodel_with_estimators)
与后的比，它需要更少的参数，是由于保存在 SavedModel 中的附加数据所致。
 `--input_arrays`和 `--output_arrays` 所需要的是 [MetaGraphDef](https://www.tensorflow.org/saved_model#apis_to_build_and_load_a_savedmodel) 中 [SignatureDefs](../../serving/signature_defs.md) 里的一个聚合起来的，按照字母序排列的入出列表，它由`saved_model_tag_set`指定。
 和 GraphDef 一, `input_shapes` 的也在可能被自定。

段不提供不 SignatureDef 的 MetaGraphDefs 或是
使用[`assets/`directory](https://www.tensorflow.org/guide/saved_model#structure_of_a_savedmodel_directory) 的 MetaGraphDefs 的支持。

###  tf.Keras 模型 <a name="keras"></a>

以下部分展示如何将一个 `tf.keras` 模型一个 TensorFlow Lite Flatbuffer。 

 `tf.keras` 文件必同包含模型和重。

```
tflite_convert \
  --output_file=/tmp/foo.tflite \
  --keras_model_file=/tmp/keras_model.h5
```

## 量化

### 将一个TensorFlow GraphDef 量化的推理 <a name="graphdef_quant"></a>

TensorFlow Lite Converter 兼容定点量化模型，情[里](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/quantize/README.md)。
浮点模型中有 `FakeQuant*` ops ，它被插入在混合的界来最大最小的范信息。

生一个量化的推理工作流，它了期被使用的量化行。

下列命令从"量化的" TensorFlow GraphDef 中生量化的 TensorFlow Lite FlatBuffer。


```
tflite_convert \
  --output_file=/tmp/foo.tflite \
  --graph_def_file=/tmp/some_quantized_graph.pb \
  --inference_type=QUANTIZED_UINT8 \
  --input_arrays=input \
  --output_arrays=MobilenetV1/Predictions/Reshape_1 \
  --mean_values=128 \
  --std_dev_values=127
```

### 使用 "dummy-quantization\" 在浮点数上行量化推理 <a name="dummy_quant"></a>

了估生成量化的可能的好，器允在浮点上行 "dummy-quantization"。参数
`--default_ranges_min` 和 `--default_ranges_max` 在所有不含有最大最小信息的 array 中指定最大最小范。"Dummy-quantization" 的精度低一些，但也近似于一个精量化模型。

下方的例子展示了一个有 Relu6 激活函数的模型。由此，我可以得出一个合理的猜，大部分的激活函数的范在[0, 6]。

```
curl https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_0.50_128_frozen.tgz \
  | tar xzv -C /tmp
tflite_convert \
  --output_file=/tmp/foo.cc \
  --graph_def_file=/tmp/mobilenet_v1_0.50_128/frozen_graph.pb \
  --inference_type=QUANTIZED_UINT8 \
  --input_arrays=input \
  --output_arrays=MobilenetV1/Predictions/Reshape_1 \
  --default_ranges_min=0 \
  --default_ranges_max=6 \
  --mean_values=128 \
  --std_dev_values=127
```

## 定入和出的数

### 多入数

如下方的例子所示，参数 `input_arrays` 接受一个用逗号分隔的列表作入数。

于有多入的模型或子来是很有用的。

```
curl https://storage.googleapis.com/download.tensorflow.org/models/inception_v1_2016_08_28_frozen.pb.tar.gz \
  | tar xzv -C /tmp
tflite_convert \
  --graph_def_file=/tmp/inception_v1_2016_08_28_frozen.pb \
  --output_file=/tmp/foo.tflite \
  --input_shapes=1,28,28,96:1,28,28,16:1,28,28,192:1,28,28,64 \
  --input_arrays=InceptionV1/InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1/Relu,InceptionV1/InceptionV1/Mixed_3b/Branch_2/Conv2d_0a_1x1/Relu,InceptionV1/InceptionV1/Mixed_3b/Branch_3/MaxPool_0a_3x3/MaxPool,InceptionV1/InceptionV1/Mixed_3b/Branch_0/Conv2d_0a_1x1/Relu \
  --output_arrays=InceptionV1/Logits/Predictions/Reshape_1
```

需要注意的是， `input_shapes` 是用冒号分割的列表。其中， 个入形状于各自数中相同位置的入数。

### 多出数

如下方的例子所示，参数 `output_arrays` 接收一个用逗号分隔的列表作出数。

于有多出的模型或子来是很有用的。

```
curl https://storage.googleapis.com/download.tensorflow.org/models/inception_v1_2016_08_28_frozen.pb.tar.gz \
  | tar xzv -C /tmp
tflite_convert \
  --graph_def_file=/tmp/inception_v1_2016_08_28_frozen.pb \
  --output_file=/tmp/foo.tflite \
  --input_arrays=input \
  --output_arrays=InceptionV1/InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1/Relu,InceptionV1/InceptionV1/Mixed_3b/Branch_2/Conv2d_0a_1x1/Relu
```

### 指定子

入文件中的任何数都可以被指定入或出数，以便从入的文件中提取子。TensorFlow Lite
Converter 忽略指定子范之外的的其他部分。 可使用 [graph visualizations](#graph_visualizations) 来成所需子的入和出数。

下列命令展示怎从一个 TensorFlow GraphDef 中提取个混合。

```
curl https://storage.googleapis.com/download.tensorflow.org/models/inception_v1_2016_08_28_frozen.pb.tar.gz \
  | tar xzv -C /tmp
tflite_convert \
  --graph_def_file=/tmp/inception_v1_2016_08_28_frozen.pb \
  --output_file=/tmp/foo.pb \
  --input_shapes=1,28,28,96:1,28,28,16:1,28,28,192:1,28,28,64 \
  --input_arrays=InceptionV1/InceptionV1/Mixed_3b/Branch_1/Conv2d_0a_1x1/Relu,InceptionV1/InceptionV1/Mixed_3b/Branch_2/Conv2d_0a_1x1/Relu,InceptionV1/InceptionV1/Mixed_3b/Branch_3/MaxPool_0a_3x3/MaxPool,InceptionV1/InceptionV1/Mixed_3b/Branch_0/Conv2d_0a_1x1/Relu \
  --output_arrays=InceptionV1/InceptionV1/Mixed_3b/concat_v2
```

注意，TensorFlow Lite FlatBuffers 中的最表示的粒度往往比 TensorFlow GraphDef 非常的表示粒度更粗。例如，然在 TensorFlow GraphDef 中，一个全接通常被表示至少四个独的 op (形，矩乘法，偏置目加，Relu…)，但在器的最表示和最上的表示中，它通常被表示个“混合的op”。

由于粒度粗，一些中的数 (例如 TensorFlow GraphDef 中矩乘和偏置加之的数)将被弃。

当使用`--input_arrays` 和 `--output_arrays`指定中数，推荐（有是必）指定在混合后生成的最形式的中保留的数。它通常是激活函数的出（因在一中，所有在激活函数前出的部分都向于被混合）。

## 日志


## 可化

器可将出 Graphviz Dot 格式，可使用`--output_format` 参数或是
`--dump_graphviz_dir`参数松地行可化。下面的小概述了多个用例。

### 使用 `--output_format=GRAPHVIZ_DOT` <a name="using_output_format_graphviz_dot"></a>

第一染 Graphviz 的方式是将 GRAPHVIZ_DOT` 参数入
`、`output_format`。将生成可化。此操作降低了在 TensorFlow GraphDef 和 TensorFlow Lite FlatBuffer 的要求。当到 TFLite 的失，此操作是很有用的。

```
curl https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_0.50_128_frozen.tgz \
  | tar xzv -C /tmp
tflite_convert \
  --graph_def_file=/tmp/mobilenet_v1_0.50_128/frozen_graph.pb \
  --output_file=/tmp/foo.dot \
  --output_format=GRAPHVIZ_DOT \
  --input_shape=1,128,128,3 \
  --input_arrays=input \
  --output_arrays=MobilenetV1/Predictions/Reshape_1
```

生成的`.dot` 文件可以使用以下命令染pdf文件：

```
dot -Tpdf -O /tmp/foo.dot
```

生成的 `.dot.pdf` 文件可以在任何 PDF 器上看，但我建使用一个能在大面上放自如的看工具，例如 Google Chrome ：

```
google-chrome /tmp/foo.dot.pdf
```

可在下一中在看示例的 PDF。

### 使用 `--dump_graphviz_dir`

第二染 Graphviz 的法是入 `dump_graphviz_dir`参数，并指定保存染果文件的目目。

和前一个方法不同的是，此方法保留了原始出格式。它提供了由特定生成的可化的程。

```
curl https://storage.googleapis.com/download.tensorflow.org/models/mobilenet_v1_0.50_128_frozen.tgz \
  | tar xzv -C /tmp
tflite_convert \
  --graph_def_file=/tmp/mobilenet_v1_0.50_128/frozen_graph.pb \
  --output_file=/tmp/foo.tflite \
  --input_arrays=input \
  --output_arrays=MobilenetV1/Predictions/Reshape_1 \
  --dump_graphviz_dir=/tmp
```

此操作将在目文件中生成一些文件。 其中，个最重要的文件是 `toco_AT_IMPORT.dot` 和`/tmp/toco_AFTER_TRANSFORMATIONS.dot`。
`toco_AT_IMPORT.dot` 文件只包含的原始，此操作在入就被完成。由于个点的信息有限，

致生成的可化果不好理解。此操作在命令失十分有用。

`toco_AFTER_TRANSFORMATIONS.dot` 含有模型在被出之前，且在行了所有的之后的信息。

通常，个文件比小，且包含个点更多的信息。

和之前一，些文件可以被染PDF文件：

```
dot -Tpdf -O /tmp/toco_*.dot
```

示例出文件如下所示。需要注意的是，它展示的都是片右上角的同一个
`AveragePool` 点。

<table><tr>
  <td>
    <a target="_blank" href="https://storage.googleapis.com/download.tensorflow.org/example_images/toco_AT_IMPORT.dot.pdf">
      <img src="../images/convert/sample_before.png"/>
    </a>
  </td>
  <td>
    <a target="_blank" href="https://storage.googleapis.com/download.tensorflow.org/example_images/toco_AFTER_TRANSFORMATIONS.dot.pdf">
      <img src="../images/convert/sample_after.png"/>
    </a>
  </td>
</tr>
<tr><td>before</td><td>after</td></tr>
</table>

### 像“拍”一日志

当使用 `--dump_graphviz_dir` 命令，通常会再入一个
`dump_graphviz_video`命令。个命令使得次后，都会保存一个可化“快照”。可能致需要存非常多的可化文件。
通常，人通看些文件来了解的化程。

### 形可化的例 <a name="graphviz_legend"></a>

*   “操作”色方:
    *   大部分的操作看起来像是
        <span style="background-color:#db4437;color:white;border:1px;border-style:solid;border-color:black;padding:1px">bright
        red</span>。
    *   一些重量操作 (比如卷)看起来像是
        <span style="background-color:#c53929;color:white;border:1px;border-style:solid;border-color:black;padding:1px">darker
        red</span>。
*   数看起来像是:
    *   常量数
        <span style="background-color:#4285f4;color:white;border:1px;border-style:solid;border-color:black;padding:1px">blue</span>。
    *   激活数
        *   内部 (中) 激活数
            <span style="background-color:#f5f5f5;border:1px;border-style:solid;border-color:black;border:1px;border-style:solid;border-color:black;padding:1px">light
            gray</span>。
        *   被指定 `--input_arrays` 或`--output_arrays` 的激活数
            <span style="background-color:#9e9e9e;border:1px;border-style:solid;border-color:black;padding:1px">dark
            gray</span>。
    *   RNN 的状数是色的。 由于器式地表示RNN的回，个RNN 状被表示个色数:
        *   作RNN回入的激活数 (例如，当它的内容在被算后制到RNN的状数），此它看起来像是
            <span style="background-color:#b7e1cd;border:1px;border-style:solid;border-color:black;padding:1px">light
            green</span>。
        *   的 RNN 状数看起来像
            <span style="background-color:#0f9d58;color:white;border:1px;border-style:solid;border-color:black;padding:1px">dark
            green</span>。它是RNN回更新的目。