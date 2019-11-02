# 建立与模型

微控制器具有有限的 RAM 和存空，限制了机器学模型的模。此外，面向微控制器的 TensorFlow Lite 目前只支持有限的一部分算，因此并非所有的模型都是可行的。

本文档解了一个 TensorFlow 模型以使其可在微控制器上行的程。本文档也概述了可支持的算，并于与一个模型以使其符合内存限制出了一些指。

一个端到端的、可行的建立与模型的示例，于如下的 Jupyter notebook 中：
<a class="button button-primary" href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro/examples/hello_world/create_sine_model.ipynb">create_sine_model.ipynb</a>


## 模型

了一个已好的 TensorFlow 模型以使其可在微控制器上行，使用 [TensorFlow Lite 器 Python API](https://tensorflow.google.cn/lite/convert/python_api) 。它能将模型成 [`FlatBuffer`](https://google.github.io/flatbuffers/) 格式，小模型模，并修改模型以使用 TensorFlow Lite 支持的算。

### 量化
了得尽可能小的模型模，考使用[后量化](https://tensorflow.google.cn/lite/performance/post_training_quantization)。它会降低模型中数字的精度，从而小模型模。不，操作可能会致模型准性的下降，于小模模型来尤如此。在量化前后分析模型的准性以保失在可接受范内是非常重要的。

以下的 Python 代片段展示了如何使用量化行模型：

```python
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_quant_model)
```

### 一个 C 数
多微控制器平台没有本地文件系的支持。从程序中使用一个模型最的方式是将其以一个 C 数的形式包含并的程序。

以下的 unix 命令会生成一个以 `char` 数形式包含 TensorFlow Lite 模型的 C 源文件：

```bash
xxd -i converted_model.tflite > model_data.cc
```

其出似如下：

```c
unsigned char converted_model_tflite[] = {
  0x18, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, 0x00, 0x00, 0x0e, 0x00,
  // <Lines omitted>
};
unsigned int converted_model_tflite_len = 18200;
```

一旦已生成了此文件，可以将它包含入的程序。在嵌入式平台上，将数声明改 `const` 型以得更好的内存效率是重要的。

一个如何在的程序中包含及使用模型的例子，微型音示例中的 [`tiny_conv_micro_features_model_data.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/tiny_conv_micro_features_model_data.h) 。

## 模型与

在一个面向微控制器的模型，考模型的模、工作，以及用到的算是非常重要的。

### 模型模

一个模型必在二制和行方面都足小，以使其可以和程序的其他部分一起符合目的内存限制。

了建一个更小的模型，可以在的里使用更少和更小的。然而，小模的模型更易面欠合。意味着于多，并使用符合内存限制的尽可能大模的模型是有意的。但是，使用更大模的模型也会致理器工作的增加。

注：在一个 Cortex M3 上，面向微控制器的 TensorFlow Lite 的核心行占 16 KB。

### 工作

工作受到模型模与度的影。大模、的模型可能会致更高的占空比，即致所用理器的工作增、空短。的用，情况所来的力消耗与量出的增加可能会成一个。

### 算支持
面向微控制器的 TensorFlow Lite 目前支持有限的部分 TensorFlow 算，影了可以行的模型。我正致力于在参考和特定的化方面展算支持。

已支持的算可以在文件 [`all_ops_resolver.cc`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/kernels/all_ops_resolver.cc) 中看到。
