# 始使用 TensorFlow Lite

TensorFlow Lite 提供了 TensorFlow 模型，并在移端（mobile）、嵌入式（embeded）和物网（IoT）上行 TensorFlow 模型所需的所有工具。以下指南介了人工作流程的个，并提供了一明的接。

## 1. 一个模型

<a id="1_choose_a_model"></a>

TensorFlow Lite 允在多上行 TensorFlow 模型。TensorFlow 模型是一数据，数据包含了在解决一个特定，得到的机器学网的和知。

有多方式可以得 TensorFlow 模型，从使用模型（pre-trained models）到自己的模型。了在 TensorFlow Lite 中使用模型，模型必成一特殊格式。将在第二[模型](#2_convert_the_model_format)中解。

Note: 不是所有的 TensorFlow 模型都能在 TensorFlow Lite 中行，因解器（interpreter）只支持部分（a limited subset）TensorFlow 算符（operations）。参考第二[模型](#2_convert_the_model_format)来了解兼容性。

### 使用模型

TensorFlow Lite 提供了一系列模型（pre-trained models），用于解决各机器学。些模型已能与 TensorFlow Lite 一起使用，且可以在的用程序中使用的模型。

些模型包括：

*	[像分（Image classification）](../models/image_classification/overview.md)
*	[物体（Object detection）](../models/object_detection/overview.md)
*	[智能回（Smart reply）](../models/smart_reply/overview.md)
*	[姿估（Pose estimation）](../models/pose_estimation/overview.md)
*	[分割（Segmentation）](../models/segmentation/overview.md)

在[模型列表（Models）](../models)中看模型的完整列表。

#### 来自其他来源的模型

可以在多其他地方得到的 TensorFlow 模型，包括 [TensorFlow Hub](https://www.tensorflow.org/hub)。在大多数情况下，些模型不会以 TensorFlow Lite 格式提供，必在使用前[（convert）](#2_convert_the_model_format)些模型。

### 重新模型（移学）

移学（transfer learning）允采用好的模型并重新（re-train），以行其他任。例如，一个[像分](../models/image_classification/overview.md)模型可以重新以新的像。与从始模型相比，重新花的更少，所需的数据更少。

可以使用移学，根据的用程序定制模型。在<a href="https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android">用 TensorFlow 花</a>的 codelab 中，可以学如何行移学。

### 自定模型

如果并了自己的 TensorFlow 模型，或者了从其他来源得到的模型，在使用前，需要将此模型成 TensorFlow Lite 的格式。

## 2. 模型

<a id="2_convert_the_model_format"></a>

TensorFlow Lite 的旨在在各上高效行模型。高效部分源于在存模型，采用了一特殊的格式。TensorFlow 模型在能被 TensorFlow Lite 使用前，必成格式。

模型小了模型文件大小，并引入了不影准性（accuracy）的化措施（optimizations）。人可以在行一些取舍的情况下，一小模型文件大小，并提高行速度。可以使用 TensorFlow Lite 器（converter）要行的化措施。

因 TensorFlow Lite 支持部分 TensorFlow 算符（operations），所以并非所有模型都能。参看[ Ops 兼容性](#Ops兼容性)得更多信息。

### TensorFlow Lite 器

[TensorFlow Lite 器（converter）](../convert)是一个将好的 TensorFlow 模型成 TensorFlow Lite 格式的工具。它能引入化措施（optimizations），将在第四[化的模型](#4_optimize_your_model_optional)中介。

器以 Python API 的形式提供。下面的例子明了将一个 TensorFlow `SavedModel` 成 TensorFlow Lite 格式的程：

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
```

可以用似的方法[ TensorFlow 2.0 模型](../convert)

然也能从[命令行](../convert/cmdline_examples)使用器，但是推荐用 Python API 行。

### 

器可以从各入型模型。

当[ TensorFlow 1.x 模型](../convert/python_api.md)，些入型有：

*	[SavedModel 文件](https://www.tensorflow.org/guide/saved_model)
*	Frozen GraphDef (通[ freeze_graph.py ](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py)生成的模型)
*	[Keras](https://keras.io) HDF5 模型
*	从 `tf.Session` 得到的模型

当[ TensorFlow 2.x 模型](../convert/python_api.md)，些入型有：

*	[SavedModel 文件](https://www.tensorflow.org/guide/saved_model)
*	[`tf.keras` 模型](https://www.tensorflow.org/guide/keras/overview)
*	[具体函数（Concrete functions）](../convert/concrete_function.md)

器可以配置用各化措施（optimizations），些化措施可以提高性能，少文件大小。将在第四[化的模型](#4_optimize_your_model_optional)中介。

### Ops 兼容性

TensorFlow Lite 当前支持[一部分（limited subset）](ops_compatibility.md) TensorFlow 算符（operations）。期目是将来能支持全部的 TensorFlow 算符。

如果期望的模型中含有不受支持的算符，可以使用[ TensorFlow Select](ops_select.md) 包含来自 TensorFlow 的算符。会使得部署到上的二制文件更大。


## 3. 使用模型行推理

<a id="3_use_the_tensorflow_lite_model_for_inference_in_a_mobile_app"></a>

*推理（Inference）* 是通模型（model）行数据（data）以得（predictions）的程。个程需要模型（model）、解器（interpreter）和入数据（input data）。

### TensorFlow Lite 解器

[TensorFlow Lite 解器（interpreter）](inference.md)是一个（library），它接收一个模型文件（model file），行模型文件在入数据（input data）上定的算符（operations），并提供出（output）的。

解器（interpreter）用于多个平台，提供了一个的 API，用于从 Java、Swift、Objective-C、C++ 和 Python 行 TensorFlow Lite 模型。

下面的代示了从 Java 用解器的方式:

```java
try (Interpreter interpreter = new Interpreter(tensorflow_lite_model_file)) {
  interpreter.run(input, output);
}
```

### GPU 加速和委托

一些机器学算符提供硬件加速（hardware acceleration）。例如，大多数手机有 GPU，些 GPU 可以比 CPU 行更快的浮点矩算（floating point matrix operations）。

速度提升（speed-up）能有著（substantial）效果。例如，当使用 GPU 加速，MobileNet v1 像分模型在 Pixel 3 手机上的行速度提高了 5.5 倍。

TensorFlow Lite 解器可以配置[委托（Delegates）](../performance/delegates.md)以在不同上使用硬件加速。[GPU 委托（GPU Delegates）](../performance/gpu.md)允解器在的 GPU 上行当的算符。

下面的代示了从 Java 中使用 GPU 委托的方式:

```java
GpuDelegate delegate = new GpuDelegate();
Interpreter.Options options = (new Interpreter.Options()).addDelegate(delegate);
Interpreter interpreter = new Interpreter(tensorflow_lite_model_file, options);
try {
  interpreter.run(input, output);
}
```

要添加新硬件加速器的支持，可以[定自己的委托](../performance/delegates.md#how_to_add_a_delegate)。

### Android 和 iOS

TensorFlow Lite 解器很容易在个主要移平台上使用。要入，[ Android 快速入](android.md)和[ iOS 快速入](iOS.md)指南。个平台，都有[示例用程序](https://www.tensorflow.org/lite/examples)。

要得所需的（libraries），Android 人使用[ TensorFlow Lite AAR](android.md#use_the_tensorflow_lite_aar_from_jcenter)。iOS 人使用[ CocoaPods for Swift or Objective-C](ios.md#add_tensorflow_lite_to_your_swift_or_objective-c_project)。

### Linux

嵌入式 Linux 是一个部署机器学的重要平台。我[ Raspberry Pi ](build_rpi.md)和[基于 Arm64 的主板](build_arm64.md)，如 Odroid C2、Pine64 和 NanoPi，提供了建明。

### 微控制器

[TensorFlow Lite 微控制器（Microcontrollers）版](../microcontrollers/overview.md)是一个 TensorFlow Lite 的端口，端口只有几千字（kilobytes）内存（memory）的微控制器和其他。

### 算符

如果的模型需要 TensorFlow Lite 中尚未的 TensorFlow 算符（operations），可以使用[ TensorFlow Select ](ops_select.md)在模型中使用它。需要建一个包含 TensorFlow 算符的自定版本解器。

可以用[自定算符（Custom operators）](ops_custom.md)写自己的算符（operations），或将新算符移植（port）到 TensorFlow Lite 中。

[算符版本（Operator versions）](ops_version.md)能已有的算符添加新的功能和参数。

## 4. 化的模型

<a id="4_optimize_your_model_optional"></a>

TensorFlow Lite 提供了化模型大小（size）和性能（performance）的工具，通常准性（accuracy）影甚微。化模型可能需要稍微的（training），（conversion）或集成（integration）。

机器学化是一个不断展的域，TensorFlow Lite 的[模型化工具包（Model Optimization Toolkit）](#模型化工具包)随着新技的展而不断展。

### 性能

模型化的目是在定上，性能（performance）、模型大小（model size）和准性（accuracy）的理想平衡。
[性能最佳践（Performance best practices）](../performance/best_practices.md)可以助指完成个程。

### 量化

通降低模型中数（values）和算符（operations）的精度（precision），量化（quantization）可以小模型的大小和推理所需的。很多模型，只有小的准性（accuracy）失。

TensorFlow Lite 器量化 TensorFlow 模型得。下面的 Python 代量化了一个 `SavedModel` 并将其保存在硬中：

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_quantized_model)
```

要了解有量化的更多信息，参[后量化（Post-training quantization）](../performance/post_training_quantization.md)。

### 模型化工具包

[模型化工具包（Model Optimization Toolkit）](../performance/model_optimization.md)是一套工具和技，旨在使人可以松化它的模型。然其中的多技可以用于所有 TensorFlow 模型，并非特定于 TensorFlow Lite，但在源有限的上行推理（inference），它特有价。

## 下一

既然已熟悉了 TensorFlow Lite，探索以下一些源：

*	如果是移人，[ Android 快速入](android.md)或[ iOS 快速入](ios.md)。
*	探索我的[模型](../models)。
*	我的[示例用程序](https://www.tensorflow.org/lite/examples)。
