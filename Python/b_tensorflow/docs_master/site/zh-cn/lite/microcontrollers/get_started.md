# 微控制器入

本文档将助始使用用于微控制器的 Tensorflow Lite。

首先并行我的[示例](#示例)

注意：如果需要一个入，我建使用 
[由 Tensorflow 提供技支持的 SparkFun Edge](https://www.sparkfun.com/products/15170)。
它是与 Tensorflow Lite 合作的，在微控制器上行深度学提供了活的平台。

有行推断所需代的介，参下文的*行推断*部分

## 示例

下面几个示例演示了如何使用 Tensorflow Lite 建嵌入式机器学用程序：

### Hello World 示例

本示例旨在演示将 Tensorflow Lite 用于微控制器的基知。它包括了模型、将模型以供 Tensorflow Lite 使用以及在微控制器上行推断的完整端到端工作流程。

在个示例中，一个模型被用来模正弦函数。部署到微控制器上，其可用来 LED 或者控制画。

<a class="button button-primary" href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro/examples/hello_world">Hello
World 示例</a>

示例代包含一个演示如何和模型的 Jupyter notebook：

<a class="button button-primary" href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro/examples/hello_world/create_sine_model.ipynb">create_sine_model.ipynb</a>

指南[“建与模型”](build_convert.md)中也介了建和模型的流程。

要了解推断是如何行的，看 [hello_world_test.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/examples/hello_world/hello_world_test.cc)。

示例在以下平台上行了：

-   [由 Tensorflow 提供技支持的 SparkFun Edge(Apollo3 Blue)](https://www.sparkfun.com/products/15170)
-   [Arduino MKRZERO](https://store.arduino.cc/usa/arduino-mkrzero)
-   [STM32F746G 探索板（Discovery Board）](https://www.st.com/en/evaluation-tools/32f746gdiscovery.html)
-   Mac OS X

### 微音示例

此示例使用一个的
[音模型](https://www.tensorflow.org/tutorials/sequences/audio_recognition)
来音中的字。示例代从的麦克中捕音。模型通音行分来定是否“是”或“否一。

<a class="button button-primary" href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro/examples/micro_speech">微音示例</a>

[“行推断”](#行推断) 部分将微音示例的代并解其工作原理。

示例在以下平台上行了：

-   [由 Tensorflow 提供技支持的 SparkFun Edge(Apollo3 Blue)](https://www.sparkfun.com/products/15170)
-   [STM32F746G 探索板（Discovery Board）](https://www.st.com/en/evaluation-tools/32f746gdiscovery.html)
-   Mac OS X

注意：若要始使用 SparkFun Edge 板，我建遵循[“在使用 SparkFun Tensorflow 的微控制器上行机器学”](https://codelabs.developers.google.com/codelabs/sparkfun-tensorflow)中所描述的流程,是一个向介工作流程的代室（codelab）。

### 微示例

本示例展示了如何使用 Tensorflow Lite 行一个 25 万字的神网来由像机拍的像中的人。示例被成可以在具有少量内存的系上行，如微控制器和 DSP。

<a class="button button-primary" href="https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro/examples/micro_vision">微示例</a>

示例在以下平台上行了：

-   [由 Tensorflow 提供技支持的 SparkFun Edge(Apollo3 Blue)](https://www.sparkfun.com/products/15170)
-   [STM32F746G 探索板（Discovery Board）](https://www.st.com/en/evaluation-tools/32f746gdiscovery.html)
-   Mac OS X

## 行推断

以下部分将介[微音](#微音示例)示例中的 [main.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/examples/micro_speech/main.cc) 文件并解了它如何使用用于微控制器的 Tensorflow Lite 来行推断。

### 包含

要使用，必包含以下文件：

```C++
#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
```

-   [`all_ops_resolver.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h)
    提供解器（interpreter）用于行模型的操作。
-   [`micro_error_reporter.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/micro_error_reporter.h)
    出信息。
-   [`micro_interpreter.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/micro_interpreter.h)
    包含理和行模型的代。
-   [`schema_generated.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/schema_generated.h)
    包含 TensorFlow Lite
    [`FlatBuffer`](https://google.github.io/flatbuffers/) 模型文件格式的模式。
-   [`version.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/version.h)
    提供 Tensorflow Lite 架的版本信息。

示例包括其他一些文件。以下些是最重要的：

```C++
#include "tensorflow/lite/experimental/micro/examples/micro_speech/feature_provider.h"
#include "tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/micro_model_settings.h"
#include "tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/tiny_conv_micro_features_model_data.h"
```

-   [`feature_provider.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/feature_provider.h)
    包含从音流中提取要入到模型中的特征的代。
-   [`tiny_conv_micro_features_model_data.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/tiny_conv_micro_features_model_data.h)
    包含存 `char` 数的模型。
    [“建与模型”](build_convert.md) 来了解如何将 Tensorflow Lite 模型格式。
-   [`micro_model_settings.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/micro_model_settings.h)
    定与模型相的各常量。

### 置日志

要置日志，需要使用一个指向 `tflite::MicroErrorReporter` 例的指来建一个 `tflite::ErrorReporter` 指：

```C++
tflite::MicroErrorReporter micro_error_reporter;
tflite::ErrorReporter* error_reporter = &micro_error_reporter;
```
量被到解器（interpreter）中，解器允它写日志。由于微控制器通常具有多日志机制，`tflite::MicroErrorReporter` 的是的特定所定制的。

### 加模型

在以下代中，模型是从一个 `char` 数中例化的，`g_tiny_conv_micro_features_model_data` （要了解其是如何建的，参[“建与模型”](build_convert.md)）。 随后我模型来保其架版本与我使用的版本所兼容：

```C++
const tflite::Model* model =
    ::tflite::GetModel(g_tiny_conv_micro_features_model_data);
if (model->version() != TFLITE_SCHEMA_VERSION) {
  error_reporter->Report(
      "Model provided is schema version %d not equal "
      "to supported version %d.\n",
      model->version(), TFLITE_SCHEMA_VERSION);
  return 1;
}
```
### 例化操作解析器

解器（interpreter）需要一个 [`AllOpsResolver`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h) 例来 Tensorflow 操作。可以展此以向的目添加自定操作：

```C++
tflite::ops::micro::AllOpsResolver resolver;
```

### 分配内存

我需要先入、出以及中数分配一定的内存。分配的内存是一个大小 `tensor_arena_size` 的 `uint8_t` 数，它被 `tflite::SimpleTensorAllocator` 例：

```C++
const int tensor_arena_size = 10 * 1024;
uint8_t tensor_arena[tensor_arena_size];
tflite::SimpleTensorAllocator tensor_allocator(tensor_arena,
                                               tensor_arena_size);
```

注意：所需内存大小取决于使用的模型，可能需要通来定。

### 例化解器（Interpreter）

我建一个 `tflite::MicroInterpreter` 例，之前建的量：

```C++
tflite::MicroInterpreter interpreter(model, resolver, &tensor_allocator,
                                     error_reporter);
```

### 入度

`MicroInterpreter` 例可以通用 `.input(0)` 我提供一个指向模型入量的指，其中 `0` 代表第一个（也是唯一一个）入量。我个量以它的度与型是我所期望的：

```C++
TfLiteTensor* model_input = interpreter.input(0);
if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
    (model_input->dims->data[1] != kFeatureSliceCount) ||
    (model_input->dims->data[2] != kFeatureSliceSize) ||
    (model_input->type != kTfLiteUInt8)) {
  error_reporter->Report("Bad input tensor parameters in model");
  return 1;
}
```

在个代段中，量 `kFeatureSliceCount` 和 `kFeatureSliceSize` 与入的属性相，它定在 [`micro_model_settings.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/micro_model_settings.h) 中。枚 `kTfLiteUInt8` 是 Tensorflow Lite 某一数据型的引用，它定在 [`c_api_internal.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/c_api_internal.h) 中。

### 生成特征

我入到模型中的数据必由微控制器的音入生成。[`feature_provider.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/examples/micro_speech/micro_features/feature_provider.h) 中定的 `FeatureProvider` 捕音并将其一将被入模型的特征集合。当被例化，我用之前取的 `TfLiteTensor` 来入一个指向入数的指。`FeatureProvider` 使用它来填充将模型的入数据：

```C++
  FeatureProvider feature_provider(kFeatureElementCount,
                                   model_input->data.uint8);
```

以下代使 `FeatureProvider` 从最近一秒的音生成一特征并填充入量：

```C++
TfLiteStatus feature_status = feature_provider.PopulateFeatureData(
    error_reporter, previous_time, current_time, &how_many_new_slices);
```

在此例子中，特征生成和推断是在一个循中生的，因此能不断地捕捉和理新的音。

当在写自己的程序，可能会以其它的方式生成特征，但需要在行模型之前就用数据填充入量。

### 行模型

要行模型，我可以在 `tflite::MicroInterpreter` 例上用 `Invoke()`：

```C++
TfLiteStatus invoke_status = interpreter.Invoke();
if (invoke_status != kTfLiteOk) {
  error_reporter->Report("Invoke failed");
  return 1;
}
```

我可以返回 `TfLiteStatus` 以定行是否成功。在 [`c_api_internal.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/c_api_internal.h) 中定的 `TfLiteStatus` 的可能有 `kTfLiteOk` 和 `kTfLiteError`。

### 取出

模型的出量可以通在 `tflite::MicroIntepreter` 上用 `output(0)` 得，其中 `0` 代表第一个（也是唯一一个）出量。

在示例中，出是一个数，表示入属于不同（“是”（yes）、“否”（no）、“未知”（unknown）以及“静默”（silence））的概率。由于它是按照集合序排列的，我可以使用的来定概率最高的：

```C++
    TfLiteTensor* output = interpreter.output(0);
    uint8_t top_category_score = 0;
    int top_category_index;
    for (int category_index = 0; category_index < kCategoryCount;
         ++category_index) {
      const uint8_t category_score = output->data.uint8[category_index];
      if (category_score > top_category_score) {
        top_category_score = category_score;
        top_category_index = category_index;
      }
    }
```

在示例其他部分中，使用了一个更加的算法来平滑多的果。部分在 [recognize_commands.h](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/examples/micro_speech/recognize_commands.h) 中有所定。在理任何的数据流，也可以使用相同的技来提高可靠性。

## 下一

建并行示例后，以下文档：

*   在[“建与模型”](build_convert.md)中了解如何使用模型。
*   在[“了解C++”](library.md)中了解更多于 C++ 的内容。
