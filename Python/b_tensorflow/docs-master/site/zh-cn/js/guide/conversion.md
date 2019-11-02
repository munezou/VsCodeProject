# 模型

TensorFlow.js 配了各模型，些模型可以在器中使用，[模型](https://github.com/tensorflow/tfjs-models) 中有相介。但是，可能已在其他地方找到或建了一个 TensorFlow 模型，并希望在 web 用程序中使用模型。此，TensorFlow.js 提供了一个 [模型器](https://github.com/tensorflow/tfjs-converter) 。TensorFlow.js 器有个件:

1. 一个命令行程序，用于 Keras 和 TensorFlow 模型以在 TensorFlow.js 中使用。
2. 一个 API ，用于在器中使用 TensorFlow.js 加和行模型。

## 的模型

TensorFlow.js 器可以以下几格式的模型:

**SavedModel**: 保存 TensorFlow 模型的默格式。SavedModel 的格式 [此](https://www.tensorflow.org/guide/saved_model)。

**Keras model**: Keras 模型通常保存 HDF5 文件。有保存 Keras 模型的更多信息， [此](https://keras.io/getting-started/faq/#savingloading-whole-models-architecture-weights-optimizer-state)。

**TensorFlow Hub module**: 些是打包后在 TensorFlow Hub 中行分的模型，TensorFlow Hub 是一个共享和模型的平台。模型 [此](tfhub.dev)。

取决于的模型的格式，需要将不同的参数器。比如，假保存了一个名 `model.h5` 的 Keras 模型到 `tmp/` 目。了使用 TensorFlow.js 器模型，可以行以下命令: 

    $ tensorflowjs_converter --input_format=keras /tmp/model.h5 /tmp/tfjs_model

会将路径 `/tmp/model.h5` 的模型并出 `model.json` 文件及其二制重文件到目 `tmp/tfjs_model/` 中。

有不同格式的模型相的命令行参数的更多信息，参 TensorFlow.js 器 [自述文件](https://github.com/tensorflow/tfjs-converter)。

在程中，我会遍模型形并 TensorFlow.js 是否支持个操作。如果是支持的，我将形成器可以使用的格式。我通将重分成 4MB 的文件来化模型以便在 web 上使用 - 它就可以被器存。我也使用源工程 [Grappler](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/core/grappler) 化模型形。形的化包括相操作的折，消除常子像等。些更改模型的出没有影。了一化，用可以入参数以指示器将模型量化到特定的字大小。量化是一少模型大小的技，它是通用更少的比特表示重的。用必保量化后模型的准度保持在可接受范内。
如果在程中遇到了不支持的操作，程失，我将用打印出操作的名称。将此提交到我的 [GitHub](https://github.com/tensorflow/tfjs/issues)  - 我会根据用的需求更多新的操作。

### 最佳做法

然在程中我尽力化的模型，但通常保的模型利行的最佳方式是在考源受限的境下建。意味着避免于的建和尽可能少参数（重）的数目。

## 行的模型

成功模型之后，将得到一重文件和一个模型拓文件。TensorFlow.js 提供模型加 APIs ，可以使用些接口取模型并且在器中行推断。

以下是加后的 TensorFlow SavedModel 或 TensorFlow Hub 模的 API :

```js
const model = await tf.loadGraphModel(‘path/to/model.json’);
```

以下是后的 Keras 模型的 API :

```js
const model = await tf.loadLayersModel(‘path/to/model.json’);
```

`tf.loadGraphModel` API 返回 `tf.FrozenModel`，意味着各参数是固定的并且不能使用新数据模型行微。`tf.loadLayersModel` API 返回可的 tf.Model。有 tf.Model 的相信息，参[模型指南](train_models.md)。

在之后，我建行几次推断并且模型的速度行基准。基于个目的，我有一个独立的基准面: https://github.com/tensorflow/tfjs-core/blob/master/integration_tests/benchmarks/benchmark.html。 可能注意到我弃了初始行中的量 - 是因（通常情况）下，由于建理和着色器的源消耗，的模型的第一次的推断将比后推断慢几倍。
