# 性能的最佳践

由于移和嵌入式的算能力有限，所以保持用的源被高效利用是非常重要的。我已写了一最佳践和策略的清，能用它来化的 TensorFlow Lite 模型和用。

## 任最佳的模型

根据任的不同，会需要在模型度和大小之做取舍。如果的任需要高准率，那可能需要一个大而的模型。于精度不高的任，就最好使用小一点的模型，因小的模型不占用更少的磁和内存，也一般更快更高效。比如，下展示了常的像分模型中准率和延模型大小的影。

![模型大小和准度的系](../images/performance/model_size_vs_accuracy.png "模型大小和准度")

![准度和延的系](../images/performance/accuracy_vs_latency.png "准度和延")

一个移化的示例模型就是 [MobileNets](https://arxiv.org/abs/1704.04861)，模型是了移端用而化的。我的 [模型列表](../models/hosted.md) 列出了外几移和嵌入式化的模型。

可以用自己的数据通移学再些模型。看我的移学教程：[像分](https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0) 和 [物体](https://medium.com/tensorflow/training-and-serving-a-realtime-mobile-object-detector-in-30-minutes-with-cloud-tpus-b78971cf1193)。

## 的模型

在了一个合的任的模型之后，模型和立基准很好的行。TensorFlow Lite [工具](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark) 有内置的器，可展示一个算符的数据。能助理解性能瓶和些算符主了算。

## 和化（graph）中的算符

如果某个特定的算符繁出在模型中，并且基于个算符消耗了大部分，那可以研究如何化个算符。情况非常少，因 TensorFlow Lite 中的大部分算符都是化的版本。然而，如果了解算符的行限制，或可以写一个自定的更快的版本。看我的 [自定算符文档](../custom_operators.md)。

## 化的模型

模型旨在建更小的模型，并且通常更快、更高效能。因此它能被部署到移上。

### 量化

如果的模型使用浮点重或者激励函数，那模型大小或可以通量化少75%，方法有效地将浮点重从32字化8字。量化分：[后量化](post_training_quantization.md) 和 [量化](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/quantize/README.md){:.external}。前者不需要再模型，但是在少情况下会有精度失。当精度失超了可接受范，使用量化。

我烈推荐立基准以保模型期准率没有被影。看信息：[模型化文档](model_optimization.md)。

## 整程数

TensorFlow Lite 支持多算符使用多程内核。可以增加程数以提高算符行速度。然而，增加程数会使的模型使用更多的源和能源。

有些用来，延或比能源效率更重要。可以通定 [解器](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/interpreter.h#L346) 的数量来增加程数。然而，根据同行的其他操作不同，多程行会增加性能的可性。比如，隔可能示多程的速度是程的倍，但如果同有一个用在行的，性能果可能比程更差。

## 清除冗余副本

如果的用没有被很好地，在入模型和取模型出可能会有冗余副本。保清除冗余副本。如果在使用高 API，如 Java，保仔性能注意事。比如，如果使用 ByteBuffers 作[入](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/java/src/main/java/org/tensorflow/lite/Interpreter.java#L175)，Java API 会快得多。

## 用平台特定工具的用

平台特定工具，如 [Android profiler](https://developer.android.com/studio/profile/android-profiler) 和 [Instruments](https://help.apple.com/instruments/mac/current/)，提供了富的可被用于用的信息。有性能可能不出自于模型，而是与模型交互的用代。保熟悉平台特定工具和平台最好的方法。

## 估的模型是否受益于使用上可用的硬件加速器

TensorFlow Lite 增加了新的方法来配合更快的硬件加速模型，比如 GPU、DSP 和神加速器。一般来，些加速器通 [代理](delegates.md) 子模暴露，些子模接管部分解器行。TensorFlow Lite 能通以下方法使用代理：

*   使用 Android 的 [神网 API](https://developer.android.com/ndk/guides/neuralnetworks/)。可以利用些硬件加速器后台来提升模型速度和效率。要用神网 API，在解器例内用 [UseNNAPI](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/interpreter.h#L343)。
*   我已布了一个限二制的 GPU 代理，Android 和 iOS 分使用 OpenGL 和 Metal。要用它，看 [GPU 代理教程](gpu.md) 和 [文档](gpu_advanced.md)。
*   如果能非准硬件，也可以建自己的代理。更多信息，看 [TensorFlow Lite 代理](delegates.md)。

注意，有的加速器在某些模型效果更好。个代理立基准以出最的是很重要的。比如，如果有一个非常小的模型，那可能没必要将模型委托 NN API 或 GPU。相反，于具有高算度的大模型来，加速器就是一个很好的。

## 需要更多助？

TensorFlow 非常意助断和定位具体的性能。在 [GitHub](https://github.com/tensorflow/tensorflow/issues) 提出并描述。
