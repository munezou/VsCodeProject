# 从 TensorFlow 中算符

注意：功能是性的。

TensorFlow Lite 已内置了很多算符，并且在不断展，但是仍然有一部分 TensorFlow 算符没有被 TensorFlow Lite 原生支持。些不被支持的算符会 TensorFlow Lite 的模型来一些阻力。了少模型的阻力，TensorFlow Lite 最近一直致力于一个性功能的。

篇文档介了怎在 TensorFlow Lite 使用 TensorFlow 算符。*注意，只是一个性的功能，并且在中。* 在使用功能的候，住些[已知的局限性](#已知的局限性)，并且将使用中遇到的反至 tflite@tensorflow.org。

TensorFlow Lite 会移和嵌入式化[内置的算符](ops_compatibility.md)。但是在，当 TensorFlow Lite 内置的算符不的候，TensorFlow Lite 模型可以使用部分 TensorFlow 的算符。

TensorFlow Lite 解器在理后的包含 TensorFlow 算符的模型的候，会比理只包含 TensorFlow Lite 内置算符的模型占用更多的空。并且，TensorFlow Lite 模型中包含的任何 TensorFlow 算符，性能都不会被化。

篇文档介了怎不同的平台[](#模型)和[行](#行模型)包含 TensorFlow 算符的 TensorFlow Lite 模型。并且了一些[已知的局限性](#已知的局限性)、此功能制定的[未来的](#未来的)以及基本的[性能和空指](#性能和空指)。

## 模型

了能包含 TensorFlow 算符的 TensorFlow Lite 模型，可使用位于 [TensorFlow Lite 器](../convert/) 中的 `target_spec.supported_ops` 参数。`target_spec.supported_ops` 的可如下：

*   `TFLITE_BUILTINS` - 使用 TensorFlow Lite 内置算符模型。
*   `SELECT_TF_OPS` - 使用 TensorFlow 算符模型。已支持的 TensorFlow 算符的完整列表可以在白名
    `lite/delegates/flex/whitelisted_flex_ops.cc` 中看。

注意：`target_spec.supported_ops` 是之前 Python API 中的 `target_ops`。

我先推荐使用 `TFLITE_BUILTINS` 模型，然后是同使用 `TFLITE_BUILTINS,SELECT_TF_OPS` ，最后是只使用 `SELECT_TF_OPS`。同使用个（也就是 `TFLITE_BUILTINS,SELECT_TF_OPS`）会用 TensorFlow Lite 内置的算符去支持的算符。有些 TensorFlow 算符 TensorFlow Lite 只支持部分用法，可以使用 `SELECT_TF_OPS` 来避免局限性。

下面的示例展示了通 Python API 中的 [`TFLiteConverter`](./convert/python_api.md) 来使用功能。

```
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
```

下面的示例展示了在命令行工具 [`tflite_convert`](../convert/cmdline_examples.md) 中通 `target_ops` 来使用功能。

```
tflite_convert \
  --output_file=/tmp/foo.tflite \
  --graph_def_file=/tmp/foo.pb \
  --input_arrays=input \
  --output_arrays=MobilenetV1/Predictions/Reshape_1 \
  --target_ops=TFLITE_BUILTINS,SELECT_TF_OPS
```

如果直接使用 `bazel` 和行 `tflite_convert`，入参数 `--define=with_select_tf_ops=true`。

```
bazel run --define=with_select_tf_ops=true tflite_convert -- \
  --output_file=/tmp/foo.tflite \
  --graph_def_file=/tmp/foo.pb \
  --input_arrays=input \
  --output_arrays=MobilenetV1/Predictions/Reshape_1 \
  --target_ops=TFLITE_BUILTINS,SELECT_TF_OPS
```

## 行模型

如果 TensorFlow Lite 模型在的候支持 TensorFlow select 算符，那在使用的候 Tensorflow Lite 行必包含 TensorFlow 算符的。

### Android AAR

了便于使用，新增了一个支持 TensorFlow select 算符的Android AAR。如果已有了<a href="android.md">可用的 TensorFlow Lite
境</a>，可以按照下面的方式支持使用 TensorFlow select 算符的 Android AAR：

```sh
bazel build --cxxopt='--std=c++11' -c opt             \
  --config=android_arm --config=monolithic          \
  //tensorflow/lite/java:tensorflow-lite-with-select-tf-ops
```

上面的命令会在 `bazel-genfiles/tensorflow/lite/java/` 目下生成一个 AAR 文件。可以直接将个 AAR 文件入到目中，也可以将其布到本地的 Maven ：

```sh
mvn install:install-file \
  -Dfile=bazel-genfiles/tensorflow/lite/java/tensorflow-lite-with-select-tf-ops.aar \
  -DgroupId=org.tensorflow \
  -DartifactId=tensorflow-lite-with-select-tf-ops -Dversion=0.1.100 -Dpackaging=aar
```

最后，在用的 `build.gradle` 文件中需要保有 `mavenLocal()` 依，并且需要用支持 TensorFlow select 算符的 TensorFlow Lite 依去替准的 TensorFlow Lite 依：

```
allprojects {
    repositories {
        jcenter()
        mavenLocal()
    }
}

dependencies {
    implementation 'org.tensorflow:tensorflow-lite-with-select-tf-ops:0.1.100'
}
```

### iOS

如果安装了 XCode 命令行工具，可以用下面的命令支持 TensorFlow select 算符的 TensorFlow Lite：

```sh
tensorflow/contrib/makefile/build_all_ios_with_tflite.sh
```

条命令会在 `tensorflow/contrib/makefile/gen/lib/` 目下生成所需要的静接。

TensorFlow Lite 的相机示例用可以用来行。一个新的支持 TensorFlow select 算符的 TensorFlow Lite XCode 目已添加在 `tensorflow/lite/examples/ios/camera/tflite_camera_example_with_select_tf_ops.xcodeproj` 中。

如果想要在自己的目中使用个功能，可以克隆示例目，也可以按照下面的方式目行置：

*   在 Build Phases -> Link Binary With Libraries 中，添加 `tensorflow/contrib/makefile/gen/lib/` 目中的静：
    *   `libtensorflow-lite.a`
    *   `libprotobuf.a`
    *   `nsync.a`
*   在 Build Settings -> Header Search Paths 中，添加下面的路径：
    *   `tensorflow/lite/`
    *   `tensorflow/contrib/makefile/downloads/flatbuffer/include`
    *   `tensorflow/contrib/makefile/downloads/eigen`
*   在 Build Settings -> Other Linker Flags 中，添加 `-force_load
    tensorflow/contrib/makefile/gen/lib/libtensorflow-lite.a`。
    
未来会布支持 TensorFlow select 算符的 CocoaPod 。

### C++

如果使用 bazel  TensorFlow Lite ，可以按照下面的方式添加和支持外的 TensorFlow 算符的。

*   如果需要体，可以添加 `--config=monolithic` 。
*   从下面的方案中一个：
    *   在用 `bazel build` 命令 TensorFlow Lite 添加 `--define=with_select_tf_ops=true` 。
    *   在依中添加 TensorFlow 算符依 `tensorflow/lite/delegates/flex:delegate`。

注意，只要委托接到了客端，在行建解器的候就会自安装所需的 `TfLiteDelegate`，而不需要像其他委托型去式安装委托例。

### Python pip Package

 Python 的支持在当中。

## 性能和空指

### 性能

如果 TensorFlow Lite 模型是同混合使用内置算符和 TensorFlow select 算符行的，那模型依然可以使用 TensorFlow Lite 的化以及内置的化内核。

下表列出了在 Pixel 2 上 MobileNet 的平均推断。表中的是 100 次行的平均。在 Android 平台的候添加了 `--config=android_arm64 -c opt` 。

                               | 推断 (milliseconds)
------------------------------------ | -------------------
Only built-in ops (`TFLITE_BUILTIN`) | 260.7
Using only TF ops (`SELECT_TF_OPS`)  | 264.5

### 二制文件大小

下表列出了不同方式生成的 TensorFlow Lite 二制文件的大小。在 Android 平台的候添加了 `--config=android_arm -c opt` 。

                 | C++ 二制文件大小 | Android APK 大小
--------------------- | --------------- | ----------------
Only built-in ops     | 796 KB          | 561 KB
Built-in ops + TF ops | 23.0 MB         | 8.0 MB

## 已知的局限性

下面列出了一些已知的局限性：

*   目前不支持控制流算符。
*   目前不支持 TensorFlow 算符的 [`post_training_quantization`](https://www.tensorflow.org/performance/post_training_quantization) ，所以不会任何 TensorFlow 算符行重量化。如果模型中既包含 TensorFlow Lite 算符又包含 TensorFlow 算符，那 TensorFlow Lite 内置的算符的重是可以被量化的。
*   目前不支持像 HashTableV2 需要式用源行初始化的算符。
*   某些 TensorFlow 操作可能不支持 TensorFlow 中整套常可用入/出操作。

## 未来的

下面列出了正在中的功能的一些改：

*   *性注册* - 有一正在完成的工作是，生成只包含特定模型集合所需的 Tensorflow 算符的 TensorFlow Lite 二制文件得更。
*   *提升可用性* - 模型的程将被化，只需要一次性完成。 并且会提供的 Android AAR 和 iOS CocoaPod 二制文件。
*   *提升性能* - 有一正在完成的工作是，使用 TensorFlow 算符的 TensorFlow Lite 具有与 TensorFlow Mobile 同等的性能。
