# Android 快速上手

要在Android上使用TensorFlow Lite，我推荐探索下面的例子。

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android">Android
像分示例</a>

有源代的明，
[TensorFlow Lite Android 像分](https://www.tensorflow.org/lite/models/image_classification/android).

示例用程序使用
[像分](https://www.tensorflow.org/lite/models/image_classification/overview)
来地的后置像所看到的内容行分。
用程序可以行在真或者模器上。

使用 TensorFlow Lite Java API 来行推理。演示用程序地像行分，示最可能的分果。它允用浮点或
[量化](https://www.tensorflow.org/lite/performance/post_training_quantization)
模型，程数，并决定行在CPU，GPU上，亦或是通
[NNAPI](https://developer.android.com/ndk/guides/neuralnetworks)行。

注意: 些[示例](https://www.tensorflow.org/lite/examples)提供了其他的在多用例中演示使用TensorFlow Lite的用程序。 

## 在Android Studio中建

如果要在Android Studio 建例子，遵循
[README.md](https://github.com/tensorflow/examples/blob/master/lite/examples/image_classification/android/README.md)中的明。

## 建自己的Android用程序

如果想快速写的Android代, 我推荐使用
[Android 像分代例子](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android)
作起始点。

下面的部分包含了一些有如何在Android上使用TensorFlow Lite的有用信息。

### 使用JCenter中的TensorFlow Lite AAR

如果要在的Android用程序中使用TensorFlow Lite，我推荐使用
[在JCenter中托管的TensorFlow Lite AAR](https://bintray.com/google/tensorflow/tensorflow-lite)。

可以像下面在的`build.gradle`依中指定它:

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly'
}
```

个AAR包含了
[Android ABIs](https://developer.android.com/ndk/guides/abis)中的所有的二制文件。可以通只包含需要支持的ABIs来少用程序的二制文件大小。

我推荐大部分的者 `x86`，`x86_64`，和`arm32` 的ABIs。可以通如下的Gradle配置，个配置只包括了 `armeabi-v7a`和`arm64-v8a`，配置能涵盖住大部分的代Android。

```build
android {
    defaultConfig {
        ndk {
            abiFilters 'armeabi-v7a', 'arm64-v8a'
        }
    }
}
```

想要了解更多有 `abiFilters`的信息, 看Android Gradle文档中的
[`NdkOptions`](https://google.github.io/android-gradle-dsl/current/com.android.build.gradle.internal.dsl.NdkOptions.html)。

### 在本地建TensorFlow Lite

在某些情况下，可能希望使用一个本地建的TensorFlow Lite. 比如，可能正在建一个自定的包含了
[从TensorFlow中的操作](https://www.tensorflow.org/lite/guide/ops_select)的二制文件。

在情况下，参照
[自定 AAR 建明](https://www.tensorflow.org/lite/guide/ops_select#android_aar)
来建自己的AAR并将其包含在的APP中.
