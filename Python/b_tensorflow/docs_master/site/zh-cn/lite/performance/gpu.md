# TensorFlow Lite GPU 代理

[TensorFlow Lite](https://www.tensorflow.org/lite) 支持多硬件加速器。本文档描述了如何在 Android 和 iOS 上使用 TensorFlow Lite 的代理 APIs 来性的 GPU 后端功能。

GPU 是用来完成高吐量的大模并行工作的。因此，它非常合用在包含大量算符的神网上，一些入量可以容易的被分更小的工作且可以同行，通常会致更低的延。在最佳情况下，用 GPU 在用程序上做推理算已可以行的足快，而在以前是不可能的。

不同于 CPU 的是，GPU 可以算 16 位浮点数或者 32 位浮点数并且 GPU 不需要量化来得最佳的系性能。

使用 GPU 做推理算有一个好就是它的能源效率。GPU 可以以非常高效和化的方式下行算，所以 GPU 在完成和 CPU 一的任可以消耗更少的力和生更少的量。

## 演示用程序教程

最的 GPU 代理的方法就是跟着下面的教程，教程将串我整个使用 GPU 建的分演示用程序。GPU 代在只有二制的形式，但是很快就会源。一旦理解了如何把我的演示程序行起来，就可以在自己的模型上。

### Android（使用 Android Studio）

如果需要一个分教程, 看
[用于 Android 的性 GPU 代理](https://youtu.be/Xkhgre8r5G0) 的。

注意：需要 OpenGL ES 3.1或者更高版本

#### 第一 克隆 TensorFlow 的源代并在 Android Studio 中打

```
git clone https://github.com/tensorflow/tensorflow
```

#### 第二  `app/build.gradle` 文件来使用 nightly 版本的 GPU AAR

在有的 `dependencies` 模已有的 `tensorflow-lite` 包的位置下添加 `tensorflow-lite-gpu` 包。

```
dependencies {
    ...
    implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly'
    implementation 'org.tensorflow:tensorflow-lite-gpu:0.0.0-nightly'
}
```

#### 第三. 和行

点 Run 按来行用程序。当行用程序的候会看到一个用 GPU 的按。将用程序从量化模式改浮点模式后点 GPU 按后，程序将在 GPU 上行。

![行 Android gpu 演示用程序和切到 GPU](images/android_gpu_demo.gif)

### iOS (使用 XCode)

如果需要一个分教程, 看
[用于 iOS 的性 GPU 代理](https://youtu.be/Xkhgre8r5G0) 的。

注意：需要 XCode 10.1 或者更高版本

#### 第一. 取演示用程序的源并保它已被

遵照我的 iOS 演示用程序[教程](https://www.tensorflow.org/lite/demo_ios)。会告没有修改的iOS相机用程序是如何在我的手机上行的。

#### 第二部. 修改 Podfile 文件来使用 TensorFlow Lite GPU CocoaPod

我建了一个包含 GPU 代理的二制 CocoaPod 文件。如果需要切到工程并使用它，修改
`tensorflow/tensorflow/lite/examples/ios/camera/Podfile` 文件来使用  `TensorFlowLiteGpuExperimental` 的 pod 替代 `TensorFlowLite`。

```
target 'YourProjectName'
  # pod 'TensorFlowLite', '1.12.0'
  pod 'TensorFlowLiteGpuExperimental'
```

#### 第三. 用 GPU 代理

了保代会使用 GPU 代理，需要将 `CameraExampleViewController.h` 的
`TFLITE_USE_GPU_DELEGATE` 从 0 修改 1 。

```c
#define TFLITE_USE_GPU_DELEGATE 1
```

#### 第四. 和行演示用程序

如果完成了上面的，已可以行个用程序了。

#### 第五. 布模式

在第四是在模式下行的用程序，了得更好的性能表，使用当的最佳 Metal 置将用程序改布版本。特需要注意的是，需要修改些置 `Product > Scheme > Edit
Scheme...`， ` Run `，在 ` Info ` 一，修改 ` Build Configuration `，从 `Debug ` 改 ` Release `，取消 ` Debug executable`。

![置布](images/iosdebug.png)

然后点 `Options` 然后将 `GPU Frame Capture` 修改成 `Disabled`，并将 `Metal API Validation` 修改成 `Disabled`。

![置 metal ](images/iosmetal.png)

最后需要保布版本只能在 64 位系上建。在 `Project
navigator -> tflite_camera_example -> PROJECT -> tflite_camera_example -> Build
Settings` 上将 `Build Active Architecture Only > Release` Yes。

![置布](images/iosrelease.png)

## 在自己的模型上使用GPU代理

### Android

看演示用程序来了解如何添加代理。在的用程序中，像上面一添加 AAR ，入`org.tensorflow.lite.gpu.GpuDelegate` 模，并使用 `addDelegate` 功能将GPU代理注册到解器中。

```java
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

// 初始化使用 GPU 代理的解器
GpuDelegate delegate = new GpuDelegate();
Interpreter.Options options = (new Interpreter.Options()).addDelegate(delegate);
Interpreter interpreter = new Interpreter(model, options);

// 行推理
while (true) {
  writeToInput(input);
  interpreter.run(input, output);
  readFromOutput(output);
}

// 清理
delegate.close();
```

### iOS
 
在的用程序代中，引入 GPU 代理文件来`Interpreter::ModifyGraphWithDelegate` 功能将 GPU 代理注册到解器中。

```cpp
#import "tensorflow/lite/delegates/gpu/metal_delegate.h"

// 初始化使用 GPU 代理的解器
std::unique_ptr<Interpreter> interpreter;
InterpreterBuilder(*model, resolver)(&interpreter);
auto* delegate = NewGpuDelegate(nullptr);  // default config
if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;

// 行推理 
while (true) {
  WriteToInputTensor(interpreter->typed_input_tensor<float>(0));
  if (interpreter->Invoke() != kTfLiteOk) return false;
  ReadFromOutputTensor(interpreter->typed_output_tensor<float>(0));
}

// 清理
interpreter = nullptr;
DeleteGpuDelegate(delegate);
```

## 支持的模型和 Ops

在 GPU 代理布后，我提供了少数可以在后端行的模型：

* [MobileNet v1 (224x224)像分](https://ai.googleblog.com/2017/06/mobilenets-open-source-models-for.html) [[下]](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/mobilenet_v1_1.0_224.tflite)
<br /><i>(移和嵌入式用的像分模型)</i>
* [DeepLab 分割 (257x257)](https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html) [[下]](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite)
<br /><i>(将入像的个像素指定（例如，狗，猫。汽的像分割模型)</i>
* [MobileNet SSD 物体](https://ai.googleblog.com/2018/07/accelerated-training-and-inference-with.html) [[下]](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/mobile_ssd_v2_float_coco.tflite)
<br /><i>(用于多个有框的象的像分模型)</i>
* [PoseNet用于姿估](https://github.com/tensorflow/tfjs-models/tree/master/posenet) [[下]](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/multi_person_mobilenet_v1_075_float.tflite)
<br /><i>(用于估像或中人物的姿的模型)</i>

如果需要完整的支持的 Ops 的列表，看[文档](gpu_advanced.md)。

## 不支持的模型和 ops

如果一些 ops 并不支持 GPU 代理，框架只会在 GPU 上行形的一部分，剩下的部分会在 CPU 上行。因会致 CPU/GPU 同出很高的使用率，像的分行模式会致行起来比整个网在 CPU 上行要慢。在情况下，用会收到一个像的警告：

```
WARNING: op code #42 cannot be handled by this delegate.
```

```
警告：此代理无法理#42操作
```

我没有失提供回，因不是真的行，但是个是者可以注意到的，他可以将整个网在代理上行。

## 化建

一些在 CPU 上的碎的的操作可能在 GPU 上会有很高的占用。其中的一操作就是很多形式的 reshape 操作，像 `BATCH_TO_SPACE`, `SPACE_TO_BATCH`, `SPACE_TO_DEPTH` 等等。如果些 ops 只是了方便网架的思考而放置在网中，了更好的性能将他在网中移除是得的。

在 GPU 上，量数据被分成4个通道。因此，算一个 `[B,H,W,5]` 的量和算 `[B,H,W,8]`的效果是一的，但是它都比行 `[B,H,W,4]` 的性能要差的多。

从个意上，如果相机硬件支持 RGBA 形式像，4 通道入明更快因可以避免内存制(从 3 通道 RGB 到 4 通道 RGBX）。

了得最佳性能，不要犹豫，使用移化的网架来重新的分器。是化推断性能的重要部分。
