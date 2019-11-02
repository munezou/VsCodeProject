#  TensorFlow Lite 在GPU境下

[TensorFlow Lite](https://www.tensorflow.org/mobile/tflite/)支持多硬件加速器。本文档介如何在安卓系（要求OpenGL ES 3.1或更高版本）和iOS（要求iOS 8 或更高版本）的GPU后端（backend）使用TensorFLow Lite delegate APIs。

## 使用GPU加速的

### 速度

GPUs 具有高吐量、可大模并行化的工作（workloads）。因此，它非常合于一个由大量算符成的深度神网，其中一个GPU都可以理一些入量（tensor）并且容易分小的工作（workloads），然后并行行。并行性通常能有低的延。在最好的情况下，在GPU上推断（inference）可以行得足快，以程序，在以前是不可能的。

### 精度

GPU使用16位或32位浮点数行算，并且（与CPU不同）不需要量化（quantization）以得最佳的性能。如果精度降低使得模型的量化（quantization）无法到要求，那在GPU上行神网可能可以消除担。

### 能效

使用GPU行推断（inference）的一个好在于它的能效。GPU能以非常有效和化方法来行算，比在CPU上行相同任消耗更少的能源并生更少的量。

### 支持的Ops

TensorFlow Lite 在GPU上支持16位和32位浮点精度中的以下操作：

* `ADD v1`
* `AVERAGE_POOL_2D v1`
* `CONCATENATION v1`
* `CONV_2D v1`
* `DEPTHWISE_CONV_2D v1-2`
* `FULLY_CONNECTED v1`
* `LOGISTIC v1`
* `MAX_POOL_2D v1`
* `MUL v1`
* `PAD v1`
* `PRELU v1`
* `RELU v1`
* `RELU6 v1`
* `RESHAPE v1`
* `RESIZE_BILINEAR v1`
* `SOFTMAX v1`
* `STRIDED_SLICE v1`
* `SUB v1`
* `TRANSPOSE_CONV v1`

## 基本用法

### Android (Java)

使用`TfLiteDelegate`在GPU上行TensorFlow Lite，在Java中，可以通`Interpreter.Options`来指定GpuDelegate。

```java
// NEW: Prepare GPU delegate.
GpuDelegate delegate = new GpuDelegate();
Interpreter.Options options = (new Interpreter.Options()).addDelegate(delegate);

// Set up interpreter.
Interpreter interpreter = new Interpreter(model, options);

// Run inference.
writeToInputTensor(inputTensor);
interpreter.run(inputTensor, outputTensor);
readFromOutputTensor(outputTensor);

// Clean up.
delegate.close();
```

### Android (C/C++)

在Android GPU上使用C/C++言的TensorFlow Lite，可以使用`TfLiteGpuDelegateCreate()`建，并使用`TfLiteGpuDelegateDelete()`。

```c++
// Set up interpreter.
auto model = FlatBufferModel::BuildFromFile(model_path);
if (!model) return false;
ops::builtin::BuiltinOpResolver op_resolver;
std::unique_ptr<Interpreter> interpreter;
InterpreterBuilder(*model, op_resolver)(&interpreter);

// NEW: Prepare GPU delegate.
const TfLiteGpuDelegateOptions options = {
  .metadata = NULL,
  .compile_options = {
    .precision_loss_allowed = 1,  // FP16
    .preferred_gl_object_type = TFLITE_GL_OBJECT_TYPE_FASTEST,
    .dynamic_batch_enabled = 0,   // Not fully functional yet
  },
};
auto* delegate = TfLiteGpuDelegateCreate(&options);
if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;

// Run inference.
WriteToInputTensor(interpreter->typed_input_tensor<float>(0));
if (interpreter->Invoke() != kTfLiteOk) return false;
ReadFromOutputTensor(interpreter->typed_output_tensor<float>(0));

// NEW: Clean up.
TfLiteGpuDelegateDelete(delegate);
```

用于Android C / C ++的TFLite GPU使用[Bazel](https://bazel.io)建系。例如，可以使用以下命令建委托（delegate）：

```sh
bazel build -c opt --config android_arm64 tensorflow/lite/delegates/gpu:gl_delegate                  # for static library
bazel build -c opt --config android_arm64 tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_gl.so  # for dynamic library
```

### iOS(ObjC++)

要在GPU上行TensorFlow Lite，需要通`NewGpuDelegate()`GPU委托（delegate），然后将其`Interpreter::ModifyGraphWithDelegate()`（而不是用`Interpreter::AllocateTensors()`）

```c++
// Set up interpreter.
auto model = FlatBufferModel::BuildFromFile(model_path);
if (!model) return false;
tflite::ops::builtin::BuiltinOpResolver op_resolver;
std::unique_ptr<Interpreter> interpreter;
InterpreterBuilder(*model, op_resolver)(&interpreter);

// NEW: Prepare GPU delegate.

const GpuDelegateOptions options = {
  .allow_precision_loss = false,
  .wait_type = kGpuDelegateOptions::WaitType::Passive,
};

auto* delegate = NewGpuDelegate(options);
if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;

// Run inference.
WriteToInputTensor(interpreter->typed_input_tensor<float>(0));
if (interpreter->Invoke() != kTfLiteOk) return false;
ReadFromOutputTensor(interpreter->typed_output_tensor<float>(0));

// Clean up.
DeleteGpuDelegate(delegate);
```

注意：用`Interpreter::ModifyGraphWithDelegate()`或`Interpreter::Invoke()`，用者必在当前程中有一个`EGLContext`，并且从同一个`EGLContext`中用`Interpreter::Invoke()`。如果`EGLContext`不存在，委托（delegate）将在内部建一个，但是人必保始从用`Interpreter::Invoke()`的同一个程用`Interpreter::ModifyGraphWithDelegate()`。

## 高用法

### 委托（Delegate）iOS 

`NewGpuDelegate()`接受一个 `struct` 。

```c++
struct GpuDelegateOptions {
  // Allows to quantify tensors, downcast values, process in float16 etc.
  bool allow_precision_loss;

  enum class WaitType {
    // waitUntilCompleted
    kPassive,
    // Minimize latency. It uses active spinning instead of mutex and consumes
    // additional CPU resources.
    kActive,
    // Useful when the output is used with GPU pipeline then or if external
    // command encoder is set
    kDoNotWait,
  };
  WaitType wait_type;
};
```

将`nullptr``NewGpuDelegate()`，并置默（即在上面的基本用法示例中述）。

```c++
// THIS:
const GpuDelegateOptions options = {
  .allow_precision_loss = false,
  .wait_type = kGpuDelegateOptions::WaitType::Passive,
};

auto* delegate = NewGpuDelegate(options);

// IS THE SAME AS THIS:
auto* delegate = NewGpuDelegate(nullptr);
```

然使用`nullptr`很方便，但我建指定置，以避免在以后更改默出任何常情况。

### 入/出冲器

要想在GPU上行算，数据必能GPU可。通常需要行内存制。如果可以的，最好不要交叉CPU / GPU内存界，因会占用大量。通常来，交叉是不可避免的，但在某些特殊情况下，可以忽略其中一个。

如果网的入是已加到GPU内存中的像（例如，包含相机的GPU理），那可以直接保留在GPU内存中而无需入到CPU内存。同，如果网的出采用可染像的格式（例如， [image style transfer](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf)_)，那它可以直接示在屏幕上。

了得最佳性能，TensorFlow Lite用可以直接取和写入TensorFlow硬件冲区并可避免的内存副本。

#### Android

假像送入在GPU存器中，必首先将其OpenGL着色器存冲区象（SSBO）。可以使用`Interpreter.bindGlBufferToTensor()`将TfLiteTensor与用准的SSBO相。注意：`Interpreter.bindGlBufferToTensor()`必在`Interpreter.modifyGraphWithDelegate()`之前用。

```java
// Ensure a valid EGL rendering context.
EGLContext eglContext = eglGetCurrentContext();
if (eglContext.equals(EGL_NO_CONTEXT)) return false;

// Create an SSBO.
int[] id = new int[1];
glGenBuffers(id.length, id, 0);
glBindBuffer(GL_SHADER_STORAGE_BUFFER, id[0]);
glBufferData(GL_SHADER_STORAGE_BUFFER, inputSize, null, GL_STREAM_COPY);
glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);  // unbind
int inputSsboId = id[0];

// Create interpreter.
Interpreter interpreter = new Interpreter(tfliteModel);
Tensor inputTensor = interpreter.getInputTensor(0);
GpuDelegate gpuDelegate = new GpuDelegate();
// The buffer must be bound before the delegate is installed.
gpuDelegate.bindGlBufferToTensor(inputTensor, inputSsboId);
interpreter.modifyGraphWithDelegate(gpuDelegate);

// Run inference; the null input argument indicates use of the bound buffer for input.
fillSsboWithCameraImageTexture(inputSsboId);
float[] outputArray = new float[outputSize];
interpreter.runInference(null, outputArray);
```

似的方法可以用于出量(tensor)。在情况下，`Interpreter.Options.setAllowBufferHandleOutput(true)`被用来，来禁用从GPU内存到CPU内存的网出制的默操作。

```java
// Ensure a valid EGL rendering context.
EGLContext eglContext = eglGetCurrentContext();
if (eglContext.equals(EGL_NO_CONTEXT)) return false;

// Create a SSBO.
int[] id = new int[1];
glGenBuffers(id.length, id, 0);
glBindBuffer(GL_SHADER_STORAGE_BUFFER, id[0]);
glBufferData(GL_SHADER_STORAGE_BUFFER, outputSize, null, GL_STREAM_COPY);
glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);  // unbind
int outputSsboId = id[0];

// Create interpreter.
Interpreter.Options options = (new Interpreter.Options()).setAllowBufferHandleOutput(true);
Interpreter interpreter = new Interpreter(tfliteModel, options);
Tensor outputTensor = interpreter.getOutputTensor(0);
GpuDelegate gpuDelegate = new GpuDelegate();
// The buffer must be bound before the delegate is installed.
gpuDelegate.bindGlBufferToTensor(outputTensor, outputSsboId);
interpreter.modifyGraphWithDelegate(gpuDelegate);

// Run inference; the null output argument indicates use of the bound buffer for output.
ByteBuffer input = getCameraImageByteBuffer();
interpreter.runInference(input, null);
renderOutputSsbo(outputSsboId);
```

#### iOS

假像送入在GPU存器中，必首先将其Metal的`MTLBuffer`象。可以将TfLiteTensor与用准的`MTLBuffer`和`BindMetalBufferToTensor()`相。注意：必在`Interpreter::ModifyGraphWithDelegate()`之前用`BindMetalBufferToTensor()`。此外，默情况下，推断（inference）果的出，会从GPU内存制到CPU内存。在初始化期用`Interpreter::SetAllowBufferHandleOutput(true)`可以操作。

```c++
// Prepare GPU delegate.
auto* delegate = NewGpuDelegate(nullptr);
interpreter->SetAllowBufferHandleOutput(true);  // disable default gpu->cpu copy
if (!BindMetalBufferToTensor(delegate, interpreter->inputs()[0], user_provided_input_buffer)) return false;
if (!BindMetalBufferToTensor(delegate, interpreter->outputs()[0], user_provided_output_buffer)) return false;
if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;

// Run inference.
if (interpreter->Invoke() != kTfLiteOk) return false;
```

注意：一旦从GPU内存制到CPU内存的操作后，将推断（inference）果出从GPU内存制到CPU内存需要个出量式用`Interpreter::EnsureTensorDataIsReadable()`。

## 提示与技巧

* 在CPU上行一些微不足道的操作可能需要非常高的代价，譬如各形式的reshape操作（包括`BATCH_TO_SPACE`，`SPACE_TO_BATCH`，`SPACE_TO_DEPTH`和其他似的操作）。如果不需要些操作（比如使用些操作是了助理解网架和了解整个系但不会影出），那得除它以提高性能。
* 在GPU上，量（tensor）数据被分4个通道（channel）。因此形状`[B, H, W, 5]` 的量（tensor）的算量大致与`[B, H, W, 8]`相同，但明比`[B, H, W, 4]`要大。
  * 比如：如果相机的硬件支持RGBA，那4通道（channel）数据的速度要快得多，因可以避免内存制（从3通道RGB到4通道RGBX）。
* 了得最佳性能，不要犹豫使用移化（mobile-optimized）的网架重新的分器。 是推断（inference）化的重要部分。

