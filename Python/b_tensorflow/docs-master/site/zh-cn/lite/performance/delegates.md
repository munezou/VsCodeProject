# TensorFlow Lite 代理

_明：Delegate API 仍于段并将随行整。_

## 什是 TensorFlow Lite 代理？

TensorFlow Lite 代理是一将部分或全部的形算委托予一程行的方法。

## 什使用代理？

由于移的理能力不足以及量受限，在其之上行高算力的机器学模型的演算是不可行的。

了避免加重 CPU（中央理器）的担，一些具有如 GPU（形理器）或 DSP（数字信号理器）等的硬件加速器以求取更佳的性能与更高的能效。

## 使用 GPU 代理

TensorFlow Lite 具 GPU 的提供了一个 GPU 代理用以模型算的加速。

有 GPU 代理的概述，看
[TensorFlow Lite 在 GPU 境下](https://www.tensorflow.org/lite/performance/gpu_advanced) 。
有在 Android 和 iOS 上使用 GPU 代理的教程，看
[TensorFlow Lite GPU 代理](https://www.tensorflow.org/lite/performance/gpu) 。

## 代理是如何作的？

假我将一个化的形本行如下所示的操作：

![原生形本](../images/performance/tflite_delegate_graph_1.png "原生形本")

如果把一个代理用于行具体操作，那 TensorFlow Lite 会将形分割多个交由代理行理的子。

若使用一个有高效理 Conv2D（卷）和算 Mean（平均）操作的能力且名“MyDelegate”的代理，那它将致主更行如下所示的操作。

![使用代理的形本](../images/performance/tflite_delegate_graph_2.png "使用代理的形本")

在返回中，个交由代理行理的子将会被更替估子的点。

根据不同的模型，末可以一个点，意味着所有的将被代理或以多个点的子行理。一般而言，当次从代理切至主而不希望采用由代理理的混合子，将会造成由子主的耗。竟，内存交并非是安全的。

## 如何添置一个代理

_注意以下所采用的 API 仍于段并将随行整。_

基于上所述，添置一个代理需要完成以下：

1.  定一个用于估代理子的核心点
2.  建一个用于注册核心点以及明代理可用点的例 [TensorFlow Lite 代理](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/c_api_internal.h#L545)

了使用代行明，我定一个可快速行 Conv2D 和算 Mean 操作的代理并将其命名“MyDelegate”。

```
// 是行操作或整个形的始。
// 具有一个空，作体的声明。
class MyDelegate {
 public:
  // 如果代理可以理此操作，返回“true”。
  static bool SupportedOp(const TfLiteRegistration* registration) {
    switch (registration->builtin_code) {
      case kTfLiteBuiltinConv2d:
      case kTfLiteBuiltinMean:
        return true;
      default:
        return false;
    }
  }

  // 代初始化
  bool Init() {}
  // 初始工作分配（例如：分配冲区）
  bool Prepare(TfLiteContext* context, TfLiteNode* node) {}
  // 代理子始行。
  bool Invoke(TfLiteContext* context, TfLiteNode* node) {}
  // ... 添加其他所需的方法
};

// 核心点建一个替代主 TfLite 中的子的 TfLiteRegistration。
TfLiteRegistration GetMyDelegateNodeRegistration() {
  // 是了取被添加至 TFLite 而非替它的子的代理点的初始化
  // 它被一个操作点。
  // 但在此，Init 函数将用于初始化代理，而 Invoke 函数将用于行代理。
  // 冲。
  // 放内存。
  TfLiteRegistration kernel_registration;
  kernel_registration.builtin_code = kTfLiteBuiltinDelegate;
  kernel_registration.custom_name = "MyDelegate";
  kernel_registration.free = [](TfLiteContext* context, void* buffer) -> void {
    delete reinterpret_cast<MyDelegate*>(buffer);
  };
  kernel_registration.init = [](TfLiteContext* context, const char* buffer,
                                   size_t) -> void* {
    // 在点的初始化段中，初始化“MyDelegate”例。
    const TfLiteDelegateParams* delegate_params =
        reinterpret_cast<const TfLiteDelegateParams*>(buffer);
    MyDelegate* my_delegate = new MyDelegate;
    if (!my_delegate->Init(context, params)) {
      return nullptr;
    }
    return my_delegate;
  };
  kernel_registration.invoke = [](TfLiteContext* context,
                                   TfLiteNode* node) -> TfLiteStatus {
    MyDelegate* kernel = reinterpret_cast<MyDelegate*>(node->user_data);
    return kernel->Invoke(context, node);
  };
  kernel_registration.prepare = [](TfLiteContext* context,
                                    TfLiteNode* node) -> TfLiteStatus {
    MyDelegate* kernel = reinterpret_cast<MyDelegate*>(node->user_data);
    return kernel->Prepare(context, node);
  };

  return kernel_registration;
}

//  TfLiteDelegate 方法

TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate) {
  // 明所有可被代理估的点以及求框架使用代理核心替。
  // 当我需要取点的大小，保留一个点。
  std::vector<int> supported_nodes(1);
  TfLiteIntArray* plan;
  TF_LITE_ENSURE_STATUS(context->GetExecutionPlan(context, &plan));
  TfLiteNode* node;
  TfLiteRegistration* registration;
  for (int node_index : TfLiteIntArrayView(plan)) {
    TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
        context, node_index, &node, &registration));
    if (MyDelegate::SupportedOp(registration)) {
      supported_nodes.push_back(node_index);
    }
  }
  // 置替所有点的点。
  supported_nodes[0] = supported_nodes.size() - 1;
  TfLiteRegistration my_delegate_kernel_registration =
      GetMyDelegateNodeRegistration();

  // 返回将分割子，于子，它将被代理一个  
  // ‘my_delegate_kernel_registration’行理。
  return context->ReplaceNodeSubsetsWithDelegateKernels(
      context, my_delegate_kernel_registration,
      reinterpret_cast<TfLiteIntArray*>(supported_nodes.data()), delegate);
}

void FreeBufferHandle(TfLiteContext* context, TfLiteDelegate* delegate,
                      TfLiteBufferHandle* handle) {
  // 用于放内存的方法。
}

TfLiteStatus CopyToBufferHandle(TfLiteContext* context,
                                TfLiteDelegate* delegate,
                                TfLiteBufferHandle buffer_handle,
                                TfLiteTensor* tensor) {
  // 若有所需，制 tensor（量）的数据至代理的冲区。
  return kTfLiteOk;
}

TfLiteStatus CopyFromBufferHandle(TfLiteContext* context,
                                  TfLiteDelegate* delegate,
                                  TfLiteBufferHandle buffer_handle,
                                  TfLiteTensor* tensor) {
  // 从代理的冲区存入数据至 tensor 的原始内存区域。
  return kTfLiteOk;
}

// 回函数取返回指的所有。
TfLiteDelegate* CreateMyDelegate() {
  TfLiteDelegate* delegate = new TfLiteDelegate;

  delegate->data_ = nullptr;
  delegate->flags = kTfLiteDelegateFlagsNone;
  delegate->Prepare = &DelegatePrepare;
  // 不可空。
  delegate->CopyFromBufferHandle = &CopyFromBufferHandle;
  // 可空。
  delegate->CopyToBufferHandle = &CopyToBufferHandle;
  // 可空。
  delegate->FreeBufferHandle = &FreeBufferHandle;

  return delegate;
}

// 添加所需用的代理

auto* my_delegate = CreateMyDelegate();
if (interpreter->ModifyGraphWithDelegate(my_delegate) !=
        kTfLiteOk) {
  // 用于解决常的方法
} else {
  interpreter->Invoke();
}
...
// 最后千万要住注代理。
delete my_delegate;
```
