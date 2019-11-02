TensorFlow Lite 操作(operator)的版本

本文档描述了TensorFlow Lite的操作(operator)版本架。 操作(operator)的版本使人能将新功能和参数添加到有操作中。 此外，它保以下内容：

* 向后兼容性：新版本的 TensorFlow Lite 方式可以理旧的模型文件。
* 向前兼容性：只要没有使用新功能，旧版本的 TensorFlow Lite 方式可以理由新版 TOCO 生成的新版本的模型文件。 
* 前向兼容性：如果旧的 TensorFlow Lite 取包含不支持的新版本的模型，告。

##示例：将膨(Dilation)添加到卷操作中
本文档的其余部分通展示如何在卷操作中添加膨系数来解 TFLite 中操作(operator)的版本。

了解本文档内容并不需要了解卷核膨的知。需要注意的是：

* 将添加2个新的整数参数：'dilation_width_factor' 和 'dilation_height_factor'。  
* 不支持膨的旧卷核相当于将因子膨系数置1。

### 更改 FlatBuffer 架(Schema)

要将新参数添加到操作(operator)中，更改`lite/schema/schema.fbs`中的表 。

例如，卷的表如下所示：

```
table Conv2DOptions {
  padding:Padding;
  stride_w:int;
  stride_h:int;
  fused_activation_function:ActivationFunctionType;
}
```

在添加新参数：

* 添加注，指明个版本支持些参数。
* 当新的取新添加的参数的默，它与旧完全相同。

添加新参数后，参数表如下所示：

```
table Conv2DOptions {
  // 版本1支持的参数：
  padding:Padding;
  stride_w:int;
  stride_h:int;
  fused_activation_function:ActivationFunctionType;

  // 版本2支持的参数：
  dilation_width_factor:int = 1;
  dilation_height_factor:int = 1;
}
```

### 更改C中的体和内核

在TensorFlow Lite中，内核与FlatBuffer定是分。 内核从`lite/builtin_op_data.h`中定的C的体中取参数。

原始卷参数如下：

```
typedef struct {
  TfLitePadding padding;
  int stride_width;
  int stride_height;
  TfLiteFusedActivation activation;
} TfLiteConvParams;
```

与FlatBuffer架(Schema)一，通添加注，指明从个版本始支持些参数。果如下：

```
typedef struct {
  // 版本1支持的参数：
  TfLitePadding padding;
  int stride_width;
  int stride_height;
  TfLiteFusedActivation activation;

  // 版本2支持的参数：
  int dilation_width_factor;
  int dilation_height_factor;
} TfLiteConvParams;
```

外，更改内核从C体中取新添加的参数。 在此不再述。

### 更改 FlatBuffer 代以取新参数

取 FlatBuffer 并生成 C 体的是由 `lite/model.cc` 的。

更新文件以理新参数，如下所示：

```
case BuiltinOperator_CONV_2D: {
  TfLiteConvParams* params = MallocPOD<TfLiteConvParams>();
  if (auto* conv_params = op->builtin_options_as_Conv2DOptions()) {
    params->padding = parse_padding(conv_params->padding());
    params->stride_width = conv_params->stride_w();
    params->stride_height = conv_params->stride_h();
    params->activation =
        parse_activation(conv_params->fused_activation_function());
    params->dilation_width_factor = conv_params->dilation_width_factor();
    params->dilation_height_factor = conv_params->dilation_height_factor();
  }
  *builtin_data = reinterpret_cast<void*>(params);
  break;
}
```

里不需要操作版本。 当新取缺少因子的旧模型文件，它将使用1作默，并且新内核将与旧内核一致地工作。

### 更改内核注册
MutableOpResolver（在`lite/op_resolver.h`中定）提供了一些注册操作(operator)内核的函数。默情况下，最小和最大版本都1：
```
void AddBuiltin(tflite::BuiltinOperator op, TfLiteRegistration* registration,
                int min_version = 1, int max_version = 1);
void AddCustom(const char* name, TfLiteRegistration* registration,
               int min_version = 1, int max_version = 1);
```

内置的操作在 `lite/kernels/register.cc` 中注册。 在个例子中，我了一个新的操作内核，它可以理 `Conv2D` 的版本1和版本2，所以我需要将下面行：

```
AddBuiltin(BuiltinOperator_CONV_2D, Register_CONV_2D());
```

修改：

```
AddBuiltin(BuiltinOperator_CONV_2D, Register_CONV_2D(), 1, 2);
```

### 改 TOCO TFLite 的出

最后一是 TOCO 填充(populate)行操作(operator)所需的最低版本。在个例子中，它意味着：

* 当膨系数均1，填充 版本=1。
* 除此之外，填充 版本=2。

此，需要在`lite/toco/tflite/operator.cc`中重写定操作(operator)的(class)中的`GetVersion`函数。

于只有一个版本的操作，它的 `GetVersion` 函数被定：
```
int GetVersion(const Operator& op) const override { return 1; }
```

当支持多个版本，参数并定op的版本，如以下示例所示：

```
int GetVersion(const Operator& op) const override {
  const auto& conv_op = static_cast<const ConvOperator&>(op);
  if (conv_op.dilation_width_factor != 1 ||
      conv_op.dilation_height_factor != 1) {
    return 2;
  }
  return 1;
}
```

### 委托

TensorFlow Lite 提供了一个委托 API，可以将操作委派硬件后端。在 Delegate 的 Prepare 函数中，版本是否支持委派代中的个点。
```
const int kMinVersion = 1;
TfLiteNode* node;
TfLiteRegistration;
context->GetNodeAndRegistration(context, node_index, &node, &registration);

if (registration->version > kMinVersion) {
  // 如果不支持版本，拒点。
}
```
即使委派支持版本1的操作，也是必需的，使委派可以在得更高版本操作到不兼容性。
