# 量化模型
本文提供有如何量化 TensorFlow Lite 模型的信息。信息，参[模型化](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/performance/model_optimization.md)。

# 后：特定 CPU 型号的量化模型
建小模型的最方法是在推理期将重量化 8 位并“在行中”量化入/激活。具有延，但先考小尺寸。

在期，将 optimizations 志置大小行化：
```
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
```

# 程中：用于整数行的量化模型
用于整数行的量化模型得具有更低延，更小尺寸和整数加速器兼容模型的模型。目前，需要具有["假量化"点](https://github.com/tensorflow/tensorflow/tree/r1.13/tensorflow/contrib/quantize)的模型 。

表：
```
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
input_arrays = converter.get_input_arrays()
converter.quantized_input_stats = {input_arrays[0] : (0., 1.)}  # mean, std_dev
tflite_model = converter.convert()
```
于全整数模型，入 uint8。mean 和 std_dev values 指定在模型些 UINT8 的是如何映射到入的浮点。

mean 是 0 到 255 之的整数，映射到浮点数 0.0f。std_dev = 255 /（float_max - float_min）

于大多数用，我建使用后量化。我正在研究用于后期和量化的新工具，我希望将化生成量化模型。
