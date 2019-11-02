# 模型化

Tensorflow Lite 和 [Tensorflow Model Optimization Toolkit](https://www.tensorflow.org/model_optimization) (Tensorflow模型化工具包)提供了最小化推理性的工具。

于移和物网 (IoT) 等,推理效率尤其重要。些在理，内存，能耗和模型存方面有多限制。
此外，模型化解了定点硬件 (fixed-point hardware) 和下一代硬件加速器的理能力。

## 模型量化

深度神网的量化使用了一些技，些技可以降低重的精表示，并且可的降低存和算的激活。量化的好有:

* 有 CPU 平台的支持。
* 激活得的量化降低了用于取和存中激活的存器成本。
* 多 CPU 和硬件加速器提供 SIMD 指令功能，量化特有益。

TensorFlow Lite 量化提供了多的量化支持。

* Tensorflow Lite [post-training quantization](post_training_quantization.md) 量化使重和激活的 Post training 更。
* [Quantization-aware training](https://github.com/tensorflow/tensorflow/tree/r1.13/tensorflow/contrib/quantize){:.external} 可以以最小精度下降来网；用于卷神网的一个子集。

### 延和准性果

以下是一些模型 post-training quantization 和 quantization-aware training 后的延和准性果。所有延数都是在使用个大内核的 Pixel 2 上量的。随着工具包的改，些数字也会随之提高:

<figure>
  <table>
    <tr>
      <th>模型</th>
      <th>Top-1 精性(初始) </th> 
      <th>Top-1 精性(Post Training量化) </th>
      <th>Top-1 精性 (Quantization Aware Training) </th>
      <th>延 (初始) (ms) </th> 
      <th>延 (Post Training量化) (ms) </th>
      <th>延 (Quantization Aware) (ms) </th>
      <th> 大小 (初始) (MB)</th>
      <th> 大小 (化后) (MB)</th>
    </tr> <tr><td>Mobilenet-v1-1-224</td><td>0.709</td><td>0.657</td><td>0.70</td>
      <td>124</td><td>112</td><td>64</td><td>16.9</td><td>4.3</td></tr>
    <tr><td>Mobilenet-v2-1-224</td><td>0.719</td><td>0.637</td><td>0.709</td>
      <td>89</td><td>98</td><td>54</td><td>14</td><td>3.6</td></tr>
   <tr><td>Inception_v3</td><td>0.78</td><td>0.772</td><td>0.775</td>
      <td>1130</td><td>845</td><td>543</td><td>95.7</td><td>23.9</td></tr>
   <tr><td>Resnet_v2_101</td><td>0.770</td><td>0.768</td><td>N/A</td>
      <td>3973</td><td>2868</td><td>N/A</td><td>178.3</td><td>44.9</td></tr>
 </table>
  <figcaption>
    <b>Table 1</b> 模型量化CNN模型的好
  </figcaption>
</figure>

## 工具

首先， [hosted models](../guide/hosted_models.md) 中的模型是否合的用程序。如果没有，我建用从 [post-training quantization tool](post_training_quantization.md) 始，因它广泛用的，且无需数据。

于精度和延目没有到，或者需要硬件加速器支持情况， [quantization-aware training](https://github.com/tensorflow/tensorflow/tree/r1.13/tensorflow/contrib/quantize) {:.external} 是更好的。参 Tensorflow 模型化工具包[Tensorflow Model Optimization Toolkit](https://www.tensorflow.org/model_optimization) 中的的其他化技。

注意: Quantization-aware training 支持卷神网体系的子集。
