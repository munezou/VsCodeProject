# Segmentation

![segmentation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/models/images/segmentation.png)

## Get started

DeepLab 是用于像分割的最先的深度学模型，其目是像中的个像素分配(例如人，狗，猫)。

[Download starter model](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite)

## How it works

像分割像的个像素是否与某个相。与矩形区域中目[目]的任(https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/models/object_detection/overview.md)和整个像行分[像分]的任(https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/models/image_classification/overview.md)形成照。

当前的包括以下功能：

1. DeepLabv1 :我使用 atrous convolution 来示地控制在深度卷神网中算特征部分的分辨率。
2. DeepLabv2 :我用 atrous spatial pyramid pooling(ASPP) ,使用多个采率和有效野的器,在多尺度上健地分割目象。
3. DeepLabv3 :我使用像特征[5,6]来展ASPP模以捕更距的信息。我增加了批准化[7]参数以加快。特的，在和估期我用 atrous convolution 来提取不同出幅的出特征，在出幅等于16有效的促了批准化,并在出幅8得到了更高的估效果。
4. DeepLabv3+ :我展了 DeepLabv3 ,增加了一个但有效的解器模，以化分果，尤其是沿着象界。此外，在器-解器中，可以通 atrous convolution 任意地控制所提取的器特征的分辨率，以折衷精度和行。

## Example output

模型将以很高的精度在目象上建掩膜。
![segmentation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/models/segmentation/images/segmentation.gif)

## Read more about segmentation

* [Semantic Image Segmentation with DeepLab in TensorFlow](https://ai.googleblog.com/2018/03/semantic-image-segmentation-with.html)
* [TensorFlow Lite Now Faster with Mobile GPUs (Developer Preview)](https://medium.com/tensorflow/tensorflow-lite-now-faster-with-mobile-gpus-developer-preview-e15797e6dee7)
* [DeepLab: Deep Labelling for Semantic Image Segmentation](https://github.com/tensorflow/models/tree/master/research/deeplab)
