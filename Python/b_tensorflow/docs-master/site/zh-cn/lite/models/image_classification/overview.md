# 像分

<img src="../images/image.png" class="attempt-right">

使用化的模型来上百象，包括人、活、物、植物和地点。

## 始

如果像分的概念不熟悉，先 <a href="#what_is_image_classification">什是像分？</a>

于如何在移用中使用像分，推荐看我提供的 <a href="#example_applications_and_guides">示例用和指</a>。

如果使用 Android 和 iOS 之外的平台，或者已熟悉了 TensorFlow Lite 接口，可以直接下我的新手像分模型及其附的。

<a class="button button-primary" href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip">下新手像分及</a>

当新手模型在的目行起来之后，可以其他模型，在性能、准率以及模型体找到最佳的平衡点。 <a href="#choose_a_different_model">不同模型</a>。

### 示例用和指

我在 Android 和 iOS 平台上都有像分的示例用，并解了它的工作原理。

#### Android

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android">看Android示例</a>

 [Android example guide](android.md) 以了解用工作原理。

#### iOS

<a class="button button-primary" href="https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios.md">看iOS示例</a>

 [iOS example guide](ios.md) 以了解用工作原理。

#### 截屏

下面的截屏 Android 像分示例用。

<img src="images/android_banana.png" alt="Screenshot of Android example" width="30%">

## 什是像分？

机器学的一个常用是像。比如，我可能想要知道下中出了物。

<img src="images/dog.png" alt="dog" width="50%">

像的任被称 _像分_ 。像分模型的目的是各像。比如，一个模型可能被用于三物的特征：兔子、鼠和狗。

当我提供一新的片模型，它会出片含有三物的概率。以下是一个出示例：

<table style="width: 40%;">
  <thead>
    <tr>
      <th>物</th>
      <th>概率</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>兔子</td>
      <td>0.07</td>
    </tr>
    <tr>
      <td>鼠</td>
      <td>0.02</td>
    </tr>
    <tr>
      <td style="background-color: #fcb66d;">狗</td>
      <td style="background-color: #fcb66d;">0.91</td>
    </tr>
  </tbody>
</table>

基于出，我能看到分模型出，片有很大概率表示的是一条狗。

注意：像分只能告片里出的及其概率，并且只能是被的。它不能告片里象的位置或者名称。
如果需要片里象的名称及位置，使用 <a href="../object_detection/overview.md">物体</a> 模型。

### 、和推断

在中，用像和其的 __ 投一个像分模型。个是一个概念或的名字。个模型就要学会去些。

予足多的数据（通常一个数以百的片），个像分模型就能学去新的片是否属于数据中的某些。个的程被称 _推断_ 。

了行推断，一片被入模型中。接着，模型将出一串代表概率的数，元素大小介于 0 和 1 之。合我的示例模型，个程可能如下所示：

<table style="width: 60%">
  <tr style="border-top: 0px;">
    <td style="width: 40%"><img src="images/dog.png" alt="dog"></td>
    <td style="width: 20%; font-size: 2em; vertical-align: middle; text-align: center;">→</td>
    <td style="width: 40%; vertical-align: middle; text-align: center;">[0.07, 0.02, 0.91]</td>
</table>

出中的个数字都数据中的一个。将我的出和三个，我能看出，个模型了片中的象有很大概率是一条狗。

<table style="width: 40%;">
  <thead>
    <tr>
      <th></th>
      <th>概率</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>兔子</td>
      <td>0.07</td>
    </tr>
    <tr>
      <td>鼠</td>
      <td>0.02</td>
    </tr>
    <tr>
      <td style="background-color: #fcb66d;">狗</td>
      <td style="background-color: #fcb66d;">0.91</td>
    </tr>
  </tbody>
</table>

可能注意到些概率的和（兔子，鼠和狗的概率）是 1。是多分模型的常出。（：<a href="https://developers.google.com/machine-learning/crash-course/multi-class-neural-networks/softmax">Softmax</a>）

### 模糊不清的果

既然概率的和是等于 1，那如果片没有被模型出来，也就是不属于被的，可能会它的几个都没有特大的概率。

比如，下表可能表示了一个模糊不清的果：

<table style="width: 40%;">
  <thead>
    <tr>
      <th></th>
      <th>概率</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>兔子</td>
      <td>0.31</td>
    </tr>
    <tr>
      <td>鼠</td>
      <td>0.35</td>
    </tr>
    <tr>
      <td>狗</td>
      <td>0.34</td>
    </tr>
  </tbody>
</table>

### 使用和限制

我提供的些形分模型分很有用。分是指像最有可能表示的某一个。些模型被用于 1000 像。完整的列表：<a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip">模型包</a>

如果想要模型新的：<a href="#customize_model">自定模型</a>.

以下使用案例，采用不同的模型：

<ul>
  <li>片里的一个或多个象的和位置（：<a href="../object_detection/overview.md">物体</a>）</li>
  <li>像的成，比如主体与背景（：<a href="../segmentation/overview.md">分割</a>）</li>
</ul>


当新手模型在的目行起来之后，可以其他模型，在性能、准率以及模型体找到最佳的平衡点。：<a href="#choose_a_different_model">不同模型</a>。

## 不同模型

我的 <a href="../../guide/hosted_models.md">模型列表</a> 中有多像分模型供。
在它的性能、准率和模型体之行衡，以来最的模型。

### 性能

我根据在同的硬件条件下，一个模型行推断所花的来衡量性能。越短，模型越快。

需要的性能取决于的用。用来，性能可能非常重要。因需要在下一制完之前及分析一（例如：推断用必少于 33 ms 才能推断 30 fps 的流）。

我量化的MobileNet 模型的性能范 3.7 ms 至 80.3 ms。

### 准率

我根据模型正分像的率来衡量准度。比如，一个准率 60% 的模型平均有 60% 的能正分一片。

我的 <a href="../../guide/hosted_models.md">模型列表</a> 提供 Top-1 和 Top-5 准率数据。Top-1 是指模型出正的概率最高的率。Top-5 是指模型出正的概率在前五的率。

我量化的 MobileNet 模型的准率范 64.4% 至 89.9%。

### 体

磁上模型的体因其性能和准性而。体可能移（可能影用的下体）或者硬件（可用存可能是有限的）很重要。

我量化的 MobileNet 模型的准率范 0.5 Mb 至 3.4 Mb。

### 模型

<a href="../../guide/hosted_models.md">模型列表</a> 中的模型有不同的，从模型名可以看出，比如，可以 MobileNet、Inception 或者其他的。

模型的影它的性能、准率和体。我提供的模型都是用同的数据的，意味着可以通我提供的数据比些模型，来最合的用的。

注意：我提供的像分模型接受的入尺寸不同。有些模型将其注在文件名上。比如，Mobilenet_V1_1.0_224 模型接受 224x224 像素的入。<br /><br />
所有模型都要求个像素有三个色通道（、、）。量化的模型中个通道需要 1 个字，浮点模型中个通道需要 4 个字。<br /><br />
我的 <a href="android.md">Android</a> 和 <a href="ios.md">iOS</a> 代本展示了如何将全尺寸相机像理个模型需要的格式。

## 自定模型

我提供的模型被用于 1000 像。完整的列表：<a href="https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_quant_and_labels.zip">模型包</a>。

能使用 _移学_ 技来再(re-train)一个模型，以新的。比如能再一个模型来区分不同品的，尽管原始数据中并没有。了到个目的，的个新都需要一片。

学如何移学：<a href="https://codelabs.developers.google.com/codelabs/recognize-flowers-with-tensorflow-on-android/#0">用 TensorFlow 花卉</a> codelab。