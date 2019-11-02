# 将Keras模型入Tensorflow.js

Keras模型（通常通Python API建）可能被保存成[多格式之一](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model). 整个模型格式可以被Tensorflow.js的(Layer)格式，个格式可以被加并直接用作Tensorflow.js的推断或是一的。

后的TensorFlow.js(Layer)格式是一个包含model.json文件和一二制格式的分片重文件的目。 model.json文件包含模型拓（又名“架(architecture)”或“形(graph)”：它是(Layer)及其接方式的描述）和重文件的清。

## 要求

程要求Python的程境，可能需要独立的使用[pipenv](https://github.com/pypa/pipenv)或是[virtualenv](https://virtualenv.pypa.io)。并使用 `pip install tensorflowjs`安装器

将Keras模型入Tensorflow.js需要程。首先，将已有Keras模型成TF.js(Layer)格式，然后将其加Tensorflow.js。

## Step 1. 将已有Keras模型成TF.js(Layer)格式

Keras模型通常通 `model.save(filepath)`行保存，做会生一个同含有模型拓以及重的HDF5(.h5)文件。如需要一个文件成TF.js格式，可以行以下代。里的`path/to/my_model.h5`Keras .h5文件地址，而`path/to/tfjs_target_dir`是出的TF.js目。

```sh
# bash

tensorflowjs_converter --input_format keras \
                       path/to/my_model.h5 \
                       path/to/tfjs_target_dir
```

## 一方式: 使用 Python API 直接出 TF.js (Layer)格式

如果有一个Python的Keras模型，可以用以下方法直接出一个Tensoflow.js(Layers)格式:


```py
# Python

import tensorflowjs as tfjs

def train(...):
    model = keras.models.Sequential()   # for example
    ...
    model.compile(...)
    model.fit(...)
    tfjs.converters.save_keras_model(model, tfjs_target_dir)
```

## Step 2: 将模型加Tensorflow.js

使用一个web服器在1中生成的后的模型文件提供服。注意，可能需要将的服器配置[允跨源源共享(CORS)](https://enable-cors.org/), 以允在 JavaScript 中提取文件。

然后通提供model.json文件的URL将模型加到TensorFlow.js中：


```js
// JavaScript

import * as tf from '@tensorflow/tfjs';

const model = await tf.loadLayersModel('https://foo.bar/tfjs_artifacts/model.json');
```

在，模型已准好行推理(inference)，估(evaluation)或重新(re-training)。例如，模型完成加后可以立即行(predict)：


```js
// JavaScript

const example = tf.fromPixels(webcamElement);  // for example
const prediction = model.predict(example);
```

很多[Tensorflow.js例](https://github.com/tensorflow/tfjs-examples)采用方法，使用已在 Google 云存上和托管的模型。

注意，使用`model.json`文件名引用整个模型。`loadModel(...)` 取 `model.json`，并且通外的HTTP(S)求以取`model.json`重清中引用的分片重文件。 此方法允器将些文件全部存(可能被存在互网上其他存服器中)。是因 `model.json`和重分都小于典型的存文件大小限制。因此个模型可能在随后的景中加地更快。


## 已支持的特性

TensorFlow.js的(Layers)目前支持基于准Keras的Keras模型。 使用不支持的操作(ops)或(layers)的模型 - 例如 自定，Lambda，自定失(loss)或自定指(metrics)无法自入，因它依于无法被可靠地JavaScript的Python代。
