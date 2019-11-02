# 用于 Keras 用使用的 TensorFlow.js layers API

TensorFlow.js 的Layers API以Keras模型。考到 JavaScript 和 Python 之的差，我努力使[Layers API](https://js.tensorflow.org/api/latest/) 与Keras 似。具有使用PythonKeras模型的用可以更松地将目移到 JavaScript中的TensorFlow.js Layers。例如，以下 Keras 代 JavaScript：

```python
# Python:
import keras
import numpy as np

# 建立并模型.
model = keras.Sequential()
model.add(keras.layers.Dense(units=1, input_shape=[1]))
model.compile(optimizer='sgd', loss='mean_squared_error')

# 生成一些用于的数据.
xs = np.array([[1], [2], [3], [4]])
ys = np.array([[1], [3], [5], [7]])

# 用 fit() 模型.
model.fit(xs, ys, epochs=1000)

# 用 predict() 推理.
print(model.predict(np.array([[5]])))
```

```js
// JavaScript:
import * as tf from '@tensorlowjs/tfjs';

// 建立并模型.
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

// 生成一些用于的数据.
const xs = tf.tensor2d([[1], [2], [3], [4]], [4, 1]);
const ys = tf.tensor2d([[1], [3], [5], [7]], [4, 1]);

// 用 fit() 模型.
await model.fit(xs, ys, {epochs: 1000});

// 用 predict() 推理.
model.predict(tf.tensor2d([[5]], [1, 1])).print();
```

但是，我希望在本文档中明并解一些差。一旦理解了些差及其背后的基本原理，将的程序从Python 移到JavaScript（或反向移）会是一相平的体。

## 造函数将 JavaScript 象作配置

比上面示例中的以下 Python 和 JavaScript 代：它都建了一个[全接](https://keras.io/layers/core/#dense)。

```python
# Python:
keras.layers.Dense(units=1, inputShape=[1])
```

```js
// JavaScript:
tf.layers.dense({units: 1, inputShape: [1]});
```

JavaScript函数在Python 函数中没有等效的字参数。我希望避免在 JavaScript 中造函数作位置参数，于和使用具有大量字参数的造函数（如[LSTM](https://keras.io/layers/recurrent/#lstm)尤其麻 。就是我使用JavaScript 配置象的原因。些象提供与Python字参数相同的位置不性和活性。

Model 的一些方法（例如，[`Model.compile()`](https://keras.io/models/model/#model-class-api)）也将 JavaScript 配置象作入。但是，住 Model.fit()、Model.evaluate() 和 Model.predict() 略有不同。因些方法将制 x（feature 特征）和 y（label 或 target 目）数据作入；x 和 y 是与后配置象分的位置参数，属于字参数。例如：


## Model.fit()是的

`Model.fit()` 是用在Tensorflow.js中行模型的主要方法。个方法往往是行的（持数秒或数分）。因此，我利用了JavaScript言的“”特性。所以在器中行，使用此函数就不会阻塞主UI程。和JavaScript中其他可能期行的函数似，例如`async`[取](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API)。需要注意`async`是一个在python中不存在的造。当[`fit()`](https://keras.io/models/model/#model-class-api)方法在keras中返回一个史象, 在JavaScript中`fit()`方法的返回一个包含史的[Promise](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise)个答可以[await(等待)](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/await)，也可以与then()方法一起使用。


## TensorFlow.js 中没有 NumPy

Python Keras 用常使用[NumPy](http://www.numpy.org/)来行基本的数和数的操作，例如在上面的示例中生成 2D 量。

```python
# Python:
xs = np.array([[1], [2], [3], [4]])
```

在 TensorFlow.js 中，基本的数字的操作是使用包本身完成的。例如：

```js
// JavaScript:
const xs = tf.tensor2d([[1], [2], [3], [4]], [4, 1]);
```

 tf.* 命名空提供数和性代数的operations（操作），如矩乘法。有更多信息，参 [TensorFlow.js核心文档](https://js.tensorflow.org/api/latest/)。

## 使用factory(工厂)方法，而不是造函数

Python 中的一行（来自上面的例子）是一个造函数用：

```python
# Python:
model = keras.Sequential()
```

如果格 JavaScript，等效造函数用将如下所示：

```js
// JavaScript:
const model = new tf.Sequential();  // 不！ 要！ ！ ！ 做！ 
```

然而，我决定不使用“new”造函数，因 1)“new”字会使代更加膨；2)“new”造函数被 JavaScript 的“bad part”：一个潜在的陷，如在[*JavaScript: the Good Parts*](http://archive.oreilly.com/pub/a/javascript/excerpts/javascript-good-parts/bad-parts.html).中的争。要在 TensorFlow.js 中建模型和 Layer ，可以用被称 lowerCamelCase（小峰命名）的工厂方法，例如：

```js
// JavaScript:
const model = tf.sequential();

const layer = tf.layers.batchNormalization({axis: 1});
```

## 字符串小峰命名，而不是 snake_case

在 JavaScript 中，与 Python 相比，更常的是使用小峰作符号名称（例如，[Google JavaScript Style Guide](https://google.github.io/styleguide/jsguide.html#naming-camel-case-defined)），而 Python 中 snake_case 很常（例如，在 Keras 中）。因此，我决定使用小峰命名作的字符串，包括以下内容：

* DataFormat，例如，channelsFirst 而不是 channels_first
* Initializer，例如，glorotNormal 而不是 glorot_normal
* Loss and metrics，例如，meanSquaredError 而不是 mean_squared_error，categoricalCrossentropy 而不是 categorical_crossentropy。

例如，如上例所示：

```js
// JavaScript:
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
```

于模型序列化和反序列化，放心。放心。TensorFlow.js 的内部机制保正理 JSON 象中的 snake_case ，例如，从 Python Keras 加模型。


## 使用 apply() 行 Layer 象，而不是将其作函数用

在 Keras 中，Layer 象定了`__call__`方法。因此，用可以通将象作函数用来用 Layer 的，例如:

```python
# Python:
my_input = keras.Input(shape=[2, 4])
flatten = keras.layers.Flatten()

print(flatten(my_input).shape)
```

个 Python 法糖在 TensorFlow.js 中以 apply() 方法：

```js
// JavaScript:
const myInput = tf.input{shape: [2, 4]});
const flatten = tf.layers.flatten();

console.log(flatten.apply(myInput).shape);
```

## Layer.apply() 支持具体 Tensor（量）的命令式（Eager）行

目前，在 Keras 中，`__call__`方法只能（Python）TensorFlow 的 `tf.Tensor` 象行操作（假 TensorFlow 是后端），些象是符号化的并且不包含的数。就是上一中的示例中所示的内容。但是，在 TensorFlow.js 中，Layer 的  `apply()` 方法可以在符号和命令模式下行。如果用 SymbolicTensor 用 `apply()`（似于 tf.Tensor）用，返回将 SymbolicTensor。通常生在模型建期。但是如果用的具体 Tensor（量）用  `apply()`，将返回一个具体的 Tensor（量）。例如：

```js
// JavaScript:
const flatten = tf.layers.flatten();

flatten.apply(tf.ones([2, 3, 4])).print();
```

个特性人想到（Python）TensorFlow 的[Eager Execution](https://www.tensorflow.org/guide/eager)。它在模型期提供了更大的交互性和可性，并且成神网打了大。

## Optimizers（化器）在 train.* 下，而不是 optimizers.*

在 Keras 中，Optimizer（化器）象的造函数位于 keras.optimizers.* 命名空下。在 TensorFlow.js Layer 中，Optimizer（化器）的工厂方法位于 tf.train.* 命名空下。例如：

```python
# Python:
my_sgd = keras.optimizers.sgd(lr=0.2)
```

```js
// JavaScript:
const mySGD = tf.train.sgd({lr: 0.2});
```

## loadLayersModel() 从 URL 加，而不是 HDF5 文件

在 Keras 中，模型通常[保存](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) HDF5（.h5）文件，然后可以使用 `keras.models.load_model()`方法加 。方法采用 .h5 文件的路径。TensorFlow.js 中的 load_model() 的是[`tf.loadLayersModel()`](https://js.tensorflow.org/api/latest/#loadLayersModel)。由于 HDF5 文件格式器并不友好，因此 tf.loadLayersModel() 采用 TensorFlow.js 特定的格式。tf.lloadLayersModel() 将 model.json 文件作其入参数。可以使用 tensorflowjs 的 pip 包从 Keras HDF5 文件 model.json。

```js
// JavaScript:
const model = await tf.loadLayersModel('https://foo.bar/model.json');
```

要注意的是`tf.loadLayersModel()`返回的是[`tf.Model`](https://js.tensorflow.org/api/latest/#class:Model)的[答`](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise)。

通常，tf.Model在 TensorFlow.js中保存和加分使用`tf.Model.save`和`tf.loadLayersModel`方法。我将些 API 似于Keras[the save and load_model API](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model)。但是器境与 Keras 等主要深度学框架行的后端境完全不同，特是用于持久化和数据的路由数中。因此，TensorFlow.js 和 Keras 中的 save/load API 之存在一些有趣的差。有更多信息，参我于 [保存和加tf.Model](./save_load.md)的教程。

## 用`fitDataset()`模型使用`tf.data.Dataset`象

在python版本的tensorflow keras中， 一个模型可以使用[Dataset](https://www.tensorflow.org/guide/datasets)象行。模型的`fit()`方法直接接受的象。一个Tensorflow.js方法可以使用相当于Dataset象的Javascript行，[TensorFlow.js的tf.data API文档](https://js.tensorflow.org/api/latest/#Data)。然而，与python不同， 基于Dataset的是通一个的方法来完成的个方法称之[fitDataset](https://js.tensorflow.org/api/0.15.1/#tf.Model.fitDataset)。[fit()](https://js.tensorflow.org/api/latest/#tf.Model.fitDataset) 只基于Tensor(量)的模型。

## Layer()象和Model(模型)象的内存管理

TensorFlow.js在器中的WebGL上行，其中和模型象的重由WebGL理支持。然而WebGL并不支持内置的收集。在推理和的程中，Layer()和Model(模型)象用在内部管理Tensor(量)内存。但是它也允用清理它以放它占用的WebGL内存。于在加程中建和放多模型例的情况很有用。想要清理一个Layer()和Model(模型)象，使用`dispose()` 方法。
