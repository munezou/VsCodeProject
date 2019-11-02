# 模型和

机器学中，一个 _model_ 是一个有可[参数](https://developers.google.com/machine-learning/glossary/#parameter)的函数。个函数将入化出。通俗的来，个函数表了入和出之的系。我通在数据集上模型来得最佳参数。好的模型可以精的将入数据我想得到的出。

TensorFlow.js有建机器学的方法：

1.  用 Layers API（用 _layers_ 来建模型）
2.  用 Core API（底端算子，例如 `tf.matMul()`或`tf.add()`等）来建立模型

我首先会用高API：Layers API来建立模型。然后，我会展示如何用Core API来搭建相同的模型。

## 用Layers API建模型

Layers API有方式建模型：第一是建 _sequential_ 模型，第二是建 _functional_ 模型。下面段会分解模型建方式。

### 使用sequential model

最常的模型是<code>[Sequential](https://js.tensorflow.org/api/0.15.1/#class:Sequential)</code>模型。Sequential模型将网的一的在一起。可以将需要的按序写在一个列表里，然后将列表作<code>[sequential()](https://js.tensorflow.org/api/0.15.1/#sequential)</code> 函数的入：

```js
const model = tf.sequential({
 layers: [
   tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'}),
   tf.layers.dense({units: 10, activation: 'softmax'}),
 ]
});
```

或用 `add()` 方法：

```js
const model = tf.sequential();
model.add(tf.layers.dense({inputShape: [784], units: 32, activation: 'relu'}));
model.add(tf.layers.dense({units: 10, activation: 'softmax'}));
```

> 注意：模型的第一需要“入形状”参数（`inputShape`）。不要在“入型状”中包含batch size（批次大小）。假要向模型入一个形状`[B, 784]`的量（`B`是任意batch size），只需要将“入型状”`[784]`。

可以通`model.layers`来使用模型中的一。例如，可以用`model.inputLayers`和`model.outputLayers`来用入和出。

### 使用functional model

我也可以通`tf.model()`来建`LayersModel`。`tf.model()`和`tf.sequential()`的主要区，可以用`tf.model()`来建任何非的算。

以下是一段如何用`tf.model()` API 建立和上文相同模型的列子：

```js
// 用apply()方法建任意算
const input = tf.input({shape: [784]});
const dense1 = tf.layers.dense({units: 32, activation: 'relu'}).apply(input);
const dense2 = tf.layers.dense({units: 10, activation: 'softmax'}).apply(dense1);
const model = tf.model({inputs: input, outputs: dense2});
```

我在一用`apply()`将上一的出作本的入。`apply()`返回一个`SymbolicTensor`（似于量，但不包含任何数）

不同于sequential model使用`inputShape`来定第一的入，我用`tf.input()`建的`SymbolicTensor`作第一的入

如果向`apply()`入一个数量，它会行算并返一个数量：

```js
const t = tf.tensor([-2, 1, 0, 5]);
const o = tf.layers.activation({activation: 'relu'}).apply(t);
o.print(); // [0, 1, 0, 5]
```

个方式用于独一并它的出。

和sequential model一，可以通`model.layers`来使用模型中的一。例如，可以用`model.inputLayers`和`model.outputLayers`来用入和出。

## 

Sequential model和functional model都属于`LayersModel`。使用`LayersModels`更方便：它要求定入形状，并用定的形状来模型的入。`LayersModel`会自算模型中所有量的形状。知道量的形状后，模型就可以自建它所需要的参数。也可以用形状信息来判断相的是否相互兼容。

## 模型

使用`model.summary()`可以示很多模型的重要信息，包括：

*   一的名字和型
*   一的出形状
*   一的重数量
*   一的入
*   一个模型有的可参数量，和不可参数量

用前面定的模型来做例子，我可以在命令行中得到以下信息：

<table>
  <tr>
   <td>Layer (type)
   </td>
   <td>Output shape
   </td>
   <td>Param #
   </td>
  </tr>
  <tr>
   <td>dense_Dense1 (Dense)
   </td>
   <td>[null,32]
   </td>
   <td>25120
   </td>
  </tr>
  <tr>
   <td>dense_Dense2 (Dense)
   </td>
   <td>[null,10]
   </td>
   <td>330
   </td>
  </tr>
  <tr>
   <td colspan="3" >Total params: 25450<br/>Trainable params: 25450<br/> Non-trainable params: 0
   </td>
  </tr>
</table>

注意：一的出形状中都含有`null`。模型的入形状包含了批次大小，而批次大小是可以活更的，所以批次的在量形状中以`null`示。

## 序列化

相于底端API而言，使用`LayersModel`的一个好是方便存、加模型。`LayersModel`包含如下信息：

*   可用于重建模型的模型架信息
*   模型的重
*   配置（例如失函数，化器和估方式）
*   化器的状（可用于模型）

存和加模型只需要一行代：

```js
const saveResult = await model.save('localstorage://my-model-1');
const model = await tf.loadLayersModel('localstorage://my-model-1');
```

在个例子中，模型被存在器的本地存里。<code>[model.save()](https://js.tensorflow.org/api/latest/#tf.Model.save)</code>和[save and load](save_load.md)了解如何把模型保存在不同的媒介中（例如 file storage, <code>IndexedDB</code>, 触下到器等等）。

## 自定

是建模型的基。如果的模型需要定制化算模，可以写一个自定并插入模型中。下面的例子是一个算平方和的自定：

```js
class SquaredSumLayer extends tf.layers.Layer {
 constructor() {
   super({});
 }
 // In this case, the output is a scalar.
 computeOutputShape(inputShape) { return []; }

 // call() is where we do the computation.
 call(input, kwargs) { return input.square().sum();}

 // Every layer needs a unique name.
 getClassName() { return 'SquaredSum'; }
}
```

可以用`apply()`方法在一个量上个自定

```js
const t = tf.tensor([-2, 1, 0, 5]);
const o = new SquaredSumLayer().apply(t);
o.print(); // prints 30
```

> 注意：如果在模型中包含了自定，模型将不能序列化

## 用Core API建模型

本文提到了在TensorFlow.js中建立模型的方法。最常用的方式是使用 Layers API，因它的模式是基于广泛用的Keras API（情 [best practices and reduces cognitive load](https://keras.io/why-use-keras/)）。Layers API提供了大量方便的工具，例如重初始化，模型序列化，，可移性和安全。

当遇到如下情况，可能会需要使用Core API：

*   需要更多活性和控制
*   不需要序列化或可以造自己的序列化方法

用Core API写的模型包含了一系列的函数。些函数以一个或多个量作入，并出一个量。我可以用Core API来重写之前定的模型：

```js
// The weights and biases for the two dense layers.
const w1 = tf.variable(tf.randomNormal([784, 32]));
const b1 = tf.variable(tf.randomNormal([32]));
const w2 = tf.variable(tf.randomNormal([32, 10]));
const b2 = tf.variable(tf.randomNormal([10]));

function model(x) {
  return x.matMul(w1).add(b1).relu().matMul(w2).add(b2).softmax();
}
```

在Core API中，我需要自己建和初始化重。个重都是一个`Variable`，TensorFlow.js会把`Variable`重可量。可以用[tf.variable()](https://js.tensorflow.org/api/latest/#variable)建`Variable`或把一个已存在的量放到`Variable`中。

本文介了如何用Layers和Core API建模型。接下来，看[training models](train_models.md)学如何模型。
