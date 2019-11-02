# 保存并加 tf.Model

TensorFlow.js提供了保存和加模型的功能，些模型可以是使用[`Layers`](https://js.tensorflow.org/api/0.14.2/#Models)API建的或从有TensorFlow模型来的。可能是自己的模型，也可能是人的模型。使用Layers API的一个主要好是使用它建的模型是可序列化的，就是我将在本教程中探的内容。

本教程将会介如何在 TensorFlow.js 中保存和加模型(可通JSON文件)。我同可以入Tensorflow Python模型。

以下个教程介了加些模型：

- [入Keras模型](../tutorials/conversion/import_keras.md)
- [入Graphdef模型](../tutorials/conversion/import_saved_model.md)


## 保存 tf.Model

[`tf.Model`](https://js.tensorflow.org/api/0.14.2/#class:Model) 和 [`tf.Sequential`](https://js.tensorflow.org/api/0.14.2/#class:Model)
同提供了函数 [`model.save`](https://js.tensorflow.org/api/0.14.2/#tf.Model.save) 允保存一个模型的
_拓(topology)_ 和 _重(weights)_ 。

-  拓(Topology): 是一个描述模型的文件（例如它使用的了些操作）。它包含存在外部的模型重的引用。

-  重(Weights): 些是以有效格式存定模型重的二制文件。它通常存在与拓相同的文件中。

我看看保存模型的代是什子的

```js
const saveResult = await model.save('localstorage://my-model-1');
```

一些需要注意的地方:

- `save`  方法采用以 scheme 字符串的 URL 字符串参数（下文称 scheme）。它描述了我想保存模型的地址的型。 在本例中我使用 localstorage:// scheme 将模型保存到本地存。
- 在 scheme 之后是 **路径(path)**。 在上面的例子中，路径是'my-model-1'。
- `save` 方法是的。
- `model.save` 的返回是一个 JSON 象，它包含一些可能有用的信息，例如模型的拓和重的大小。
- 用于保存模型的境不会影那些可以加模型的境。在 node.js 中保存模型并不会阻碍模型在器中被加。


下面我将介以下不同方案。

### 本地存 (限器)

**Scheme:** `localstorage://`

```js
await model.save('localstorage://my-model');
```
可以在器的[本地存](https://developer.mozilla.org/en-US/docs/Web/API/Window/localStorage)中以名称 `my-model` 来保存模型。存能在器刷新后保持不，而当存空成，用或器本身可以清除本地存。 个器可以定域在本地的存空定限。

### IndexedDB (限器)

**Scheme:** `indexeddb://`

```js
await model.save('indexeddb://my-model');
```

会将模型保存到器的[IndexedDB](https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API)存中。
与本地存一，它在刷新后仍然存在，同它往往也存的象的大小有大的限制。

### 文件下 (限器)

**Scheme:** `downloads://`

```js
await model.save('downloads://my-model');
```
会器下模型文件至用的机器上，并生成个文件：
 1. 一个名 `[my-model].json` 的 JSON 文件，它包含了模型的拓和下面将要介的重文件的引用。
 2. 一个二制文件，其中包含名 `[my-model].weights.bin` 的重。

可以更 `[my-model]` 的名称以得一个不同的名称的文件。

由于`.json`使用相路径指向 `.bin`，所以个文件需要被安放在同一个文件中。

> 注意: 某些器要求用在同下多个文件之前授予限。



### HTTP(S) Request

**Scheme:** `http://` or `https://`

```js
await model.save('http://model-server.domain/upload')
```

将建一个Web求，以将模型保存到程服器。 控制程服器，以便保它能理求。
模型将通[POST](https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods/POST) 求送到指定的 HTTP 服器。
POST 求的 body 遵守称`multipart/form-data`的格式。它由以下个文件成

 1. 一个名 `model.json` 的 JSON 文件，其中包含拓和下面描述的重文件的引用。
 2. 一个二制文件，其中包含名 `[my-model].weights.bin` 的重。

注意，个文件的名称需要与上述介中的保持完全相同（因名称内置于函数中，无法更改）。 此[ api 文档](https://js.tensorflow.org/api/latest/#tf.io.browserHTTPRequest)包含一个 Python 代片段，演示了如何使用 [flask](http://flask.pocoo.org/) web 框架来理源自 `save` 的求。

通常，必向 HTTP 服器更多参数或求（例如，用于身，或者如果要指定保存模型的文件）。可以通替 `tf.io.browserHTTPRequest` 函数中的 URL字符串参数来得来自 `save` 函数的求在些方面的粒度控制。个API在控制 HTTP 求方面提供了更大的活性。

例如：

```js
await model.save(tf.io.browserHTTPRequest(
    'http://model-server.domain/upload',
    {method: 'PUT', headers: {'header_key_1': 'header_value_1'}}));
```


### 本机文件系 (限于Node.js)

**Scheme:** `file://`

```js
await model.save('file:///path/to/my-model');
```

当行Node.js后我当然可以直接文件系并且保存模型。个命令将会保存个文件在`scheme`之后指定的`path`中。

 1. 一个名 `model.json` 的 JSON 文件，其中包含拓和下面描述的重文件的引用。
1.  一个二制文件，其中包含名`model.weights.bin`. 的重。

注意，个文件的名称将始与上面指定的完全相同（名称内置于函数中）。


## 加 tf.Model

定一个使用上述方法之一保存的模型，我可以使用 `tf.loadLayersModel` API来加它。

我看一下加模型的代是什子的

```js
const model = await tf.loadLayersModel('localstorage://my-model-1');
```

一些事情得注意:
- 似于`model.save()`,  `loadLayersModel`函数使用以 **scheme**的似URL的字符串参数。它描述了我从中加模型的目型。
- scheme 由**path**指定。在上述例子中路径`my-model-1`。
- URL字符串可以被替一个符合IOHandler接口的象。
- `tf.loadLayersModel()`函数是的。
- `tf.loadLayersModel`返回的是 `tf.Model`

下面我将介可用的不同方案。


### 本地存 (限器)

**Scheme:** `localstorage://`

```js
const model = await tf.loadLayersModel('localstorage://my-model');
```

将从器的[本地存](https://developer.mozilla.org/en-US/docs/Web/API/Window/localStorage).
加一个名`my-model`模型。

### IndexedDB (限器)

**Scheme:** `indexeddb://`

```js
const model = await tf.loadLayersModel('indexeddb://my-model');
```
将从器的[IndexedDB](https://developer.mozilla.org/en-US/docs/Web/API/IndexedDB_API).
加一个模型。


### HTTP(S)

**Scheme:** `http://` or `https://`

```js
const model = await tf.loadLayersModel('http://model-server.domain/download/model.json');
```
将从HTTP端点加模型。加`json` 文件后，函数将求的`json` 文件引用的`.bin`文件。

> 注意：个工具依于[`fetch`](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch)方法。如果的境没有提供原生的fetch方法，可以提供全局方法名称[`fetch`](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch)从而足接口要求或是使用似于(`node-fetch`)[https://www.npmjs.com/package/node-fetch]的。


### 本机文件系 (限于Node.js)

**Scheme:** `file://`

```js
const model = await tf.loadLayersModel('file://path/to/my-model/model.json');
```

当行在Node.js上，我可以直接文件系并且从那里加模型。注意，在上面的函数用中，我引用model.json文件本身（而在保存，我指定一个文件）。相的`.bin`文件需要和`json` 文件在同一个文件中。

## 使用 IOHandlers 加模型

如果上述方案没有足的需求，可以使用`IOHandler`行自定的加行。Tensorflow.js的`IOHandler`提供了[`tf.io.browserFiles`](https://js.tensorflow.org/api/latest/#io.browserFiles) ，行器用在器中上文件。可以在 [文档](https://js.tensorflow.org/api/latest/#io.browserFiles)中看更多信息。

# 使用自定的 IOHandlers 保存或加模型

如果上述方案没有足的保存和加模型的需求，可以通行`IOHandler`以行自定的序列化行。

`IOHandler`是一个含有`save` 和 `load`方法的象。

`save`函数接受一个与[ModelArtifacts](https://github.com/tensorflow/tfjs-core/blob/master/src/io/types.ts#L165)接口匹配的参数并且会返回一个解析[SaveResult](https://github.com/tensorflow/tfjs-core/blob/master/src/io/types.ts#L107)的象。

`load`函数没有接受参数而回返回一个解析[ModelArtifacts](https://github.com/tensorflow/tfjs-core/blob/master/src/io/types.ts#L165)的象。和`save`的相同象。


看[BrowserHTTPRequest](https://github.com/tensorflow/tfjs-core/blob/master/src/io/browser_http.ts)取如何行IOHandler的例子。
