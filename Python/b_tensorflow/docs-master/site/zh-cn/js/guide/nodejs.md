# Node 中的 TensorFlow.js

## TensorFlow CPU

TensorFlow CPU 包，可以按如下方式入：


```js
import * as tf from '@tensorflow/tfjs-node'
```


当从个包入 TensorFlow.js ，入的模将由 TensorFlow C 二制文件加速并在 CPU 上行。CPU 上的 TensorFlow 使用硬件加速来加速内部的性代数算。

此件包用于支持 TensorFlow 的 Linux，Windows 和 Mac 平台。

> 注意：没有必要入'@tensorflow/tfjs'或者将其添加到的 package.json 文件中，是由 Node 接入的。


## TensorFlow GPU

TensorFlow GPU 包，可以按如下方式入：


```js
import * as tf from '@tensorflow/tfjs-node-gpu'
```

与 CPU 包一，入的模将由 TensorFlow C 二制文件加速，但是它将使用 CUDA 在 GPU 上行量算，因此只能行在 Linux 平台。定比其他可定可以快至少一个数量。

> 注意：此件包目前用于 CUDA。在本方案之前，需要在有 NVIDIA 的的机器上安装 CUDA。

> 注意：没有必要入'@tensorflow/tfjs'或将其添加到的 package.json 文件中，是由 Node 接入的。


## 普通 CPU

使用普通 CPU 行 TensorFlow.js 版本，可以按如下方式入：


```js
import * as tf from '@tensorflow/tfjs'
```

个包与在器中使用的包似。在个包中，些用是在 CPU 上以原生 JavaScript 行。个包比其他包小得多，因它不需要 TensorFlow 二制文件，但是速度要慢得多。

由于个件包不依于 TensorFlow，因此它可用于支持 Node.js 的更多，而不是 Linux，Windows 和 Mac平台。


## 生境考因素

Node.js Bindings  TensorFlow.js 提供了一个同地行操作的后端。意味着当用一个操作，例如 `tf.matMul(a, b)`，它将阻塞主程直到个操作完成。

因此，当前 Bindings 非常合脚本和任。如果要在用程序（如：Web 服器）中使用Node.js Bindings，置一个工作列或置一些工作程，以便的 TensorFlow.js 代不会阻止主程。


## APIs

一旦在上面的任何中将包入 tf 后，所有普通的 TensorFlow.js 符号都将出在入的模上。

### tf.browser

在普通的 TensorFlow.js 包中，`tf.browser.*` 命名空中的符号将在 Node.js 中不可用，因它使用特定器的 API。

目前，有如下 API：

*   tf.browser.fromPixels
*   tf.browser.toPixels

### tf.node

有个 Node.js 包提供了一个名称 `tf.node` 的命名空，其中包含了特定 Node 的 API。

TensorBoard 是一个特定 Node.js API 的重要例子。

是一个将的(summaries)出至Node.js的TensorBoard中的案例

```js
const model = tf.sequential();
model.add(tf.layers.dense({units: 1}));
model.compile({
  loss: 'meanSquaredError',
  optimizer: 'sgd',
  metrics: ['MAE']
});


// 了演示目的生成一些随机假数据。
const xs = tf.randomUniform([10000, 200]);
const ys = tf.randomUniform([10000, 1]);
const valXs = tf.randomUniform([1000, 200]);
const valYs = tf.randomUniform([1000, 1]);


// 始模型程。
await model.fit(xs, ys, {
  epochs: 100,
  validationData: [valXs, valYs],
   // 在里添加 tensorBoard 回。
  callbacks: tf.node.tensorBoard('/tmp/fit_logs_1')
});
```
