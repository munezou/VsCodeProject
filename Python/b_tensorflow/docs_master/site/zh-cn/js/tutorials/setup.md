# 安装

## 器安装

在基于器的目中取TensorFlow.js有以下主要方法

-   使用
    [脚本(script tags)](https://developer.mozilla.org/en-US/docs/Learn/HTML/Howto/Use_JavaScript_within_a_webpage)。
-   从[NPM](https://www.npmjs.com)安装并且使用[Parcel](https://parceljs.org/),
    [WebPack](https://webpack.js.org/)或是
    [Rollup](https://rollupjs.org/guide/en)的建工具。

如果不熟悉Web，或者从未听webpack或parcel等工具，_我建使用脚本(script
tags)_。如果富或想要写更大的程序，那使用建工具行探索可能更加合。

### 使用脚本(script tags)

将以下脚本添加到的主HTML文件中：


```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.0.0/dist/tf.min.js"></script>
```

有脚本的置，参代示例：

<section class="expandable">
  <h4 class="showalways">See code sample script tag setup</h4>
  <pre class="prettyprint">
//定一个性回模型。
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

// 生成一些合成数据
const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

// 使用数据模型
model.fit(xs, ys, {epochs: 10}).then(() => {
// 在模型从未看到的数据点上使用模型行推理

  model.predict(tf.tensor2d([5], [1, 1])).print();
  // 打器工具看出
});
  </pre>
</section>

### 从NPM安装

可以使用
[npm cli](https://docs.npmjs.com/cli/npm)工具或是[yarn](https://yarnpkg.com/en/)安装TensorFlow.js。

```
yarn add @tensorflow/tfjs
```

_或者_

```
npm install @tensorflow/tfjs
```

<section class="expandable">
  <h4 class="showalways">See sample code for installation via NPM</h4>
  <pre class="prettyprint">
import * as tf from '@tensorflow/tfjs';

//定一个性回模型。
const model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

// 生成一些合成数据
const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

// 使用数据模型
model.fit(xs, ys, {epochs: 10}).then(() => {
  // 在模型从未看到的数据点上使用模型行推理
  model.predict(tf.tensor2d([5], [1, 1])).print();
  //  打器工具看出
});
  </pre>
</section>


## Node.js 安装

可以使用
[npm cli](https://docs.npmjs.com/cli/npm)工具或是[yarn](https://yarnpkg.com/en/)安装TensorFlow.js。

**1:** 安装有原生C++定的TensorFlow.js。

```
yarn add @tensorflow/tfjs-node
```

_或者_

```
npm install @tensorflow/tfjs-node
```

**2:**
（限Linux）如果的系具有[支持CUDA](https://www.tensorflow.org/install/install_linux#NVIDIARequirements)的NVIDIARGPU，使用GPU包以得更高的性能。

```
yarn add @tensorflow/tfjs-node-gpu
```

_or_

```
npm install @tensorflow/tfjs-node-gpu
```

**3:** 安装JavaScript版本，是性能方面最慢的。

```
yarn add @tensorflow/tfjs
```

_or_

```
npm install @tensorflow/tfjs
```

<section class="expandable">
  <h4 class="showalways">See sample code for Node.js usage</h4>
  <pre class="prettyprint">
const tf = require('@tensorflow/tfjs');

// 可加定：
// 如果使用GPU行，使用'@tensorflow/tfjs-node-gpu'
require('@tensorflow/tfjs-node');

// 一个模型:
const model = tf.sequential();
model.add(tf.layers.dense({units: 100, activation: 'relu', inputShape: [10]}));
model.add(tf.layers.dense({units: 1, activation: 'linear'}));
model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});

const xs = tf.randomNormal([100, 10]);
const ys = tf.randomNormal([100, 1]);

model.fit(xs, ys, {
  epochs: 100,
  callbacks: {
    onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}`)
  }
});
  </pre>
</section>

### TypeScript

当使用TypeScript，如果的目使用格的空，或者在程中遇到，可能需要在的`tsconfig.json`文件中置`skipLibCheck：true`。
