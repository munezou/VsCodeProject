# 平台和境

TensorFlow.js有工作平台：器和Node.js。不同平台有很多不同的配置，平台的差影着基于平台的用。

在器平台上，TensorFlow.js既支持移，也支持台式。然之有很多差，TensorFlow.js提供的WebGL API能自并做相的化配置。

在Node.js平台上，TensorFlow.js既支持直接使用TensorFlow API，也支持更慢的CPU境。

## [](https://github.com/tensorflow/tfjs-website/blob/master/docs/guide/platform_environment.md#environments)境

当一个用TensorFlow.js的程序行，所有的配置被称境。它包含一个全局的backend，以及一些可以精控制TensorFlow.js特性的。

### [](https://github.com/tensorflow/tfjs-website/blob/master/docs/guide/platform_environment.md#backends)Backends

TensorFlow.js支持多个不同的backend，用来量的存和数学操作。任何候都只有一个backend生效。大部分，TensorFlow.js会根据当前境自使用最佳的backend。即使，仍然需要知道，如何得知当前正在使用的是个backend，以及如何在不同backend之切。

下面命令用来取当前正使用的backend
```js
console.log(tf.getBackend());
```

下面命令用来手切backend
```js
tf.setBackend(‘cpu’);
console.log(tf.getBackend());
```

#### [](https://github.com/tensorflow/tfjs-website/blob/master/docs/guide/platform_environment.md#webgl-backend)WebGL backend

WebGL backend，称“webgl”，是在器平台上最大的一个backend。它比CPU backend要快100倍。部分原因是，Tensor是作WebGL理保存的，数学算操作在WebGL shader里面。

下面是在使用个backend需要了解的一些知。

##### [](https://github.com/tensorflow/tfjs-website/blob/master/docs/guide/platform_environment.md#avoid-blocking-the-ui-thread)避免阻塞UI程
当用一个操作，如tf.matMul(a,b)，返回tf.Tensor会同返回，然而矩乘法算不一定完成。意味着返回tf.Tensor只是一个指向算的句柄。当用`x.data()`或`x.array()`，只有当算完成才能取到。因此在算程中，避免阻塞UI程，需要使用版本的`x.data()`和`x.array()`，而不是同版本的`x.dataSync()`和`x.arraySync()`。
##### [](https://github.com/tensorflow/tfjs-website/blob/master/docs/guide/platform_environment.md#memory-management)内存管理

一下，在使用WebGL backend，需要式管理内存。因存Tensor的WebGL理，不会被器的收集机制自清理。

用dispose()清理tf.Tensor占用的内存

```js
const a = tf.tensor([[1,2], [3,4]]);
a.dispose();
```

在用中，常需要把多个操作合起来。持一个所有中量的引用，然后清理其占用的内存，方法使代可性差。TensorFlow.js提供tf.tidy()方法清理函数返回不再需要的tf.Tensor，就好像函数行后，本地量都会被清理一。

```js
const a = tf.tensor([[1, 2], [3, 4]]);
const y = tf.tidy(() => {
  const result = a.square().log().neg();
  return result;
});
```

>注意：其他非WebGL境（如Node.js TensorFlow backend或CPU backend）有自回收机制，在些境下使用dispose()或tidy()没有副作用。上，主用通常会比回收的清理来更好的性能。

##### [](https://github.com/tensorflow/tfjs-website/blob/master/docs/guide/platform_environment.md#precision)精度

在移，WebGL只支持16位浮点理操作。然而，大部分机器学模型都用32位浮点的weight和activation的。由于16位浮点数字只能表示[0.000000059605， 65504]个范，当把模型移植到移，它会生精度。需要保自己模型中的weight和activation不要超出个范。
##### [](https://github.com/tensorflow/tfjs-website/blob/master/docs/guide/platform_environment.md#shader-compilation--texture-uploads)Shader& texture 上
TensorFlow.js在GPU里行WebGL的shader程序。然而些shader只有在被用才会被，即lazy-compile。程在CPU上的主程完成，致程序慢。TensorFlow.js会自存好的shader，下次再用有同shape，同入出的tensor能快很多。TensorFlow.js的用一般会多次使用同的操作，因此第二次行会快很多。

TensorFlow.js会把tf.Tensor数据存WebGL理。当一个tf.Tensor被建后，不会被立即上到GPU，而是当其被用到才做。如果个tf.Tensor被第二次使用，由于已在GPU里，因此省掉了上。在一个典型的机器学模型中，意味着weight在第一次被上，第二次就会快很多。

如果希望加快第一次的性能，我推荐模型行，即一个有同shape的入Tensor。
例如:
```js
const model = await tf.loadLayersModel(modelUrl);
// 使用真数据来模型
const warmupResult = model.predict(tf.zeros(inputShape));
warmupResult.dataSync();
warmupResult.dispose();

// 第二次行 predict() 的候将会更加快速
const result = model.predict(userData);
```

#### [](https://github.com/tensorflow/tfjs-website/blob/master/docs/guide/platform_environment.md#nodejs-tensorflow-backend)Node.js TensorFlow backend

在Node.js TensorFlow backend中，所“node”,即TensorFlow的C言API被用来加速操作。它会尽可能使用机器的硬件加速模，如CUDA。

在个backend中，和WebGL backend一，函数会同返回`tf.Tensor`。然而，与WebGL backend不同的是，当得个tensor返回，算已完成。意味着`tf.matMul(a,b)`用会阻塞UI程。

因此，如果在生境下使用个方法，需要在工作程中用，而不是主程。

更多于Node.js的信息，看相文档。
#### [](https://github.com/tensorflow/tfjs-website/blob/master/docs/guide/platform_environment.md#cpu-backend)CPU backend

个backend是性能最差的backend，然而是最的。所有操作都在vanilla JavaScript中，因此很少有并行化，并且会阻塞UI程。

个backend有用，或者是用于WebGL不能使用的。

### [](https://github.com/tensorflow/tfjs-website/blob/master/docs/guide/platform_environment.md#flags)Flags

TensorFlow.js有一套境，能自估和，保是当前平台上的最佳配置。些大部分是内部使用，其中有一些全局可以被API控制。

-   `tf.enableProdMode():`  用生模式。它会去掉模型，NaN，以及其他校操作，从而提高性能。
-   `tf.enableDebugMode()`: 用模式。它会操作的日志并出到到台，行性能信息，如内存footprint和内核行。注意将大降低用行，不可在生境中使用。

注：方法在程序的最前面用，因它影所有的其他。基于同的原因，没有相的disable方法。

注：所有在控制台都tf.ENV.features。尽管没有的公API（不需要考版本兼容），可以使用tf.ENV.set来改些，从而程序做微或断。
