# 量(Tensors) 和 操作(operations)

TensorFlow.js是一个在JavaScript中使用量来定并行算的框架。量是向量和矩向更高度的推广。

## 量(Tensors)

`tf.Tensor`是TensorFlow.js中的最重要的数据元，它是一个形状一或多数成的数的集合。`tf.Tensor`和多数其非常的相似。

一个`tf.Tensor`包含如下属性:

*   `rank`: 量的度
*   `shape`: 个度的数据大小
*   `dtype`: 量中的数据型

>注：在后文中，我将用“度（dimension）”表示`rank（秩）`。在机器学中，量的“数（dimensionality）”有也指特定度的大小。（例如，一个形状[10, 5]的矩是一个rank-2 的量，或者可以成一个2-的量。第一个度的数是10。所以在里用注的方式，描述一下个的双重用法，避免之后的理解。）

我可以用`tf.tensor()`方法将一个数(array)建一个`tf.Tensor`：

```js
// 从一个多数建一个rank-2的量矩
const a = tf.tensor([[1, 2], [3, 4]]);
console.log('shape:', a.shape);
a.print();
// 或者可以用一个一数并指定特定的形状来建一个量
const shape = [2, 2];
const b = tf.tensor([1, 2, 3, 4], shape);
console.log('shape:', b.shape);
b.print();
```

在默的情况下，`tf.Tensor`的数据型也就是
`dtype`32位浮点型(`float32`)。当然`tf.Tensor`也可以被建以下数据型：布(`bool`), 32位整型(`int32`),
64位数(`complex64`), 和字符串(`string`)：

```js
const a = tf.tensor([[1, 2], [3, 4]], [2, 2], 'int32');
console.log('shape:', a.shape);
console.log('dtype', a.dtype);
a.print();
```

TensorFlow.js同也提供了一系列方便的模式用作建随机量，比如将量填入特定的数或是从`HTMLImageElement`中取量。当然可以在文档中找到更多的[方法](https://js.tensorflow.org/api/latest/#Tensors-Creation)。

#### 修改量的形状

`tf.Tensor`中的元素数量是个量的形状的乘(例如一个形状[2,3]的量所含有的元素个数2*3=6个)。所以在大部分候不同形状的量的大小却是相同的,那将一个`tf.Tensor`改形状(reshape)成外一个形状通常是有用且有效的。上述操作可以用`reshape()`
方法:

```js
const a = tf.tensor([[1, 2], [3, 4]]);
console.log('a shape:', a.shape);
a.print();

const b = a.reshape([4, 1]);
console.log('b shape:', b.shape);
b.print();
```

#### 取量的

如果想要取一个`tf.Tensor`的，可以使用`Tensor.array()` or `Tensor.data()`个方法:

```js
 const a = tf.tensor([[1, 2], [3, 4]]);
 //返回多数的
 a.array().then(array => console.log(array));
 // 返回量所包含的所有的一数
 a.data().then(data => console.log(data));
```

我同也提供了些方法能更用的同行版本，但是些方法可能会致的用程序遇到一些性能瓶。在生境的用程序中，始先使用方法。

```js
const a = tf.tensor([[1, 2], [3, 4]]);
 //返回多数的
console.log(a.arraySync());
// 返回量所包含的所有的一数
console.log(a.dataSync());
```

## 操作

可以使用量存数据，而操作(operation)可以操作些数据。TensorFlow.js提供了多能在量上行，用于性代数和机器学的操作。

例1: `tf.Tensor`中所有的元素行x<sup>2</sup>函数:

```js
const x = tf.tensor([1, 2, 3, 4]);
const y = x.square();  // 相当于 tf.square(x)
y.print();
```

例2: 将个 `tf.Tensor`中的元素相加:

```js
const a = tf.tensor([1, 2, 3, 4]);
const b = tf.tensor([10, 20, 30, 40]);
const y = a.add(b);  // 相当于 tf.add(a, b)
y.print();
```

因量是不可的，所以些算并不会更改他的。相的些操作会返回一个新的`tf.Tensor`。

> 注: 大部分的操作会同返回 `tf.Tensor`,
> 然而果可能不会立刻被算出来。意味着得到的`tf.Tensor`上是算的一个句柄。当用`Tensor.data()`或是`Tensor.array()`，些方法将会等待算完成之后才将数解析出来。意味着始先些方法的版本而不是同版本，以避免在算的程中阻塞UI程。

可以在里找到更多于Tensorflow.js中[操作](https://js.tensorflow.org/api/latest/#Operations)的技支持。

## 内存

当使用WebGL后端, `tf.Tensor`的内存必以式管理。是因WebGL不足以`tf.Tensor`超出生命周期后内存被自放。

可以使用`dispose() `方法或是`tf.dispose()`方法用以放`tf.Tensor`所占用的内存:

```js
const a = tf.tensor([[1, 2], [3, 4]]);
a.dispose(); // 相当于 tf.dispose(a)
```

在一个用程序中，将多个操作接在一起是非常常的。保存所有中量的引用以放它所占用的空会降低代的可性。了解决个，TensorFlow.js提供了`tf.tidy()`方法。个方法可以清楚所有在行函数后没有返回的`tf.Tensor`,和行函数清楚一些局部量的方法有些似:

```js
const a = tf.tensor([[1, 2], [3, 4]]);
const y = tf.tidy(() => {
  const result = a.square().log().neg();
  return result;
});
```

在个例子中，`square()`和`log()`个函数因没有返回任何，所以它的果会自的被放。而`neg()`是`tf.tidy()`的返回，所以它的果不会被放。

当然可以取TensorFlow.js程序中量的数量。

```js
console.log(tf.memory());
```

`tf.memory()`将会打印出有当前分配了多少内存的信息。在[里](https://js.tensorflow.org/api/latest/#memory)可以得更多的料。
