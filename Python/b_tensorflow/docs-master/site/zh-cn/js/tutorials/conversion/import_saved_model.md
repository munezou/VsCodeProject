# 在TensorFlow.js中引入TensorFlow GraphDef模型

TensorFlow GraphDef模型（一般是通Python API建的）可以保存成以下几格式：
1. TensorFlow [SavedModel](https://www.tensorflow.org/programmers_guide/saved_model#overview_of_saving_and_restoring_models)
2. [Frozen Model](https://www.tensorflow.org/mobile/prepare_models#how_do_you_get_a_model_you_can_use_on_mobile)
3. [Session Bundle](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/session_bundle/README.md)
4. [Tensorflow Hub module](https://www.tensorflow.org/hub/)

以上所有格式都可以被[TensorFlow.js converter](https://github.com/tensorflow/tfjs-converter)成TensorFlow.js可取的模型格式，并用于推算（inference）。

（注意：TensorFlow已淘汰了session bundle格式，将的模型成SavedModel格式。）

## 必要条件

模型的工作需要用到Python境；可以用[pipenv](https://github.com/pypa/pipenv) 或 [virtualenv](https://virtualenv.pypa.io)建一个隔的境。用条命令安装模型器：

```bash
 pip install tensorflowjs
```

将TensorFlow模型引入到TensorFlow.js需要个。首先，将的模型TensorFlow.js可用的web格式，然后入到TensorFlow.js中。

## 第一：将TensorFlow模型至TensorFlow.js可用的 web 格式模型

行器提供的脚本：

用法：以SavedModel例：

```bash
tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_node_names='MobilenetV1/Predictions/Reshape_1' \
    --saved_model_tags=serve \
    /mobilenet/saved_model \
    /mobilenet/web_model
```

Frozen model 例:

```bash
tensorflowjs_converter \
    --input_format=tf_frozen_model \
    --output_node_names='MobilenetV1/Predictions/Reshape_1' \
    /mobilenet/frozen_model.pb \
    /mobilenet/web_model
```

Tensorflow Hub module 例:

```bash
tensorflowjs_converter \
    --input_format=tf_hub \
    'https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/1' \
    /mobilenet/web_model
```

|脚本参数 | 描述 |
|---|---|
|`input_path`  | saved model, session bundle 或 frozen model的完整的路径，或TensorFlow Hub模的路径。|
|`output_path` | 出文件的保存路径。|

|  | 描述
|---|---|
|`--input_format`     | 要的模型的格式。SavedModel  tf_saved_model, frozen model  tf_frozen_model, session bundle  tf_session_bundle, TensorFlow Hub module  tf_hub，Keras HDF5  keras。 |
|`--output_node_names`| 出点的名字，个名字用逗号分。|
|`--saved_model_tags` | 只SavedModel用的：入需要加的MetaGraphDef相的tag，多个tag用逗号分隔。默 `serve`。|
|`--signature_name`   | 只TensorFlow Hub module用的：要加的名，默`default`。参考 https://www.tensorflow.org/hub/common_signatures/.|

用以下命令看助信息：

```bash
tensorflowjs_converter --help
```

### 器生的文件

脚本会生文件：

* `model.json` （数据流和重清）
* `group1-shard\*of\*` （二制重文件）

里例Mobilenet v2模型后出的文件：

```html
  output_directory/model.json
  output_directory/group1-shard1of5
  ...
  output_directory/group1-shard5of5
```

## 第二：在器加和行模型

1. 安装tfjs-convert npm包：

`yarn add @tensorflow/tfjs` 或 `npm install @tensorflow/tfjs`

2. 建 [FrozenModel class](https://github.com/tensorflow/tfjs-converter/blob/master/src/executor/frozen_model.ts) 并始推算：

```js
import * as tf from '@tensorflow/tfjs';
import {loadGraphModel} from '@tensorflow/tfjs-converter';

const MODEL_URL = 'model_directory/model.json';

const model = await loadGraphModel(MODEL_URL);
const cat = document.getElementById('cat');
model.execute(tf.fromPixels(cat));
```

具体代参考 [MobileNet 演示](https://github.com/tensorflow/tfjs-converter/tree/master/demo/mobilenet).

`loadGraphModel` API中的`LoadOptions`参数可以用来送密或者自定求中的文件。更多信息参考 [loadGraphModel() 文档](https://js.tensorflow.org/api/1.0.0/#loadGraphModel)。

## 支持的操作

目前，TensorFlow.js只支持部分TensorFlow算子。若的模型包含了不被支持的算子，`tensorflowjs_converter`脚本会并列出的模型中不被支持的算子。在github上起 [issue](https://github.com/tensorflow/tfjs/issues)我知道需要支持的算子。

## 加模型重

若只需要加模型的重，参考以下代：

```js
import * as tf from '@tensorflow/tfjs';

const weightManifestUrl = "https://example.org/model/weights_manifest.json";

const manifest = await fetch(weightManifestUrl);
this.weightManifest = await manifest.json();
const weightMap = await tf.io.loadWeights(
        this.weightManifest, "https://example.org/model");
```
