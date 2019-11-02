# 器命令行参考

本 TensorFlow 1.9 至 TensorFlow 最新版本中 TensorFlow Lite 器命令行使用的命令行参数提供全面参考。

## 高命令行参数

下列高命令行参数指定入文件和出文件的。命令行参数 `--output_file` 是需要指定。此外，`--graph_def_file`，`--saved_model_dir` 和 `--keras_model_file` 至少需要指定一个。

* `--output_file`。型：字符串。指定出文件的全路径。

* `--graph_def_file`。型：字符串。指定入 GraphDef 文件（使用 [freeze_graph.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/freeze_graph.py)）的全路径。

* `--saved_model_dir`。型：字符串。指定包含 SavedModel 的目的全路径。

* `--keras_model_file`。型：字符串。指定包含 tf.keras 模型的 HDF5 文件的全路径。 

* `--output_format`。型：字符串。缺省：`TFLITE`。指定出文件的格式。允下列：

    * `TFLITE`：TensorFlow Lite FlatBuffer 格式。    
    * `GRAPHVIZ_DOT`：GraphViz `.dot` 格式包含后生成一个可化。 

* 注意，将 `--output_format`  `GRAPHVIZ_DOT` 会 TFLite 特定造成失。因此，所得的可化可能无法反映最的。如果想得反映所有的最可化，使用 `--dump_graphviz_dir`。

以下命令行参数指定使用 SavedModels 的可函数参数。

* `--saved_model_tag_set`。型：字符串。缺省： [kSavedModelTagServe](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/cc/saved_model/tag_constants.h)。指定一以逗号分隔的，用于要分析的 SavedModel 内的 MetaGraphDef。中的所有都必指定。

* `--saved_model_signature_key`：型: 字符串。缺省：`tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY`。指定包含入和出的 SignatureDef 的。

## 模型命令行参数

模型命令行参数提供有存在入文件中的模型的外信息。

* `--input_arrays`。型：以逗号分隔的字符串列表。指定一个包含入激活量名称的列表。

* `--output_arrays`。型：以逗号分隔的字符串列表。 指定一个包含出激活量名称的列表。

以下命令行参数定入量的属性。命令行参数 `--input_arrays` 中的一根据索引以下命令行参数中的一。

* `--input_shapes`。型：以冒号分隔的列表，列表由以逗号分隔的整数成的子列表成。个子列表指定一个入数的形状，形状的格式参 [TensorFlow 例](https://www.tensorflow.org/guide/tensor#shape)。

* 例： `--input_shapes=1,60,80,3` 于一个典型的模型，表示批量大小 1 ， 入像的高 60 ， 入像的 80 ， 入像的深 3 （代表通道）。

* 例： `--input_arrays=foo,bar --input_shapes=2,3:4,5,6` 表示 "foo" 的形状 [2, 3]， "bar" 的形状 [4, 5, 6]。
    
* `--std_dev_values`， `--mean_values`。 型：以逗号分隔的浮点列表。它指定当入数量化，入数的量化（或反量化）函数参数。只有当 `inference_input_type` 被指定 `QUANTIZED_UINT8` ，才需要定它。

    * `mean_values` 与 `std_dev_values` 的意如下：量化入数中的个量化将根据如下公式被解一个数学数（即一个入激活）：

    * `real_value = (quantized_input_value - mean_value) / std_dev_value`。

* 当一个量化入行浮点推断 （`--inference_type=FLOAT`） ，在行浮点推断之前，推断代将立即根据上述公式量化入行反量化。

* 当行量化推断 （`--inference_type=QUANTIZED_UINT8`） ，推断代不会行反量化。然而，所有数的量化函数参数，包括入数通 `mean_value` 和 `std_dev_value` 指定的量化函数参数，决定了量化推断代中使用的不点乘数。`mean_value` 在行量化推断必是整数。

## 命令行参数

命令行参数指定用在上的可，即它指定出文件具有些属性。

* `--inference_type`。型：字符串。缺省：`FLOAT`。出文件中所有数数的数据型，入数 （用 `--inference_input_type` 指定）除外。必是 `{FLOAT, QUANTIZED_UINT8}`。

    个命令行参数只影数数，包括浮点数和量化数。不包括其他所有数据型，包括通常整数（plain integer）数和字符串数。具体如下：
    
    * 如果指定 `FLOAT`，那出文件中的数数将是浮点型。如果它在入文件中被量化，它将被反量化。

    * 如果指定 `QUANTIZED_UINT8`，那出文件中的数数将被量化 uint8。如果它在入文件中是浮点型，它将被量化。

* `--inference_input_type`。型：字符串。出文件中的一个数入数的数据型。所有入数的数据型的缺省是与 `--inference_type`的指定相同。个命令行参数的主要目的是生成一个具有量化入数的浮点。在入数之后接着添加一个反量化算子。必是 `{FLOAT, QUANTIZED_UINT8}`。

    个命令行参数主要用于的模型：入是位，但是要求浮点推断。于的像模型，其 uint8 入将被量化，并且的入数使用的量化函数参数是它的 `mean_value` 和 `std_dev_value` 函数参数。

* `--default_ranges_min`， `--default_ranges_max`。型：浮点型。指定缺省（最小，最大）区，用于所有没有指定区的数。允用未量化的入文件或者量化的入文件行量化。些命令行参数致模型准率降低。它的目的在于通“虚量化”来一下量化。

* `--drop_control_dependency`。型：布型。 缺省：True。指定是否静默弃控制依。是由于 TensorFlow Lite 不支持控制依。

* `--reorder_across_fake_quant`。型：布型。 缺省：False。指定是否料之外的位置上的 FakeQuant 点行重新排序。用于 FakeQuant 点的位置阻碍，以至于影的情况。它会致生成的与量化不同，有可能会造成不同的算行。

* `--allow_custom_ops`。型：字符串。 缺省：False。指定是否允自定操作。当定 false ，所有未知操作都会。当定 true ，所有未知操作会生成自定操作。者需通在 TensorFlow Lite runtime 配置自定解析器来提供些信息。

* `--post_training_quantize`。型：布型。 缺省：False。指定是否量化被的浮点模型的重。模型将小，延将改善（以准率降低代价）。

## 日志命令行参数

下列命令行参数在程中的多个点生成 [GraphViz](https://www.graphviz.org/) `.dot` 文件的可化。

- `--dump_graphviz_dir`。型：字符串。指定 GraphViz `.dot` 文件出到的目的全路径。在入之后，以及所有完成之后，会出。

- `--dump_graphviz_video`。型：布型。指定是否在次之后出 GraphViz 文件。它要求 `--dump_graphviz_dir` 有指定。
