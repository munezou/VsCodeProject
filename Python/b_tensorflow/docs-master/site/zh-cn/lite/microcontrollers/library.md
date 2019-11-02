# 理解 C++ 

微控制器版 TensorFlow Lite C++ 是
[TensorFlow ](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro)
的一部分。它可、易修改，果良好，易整合，并且与准 TensorFlow Lite 兼容。

下面的文档列出了 C++ 的基本，提供了所需的命令，并出了将程序写入新的概。

在
[README.md](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/README.md#如何将用于微控制器的TensorFlow-Lite写入一个新的平台)
中包含更多于所有些的更多深入信息。

## 文件

在
[`micro`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro)
根目中有一个相的。然而，因它位于巨大的 TensorFlow 中，所以我建了一些脚本和生成的目文件，多嵌入式境（如 Arduino, Keil, Make 和 Mbed）提供分的相源文件。

### 文件

使用微控制器版 TensorFlow Lite 解器最重要的文件在目的根目中，并附：

-   [`all_ops_resolver.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h)
    提供解器行模型的算符。
-   [`micro_error_reporter.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/micro_error_reporter.h)
    出信息。
-   [`micro_interpreter.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/micro_interpreter.h)
    包含控制和行模型的代。

在 [始使用微控制器](get_started.md) 可以找到典型的用途的展示。

建系提供某些文件在特定平台的。它在以平台名称命名的目下，例如：
[`sparkfun_edge`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro/sparkfun_edge)。

有多其他目，包括：

-   [`kernel`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro/kernels),
    包含算符的和相代。
-   [`tools`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro/tools),
    包含建工具和它的出。
-   [`examples`](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro/examples),
    包含示例代。

### 生成目文件

目中的 `Makefile` 能生成包含所有必需源文件的独立目，些源文件可以被入嵌入式境。目前被支持的境是 Arduino, Keil, Make 和 Mbed。

注意：我其中一些境托管建目。参
[支持的平台](overview.md#supported-platforms)
以下。

要在 Make 中生成目，使用如下指令：

```bash
make -f tensorflow/lite/experimental/micro/tools/make/Makefile generate_projects
```

需要几分，因它需要下一些大型工具依。束后，看到像的路径中，建了一些文件：
`tensorflow/lite/experimental/micro/tools/make/gen/linux_x86_64/prj/` （切的路径取决于的主机操作系）。些文件包含生成的目和源文件。例如：
`tensorflow/lite/experimental/micro/tools/make/gen/linux_x86_64/prj/keil`
包含了 Keil uVision 目。

## 建

如果在使用一个已生成的目，参它内含的 README 以取建指南。

要建并从主 TensorFlow 存中行，行以下命令：

1.  从 GitHub 中把 TensorFlow 存克隆到方便的地方。

    ```bash
    git clone --depth 1 https://github.com/tensorflow/tensorflow.git
    ```

2.  入上一建的目。

    ```bash
    cd tensorflow
    ```

3.  用 `Makefile` 来建目并行。
    注意将会下所有需要的依：

    ```bash
    make -f tensorflow/lite/experimental/micro/tools/make/Makefile test
    ```

## 写入新

把微控制器版 TensorFlow Lite 写入新平台和的指南，可在
[README.md](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro#如何将用于微控制器的TensorFlow-Lite写入一个新的平台)
中看。
