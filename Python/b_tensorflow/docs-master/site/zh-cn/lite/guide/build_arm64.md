# 在ARM64板上建Tensorflow Lite

## 交叉

### 安装工具

```bash
sudo apt-get update
sudo apt-get install crossbuild-essential-arm64
```

> 如果使用docker，可能不需要加上`sudo`

### 建

制Tensorflow代。在代根目下行下面的脚本来下依：

> 也可以使用docker像`tensorflow/tensorflow:nightly-devel`，
> tensorflow代在`/tensorflow`

```bash
./tensorflow/lite/tools/make/download_dependencies.sh
```

注意只需要做一次个操作

:

```bash
./tensorflow/lite/tools/make/build_aarch64_lib.sh
```

会出一个静在：
`tensorflow/lite/tools/make/gen/aarch64_armv8-a/lib/libtensorflow-lite.a`.

## 原生

以下在 HardKernel Odroid C2 和gcc 5.4.0版本上.

登的板，安装工具

```bash
sudo apt-get install build-essential
```

首先，制Tensorflow代。在代根目下行：

```bash
./tensorflow/lite/tools/make/download_dependencies.sh
```

注意只需要做一次个操作

:

```bash
./tensorflow/lite/tools/make/build_aarch64_lib.sh
```

会出一个静在：
`tensorflow/lite/tools/make/gen/aarch64_armv8-a/lib/libtensorflow-lite.a`.
