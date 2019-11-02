# 在莓派上建 TensorFlow Lite
## 交叉
### 安装工具
此功能已在64位的 ubuntu 16.04.3 和 Tensorflow devel docker image [tensorflow/tensorflow:nightly-devel](https://hub.docker.com/r/tensorflow/tensorflow/tags/) 上。
要使用 TensorFlow Lite 交叉功能，先安装工具和相的。
```bash
sudo apt-get update
sudo apt-get install crossbuild-essential-armhf
```
如果使用 Docker，可能无法使用 `sudo`。
### 建
克隆此Tensorflow，在的根目下行此脚本以下所有依：
> tensorflow在 /tensorflow 下。如果使用的是 docker 像 tensorflow/tensorflow:nightly-develimage，会使用Tensorflow存，使用以下命令即可。
```bash
./tensorflow/lite/tools/make/download_dependencies.sh
```
注意，只需要行一次此操作。
然后便能：
```bash
./tensorflow/lite/tools/make/build_rpi_lib.sh
```
将一个静,它位于：
tensorflow/lite/tools/make/gen/rpi_armv7l/lib/libtensorflow-lite.a.
## 本地
已在Raspberry Pi 3b，Raspbian GNU / Linux 9.1（stretch），gcc版本6.3.0 20170516（Raspbian 6.3.0-18 + rpi1）上行了。
登Raspberry Pi，安装工具。
```bash
sudo apt-get install build-essential
```
首先，克隆TensorFlow。在的根目行：
```bash
./tensorflow/lite/tools/make/download_dependencies.sh
```
注意，只需要行一次此操作。
然后便能：
```bash
./tensorflow/lite/tools/make/build_rpi_lib.sh
```
将一个静,它位于：
tensorflow/lite/tools/make/gen/lib/rpi_armv7/libtensorflow-lite.a 。
