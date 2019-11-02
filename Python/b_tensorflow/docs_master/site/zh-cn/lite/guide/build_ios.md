# 在 iOS 上建 TensorFlow Lite

本文档描述了如何建 TensorFlow Lite iOS 。如果需使用，可以直接使用 TensorFlow Lite CocoaPod 版本。参 [TensorFlow Lite iOS Demo](ios.md) 取示例。

## 建

TensorFlow Lite 的通用 iOS 需要在 MacOS 机器上，通 Xcode 的命令行工具来建。
如果没有配置好境，可以通 `xcode-select` 来安装 Xcode 8(或更高版本) 和工具:

```bash
xcode-select --install
```

如果是第一次安装，需要先行一次 XCode 并同意它的可。

(也需要安装好 [Homebrew](http://brew.sh/))

下面安装 [automake](https://en.wikipedia.org/wiki/Automake)/[libtool](https://en.wikipedia.org/wiki/GNU_Libtool):

```bash
brew install automake
brew install libtool
```

如果遇到了 automake  和 libtool 已安装但未正接的，首先入以下命令:
```bash
sudo chown -R $(whoami) /usr/local/*
```
然后使用下面的命令来使接生效:
```bash
brew link automake
brew link libtool
```

接着需用通 shell 脚本来下所需的依:
```bash
tensorflow/lite/tools/make/download_dependencies.sh
```

会从网上取和数据的拷，并安装在`tensorflow/lite/downloads`目

所有的依都已建完，在可以在 iOS 上五个支持的体系架建:

```bash
tensorflow/lite/tools/make/build_ios_universal_lib.sh
```

它使用 `tensorflow/lite` 中的 makefile 来建不同版本的，然后用 `lipo` 将它到包含 armv7, armv7s, arm64, i386, 和 x86_64 架的通用文件中。生成的在: `tensorflow/lite/tools/make/gen/lib/libtensorflow-lite.a`

如果在行 `build_ios_universal_lib.sh` ，遇到了如 `no such file or directory: 'x86_64'` 的:
打 Xcode > Preferences > Locations，保在"Command Line Tools"下拉菜中有一个中。

## 在用中使用

需要更新一些的用置来接 TensorFlow Lite。可以在示例目
`tensorflow/lite/examples/ios/simple/simple.xcodeproj` 看些置，
但下面提供了一个完整的要:

-   需要将 `tensorflow/lite/gen/lib/libtensorflow-lite.a` 加入的接建段，并且在 Search Paths 的 Library Search Paths 置中添加 `tensorflow/lite/gen/lib`

-   _Header Search_ 路径需要包含:

    -   tensorflow 的根目,
    -   `tensorflow/lite/downloads`
    -   `tensorflow/lite/downloads/flatbuffers/include`

-   置 `C++ Language Dialect`  `GNU++11` (或 `GNU++14`), 同置 `C++ Standard Library`  `libc++` 来用 C++11 支持 (或更高版本)
