# TensorFlow Lite for Microcontrollers

TensorFlow Lite for Microcontrollers 是 TensorFlow Lite 的一个性移植版本，它用于微控制器和其他一些有数千字内存的。

它可以直接在“裸机”上行，不需要操作系支持、任何准 C/C++ 和内存分配。核心行(core runtime)在 Cortex M3 上行需 16KB，加上足以用来行音字模型的操作，也只需 22KB 的空。

## 始

要快速入并行 TensorFlow Lite for Microcontrollers，[微控制器入](get_started.md)。

## 什微控制器很重要

微控制器通常是小型、低能耗的算，常嵌入在只需要行基本算的硬件中，包括家用器和物网等。年都有数十个微控制器被生出来。

微控制器通常低能耗和小尺寸行化，但代价是降低了理能力、内存和存。一些微控制器具有用来化机器学任性能的功能。

通在微控制器上行机器学推断，人可以在不依于网接的情况下将 AI 添加到各各的硬件中，常用来克服、功率以及由它所致的高延而造成的束。在上行推断也可以助保私，因没有数据从中送出去。

## 功能和件

* C++ API，其行(runtime)在 Cortex M3 上需 16KB
* 使用准的 TensorFlow Lite [FlatBuffer](https://google.github.io/flatbuffers/) 架(schema)
*  Arduino、Keil 和 Mbed 等流行的嵌入式平台生成的目文件
* 多个嵌入式平台化
* 演示口的[示例代](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro/examples/micro_speech)

## 工作流程

是将 TensorFlow 模型部署到微控制器的程：

1. **建或取 TensorFlow 模型**

    模型必非常小，以便在后合的目。它只能使用[支持的操作](build_convert.md#支持的操作)。如果要使用当前不被支持的操作，可以提供自己的。

2. **将模型 TensorFlow Lite FlatBuffer**

    将使用 [TensorFlow Lite 器](build_convert.md#模型)来将模型准 TensorFlow Lite 格式。可能希望出量化模型，因它的尺寸更小、行效率更高。

3. **将 FlatBuffer  C byte 数**

    模型保存在只程序存器中，并以的 C 文件的形式提供。准工具可用于[将 FlatBuffer  C 数](build_convert.md#-C-数)。

4. **集成 TensorFlow Lite for Microcontrollers 的 C++ **

    写微控制器代以使用 [C++ ](library.md)行推断。

5. **部署到的**

    建程序并将其部署到的。

## 支持的平台

嵌入式件的挑之一是存在多不同的体系、、操作系和建系。我的目是尽可能多地支持流行的合，并尽可能地其他添加支持得。

如果是品人，可以下我提供的以下平台的建明或生成的目文件：

                                                                                           | Mbed                                                                           | Keil                                                                           | Make/GCC
---------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ | --------
[STM32F746G Discovery Board](https://www.st.com/en/evaluation-tools/32f746gdiscovery.html)     | [下](https://drive.google.com/open?id=1OtgVkytQBrEYIpJPsE8F6GUKHPBS3Xeb)     | -                                                                              | [下](https://drive.google.com/open?id=1u46mTtAMZ7Y1aD-He1u3R8AE4ZyEpnOl)
["Blue Pill" STM32F103 兼容板](https://github.com/google/stm32_bare_lib)                   | -                                                                              | -                                                                              | [明](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/README.md#building-for-the-blue-pill-stm32f103-using-make)
[Ambiq Micro Apollo3Blue EVB（使用 Make）](https://ambiqmicro.com/apollo-ultra-low-power-mcus/)| -                                                                              | -                                                                              | [明](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/README.md#building-for-ambiq-micro-apollo3blue-evb-using-make)
[Generic Keil uVision Projects](http://www2.keil.com/mdk5/uvision/)                            | -                                                                              | [下](https://drive.google.com/open?id=1Lw9rsdquNKObozClLPoE5CTJLuhfh5mV)     | -
[Eta Compute ECM3531 EVB](https://etacompute.com/)                                             | -                                                                              | -                                                                              | [明](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/README.md#Building-for-the-Eta-Compute-ECM3531-EVB-using-Make)

如果的尚未被支持，添加支持也并不困。可以在 [README.md](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/micro/README.md#how-to-port-tensorflow-lite-micro-to-a-new-platform) 中了解程。

### 可移植参考代

如果没有考具体的的微控制器平台，或者只想在始移植之前用代，最的方法是[下与平台无的参考代](https://drive.google.com/open?id=1cawEQAkqquK_SO4crReDYqf_v7yAwOY8)。

档中有很多文件，个文件只包含建一个二制文件所需的源文件。个文件都有一个的 Makefile 文件，能将文件加到几乎任何 IDE 中并建它。我提供了已置好的 [Visual Studio Code](https://code.visualstudio.com/) 目文件，因此可以松地在跨平台 IDE 中代。

## 目

我的目是使框架可、易于修改、良好、易于集成，并通一致的文件架、解器、API 和内核接口与 TensorFlow Lite 完全兼容。

可以更多在[目和衡](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/micro#goals)方面有的信息。

## 限制

TensorFlow Lite for Microcontrollers 微控制器中的特殊限制而。如果正在使用更大的（例如像 Raspberry Pi 的嵌入式 Linux ），准的 TensorFlow Lite 框架可能更容易集成。

考以下限制：

* 支持 TensorFlow 操作的[有限子集](build_convert.md#支持的操作)
* 支持有限的一些
* 低 C++ API 需要手内存管理
