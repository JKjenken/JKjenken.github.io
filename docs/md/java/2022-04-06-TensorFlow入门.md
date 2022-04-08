# TensorFlow 概述
## 含义
- 用计算机程序表示出向量、矩阵和张量等数学概念，并方便地进行运算；（数学概念和运算的程序化表达）
- 对任意可导函数 f(x) ，求在自变量 x = x_0 给定时的梯度（“符号计算” 的能力）

## 能帮助我们做什么？
### 训练流程
数据的处理 ：使用 tf.data 和 TFRecord 可以高效地构建和预处理数据集，构建训练数据流。同时可以使用 TensorFlow Datasets 快速载入常用的公开数据集。

模型的建立与调试 ：使用即时执行模式和著名的神经网络高层 API 框架 Keras，结合可视化工具 TensorBoard，简易、快速地建立和调试模型。也可以通过 TensorFlow Hub 方便地载入已有的成熟模型。

模型的训练 ：支持在 CPU、GPU、TPU 上训练模型，支持单机和多机集群并行训练模型，充分利用海量数据和计算资源进行高效训练。

模型的导出 ：将模型打包导出为统一的 SavedModel 格式，方便迁移和部署。
### 部署流程
服务器部署 ：使用 TensorFlow Serving 在服务器上为训练完成的模型提供高性能、支持并发、高吞吐量的 API。

移动端和嵌入式设备部署 ：使用 TensorFlow Lite 将模型转换为体积小、高效率的轻量化版本，并在移动端、嵌入式端等功耗和计算能力受限的设备上运行，支持使用 GPU 代理进行硬件加速，还可以配合 Edge TPU 等外接硬件加速运算。

网页端部署 ：使用 TensorFlow.js，在网页端等支持 JavaScript 运行的环境上也可以运行模型，支持使用 WebGL 进行硬件加速。

# TensorFlow安装与环境配置
## anaconda操作指南
anaconda下载
官方下载地址:https://www.anaconda.com/download/#download  
国内清华源下载地址:https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/

执行conda命令时不直接使用cmd执行，使用anaconda提供的命令行工具执行  
开始菜单 - Anaconda3 - Anaconda Prompt

修改为清华同方源
```bash
conda config --add http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
conda config --add channels http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/
conda config --add channels http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes
```
或者直接修改.condarc内容
``` txt
channels:
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/bioconda/
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
show_channel_urls: true
```
检查显卡的gpu计算能力是否符合CUDA（cuda要求计算能力大于等于3.5）  
https://developer.nvidia.com/cuda-gpus

在conda中创建 tensorflow 环境（conda虚拟环境）
```bash
conda create -n tensorflow python=3.7
```
安装tensorflow-gpu
```bash
conda install tensorflow-gpu==2.2.0
```
安装tensorflow-gpu（上面方法失败时可以选择通过pip安装） 
```bash
pip install tensorflow-gpu==2.2.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```
安装cuda
```bash
conda install cudatoolkit=10.1 cudnn=7.6.5
```

测试代码
```python
import tensorflow as tf
import timeit
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 代码用于忽略级别 2 及以下的消息（级别 1 是提示，级别 2 是警告，级别 3 是错误）。

with tf.device('/cpu:0'):
    cpu_a = tf.random.normal([10000, 1000])
    cpu_b = tf.random.normal([1000, 2000])
    print(cpu_a.device, cpu_b.device)

with tf.device('/gpu:0'):
    gpu_a = tf.random.normal([10000, 1000])
    gpu_b = tf.random.normal([1000, 2000])
    print(gpu_a.device, gpu_b.device)


def cpu_run():
    with tf.device('/cpu:0'):
        c = tf.matmul(cpu_a, cpu_b)
    return c


def gpu_run():
    with tf.device('/gpu:0'):
        c = tf.matmul(gpu_a, gpu_b)
    return c


# warm up
cpu_time = timeit.timeit(cpu_run, number=10)
gpu_time = timeit.timeit(gpu_run, number=10)
print('warmup:', cpu_time, gpu_time)

cpu_time = timeit.timeit(cpu_run, number=10)
gpu_time = timeit.timeit(gpu_run, number=10)
print('run time:', cpu_time, gpu_time)

print('GPU', tf.test.is_gpu_available())

```

另一个兼容的版本号
安装tensorflow-gpu=2.4.0，使用python版本为3.7或者3.8，cudatoolkit=11.0，cudnn=8.0

## IDE设置
建议使用 PyCharm 作为 Python 开发的 IDE

在新建项目时，需要选定项目的 Python Interpreter。在安装部分，你所建立的每个 Conda 虚拟环境其实都有一个自己独立的 Python Interpreter，只需要将它们添加进来即可。选择 “Add”，并在接下来的窗口选择 “Existing Environment”，在 Interpreter 处选择 Anaconda安装目录/envs/所需要添加的Conda环境名字/python.exe （Linux 下无 .exe 后缀）并点击 “OK” 即可。如果选中了 “Make available to all projects”，则在所有项目中都可以选择该 Python Interpreter。注意，在 Windows 下 Anaconda 的默认安装目录比较特殊，一般为 C:\Users\用户名\Anaconda3\ 或 C:\Users\用户名\AppData\Local\Continuum\anaconda3 。此处 AppData 是隐藏文件夹。

其他可选方案
 Visual Studio Code
 Google Colab （在线交互式python环境）

## 安装tip注意事项

可以使用 conda install tensorflow 来安装 TensorFlow，不过 conda 源的版本往往更新较慢，难以第一时间获得最新的 TensorFlow 版本；

从 TensorFlow 2.1 开始，pip 包 tensorflow 即同时包含 GPU 支持，无需通过特定的 pip 包 tensorflow-gpu 安装 GPU 版本。如果对 pip 包的大小敏感，可使用 tensorflow-cpu 包安装仅支持 CPU 的 TensorFlow 版本。（本文档是直接安装tensorflow-gpu）

在 Windows 下，需要打开开始菜单中的 “Anaconda Prompt” 进入 Anaconda 的命令行环境；

如果默认的 pip 和 conda 网络连接速度慢，可以尝试使用镜像，将显著提升 pip 和 conda 的下载速度（具体效果视您所在的网络环境而定）；

-{zh-hant: 北京清華大學；zh-hans: 清华大学；}- 的 pypi 镜像：https://mirrors.tuna.tsinghua.edu.cn/help/pypi/

-{zh-hant: 北京清華大學；zh-hans: 清华大学；}- 的 Anaconda 镜像：https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/

如果对磁盘空间要求严格（比如服务器环境），可以安装 Miniconda ，仅包含 Python 和 Conda，其他的包可自己按需安装。Miniconda 的安装包可在 这里 获得。

如果在 pip 安装 TensorFlow 时出现了 “Could not find a version that satisfies the requirement tensorflow” 提示，比较大的可能性是你使用了 32 位（x86）的 Python 环境。请更换为 64 位的 Python。可以通过在命令行里输入 python 进入 Python 交互界面，查看进入界面时的提示信息来判断 Python 是 32 位（如 [MSC v.XXXX 32 bit (Intel)] ）还是 64 位（如 [MSC v.XXXX 64 bit (AMD64)] ）来判断 Python 的平台。


# TensorFlow 基础
## 前置知识
- Python 基本操作 （赋值、分支及循环语句、使用 import 导入库）；
- Python 的 With 语句 ；
- NumPy ，Python 下常用的科学计算库。TensorFlow 与之结合紧密；
- 向量和矩阵运算（矩阵的加减法、矩阵与向量相乘、矩阵与矩阵相乘、矩阵的转置等；
- 函数的导数 ，多元函数求导；
- 线性回归；
- 梯度下降方法 求函数的局部最小值。

## 相关内容链接
内容包括TensorFlow 1+1、自动求导机制、基础示例：线性回归  
[TensorFlow 基础](https://tf.wiki/zh_hans/basic/basic.html)

# TensorFlow 模型建立与训练
如何使用 TensorFlow 快速搭建动态模型。
## 前置知识
- Python面向對象编程 （在 Python 内定义类和方法、类的继承、构造和析构函数，使用 super () 函数调用父类方法 ，使用__call__() 方法对实例进行调用 等）；
- 多层感知机、卷积神经网络、循环神经网络和强化学习；
- Python 的函数装饰器 （非必须）。

## 相关内容链接
[TensorFlow 模型建立与训练](https://tf.wiki/zh_hans/basic/models.html)
# TensorFlow 常用模块
## 前置知识
Python 的序列化模块 Pickle （非必须）
Python 的特殊函数参数（非必须）
Python 的迭代器

## 相关内容链接
[TensorFlow 常用模块](https://tf.wiki/zh_hans/basic/tools.html)

### 数据集的构建与预处理
[实例：猫狗图像分类](https://tf.wiki/zh_hans/basic/tools.html#zh-hans-cats-vs-dogs)


# TensorFlow 模型导出
## 相关内容链接
[TensorFlow模型导出](https://tf.wiki/zh_hans/deployment/export.html)

# TensorFlow Serving
## 相关内容链接
[TensorFlow Serving](https://tf.wiki/zh_hans/deployment/serving.html)

# 参考资料与推荐阅读 
本文为参考https://tf.wiki 给出的技术手册入门知识精简后整理的入门笔记，参考和借鉴了csdn上多个博客提供的安装部署入门文档，成功搭建了TensorFlow环境并对其训练流程和部署流程进行熟悉，完成猫狗图像分类的实例训练。

后续若想将该技术应用于实际业务中，还需要针对实践的内容补充更多的机器学习和深度学习的理论知识。

原理性相关书籍：
* 李航. `统计学习方法 <https://book.douban.com/subject/10590856/>`_ . 清华大学出版社, 2012. （有课件可在 `清华大学出版社网站 <http://www.tup.tsinghua.edu.cn/booksCenter/book_08132901.html>`_ 下载，点击“资源下载”-“网络资源”即可。有书中算法的 `GitHub开源代码实现 <https://github.com/fengdu78/lihang-code>`）
* 周志华. `机器学习 <https://book.douban.com/subject/26708119/>`_ . 清华大学出版社, 2016. （有辅助在线资料 `南瓜书 <https://datawhalechina.github.io/pumpkin-book>`_ ，`后续进阶版《机器学习理论导引》 <https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/MLthybook2020.htm>`_ ）
* 邱锡鹏. `神经网络与深度学习 <https://nndl.github.io/>`_ . 机械工业出版社, 2020.（有在线版本可阅读）

更具实践性的内容书籍：
* Aurélien Géron. `机器学习实战：基于Scikit-Learn和TensorFlow <https://book.douban.com/subject/30317874/>`_ . 机械工业出版社, 2018.
* 郑泽宇, 梁博文, and 顾思宇. `TensorFlow：实战Google深度学习框架（第2版） <https://book.douban.com/subject/30137062/>`_ . 电子工业出版社, 2018.
* 阿斯顿·张（Aston Zhang）, 李沐（Mu Li）, 扎卡里·C. 立顿 等. `动手学深度学习 <https://zh.d2l.ai/index.html>`_ . 人民邮电出版社, 2019. （有在线版本可阅读）

相对生动的视频讲解：
* `台湾大学李宏毅教授的《机器学习》课程 <https://www.bilibili.com/video/av10590361>`（ `讲义点此 <http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML17.html>` 中文，讲解生动且更新及时）
* `谷歌的《机器学习速成课程》 <https://developers.google.cn/machine-learning/crash-course/>`_ （内容已全部汉化，注重实践）
* `Andrew Ng的《机器学习》课程 <https://www.bilibili.com/video/av29430384>`_ （英文含字幕，经典课程，较偏理论，网络上可搜索到很多课程笔记）
