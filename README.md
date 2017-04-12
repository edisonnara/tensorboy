# Tensorboy
这是一个用于tensorflow ( python3 + tensorflow1 ) 入门学习的project

非常入门, 只适合菜鸟看.

一来帮助自己理清思路, 二来为新手提供参考, 快速上手。

之所以取名"tensor boy", 是把tensorflow的学习过程比作一个学习的小男孩，帮助理解抽象的概念

# 环境
OS: Mac OS X 10.12

python环境：anaconda4.3 for mac

python版本：3.6

tensorflow: 1.0.1

IDE: PyCharm Community Edition 2017.1

# 安装
Step1: 安装anaconda. 

注意：mac下，我的root用户安装后无法使用，建议采用普通用户

前往官网https://www.continuum.io/downloads/下载mac版安装器

安装完成，默认Python版本自动切换为3.6

Step2: 安装tensorflow. 

进入命令行执行：

pip install --upgrade https://storage.googleapis.com/tensorflow/mac/tensorflow-1.0.1-py3-none-any.whl

Collecting tensorflow==1.0.1 from https://storage.googleapis.com/tensorflow/mac/tensorflow-1.0.1-py3-none-any.whl

注意，以上安装完成， tensorflow版本为1.0.1, 基于python3，mac，仅cpu版

# Kick Off
打开pycharm, VSC-->checkout-->github

是不是很简单，假如你的环境和笔者类似，那么就准备就绪，开工吧

# 代码学习路线
1，实现训练简单神经网络模型, 正确率92%, 并保存模型 - softmax.py

2, 加载之前训练好简单神经网络模型, 识别自己的手写数字 - restore_softmax.py

3，实现训练卷积神经网络模型, 正确率99.2%,  并保存模型 - cnn.py

4, 加载之前训练好卷积神经网络模型, 识别自己的手写数字 - restore_cnn.py


# 参考资料
tensorflow中文站 - http://www.tensorfly.cn/

《tensorflow实战》- 电子工业出版社

https://github.com/wlmnzf/tensorflow-train

# 联系
mail: 71226088@qq.com

QQ: 71226088



