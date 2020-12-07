



@茹姐： 
  #图片风格迁移原理：
以风格为目标利用模型对输入做推理的过程。
执行推理用到两个网络：生成网络层和损失网络层
推理过程：
   输入IMG -->生成网络-->VGG-16(损失网络)-->输出
   通过OLAVHN的网络进行推理（原作者GIT:https://github.com/OlavHN/fast-neural-style）
同样产生噪点，其原因是PADDING过小，但如果要修改PADING要连同层和模型一起修改。
有无其他办法呢？有，先将图片放大再进行卷积。（什么是卷积？卷积核就是图像处理时，给定输入图像，输入图像中一个小区域中像素加权平均后成为输出图像中的每个对应像素，其中权值由一个函数定义，这个函数称为卷积核。）
卷积层只是神经网络模型层设计的一种，大致列举一些常规层如后，通过各个层推理计算后，
最后进行归一和输出处理。为了使输出接近目标结果及防止过拟合，TENSORFLOW 中y = f(x1 × w1 + x2 × w2 + b)
引入了alpha和Beta概念，在输出中也会用到MNS非极大值抑制之类的手段。如MOBILE net中各层的权重都由
alpha和Beta来控制权重以减少输入特征用于减少推理运算。y=f(a*x1*w1+a*x2*w2+b）
其中的X相当于输入层因子，W1为该因子的权重，X2，Xn同理。
（基本原理参考文档：https://blog.csdn.net/weixin_33788244/article/details/94586637）
模型中的基本层设计，入后，详细运算方式，百度。
 ===================================================
   CONVOLUTIONAL, 卷积层 产特征 离线二维滤波器
    DECONVOLUTIONAL, 解卷积层
    CONNECTED,  连接层
    MAXPOOL, 最大池化层 取大值
    SOFTMAX,回归层 特征分类概率
    DETECTION,概率偏导置信度损失
    DROPOUT,子层筛选（训练使用，推理不用）
    CROP,裁剪层
    ROUTE,多层嵌套路由
    COST,代价函数
    NORMALIZATION,归一化变量
    AVGPOOL,平均池化层 步长为间隔，池化size为长度
    LOCAL,局部独立卷积
    SHORTCUT,直连层
    ACTIVE,激活函数
    RNN,循环网络
    GRU,门循环单元LSTM变体
    CRNN,端到端网络层
    BATCHNORM,归一化减少震荡加速训练
 =====================================================
通过使用TENSORFLOW扩展库SLIM及resize_conv2d修改图片尺寸
SLIM地址：https://github.com/tensorflow/models/tree/master/research/slim
resize_conv2d：https://blog.csdn.net/liangjiu2009/article/details/106549926
#通过RESIZE conv2d相当于扩充了视野，因此而在输出中减少了噪声。
======================================================================================================

# fast-neural-style-tensorflow

A tensorflow implementation for [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155).

This code is based on [Tensorflow-Slim](https://github.com/tensorflow/models/tree/master/slim) and [OlavHN/fast-neural-style](https://github.com/OlavHN/fast-neural-style).

## Samples:

| configuration | style | sample |
| :---: | :----: | :----: |
| [wave.yml](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/conf/wave.yml) |![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/style_wave.jpg)|  ![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/wave.jpg)  |
| [cubist.yml](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/conf/cubist.yml) |![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/style_cubist.jpg)|  ![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/cubist.jpg)  |
| [denoised_starry.yml](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/conf/denoised_starry.yml) |![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/style_denoised_starry.jpg)|  ![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/denoised_starry.jpg)  |
| [mosaic.yml](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/conf/mosaic.yml) |![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/style_mosaic.jpg)|  ![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/mosaic.jpg)  |
| [scream.yml](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/conf/scream.yml) |![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/style_scream.jpg)|  ![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/scream.jpg)  |
| [feathers.yml](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/conf/feathers.yml) |![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/style_feathers.jpg)|  ![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/feathers.jpg)  |
| [udnie.yml](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/conf/udnie.yml) |![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/style_udnie.jpg)|  ![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/udnie.jpg)  |

## Requirements and Prerequisites:
- Python 2.7.x
- <b>Now support Tensorflow >= 1.0</b>

<b>Attention: This code also supports Tensorflow == 0.11. If it is your version, use the commit 5309a2a (git reset --hard 5309a2a).</b>

And make sure you installed pyyaml:
```
pip install pyyaml
```

## Use Trained Models:

You can download all the 7 trained models from [Baidu Drive](https://pan.baidu.com/s/1i4GTS4d).

To generate a sample from the model "wave.ckpt-done", run:

```
python eval.py --model_file <your path to wave.ckpt-done> --image_file img/test.jpg
```

Then check out generated/res.jpg.

## Train a Model:
To train a model from scratch, you should first download [VGG16 model](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz) from Tensorflow Slim. Extract the file vgg_16.ckpt. Then copy it to the folder pretrained/ :
```
cd <this repo>
mkdir pretrained
cp <your path to vgg_16.ckpt>  pretrained/
```

Then download the [COCO dataset](http://msvocds.blob.core.windows.net/coco2014/train2014.zip). Please unzip it, and you will have a folder named "train2014" with many raw images in it. Then create a symbol link to it:
```
cd <this repo>
ln -s <your path to the folder "train2014"> train2014
```

Train the model of "wave":
```
python train.py -c conf/wave.yml
```

(Optional) Use tensorboard:
```
tensorboard --logdir models/wave/
```

Checkpoints will be written to "models/wave/".

View the [configuration file](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/conf/wave.yml) for details.
