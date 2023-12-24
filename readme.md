# Instructor on Final Project

File Structures

```bash
├─.idea
│  └─inspectionProfiles
├─.ipynb_checkpoints
├─cpp
│  ├─AutoLinearEncoder.ipynb(Deprecated)
│  ├─LinearAutoEncoder.h
│  └─PY_KMEANS.h
├─py
│  ├─mKMeans.py
│  ├─mPCA.py
│  ├─PY_LinearAutoEncoder.py
│  └─__pycache__
├─statics
├─__pycache__
├─AdvancedKMeans.ipynb
├─BaseImageProcesser.ipynb
├─ImageClassifier.ipynb
├─ImageClassifier.ipynb
├─KMeans.ipynb
├─LinearEncoder.ipynb
├─PCA.ipynb
├─seeds_dataset.txt
└─readme.md
```

# File details

## package py

`mKMeans.py`: Implimention of KMeans, KMeans++, SoftKMeans, EnhancedKMeans(split-merge) and EnhancedSoftKMeans. About **Q1, Q2.**

`mPCA.py`: PCA. About **Q5**.

`PY_LinearAutoEncoder.py`: Linear autoencoder. About **Q6.**

## package cpp

`AutoLinearEncoder.ipynb`: **Deprecated.** A test sample for binary classification of characters.

`LinearAutoEncoder.h` Infact, this is an encoder **with Non-linear method**. However, the jupyter notebook contains no refactor methods at once, making it hard to change the name, so I kept its name. It's **Non-linear encoder.**

`PY_KMEANS.h` Kmeans implimention on cpp.

## root package

* `AdvancedKMeans.ipynb`: classify wheel seeds. About **Q3**, **Q4**
* `BaseImageProcesser.ipynb`: PCA method as well as the linear autoencoder method to find the basis of principal components. About **Q7.**

* `ImageClassifier.ipynb`: Deprecated, bcus the author hasn't get any result in this file...QwQ
* `ImageEncoder.ipynb`: Classify characters in Genshin Impact.
* `KMeans.ipynb`: Simple test on KMeans Algorithm. About **Q1, Q2**, but it's not easy to read...
* `LinearEncoder.ipynb`: Test on linear encoder, for **Q6**.
* `PCA.ipynb`: Deprecated.

# Questions guild

**Q1 & Q2:** `py\mKMeans.py` 

**Q3 & Q4:** `AdvancedKMeans.ipynb`. To see realization of merge-split method, refers to `py\mKMeans.py`.

**Q5:**  `py\mPCA.py`,  `BaseImageProcesser.ipynb`(lower the dimension of the seeds) and `BaseImageProcesser.ipynb`

**Q6:** `LinearEncoder.ipynb`, `BaseImageProcesser.ipynb` and `py\PY_LinearAutoEncoder.py`

**Q7:** `BaseImageProcesser.ipynb`

# Advanced Discussions

既然是超前讨论，那么就用中文来讲了。

* 原神角色识别

本次有对原神角色头像识别的讨论，为了获得更高的效率，使用到了基于C++的非线性编码器，如果希望尝试，可能需要您配置C++的环境。另外笔者要声明的是，这是一个进一步的讨论，如果您没有配置出C++的环境而没有跑出这一部分的实验结果，请不要就此给实验者扣分。安全起见，作者直接将其安放在头文件中，从而解决了需要编写cmakelist可能造成包冲突、不兼容等问题

cppyy的安装：

```bash
pip install cppyy
```

在`ImageEncoder.ipynb`中，有基于非线性编码器对原神角色头像识别的实现。不过它是基于黑白图像来进行的几何特征识别，对于二次元画风相对统一的角色来说，训练失误率较高，尤其是在没有颜色作为训练输入的情况下，因此测试的时候请尽量选用差距较大一些的角色头像（主要是我太菜了写不了好的classifier呜呜呜）来测试。

超参数尽量别改（主要是我不会改TvT，只会暴力测试），风险很大，维度太高了，没有写比较好的特征提取算法或者降维的算法，到处都是局部最优解，不出意外的话很容易翻车。运气好十次里大概能跑出七八次成功。

* Hebbian学习法则的探索

在`__init__.py`中有一个搜罗来的Hebbian学习法则的探索。通过对参图案进行学习，将其部分特征遮盖后，hebbian神经网络会将其往收敛的方向进行计算，还原原来的图案。由于时间和精力有限，这一部分被带过。

# Requirements

`skylearn`, `numpy`, `cv2`, `Image`, `matplotlib`, `cppyy`(optional), `ctypes`(optional)

Details， ples refer: [Ethylene9160/SDM274_Final: Final project of SDM274 2023 fall (github.com)](https://github.com/Ethylene9160/SDM274_Final).

# Else

图片放的有点多，后面有的图已经排到老后面去了qwq，文字描述可能在图的前几页了qwq，呜呜呜。同时也因为时间不够，没空整理代码了，所以附录里只放了几个陈年老代码emmm，详细的实现可能得去源文件的屎山里看了TvT

Best Wishes.