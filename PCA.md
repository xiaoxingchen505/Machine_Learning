

# PCA算法


## principal component analysis ( 主成分分析)
PCA（Principal Component Analysis，主成分分析）是一种常用的数据分析方法。PCA通过线性变换将原始数据变换为一组各维度线性无关的表示，可用于提取数据的主要特征分量，常用于高维数据的降维

案例 ：假设有一个100 x 4的矩阵，代表100个数据里，每个数据有4个特征，我们觉得4个特征分得不是特别好，我们的目标是把这个矩阵降到2个特征，也就是变成100 x 2的矩阵，通过矩阵乘法，我们知道通过引入一个4 x 2 的矩阵，让100 x 4矩阵和4 x 2 的矩阵相乘，最后得到的是 100 x 2的矩阵，也就是我们想要的结果。在PCA算法中，我们的核心问题就是先需要找到这个4 x 2的矩阵。

## PCA的应用范围
PCA的应用范围有：

1. 数据压缩

    * 数据压缩或者数据降维首先能够减少内存或者硬盘的使用， 如果内存不足或者计算的时候出现内存溢出等问题， 就需要使用PCA获取低维度的样本特征。

    * 其次， 数据降维能够加快机器学习的速度。 



2. 数据可视化

    * 在很多情况下， 可能我们需要查看样本特征， 但是高维度的特征根本无法观察， 这个时候我们可以将样本的特征降维到2D或者3D， 也就是将样本的特征维数降到2个特征或者3个特征， 这样我们就可以采用可视化观察数据。

## 协方差


协方差（Covariance）在概率论和统计学中用于衡量两个变量的总体误差，在下面公式中，可以计算出两个特征之间的值存在正相关或者负相关。

<img src="https://github.com/xiaoxingchen505/Machine_Learning/blob/main/images/cov1.png" width="300" height="100">

<img src="https://github.com/xiaoxingchen505/Machine_Learning/blob/main/images/cov2.png" width="300" height="100">

## PCA的计算过程

* 首先要对训练样本的特征进行归一化

```python
from sklean.preprocessing import StandardScaler
X_std = StandardScaler().fit_transform(X) # 假设X是我们的输入
```

* 计算协方差矩阵

```python
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec))/(X_std.shape[0]-1)
```