

# SVM算法
<img src="https://github.com/xiaoxingchen505/Machine_Learning/blob/main/images/svm1.png" width="600" height="300">


## 有关于向量内积 Refresh
<img src="https://github.com/xiaoxingchen505/Machine_Learning/blob/main/images/svm2.png" width="600" height="300">

## SVM的推导相关
<img src="https://github.com/xiaoxingchen505/Machine_Learning/blob/main/images/svm3.png" width="600" height="300">

<img src="https://github.com/xiaoxingchen505/Machine_Learning/blob/main/images/svm4.png" width="600" height="300">

* 这个公式左边w和b是两个参数，可以被调整，无论右边的数是什么样的，都可以通过左边的变换来变成求解w* x+b = 1和 w* x+b = -1

* 右边公式结合上图看，x1和x2是向量，x1所在的向量减去x2所在的向量，然后取模长是这图中所示两个向量上的箭头之间的距离。w是wx+b的法向量，于是||x1-x2||的值乘以cos(Θ)就等于d (其中θ是||x1-x2||与d的夹角)，这里d就是w* x+b = 1和w* x+b = -1两条线的距离，也就是SVM算法要求解的关键。


## 公式



