

# ID3、C4.5和CART决策树对比


## ID3决策树：利用信息增益来划分节点


信息熵是度量样本集合纯度最常用的一种指标。假设样本集合D中第k类样本所占的比重为pk，那么信息熵的计算则为下面的计算方式

![image](https://github.com/xiaoxingchen505/Machine_Learning/blob/main/images/eq1.png)

当这个Ent(D)的值越小，说明样本集合D的纯度就越高

有了信息熵，当我选择用样本的某一个属性a来划分样本集合D时，就可以得出用属性a对样本D进行划分所带来的“信息增益”