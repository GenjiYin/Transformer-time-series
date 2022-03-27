# Transformer-
该项目将Transform模型运用于时间序列数据预测
Transform流程直接上图, 为确保代码阅读的流畅性, 我将从原理上详细讲解transform模型， 以及项目的整个流程
<div align=center><img src="https://github.com/GenjiYin/Transformer-/blob/main/filename/transform.jpg"/></div>

## 数据预处理
处理数据依旧是二维结构化数据， 横向为特征， 纵向为时间(本文以一维数据为例, 模拟三角复合函数数据进行建模)
1. 导入数据， 截取前80%的数据作为训练集， 后10%作为检验集， 剩下的10%作为验证集；
2. 将数据转化为三维格式， tensor的维度为(N, timestep, feature_number), 其中N+timestep为样本容量(N为送入模型的样本容量，N+timestep为整个时序的样本容量)， feature_number为样本的特征数， 这样的三维tensor为xtrain，ytrain为每个送入模型样本的最后一个时点的后一时点的数据， 所以ytrain的每个时点的数据都为xtrain对应时点的下一时点的第一个timestep的数据；

## 模型建立
### 模型讲解
#### Attention
