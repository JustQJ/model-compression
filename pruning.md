## 剪枝
剪枝一般分为非结构化剪枝和结构化剪枝，主要作用就是减少模型权重。

### 非结构化剪枝
非结构化剪枝将权重矩阵的部分参数置零，从而减少参数数量。选择那些参数置零，以及如何更新剩余的参数以补偿置零参数造成的误差成为主要研究点。即优化
$$
\argmin_{M, \hat{W}} ||WX-(M \cdot \hat{W})X||_2
$$

其中$M$为mask，即$M_{ij}$为0表示将$W_{ij}$置零，即裁剪掉；否则，为1保持不变。而$\hat{W}$则表示重建后的权重矩阵，即裁剪会导致误差，而通过更新没有被裁剪掉的权重可以恢复部分误差。 因此，一般的优化要么只关注$M$如何确定，不更新$W$；要么同时更新$W$。

一般$M$的确定是通过对$W$中的每个参数计算其重要性，然后将重要性小的参数置零。一般的重要性是通过$W$的幅值确定，即$|W|$。

而$\hat{W}$则一般通过梯度或者其他方式更新。

非结构化剪枝的问题在于其可能无法得到理想的加速，即一个权重矩阵中某些参数置零，但是是不规则的，所以在实际的计算中无法得到很好的加速。

可以加速的是半结构化剪枝(semi-structured)，形成`n:m`的稀疏格式，然后可以通过Nvidia稀疏tensor核加速。`n:m`的稀疏格式就是每m个连续的权重中，由n个是0。


### 结构化剪枝
结构化剪枝一般就是去掉模型的一整个模型层或者权重矩阵的一行或者一列，这样可以直接获得加速，但是精度可能下降较多。



## SparseGPT
[SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot](https://arxiv.org/abs/2301.00774)
非结构化剪枝。






## Wanda
非结构化剪枝，通过激活值和权重的乘积来确定裁剪那些权重。即对于一个$X\in R^{L\times h}$和$W^{o\times h}$，$W_{ij}$的重要性分数计算为
$$
S_{ij} = |W_{ij}| \cdot||X_{:,j}||_2
$$
其实就是将$X$的第j列做一个L2 Norm，然后乘上对应位置的权重作为其分数。然后对于$W$的每一行选择部分的权重置0，而不是在整个权重进行选择。每行有$h$个元素，置零的比例为$\alpha$，那么就是把每一行根据$S$排序，然后将前面$h*a$个元素置零。 代码实现也简单，即
```python
import torch
h = 1024
bs_L = 512
o = 1024
ratio = 0.5
W = torch.randn((o,h))
X = torch.randn((bs_L, h))
Score = W.abs() * X.norm(p=2, dim=0)

_, sorted_idx = torch.sort(Score, dim=1) ##排序每一行
pruned_idx = sorted_idx[:, :int(h*ratio)] ## 选择前面小的元素
W.scatter_(dim=1, index=pruned_idx, src=0) ## 原本权重中对应的idx置零
```


## LLM-Pruner
结构化剪枝，去掉权重矩阵的一行或者一列。由于直接去掉矩阵的一行或者一列，会造成输入和输出的维度发生变化，所以需要根据依赖关系，多个神经元形成一个组，每次去掉整个组。

LLM-Pruner有三个阶段，包括分组，评估组的重要性，以及训练恢复。

### 分组
根据依赖关系，LLM-Pruner将Transformer层分成3种组，包括MLP中的组，MHA中的组，以及channel-wise的组。

**MLP的计算**包括三个矩阵，up, gate, down，其实就是
$$
y = down( up(x)*gate(x))
$$
分组就是up一列，gate的一列，以及down的一行形成一组。这样，删去一个或者多个组，MLP整个模块的输入和输出的维度不会发生变化。例如，删除k个组，有
$$
y^{l\times o} = (x^{l\times h1} up^{h1\times h2}) * (x^{l\times h1}gate^{h1\times h2}) down^{h2\times o} \\
y^{l\times o} = (x^{l\times h1} up^{h1\times (h2-k)}) * (x^{l\times h1}gate^{h1\times (h2-k)}) down^{(h2-k)\times o} 
$$


**MHA的计算**是以多个head为核心，即
$$
Q = q(x), K = k(x), V = v(x) \\
y = o(mAtten(Q,K,V)) \\
mAtten(Q,K,V) = Contact(Atten(Q_i,K_i,V_i)), i=1 .. n \\
Q_i = Q_{:, i*s:(i+1)*s}, K_i = K_{:, i*s:(i+1)*s} , V_i = V_{:, i*s:(i+1)*s} 
$$
这里表示有n个head，把Q, K, V按列划分成n份，每份有s列，每份单独计算一个head，然后把结果拼接起来通过输出矩阵o。这里的一个组就是以一个head为核心，一个head加上q,k,v的对应的列以及o的对应的行形成一个组。例如，删除第k个组，就是删去q,k,v的`k*s:(k+1)*s`列，以及o的`k*s:(k+1)*s`行。保证了MHA整个输入和输出的维度不变，不会影响前面和后序的计算。

**channel-wise的组**是指从模型的开始到模型的结尾，如果删除最后的head的一个输入神经元（一行），那么就会传递到整个模型对应的位置进行删除，跨越整个模型，而不是局限在一个层。

论文中将MLP的组和MHA的组归为一类（**block**），然后channel-wise的组归为一类。对于一个模型只裁剪一类，**block效果较好（依赖链短）**，channel-wise的组性能较差。

### 组的重要性评估

同样是依赖于梯度，使用**二阶（文中实验二阶效果更好）** 或者一阶导数来评估。每个组由多个$W_i$组成，即权重行或者列，一个组的重要性由所有的$W_i$的确定。即先单独计算一个组中每个$W_i$的重要性，然后再通过合并函数合并起来。$W_i$的重要性可以直接由其梯度计算，**也可以由单个元素$W_i^k$的梯度进行累加计算（文中实验这种的效果较好）**。

有了一个组中每个$W_i$的重要性，然后进行合并，文中使用了四种合并方式：（1）相加；（2）相乘；（3）取最大值；（4）**取最后一个$W_i$的重要性**，即在这个组中的计算依赖图中的最后一个$W_i$。 文中实验显示方法（4）相对效果最好。

有了重要性后，就可以对所有的组排序，然后根据预定的稀疏性删去重要性低的组。对于排序的范围，可以是整个模型，不同层，不同类别的组一起排序，**也可以是每层，同一个类别单独排序**。文中实验显示后者较好，可能因为不同层和不同的类别的重要性的相对性不同。 

**文中还显示前几层和最后一层的影响大，所以前几层和最后一层不裁剪，中间层多裁剪一些。**

### 训练恢复

使用lora进行训练恢复。


## Sheared LLaMA
[Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning](https://arxiv.org/abs/2310.06694)                  


## LoRAPrune
[LoRAPrune: Structured Pruning Meets Low-Rank Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2305.18403)