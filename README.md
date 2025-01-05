# 常见模型压缩的相关方法
量化、剪枝、知识蒸馏、低秩分解等


## 量化

**量化的主要好处：（1）减少模型的权重的内存占用；（2）减少计算过程中对内存带宽的需求。**

### 基本原理
将浮点矩阵量化到$k$位的int类型，就是做一个映射函数，即
$$
f(x) = x/s - z
$$
其中$x$ 是一个张量，$s$ 通过 $x$ 的最大最小值和$k$位int的最大最小值进行计算，即
$$
s = (max(x)- min(x)) / (2^k-1)\\
z = round(max(x)/s) - (2^k-1) \\
q_x = clip(round(x/s-z), 0, 2^k-1)
$$
这里是将$x$投射到$[0, 2^k-1]$，当然也可以是$[-2^{(k-1)}, 2^{(k-1)}-1]$，但是全是正数，实际的实现会更好。              
也可以使用对称映射，即将$[-m, m]$映射到$[-2^{(k-1)}+1, 2^{(k-1)}-1]$，有
$$
z = 0 \\
s = m/(2^{(k-1)}-1) \\
m = max(abs(max(x)), abs(min(x)))
$$



### 量化粒度

最大最小值映射是最基本的，现在的方法主要的优化基本就是优化离群值，即$x$中有一些值异常的大或者小，分布不均匀。 

**per-tensor quantization:** 一个张量矩阵使用同一个scale和zero，scale和zero占用的内存小，当然误差变大。

**per-channel quantization:** 一个通道使用一个scale和zero（对于一个二维权重矩阵，就是一行或者一列），scale和zero占用的内存大，当然误差变小。

**per-group quantization:** 对上面进行折中，将多个通道形成一个group，然后一个group使用一个scale和zero，例如一般使用128作为group的大小。也可以是多个元素形成一个group，即比per-channel粒度更小。


### 量化种类
1.只量化权重：在计算是需要先将权重反量化为激活的类型，然后计算。                      
2.权重+激活量化：在计算时不需要反量化，而是将激活也量化为权重的类型，然后进行计算。


### 基本步骤
1.使用量化函数对输入的tensor进行量化，将每个元素量化到intk的范围内。          
2.对量化的权重进行打包，使得一个32位的int可以存储多个intk的值。                      
3.使用量化的权重进行计算。可以先对量化的权重进行反量化，然后再进行计算，但是这样无法得到实际的计算加速。**为了得到实际的计算加速，我们需要实现相应的计算kernel，能够直接读取量化的权重，与输入进行计算。**     


### 参考

https://huggingface.co/docs/transformers/quantization/overview

https://iq.opengenus.org/basics-of-quantization-in-ml/

https://huggingface.co/docs/optimum/concept_guides/quantization

https://github.com/fpgaminer/GPTQ-triton/blob/main/quantize.py

https://github.com/AutoGPTQ/AutoGPTQ/blob/main/auto_gptq/nn_modules/qlinear/qlinear_tritonv2.py


## GPTQ量化
### 原理介绍
按照该算法的发展来历，依次介绍OBD->OBS->OBQ->GPTQ。
1. OBD (Optimal Brain Damage)
    OBD以网络的训练损失函数作为目标函数，研究如何使用二阶倒数来选择对目标函数影响最小的参数，然后对其进行剪枝。                                   
    使用 $L(W)$ 表示训练的损失函数，$W$表示当前训练好的网络的参数权重，是使得$L(W)$达到局部最优的局部最优解（这里把$W$看着一维列向量 $(w_0, w_1 , \cdots, w_i,\cdots)^T$，即$L(W)$是一个多元函数）。使用$\delta W$表示权重的变化，即需要找到使得$L(W)$变化最小的$\delta W$。在$W$处进行二阶泰勒展开，有
    $$
    L(W+\delta W) = L(W) + (\frac{\partial L}{\partial W})^T \delta W + \frac{1}{2} \delta W^T  H  \delta W + O(||\delta W||^3)\\
    H =  \frac{{\partial}^2 L}{\partial W^2} \\

    \delta L = (\frac{\partial L}{\partial W})^T \delta W + \frac{1}{2} \delta W^T  H  \delta W + O(||\delta W||^3)
    $$      
    由于$W$是局部最优解，所以其一阶偏导数均为0，同时高阶项忽略，得到
    $$
    \delta L = \frac{1}{2} \delta W^T  H  \delta W = \frac{1}{2} \sum_i \delta w_i^2h_{ii} + \frac{1}{2} \sum_{i!=j} \delta w_ih_{ij}\delta w_j
    $$
    **OBD假设各个参数之间是相互独立，即单独裁剪每个参数对目标函数的影响的和与同时裁剪多个参数的影响是相同的**。则Hessian矩阵是一个对角阵，非对角线上的元素均为0（$h_{ij}=0, i!=j$），所以上面的式子化简为
    $$
    \delta L = \frac{1}{2} \sum_i \delta w_i^2h_{ii}
    $$

    自此，目标就变成找到使得 $\delta L$最小的$w_i$，对其进行裁剪（即$\delta w_i = w_i, \delta w_j = 0(j!=i)$），即
    $$
    \argmin_{w_i} \frac{1}{2}  \delta w_i^2h_{ii}
    $$

    OBD的整个步骤为：            
    (1) 构建一个网络            
    (2) 训练该网络到最优解                     
    (3) 计算每个元素二阶偏导$h_{ii}$，计算方式也是梯度反向传播                        
    (4) 然后计算每个元素的 $\frac{1}{2} w_i^2h_{ii}$
    (5) 根据(4)中的结果，选择值最小的那个元素，删除。删除的方式就是设置成0，并在后序的计算过程中固定住，不参与更新。
    (6) 重复(2)到(5)直到删除的数量达到预定值。      
    

2. OBS (Optimal Brain Surgeon)
    OBD假设了不同参数互不影响，OBS放弃了这种假设，继续从假设前的公式进行推导，即
    $$
    \delta L = \frac{1}{2} \delta W^T  H  \delta W 
    $$
    假设从一个参数$w_q$进行剪枝（$w_q+\delta w_q = 0$），那么就有
    $$
    \min_q\{\min_{\delta W}(\frac{1}{2} \delta W^T  H  \delta W ) | e^T_q\delta W + w_q = 0\}
    $$
    其中$e^T_q$表示第$q$个位置为1，其余位置为0的向量。使用拉格朗日乘数去掉约束，有
    $$
    {\cal{L}} = \frac{1}{2} {\delta W^T  H  \delta W} + \lambda (e^T_q\delta W + w_q)
    $$
    求极值，偏导数为0，即
    $$
    \frac{\partial \cal{L}}{\partial \delta W} =H\delta W  +\lambda e_q = 0 \\
    \delta W = -\lambda H^{-1}  e_q \\
    \delta w_q = -\lambda [H^{-1}]_{qq}
    $$
    其中$[H^{-1}]_{qq}$表示$H$的逆矩阵的第q行q列的元素。则代入$w_q+\delta w_q = 0$，得到
    $$
    w_q = \lambda [H^{-1}]_{qq} \\
    \lambda = \frac{w_q}{[H^{-1}]_{qq}}
    $$
    然后得到
    $$
    \delta W = -\lambda H^{-1}  e_q = -\frac{w_q}{[H^{-1}]_{qq}} H^{-1}  e_q = -\frac{w_q}{[H^{-1}]_{qq}} [H^{-1}]_{:,q}
    $$
    其中 $[H^{-1}]_{:,q}$ 表示第q列。从而得到原本的损失为
    $$
    \delta L = \frac{1}{2} \delta W^T  H  \delta W  \\
    = \frac{1}{2} (-\frac{w_q}{[H^{-1}]_{qq}} H^{-1}  e_q)^T H (-\frac{w_q}{[H^{-1}]_{qq}} H^{-1}  e_q) \\
    = \frac{1}{2}  \frac{w_q^2}{([H^{-1}]_{qq})^2} e_q^T (H^{-1})^T  H H^{-1}  e_q \\ 
    = \frac{1}{2}  \frac{w_q^2}{([H^{-1}]_{qq})^2} e_q^T (H^{-1})^T e_q \\
    = \frac{1}{2}  \frac{w_q^2}{([H^{-1}]_{qq})^2} [H^{-1}]_{qq} \\
    = \frac{1}{2} \frac{w_q^2}{[H^{-1}]_{qq}}
    $$

    因此，我们得到
    $$
    \delta L  = \frac{1}{2} \frac{w_q^2}{[H^{-1}]_{qq}} \quad (1) \\
    \delta W = -\frac{w_q}{[H^{-1}]_{qq}} [H^{-1}]_{:,q} \quad (2)\\
    $$
    每次选择使得 $\delta L$最小的 $w_q$移除，然后计算出向对应的$ \delta W$，更新其余的权重，不断重复。即OBS在去除权重后，会对剩余的权重进行更新，补偿去除权重带来的误差。对比原来的OBD，目标函数从乘以$H_{qq}$变成了除以$[H^{-1}]_{qq}$，而$[H^{-1}]_{qq}$实际上是受到其他项的影响，而不是只受到$H_{qq}$的影响。        
    OBS的整个步骤为：              
    (1) 训练网络到最优解             
    (2) 计算$H^{-1}$        
    (3) 根据公式$(1)$计算每个参数的$\delta L $，选择最小参数进行裁剪         
    (4) 根据选择的$w_q$和公式$(2)$计算出$\delta W$，然后更新其他剩余参数，即 $W = W+\delta W$            
    (5) 重复(2)到(4)直到满足条件              

    $H$ 每次求逆需要有$O(d^3)$的时间复杂度，d表示参数的数量，迭代$d$次，总共为$O(d^4)$。


3. OBQ
    OBC的论文中对OBS的具体实现进一步优化，并提出将该方法应用到量化中，形成OBQ。                  
    将$W\in R^{d_{r}\times d_c}$当作一个二维矩阵权重，$\hat{W}$为剪枝后的权重，那么对该层神经网络剪枝，优化目标为  
    $$
    \argmin_{\hat{W}} ||WX-\hat{W}X||^2
    $$
    即剪枝后和原矩阵输出误差最小，其中$X\in R^{d_c \times N}$为输入数据。根据原本的OBS，得到
    $$
    \delta L  = \frac{1}{2} \frac{w_q^2}{[H^{-1}]_{qq}} \quad (1) \\
    \delta W = -\frac{w_q}{[H^{-1}]_{qq}} [H^{-1}]_{:,q} \quad (2)\\
    $$
    对于每个$W\in R^{d_{r}\times d_c}$，$H\in R^{d\times d}$，时间复杂度为$O(d^4), d=d_r * d_c$。                           

    对权重矩阵按行进行分解，得到
    $$
    \delta L = \sum_i \delta L_i = \sum_{i=1}^{d_r} ||W_{i,:}X - \hat{W}_{i,:}X||^2
    $$
    即$W$的不同行之间的元素相互不影响，所以可以单独考虑每一个行，而不是整个矩阵。本质上就是 
    $$
    y_{ij} = \sum_{k=1}^{d_c} w_{ik}x_{kj} \\
    \delta L = \sum_i \sum_j (y_{ij} - \hat{y}_{ij})^2
    $$
    $y_{ij}$ 只受到同一行的$W_{i,:}$影响，而不会受到其他行的影响。
    单独对每行求解，那么H矩阵也只需要单独对每个行计算的即可，即 $H\in R^{d_c \times d_c}$，然后有$d_r$个H。对于单独一行 $W_{i,:}$，对其求H矩阵
    $$
    \delta L_i = ||W_{i,:}X - \hat{W}_{i,:}X||^2 \\
    \frac{\partial \delta L_i}{\partial \delta W_{i}} = 2\delta W_{i} XX^T \\
    H = 2XX^T
    $$
    因此，对于矩阵$W$，其每一行的H矩阵相同，均为 $2XX^T$。          
    在迭代过程中，对于每一行的计算是独立的，所以开始时所有行的$H$均为$2XX^T$，但是在后序迭代中不同的行是不一样的。因此，每行的时间复杂度为$O(d_c * d_c^3)$，即$d_c$次迭代，每次迭代需要$O(d_c^3)$的时间计算$H^{-1}$。每次裁剪后一个元素后，需要重新计算$H$和其逆矩阵。$H$其实不需要重新计算，因为移除一个$p$位置上的元素，就是移除$H$的第$q$行和$q$列。为了更加高效地计算$H^{-1}$，而不是每次花费$O(d_c^3)$，论文给出了下面$O(d_c^2)$的更新方法，其中$H^{-1}_{-q}$表示去掉第$q$行和$q$列
    $$
    H_{-q}^{-1} = (H^{-1} - \frac{1}{[H^{-1}]_{qq}}H^{-1}_{:,q}H^{-1}_{q,:})_{-q} \quad (3)
    $$
    该公式通过高斯消元得到的，即通过高斯消元将$H^{-1}$的第q行和第q列置为0，然后移除。
    $$
    [H^{-1}_{:,q}H^{-1}_{q,:}]_{ij}  = [H^{-1}]_{ij}*[H^{-1}]_{qq} \quad \text{when} \quad i=q || j=q  \\
    [\frac{1}{[H^{-1}]_{qq}}H^{-1}_{:,q}H^{-1}_{q,:}]_{ij} = [H^{-1}]_{ij}\quad \text{when} \quad i=q || j=q
    $$
    所以相减后$H^{-1}$的第q行和第q列就变成了0。由于 $H^{-1}H=I$，在等式两边进行上面的相同的高斯变换，使得该等式变成了
    $$
    \begin{bmatrix}
    H^{-1}_{1} & 0 & H^{-1}_{2} \\
    0 & H^{-1}_{qq} & 0 \\
    H^{-1}_{3} & 0 & H^{-1}_{4}
    \end{bmatrix}  \begin{bmatrix}
    H_{1} & h_1 & H_{2} \\
    h_4 & H_{qq} & h_2 \\
    H_{3} & h_3 & H_{4}
    \end{bmatrix}  = \begin{bmatrix}
    I & i_{1} & 0 \\
    i_4 & 1 & i_{2} \\
    0 & i_3 & I
    \end{bmatrix}
    $$
    即$I$的第q行和q列除了$I_{qq}$，其余均为0，所以经过高斯变换后，只有第q行和第q列发生了变化，其余均保持不变。所以对于上面的式子，去掉H的第q行和第q列后
    $$
      \begin{bmatrix}
    H^{-1}_{1}  & H^{-1}_{2} \\
    H^{-1}_{3}& H^{-1}_{4}
    \end{bmatrix}  \begin{bmatrix}
    H_{1} & H_{2} \\
    H_{3}& H_{4}
    \end{bmatrix}  = \begin{bmatrix}
    I &  0 \\
    0 &  I
    \end{bmatrix} = I \\
    \begin{bmatrix}
    H^{-1}_{1}  & H^{-1}_{2} \\
    H^{-1}_{3}& H^{-1}_{4}
    \end{bmatrix}  = (H_{-q})^{-1}
    $$
    所以，证明了通过公式(3)得到的结果就是H去掉q行和q列的逆。

    因此，对于$W$的每一行，OBC通过如下步骤移除$k$个元素          
    (1) 计算$H^{-1} = (2XX^T)^{-1}$          
    (2) 每次通过公式(1)计算剩余元素中误差最小的                        
    (3) 通过公式(2)更新其余元素                
    (4) 通过公式(3)更新$H^{-1}$               
    (5) 移除该元素        
    (6) 重复(2)-(5)直到k个元素被移除          
    如果直接针对整个$W$进行剪枝，那么就是每次找到所有行中使得误差最小的元素，去掉该元素，更新该元素所在行的逆矩阵和其余元素。总体的复杂度为$O(d_r * d_c^3)$。


    OBQ就是将上面的思想应用到量化上面。在剪枝中，我们将$w_q$变成0，所以有$\delta w_q + w_q = 0$，而在量化中，我们将$w_q$变成了$quant(w)$，所以有$\delta w_q + w_q = quant(w)$。这里的$ quant(w)$不是指量化后的整数，而是指反量化回来的数，因为我们在实际的计算中是使用反量化的数进行计算，即
    $$
    quant(w) = scale((clamp(\frac{w}{scale} - zero))+zero)
    $$
    本质上就是$clamp$的操作丢失了一部分信息，造成$w$和$quant(w)$之间有误差。使用$\delta w_q + w_q = quant(w_q)$约束，原本的拉格朗日乘法就变成了
    $$
    {\cal{L}} = \frac{1}{2} {\delta W^T  H  \delta W} + \lambda (e^T_q\delta W + w_q-quant(w_q))
    $$
    因此可以得到
    $$
    \delta L  = \frac{1}{2} \frac{(quant(w_q)- w_q)^2}{[H^{-1}]_{qq}} \quad (1) \\
    \delta W = -\frac{w_q-quant(w_q)}{[H^{-1}]_{qq}} [H^{-1}]_{:,q} \quad (2)\\
    $$
    其实就是使用现在的$\delta w_q=(quant(w_q)- w_q)$替换掉原来的$-w_q$。          
    然后其他的步骤就和剪枝相同，只是计算的公式变了。

4. GPTQ               
    GPTQ是OBQ的加速改良版。
    **(1) 按顺序量化** 
    OBQ每一行的量化顺序不同（每行寻找使得$\delta L$最小的权重），整个时间复杂度为$O(d_r*d_c^3)$。GPTQ认为所有行按一个顺序量化，最终的效果差不多，但是可以极大的简化量化过程。因为每行按相同的顺序量化，那么所有行的H更新都一样，不需要为每行保存一个H，时间复杂度变为$O(d_c^3)$。而且按顺序量化不需要再使用公式(1)去寻找使得误差最小的权重。         
    因此，当量化到第$q$个权重$w_q$时，权重更新和H矩阵更新变成
    $$
    \delta W = -\frac{w_q-quant(w_q)}{[(H_{q:,q:})^{-1}]_{00}} [(H_{q:,q:})^{-1}]_{:,0} \quad (2)\\
    (H_{q+1:,q+1:})^{-1} = ((H_{q:,q:})^{-1}- \frac{1}{[(H_{q:,q:})^{-1}]_{00}}(H_{q:,q:})^{-1}_{:,0}(H_{q:,q:})^{-1}_{0,:})_{1:,1:} \quad (3)
    $$      
    公式(2)的可以直接应用到所有行，而不是一行一行计算。

    **(2) Cholesky 分解**
    GPTQ在实际实验中发现公式(3)不断更新，误差累积容易造成H的逆矩阵为非正定矩阵，导致权重的更新错误。H矩阵是取最小值时的二阶导数，所以是对称正定矩阵，其逆也是对称正定矩阵。则$H^{-1}$可以进行Cholesky 分解，分解为一个对角为正数的下三角矩阵和其转置的乘积，即
    $$
    H^{-1} = LL^{T}
    $$
    通过证明（**待证明**），可以有
    $$
    [(H_{q:,q:})^{-1}]_{:,0} = C_{qq}L_{q:,q}
    $$
    $C_{qq}$为一个系数，则公式(2)变为
    $$
    \delta W = -\frac{w_q-quant(w_q)}{[(H_{q:,q:})^{-1}]_{00}} [(H_{q:,q:})^{-1}]_{:,0} \\
    =-\frac{w_q-quant(w_q)}{C_{qq}L_{qq}} C_{qq}L_{q:,q} \\
    =  -\frac{w_q-quant(w_q)}{L_{qq}} L_{q:,q} \\
    = -\frac{w_q-quant(w_q)}{L_{qq}^T} L_{q,q:}^T   \quad (4)
    $$
    因此，当计算出$L$后，不再需要更新$H^{-1}$。

    **(3) 延迟计算**                
    每量化一个参数，就需要更新后面剩余的所有参数，效率较低。通过将$d_c$划分成多个block，在每个block内进行参数的更新，然后在一个block结束后，一次性更新剩余的所有block的参数。                     

    **最终的GPTQ的算法：**                
    (1) 通过$H$计算$H^{-1}$，然后计算$L$
    (2) 遍历所有的block
    (3) 在一个block内，遍历每个列      
    (4) 计算$w_q$对应的量化 $quant(w_q)$
    (5) 使用公式$(4)$更新该block内的权重              
    (6) 一个block结束后，更新剩余block的权重，返回步骤(2)

    **注：这里的列包含了所有的行。$quant(w_q)$的scale和zero是什么？**

5. 总结              
    OBD将二阶hessian矩阵应用到剪枝；OBS提出了修正权重，弥补了OBD的假设；OBQ将OBS应用到量化；GPTQ加速了OBQ的量化过程。

    **参考：**

    https://zhuanlan.zhihu.com/p/690834228

    https://zhuanlan.zhihu.com/p/692338716

    OBD: https://proceedings.neurips.cc/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf
    OBS: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=298572
    OBQ: https://arxiv.org/pdf/2208.11580
    GPTQ: https://arxiv.org/pdf/2210.17323


### 具体实现



## HQQ量化
hqq属于calibration-free的量化方法，即其不需要像GPTQ一样使用矫正数据集，直接对权重进行优化，使得权重的误差变化最小。在线性量化的情况下，目标就是找到scale和zero，即
$$
\argmin_{z,s} \phi (W-Q^{-1}_{z,s}(Q_{z,s}(W))) \\
Q_{z,s}(W) = W_q = round(W/s - z) \\
Q^{-1}_{z,s}(W_q) = s(W_q+z)
$$
其中$\phi(*)$用于衡量量化误差，一般使用$l_2$范数来衡量量化权重的误差。但是$l_2$范数无法很好地建模离群值的分布，所以hqq采用$l_{p<1}$范数。由于$l_{p<1}$范数是非凸函数，所以需要格外的求解方案。hqq采用`Half-Quadratic solver`进行求解。

### Half-Quadratic solver

待学习

https://ieeexplore.ieee.org/document/120331




https://mobiusml.github.io/hqq_blog/

## AWQ量化

根据llm.int8()，激活的部分离群值更加重要。但是llm.int8()实现的分离式计算很难取得真正的计算加速。经过AWQ的论文测量，llm.int8()的速度总是比f16的计算速度慢，只能取得内存方面的优化。

AWQ基于激活的离群值明显，难以量化，而模型权重的离群值不明显，容易量化的特点，通过一个scaling操作，将激活的离群值转移到权重上面，从而进行两者的平衡。由于激活的离群值分布在几个通道之中，那么将每个通道除以一个值，使其放缩到较小的数，同时将相应权重的行乘以这个值，保持最后的结果不变。然后再分别对激活和权重进行量化，使得两者都易于量化。这里的量化是`vector-wise`，即激活按照token维度进行量化（行），而权重按照通道维度进行量化（列）。**如果激活按照通道维度进行量化，那么就离群值就不会有影响了，因为整个通道都是离群值，在通道维度上看就没有离群值了，但是这样量化不符合矩阵乘法的计算特性，很难计算。** 总之，通过下面的变换
$$
O^{s\times o} = X^{s\times h} W^{h\times o} =  (X^{s\times h} \text{diag}(S)^{-1} ) (\text{diag}(S)W^{h\times o} ) = \hat{X} \hat{W} \\
X_{int8} = Q(\hat{X}) \\
W_{int8} = Q(\hat{W}) \\
S\in R ^h
$$ 

这里的$S$就是scaling的数，其求解为
$$
S_j = (\max|X_{:,j}|)^\alpha / \max{|W_{j,:}|}^{1-\alpha} \\
\alpha \in [0,1]
$$
其在转移量化困难性下进行折衷。即当$\alpha$为1时，将所有的激活的值都scaling到(0,1)，将量化的所有的困难转移给了权重；反之，当$\alpha$为0时，量化困难都转移给了激活。文中将$\alpha$为不同的模型设置不同，例如0.5。

通过上面的操作，通过一个校验数据集获取$S$，然后对模型进行量化。**最后在线计算时，对输入进行在线量化，最后反量化。（这里需要进一步查找具体的实现，看激活在什么地方被量化和反量化）。**

取得了和llm.int8()相似的精度和内存优化，并获得了实际的加速，比f16更快，得益于整数乘法更快。**按照其论文的量化方案，attention计算也是整数，所以kv-cache也是整数？**

### 参考

https://arxiv.org/pdf/2211.10438

## LLMINT8量化
该方法的核心思想是将离群值分解出来，不进行量化，从而保证模型的精度。                                                       
对于一个输入$X \in R^{s\times h}$ 和一个权重 $W\in R^{h\times o}$，那么其进行乘法得到输出，即
$$
O^{s\times o} = X^{s\times h} W^{h\times o}
$$
根据论文的实验验证，大部分异常值（论文中定义为绝对值大于6）都分布在$X$的同一列，而包含异常值的列的数量不超过总列数的$0.1%$，所以将这些列抽出来形成集合$X_{f16} \in R^{s \times h_1}$，并将其对应的在$W$中的行也抽取出来形成集合$W_{f16} \in R^{h_1 \times o}$。然后将剩余的部分量化为int8，即得到$X_{int8}\in R^{s \times h_2} $ 和 $W_{int8} \in R^{h_2 \times o}$，其中 $h_1+h_2 = h$。 这里的量化采用`vector-wise quanztization`和对称量化（$X_{int8}$ 是按行量化得到的，而 $W_{int8}$是按列量化得到的；零点为0， 方便反量化），并得到相应的scale参数，即$S_{x} \in R^s, S_{w} \in R^{o}$。
$$
Q(x) =  x/s, DQ(x) = sx \\
$$
计算被分为int8的计算和f16的计算两部分，即
$$

O_{int32} = X_{int8}W_{int8} \quad (1) \\
O_{f16} = X_{f16}W_{f16} \quad (2) \\
O^{s\times o} = (S_xS_w)^{s\times o}*O_{int32} + O_{f16} \quad (3)
$$
公式(1)是执行W8A8的矩阵乘法，执行完了再根据公式(3)进行反量化（公式中`*`表示对应元素相乘）。公式(2)执行正常矩阵乘法。

在实际的实现中，应该先量化权重，但是没有输入的情况下，其实是不知道$W$的那些行是重要的，应该量化和不量化那些行。 在`bitsandbytes`中，实际上的$W_{fp16}$是根据当前的输入$X$将需要使用`f16`的那些行通过$W_{int8}$反量化回来得到的。即在一开始，所有的权重量化为`int8`，然后在输入$X$时，根据$X$的离群值将其分为$ X_{int8}$ 和 $X_{f16}$，然后将离群值对应权重的行反量化，得到$W_{fp16}$。所以该实现，从一开始就使得权重出现了较大的偏差。如果是执行$A16W8$的乘法，就没有必要做分离了，因为权重已经不是原来的权重了。 因此，这个设计就是为了做一次$A8W8$。**这个操作应该是可以减少内存的消耗和加速（也不一定，多了量化反量化以及两个kerenl的相加），其精度肯定是比不上$A16W8$ 的（？）。**

在理想情况下，应该保留一份`f16`的权重，并根据当前的输入从其中抽取对应的离群值的行，而不是使用`int8`反量化。 然而，这样违背了节约内存的初衷。 **或许可以将其保留在CPU中，需要时加载，因为很小一部分，可能不会造成太多的overhead。**

**或许可以使用一个`calibration data`找出权重中那些行是重要的，但是没有这样做说明可能使用一个大型数据集进行校验时，所有的行都会被作为离群值对应的行。**

**因此，`bitsandbytes` 目前的实现是有较大问题的。**

已经有相关的issue：
https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1400                    
https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1320                     

论文：                        
https://arxiv.org/pdf/2208.07339






## GGUF量化


