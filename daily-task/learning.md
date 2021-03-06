# 学习记录

1. 岛屿计数问题，DFS、BFS、迷宫的最短路径、DFS和BFS适用场景、DFS非递归遍历
   1. BFS(广度优先搜索)
      1. 选择一个节点作为起始节点，染成灰色，其余节点为白色
      2. 将起始节点放入队列
      3. 从队列首部选择一个节点，并找出所有邻接节点，并将邻接节点放入队列尾部
      4. 已访问过的是黑色，未访问过的是白色，在队列的是灰色
      5. 同样方法处理下一个节点
   2. DFS(深度优先搜索)
      1. 选择一个节点作为起始节点
      2. 将起始节点假如队列
      3. 从队列首部选择一个节点，找到一个未访问的邻接节点并访问
      4. 重复步骤3知道无邻接节点，然后返回
   3. DFS和BFS区别
      1. BFS用来搜索最短路径；DFS用来搜索全部路径
      2. DFS空间效率高，BFS需要记录中间过程
   4. 例题：黄金矿工&封闭岛屿
2. CNN反向传播，怎么过全连接、池化层、卷积层
   1. 池化层：最大池化只需要反向对应位置即可，其他位置为0；平均池化为误差除以区域值
   2. 卷积层：原图的误差，等于卷积结果误差经过零填充后，与卷积核旋转180度后的卷积
3. 过拟合的方法
   1. 早停
   2. dropout
   3. 正则化
   4. 数据增强
4. SGD为什么可以online learning、Adam原理
   1. SGD通过牺牲一定的精度，采用单样本计算损失，达到快速更新参数的目的，在运行速度、计算方式上符合online learning的要求
   2. 自适应学习率+动量
5. 二叉树宽度、深度
   1. 二叉树宽度用队列层次遍历，保存每层的宽度
   2. 二叉树深度用递归
6. 小兔的棋盘
   1. 类似斐波那契数列，往前走到一个点，只能从上方或者左方到达因此dp[i,j]=dp[i-1,j]+dp[i,j-1],dp[i,i]=dp[i,j-1]
   2. 不越过对角线，棋盘的对称性考虑，可以直接结果乘以2，得到最终的数值
7. leetcode76、leetcode856、链表重复节点删除、leetcode72、leetcode222、leetcode448、leetcode102、42、124
8. 五局三胜和三局两胜的公平性
    1. 在水平相当，即胜率为0.5的情况下，是公平的
    2. 如果胜率大于0.5，那么局数越多越有利
9. A文件有m个专有名词，B文件有n个query，统计专有名词出现次数
   1. 类似给定歌手以及用户输入，统计歌手频次的问题
   2. 解决方法：专有名词长度排序，按照名词长度做滑动窗口去匹配query，分布式统计
10. 从访问日志中，找访问次数最多的TopK用户
    1. 先统计，再排序最小堆排序
    2. 通过hash划分为k个文件，每个文件的最多类就是
11. xgboost模型、GBDT原理，如何多分类
    1. 传统的GBDT以CART作为基分类器，Xgboost还支持线性分类器
    2. GBDT在优化时候只用到了一阶导数信息，Xgboost对代价函数进行了二阶泰勒展开，同时也支持自定义代价函数
    3. Xgboost在代价函数中添加了正则项，正则项包含叶子节点个数、叶子节点score的L2模的平方和，防止过拟合
    4. 列抽样，不仅降低过拟合，而且减少计算
    5. 对缺失值处理，可以自动学习分裂方向
    6. Xgboost支持并行，在特征维度上而不是tree维度，计算每个特征的增益，选择增益最大的特征去分裂，那么计算增益可以并行(直方图算法)
    7. 分裂方式的区别：λ存在于正则项的叶子节点数的系数以及在Gain计算中充当阈值，起到预剪枝的作用
12. 梯度消失、梯度爆炸
    1. 预训练+微调
    2. 正则：L1和L2正则
    3. 残差连接：相当于恒等映射
    4. BN：通过一定规范化手段将每层NN的激活输入值分布拉回到标准正态分布，使得激活值落在非线性函数比较敏感的区域，产生较大的梯度，梯度变大意味着收敛速度快，能加快训练速度
13. KL散度和交叉熵关系
    1. 保证真实数据分布不变的情况下，最小化KL散度等价于最小化交叉熵
    2. KL(A||B)=-S(A)+H(A, B)，A代表真实数据分布，因此保持不变
14. 现有模型需要多少额外标注数据？
    1. 取决于执行的任务、最终可接受的性能、现有的特征、数据噪声情况以及模型复杂度等
15. 几何分布的期望
    1. 定义：在伯努利实验中，时间第k次发生的概率，记做几何分布
    2. 期望等于发生概率p的倒数；方差等于1-p/p^2
16. 分类问题的指标、ROC、MAP、AUC、特征选择的方法
    1. AUC：一种是面积角度解释，通过计算ROC曲线下方面积；另一种解释是排序概率解释，从小到大排序预测样本，(正样本排序位置和-负样本n1*(n1+1)/2)/正负样本数乘积
    2. 给M个正样本、N个负样本，以及预测值P，计算AUC，如果预测值都乘以1.2，AUC如何变化
       1. 扩大1.2倍，AUC不变
       2. 基于P值进行排序，概率计算公式为：(sum(rank_M)-N*(N+1)/2)/(M*N)
    3. AUC问题：线下计算全体的AUC，user1的正样本和user2的负样本排序高，没有太大的意义；线上只关注某个user的AUC，存在一定的gap；改进：阿里的GAUC
17. MCMC采样、马尔科夫链、CRF、FM是否可以特征选择？
18. 排序算法分析、堆排序、快排
    1. 稳定排序：冒泡排序、归并排序、选择排序
    2. O(n^2)的算法：选择、冒泡、插入；O(nlogn)的算法：快排、堆排序、归并排序
19. 排序数组中，绝对值不同的个数
    1. 对所有的数取平方，保证两头大中间小，然后头尾双指针
    2. 每次删除两个指针较大的值，同时记录下来，然后移动指针
    3. 如果此次删除的值和上次删除的相同，仅移动指针，否则计数加一
    4. 删除前需要比较是否和上次删除数据相同，然后比较此次两个指针数字的大小，不等则删除大的，移动；相等则同时移动，计数
20. attention的实现、BN原理和作用
    1. attention是一种软对齐，实现方式是加权，让模型的信息视野有的放矢
    2. BN是将激活输入值的分布拉回标准正态分布，能够产生较大梯度，加速训练
21. 特征工程、PCA原理
    1. PCA原理
       1. PCA是主成分分析，通过线性变化，将n维特征映射为k维特征，实现数据降维的目的
       2. 对原始数据进行零均值化，方便计算
       3. PCA可以通过转化为SVD求解
    2. 特征工程
       1. PCA是无监督特征选择、LDA是有监督特征选择
       2. 通过信息增益、L1、拉普拉斯分数等进行特征重要性衡量，来选择特征；决策树模型也可以
22. softmax推导、交叉熵推导
    1. softmax：等价于soft max，计算的是一种概率，等价于该元素的指数与所有元素指数和的比值
    2. 一定条件下，交叉熵和KL散度是等价的
23. 数据不均衡处理方法
    1. 数据上采样、下采样
    2. 集成学习
    3. 损失函数(Focal loss)、评价指标调整(AUC)
24. 0-1矩阵，找1的最大连通域，计算其面积
    1. DFS、BFS求解
25. xgboost和GBDT的分裂方式、xgb如何处理类别特征的
    1. 还需要好好看看
26. bert的改进，定制化bert，bert原理
    1. Bert基于预训练+微调的模型，使用Mask+next sentence的方式在超大规模语料上进行预训练
    2. 算力和推理速度制约了Bert在实际生产环境的应用
    3. 模型压缩+词mask训练+XLNet
27. 如何标注数据（active learning）
    1. 主动学习：基于选择策略，选择未标注数据进行标注，然后喂给模型进行训练，直到满足终止条件
    2. 选择策略的确定是主观的，不具泛化能力；情感分析中是人工标注分类模型推理错误的样本
    3. 在保证质量的前提下增加数量，防止过度拟合噪声
    4. 主动学习区别于半监督学习的点在于人工参与
28. LR、SVM、softmax、L1、L2、Bert、Transformer、GBDT、Xgb
    1. LR模型：广义线性模型；简单、快速、易于求导
    2. L1正则为什么可以产生稀疏解？
       1. 从图形角度解释
          1. 以二维平面空间为例，c=|x|+|y|，假设c为常数，那么上述图形表现为一个倾斜的正方向，顶点在坐标轴上
          2. 以求解平面直线为例，知道点(10, 1), 10a+b=5，b=5-10a，那么通过改变c的值，可以使得两个图形有交点
          3. 在正常情况下，交点必定在坐标轴上，从而在L1正则的情况下，产生的解释稀疏
       2. 从导数角度解释
          1. 以一个参数w为例，求解参数w=0处的导数，L(w)在w=0的导数为d
          2. L2正则导数不变为d，L1的导数为d+λ和d-λ，是一个极小值点
    3. L1可以用来特征选择(由于稀疏解的原因)；L1、L2都可以正则化，防止过拟合
29. CART数
    1. CART分类树
       1. 基尼指数计算Gain(D)=sum(pn(1-pn))，集合A中每个类别的概率与1-概率乘积的和
       2. 分为等于特征值和不等于特征值两类，计算Gain(D,A)=D1/D*(Gain(D1))+D1/D*(Gain(D2))，其中D1代表等于特征值的集合，D2代表不等于的集合
       3. 遍历所有特征的所有特征值，选择Gain最小的特征的特征值，进行二分，迭代进行下去
       4. 叶子节点的类别是多数类的类别
    2. CART回归树
       1. 最小二乘回归树：min(sum(yi-c1)^2+sum(yi-c2)^2)，c1、c2代表特征j上切分点s划分的两个集合的标签均值，计算两个集合在j特征s切割点下最小均方误差
       2. 遍历所有特征所有值，找到最小均方误差的切割点，进行分类划分
       3. 叶子节点的值是标签值的均值
       4. 时间复杂度O(NFS)，其中N个特征，F个切分点，S个内部节点
    3. GBDT是基于CART数的boosting算法
       1. GBDT做回归
          1. 使用平方误差作为损失函数的时候，负梯度和残差是相等的
       2. GBDT做分类
          1. 如何做？
30. 参与派单系统顺路度优化
31. 骑手招募画像模型
    1.  背景：应对突增运力需求下，骑手运力招募困难、成本高的问题
    2.  