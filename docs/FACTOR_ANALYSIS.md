# Factor Analysis 方法说明

本文档说明当前 GUI 中 `Burst Trajectory` 和 `Multi-file FA Analysis`
采用的 Factor Analysis 方法、数据构造方式、隐状态和权重矩阵的含义，
以及各个量化图的解释。

## 1. 分析目标

MEA 记录通常包含很多通道。直接观察 `burst x time x channel` 的原始发放率
很难判断群体活动是否存在稳定结构。Factor Analysis 的目标是用少量隐变量
表示多通道活动：

```text
X ~= z W + mu + noise
```

其中：

- `X` 是观测到的群体活动向量。
- `z` 是低维隐状态。
- `W` 是从隐状态到通道空间的权重矩阵。
- `mu` 是通道均值。
- `noise` 是通道独立噪声。

直观理解：

- `z(t)` 描述当前时刻群体活动处于哪种低维状态。
- `W` 描述每个隐状态方向如何投影到各个记录通道。
- `z W` 是用低维隐状态重建出的多通道活动。

## 2. 数据矩阵如何构造

当前实现先把 spike train 转换成按时间 bin 统计的发放率矩阵。

### 2.1 Burst 模式

`Scope = Bursts` 时，每个 detected burst 是一个样本窗口。

数据形状为：

```text
burst x time_bin x channel
```

窗口起点是 burst onset，窗口长度由 GUI 中的 `Window` 参数控制。
bin 宽度由 `Bin` 参数控制。

### 2.2 All data windows 模式

`Scope = All data windows` 时，不再只分析 burst。
程序会从该记录最早 spike 到最晚 spike，按 `Window` 长度切成不重叠窗口。
每个窗口被当成一个 sample。

数据形状仍然是：

```text
window x time_bin x channel
```

这适合分析整体记录的低维活动结构，而不是只看 burst 片段。

## 3. 通道筛选

单文件分析中，通道筛选基于该文件自身活动。

多文件分析中，先把所有文件对齐到统一 channel 集合和顺序，再做全局通道筛选。
也就是说，多文件中的每个文件最终使用同一套 `global_selected_labels`，
因此不同文件的 `W` 列对应同一个 channel identity。

筛选指标包括：

- `Min activity`：通道总活动量下限。
- `Min bursts`：通道至少在多少个样本窗口中有活动。
- `Min variance`：通道发放率方差下限。
- `Max fit channels`：最多用于拟合的通道数。

筛选时 score 用于决定选哪些通道，但多文件模式下最终通道顺序固定为全局
channel 顺序，避免不同文件的 `W` 列错位。

## 4. 归一化

FA 拟合使用归一化后的状态矩阵。

当前支持：

- `Channel z-score`：每个通道在所有 sample/time 上做 z-score。
- `Log + channel z-score`：先 `log1p`，再对每个通道 z-score。
- `Per time total`：每个时间点按总活动量归一。
- `None`：直接使用原始发放率。

需要注意：GUI 中的 raw raster 和 raw PSTH 使用反变换后的原始发放率空间显示；
FA 拟合和 `Recon Metrics` 中的 R2/RMSE 使用归一化后的状态空间评价。

## 5. Factor Analysis 拟合

程序把三维矩阵：

```text
sample x time_bin x channel
```

拉平成：

```text
(sample * time_bin) x channel
```

然后使用 `sklearn.decomposition.FactorAnalysis` 拟合：

```text
X_flat ~= z_flat W + mu
```

更准确地说，当前使用的是标准线性高斯 Factor Analysis 概率模型。
对每一个样本点 `x_n`，模型假设：

```text
z_n ~ N(0, I)
x_n | z_n ~ N(W^T z_n + mu, Psi)
```

其中：

- `z_n` 是该样本点的隐状态。
- `I` 是单位协方差矩阵。
- `W` 是 factor loading matrix。
- `mu` 是观测通道均值。
- `Psi` 是对角噪声协方差，也就是每个通道有自己的独立噪声方差。

因此观测数据的边缘分布为：

```text
x_n ~ N(mu, W^T W + Psi)
```

这里的拟合目标不是直接最小化重建误差，而是最大化观测数据在上述高斯模型下的
log-likelihood。`sklearn.decomposition.FactorAnalysis` 会估计：

- `W`：保存在 `components_` 中，代码中称为 `loadings`。
- `mu`：保存在 `mean_` 中。
- `Psi`：保存在 `noise_variance_` 中。
- `z`：通过 `fit_transform` 得到，是给定数据和已估计参数后的后验隐变量均值。

可以把它理解为：模型寻找一组低维高斯隐变量，使得 `W^T W + Psi` 尽可能解释
多通道活动的协方差结构。

拟合后再把 `z_flat` reshape 回：

```text
sample x time_bin x latent_dim
```

这就是 GUI 中显示和导出的 `latent_states`。

## 6. z 的含义

`z` 是隐状态，不是某个真实电极的发放率。

每个时间 bin 都有一个低维向量：

```text
z_t = [z1, z2, ..., zk]
```

它表示该时间点群体活动在 k 个隐空间方向上的坐标。

在当前 FA 模型中，`z` 的先验是独立标准高斯。这个先验是模型假设，不是说
拟合后的每条轨迹一定严格服从标准高斯。拟合后的 `z` 会受到数据结构、通道
协方差、噪声方差和选定 latent dim 的共同影响。

更具体地说，`z ~ N(0, I)` 表示：

- 每个样本点的隐状态向量在先验上以 0 为中心。
- 不同隐变量维度在先验上互相独立。
- 每个隐变量维度在先验上的方差为 1。

这里的“先验”不是指用高斯随机数初始化一个矩阵；它是模型对隐变量分布的概率
假设。拟合完成后，GUI 中显示的 `z` 是在该模型下根据观测数据推断出的后验
隐状态均值，因此它可以表现出时间结构、burst 结构或条件差异。

当前实现没有对 `z` 加额外时序约束。也就是说：

- 不要求相邻 time bin 的 `z_t` 平滑。
- 不显式假设 `z_t = A z_{t-1} + noise`。
- 不限制 burst 内轨迹必须沿某个动力系统演化。

如果要让隐状态更像连续轨迹，后续可以扩展为 temporal smooth FA、LDS/Kalman FA
或 GPFA。

## 7. W 的含义

`W` 是权重矩阵，形状为：

```text
latent_dim x selected_channel
```

第 k 行表示第 k 个隐状态方向如何投影到所有通道。

当前实现对 `W` 的假设主要来自标准 FA 模型：

- `W` 是线性 loading matrix。
- `W` 通过极大似然从数据中估计。
- `W` 没有非负约束。
- `W` 没有稀疏约束。
- `W` 没有基于 channel map 的空间平滑约束。
- `W` 的不同 factor 不被强制正交。

因此当前的 `W` 更准确地说是“解释通道协方差的线性子空间”，而不是已经带有
生物空间约束的传播模板。

需要特别注意：FA 的 `z` 和 `W` 存在旋转、符号、顺序上的不唯一性。
例如对隐空间做一个旋转，同时对 `W` 做相反旋转，重建出的 `X` 可以几乎不变。
因此直接比较两个文件的 raw `W` 元素并不总是可靠。

这也是当前 `W Metrics` 改为子空间分析的原因。比起单个 loading 元素，
`W` 张成的子空间、奇异值谱、`W W^T` overlap 和 channel-space projection
通常更适合解释模型是否学到了稳定结构。

## 8. 重建

重建公式为：

```text
X_recon = z W + mu
```

GUI 中左上图显示一个 sample 的真实活动和重建活动对比。
右上 PSTH 是对所有通道求平均后的群体发放率曲线。

如果逐通道重建有误差，但总体 PSTH 很好，通常说明：

- 高估和低估的通道误差在平均时相互抵消。
- FA 抓住了强全局活动成分。
- PSTH 丢掉了空间分布误差，只保留群体平均时间轮廓。

因此 PSTH 拟合好不等于每个通道都拟合好。

## 9. Recon Metrics

`Recon Metrics` 用来判断重建质量。

当前包含：

- `Latent dim vs reconstruction`：
  扫描 latent dim 与重建效果的关系。横轴是隐状态维度，纵轴为 R2 和 mean RMSE。
  默认最低 4 维，最高不超过 96，同时受样本数和通道数限制。
- `Time-bin reconstruction RMSE`：
  每个时间 bin 的平均重建误差。
- `Channel RMSE`：
  每个通道的重建误差。
- `Residual distribution`：
  所有状态元素的残差分布。
- `Observed vs reconstructed state`：
  真实状态和重建状态的散点对比。

这些图用于判断模型是不是只重建了总体趋势，还是也较好保留了通道结构。

## 10. z Metrics

`z Metrics` 用于观察隐状态本身。

当前包含：

- 平均隐状态轨迹 `mean z(t)`。
- 隐变量值分布，并与标准正态分布比较。
- 每个 latent dim 的均值和标准差。
- latent dim 之间的相关矩阵。

这些图用于判断隐状态是否稳定、是否存在强相关维度、是否有某些维度几乎不活动。

## 11. W Metrics 的子空间解释

由于 `W` 有旋转不唯一性，目前 `W Metrics` 不再强调单个 raw loading 的大小，
而从子空间角度分析。

当前包含：

- `W subspace singular spectrum`：
  对 `W` 做 SVD，查看奇异值谱和累计能量。如果少数奇异值解释大部分能量，
  说明 W 的有效维度可能低于设定 latent dim。
- `Latent-factor overlap from W W^T`：
  计算 latent factor 之间在通道投影上的重叠。非对角线越大，说明不同
  latent factor 的通道模式越相似。
- `Orthonormal channel subspace basis V`：
  展示 W 行空间在 channel 空间中的正交基。
- `Channel-space projection P = V^T V`：
  表示哪些通道共同落在同一个 W 子空间中。它比 raw W 更不受隐空间旋转影响。

这些分析的意义是：即使 raw `W` 因旋转而改变，W 张成的子空间仍可能保持稳定。
因此子空间指标更适合比较不同片段或不同条件下网络结构是否相似。

## 12. 多文件 FA 和 W 对齐

`Multi-file FA Analysis` 中，每个文件独立拟合 FA，得到各自的 `z_i` 和 `W_i`。

为了比较不同文件的 `W`：

1. 所有文件先对齐到统一 channel 集合。
2. 使用全局通道筛选得到同一套 selected channels。
3. 每个文件用相同 channel 顺序拟合。
4. 将每个文件的 `W_i` 对齐到第一个文件的参考 `W_ref`。
5. 对 aligned `W` 计算 Pearson correlation。

相关性矩阵用于判断不同文件的 loading 子空间是否相似。

需要注意：当前多文件模式仍是 independent W 模型。它适合观察不同文件的结构
是否接近，但如果理论假设是“同一网络不同片段共享一个 W”，更严格的方法是
shared-W 模型：

```text
X_i ~= z_i W_shared + mu_i
```

也就是所有文件共享一个 `W_shared`，每个文件只允许 `z_i` 变化。
这是后续可以继续扩展的方向。

## 13. 当前方法的限制

当前 FA 方法有以下限制：

- `z` 没有显式时间动力学约束。
- `W` 没有稀疏约束或空间平滑约束。
- FA 假设噪声在通道间独立。
- raw `W` 不唯一，比较时应优先看对齐后相关性或子空间指标。
- PCA/FA 这类线性模型可能无法表达强非线性传播过程。

可改进方向：

- shared-W 多文件模型。
- 对 `z` 加时间平滑或 LDS/GPFA 动力学。
- 对 `W` 加稀疏约束。
- 结合 channel map 对 `W` 加空间平滑约束。
- 比较 `W` 子空间而不是 raw loading。
