# Plot Rules

所有图片的需求：

所有生成图片文字都为英文，交互的pipeline都为显示都为中文

- dpi=600
- 自动保存
- 不手动截图
背景为白色。图中需要多个类别时，颜色从左到右选择为(#A4C8E1,, #F8B400, #F26B59, #FFC0CB,#303C43)，其余额外补充颜色有#3B4252, #A3BE8C, #BF616A, #D8DEE9。如果是热图则采用#7f3f98到#e46240再到#ffde17的过渡颜色。生成的x轴y轴的字体颜色都采用深灰色.如果是箱型图，用比字体颜色更深的灰蓝色框线作为箱的外框和，中间的填充为类别颜色。
对于多类别的统计图示，需要进行显著性标注，默认为P < 0.05标记ns，*：P < 0.01；**：P < 0.001，***：P < 0.0001。方法：条件对比 → Wilcoxon signed-rank；多参数 → Linear Mixed Model；闭环趋势 → Spearman；可分性 → Permutation test。

## 连接性绘图

节点大小正比于「burst 内发放率」（除以整段 T_window）；边仅显示 STTC>0.5 且最多 900 条（按 STTC 降序）；STTC 的 Δt=50 ms；burst 为池化 ISI 检测、多于 30 个时均匀抽 30 个；仅 burst 内 spike 参与 STTC。节点颜色：行平均连接性（非对角 STTC 均值）处于全体电极前 10% 为珊瑚红 hub，其余绿色；边为浅蓝到深蓝渐变。浅冷灰背景与淡网格

# Batch Analysis Rules

每个batch_analysis必须:

- 有config.yaml
- 自动输出结果
- 自动保存参数
- 自动记录分析时间

# Utils

utils 禁止依赖具体实验，必须单独书写，
同时对于一些相似功能函数（不同连接性计算，不同burst识别等）需要统一数据接口，以及统一输出，

**工具索引与维护约定**：见 [analysis_utils.md](./analysis_utils.md)。每次新增/修改 `analysis/utils/` 公开 API 时须同步更新该文档。

禁止:

utils不断膨胀，需要进行维护，重复内容不新建