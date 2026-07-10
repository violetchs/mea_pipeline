任何分析脚本都需要详细的log显示+进度条
把关键阶段的日志细化到“当前 run / 文件 / 子任务”；
增加可见进度条（按总任务和子任务双层显示），让长时间步骤有持续反馈。

/Users/chenrongrong/Desktop/maxwell_platform/analysis/interactive_pipeline2是独立出来的
其所有相关utils放在自己文件夹下，不使用公共utils

所有刺激时间都读取h5文件内部的刺激时间，具体可以参考mea-pipeline里面的读取刺激时间的方法