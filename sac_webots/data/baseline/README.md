# Baseline report comparison

双击Compare Reports.cmd，读取本目录config.json，更新comparison.html与comparison.json，不启动训练。

默认比较同一实验最新完成训练的report与前一次独立完成训练的report。重复生成同一次训练的report不算上一轮；未完成运行不参加自动选择，缺上一轮时显示待比较。

reports.current和reports.previous可直接填写两个report的index.html路径（相对本baseline目录或绝对路径），用于手动比较。experiment可填具体英文实验目录名，也可保留auto。窗口默认最后5000环境步；原报告可在比较页面内放大，完整原始数据仍在报告/训练归档中。

比较是在线训练诊断，不是独立Evaluation。Reward公式或目标不同时应结合转速、目标误差、反向转角和跌倒解读，不能仅比Reward数值。
