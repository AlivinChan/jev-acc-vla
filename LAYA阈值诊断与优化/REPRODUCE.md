# 复查本轮实验

本仓库包含文档、实验代码、配置、汇总结果与图表；不包含模型权重、逐回合原件、完整压缩归档和运行环境。数据范围见[数据说明](../docs/DATA.md)。

四项完整验证共2550个方法／H回合；输入对齐组的420条完整记录用于部分描述，199条中止分片记录单列；随机组未启动。

## 证据链

1. `protocol/COHORT/definition.json`与冻结收据记录方法、起点、顺序、控制周期及原实验源码和模型指纹。
2. `configs/`保存分片配置。原始运行的`manifest.json`记录前后权重指纹、输入配置和环境。
3. `artifacts/*_archive.json`保留原始归档的SHA与大小；大归档和逐回合文件未发布。
4. `checks/*_audit.json`核对动作块来源、原生形状、调度和费用；初始输入检查与噪声对齐检查另列。技术资格检查不纳入有效性分母。
5. `analysis/*_summary.json`、`*_service_accounting.json`、`*_cross_horizon.json`分别保留完整组结果、费用拆分和长度比较。
6. [完整数值表](analysis/VALIDATION_READOUT.md)、[部分样本表](analysis/PARTIAL_SAMPLE_READOUT.md)与[运行清单](analysis/run_inventory.md)对应报告中的数字和计数范围。

## 分析入口

完整组分析入口为`scripts/analyze_validation_cohort.py`、`analyze_service_accounting_v2.py`、`analyze_cross_horizon.py`与`analyze_cross_delay.py`。未完成组单独使用`analyze_partial_sample.py`，只纳入原顺序中连续完整、全部方法／H齐全且审计通过的分片；不计算新增置信区间或显著性检验，也不替代完整确认。

这些分析仍需要未发布原件。仅下载本仓库不能完整重算逐动作审计、反事实分支或闭环结果。

## 代码与指纹

发布版已整理目录、删除非实验运行管理内容和日期字段；模型、任务、控制规则、种子和汇总指标保留。`EXPORT_MANIFEST.json`中的发布指纹校验本仓库文件；实验冻结收据中的指纹对应原始执行字节，两者用途不同。

重新执行时需要配置数据和模型位置、准备仿真与软件环境、重新冻结发布代码及配置并运行资格检查。原运行收据用于追溯已完成实验，不能充当新执行的冻结收据。源码仍按研究快照提供，不是一键重跑包。

两个随机种子不能当成两倍独立初态数，旧v1异常批次420回合不进入v2有效性分母；d4与d5共用配对起点。
