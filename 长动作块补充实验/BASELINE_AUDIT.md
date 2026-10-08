# 长动作块实验的基线口径审计

下列定义来自论文与官方源代码；本文件不包含尚未运行的实验成绩。

## SmolVLA 的 70% 定义

论文 §3.3 的 g=0.7 是**剩余动作数/H < 0.7**，约执行30%后触发。当前公开客户端使用 `<=`，故本实验按照公开客户端选择 H=50/100/200 首次约15/30/60步触发。论文严格小于版本则为16/31/61步；不混用“已执行70%”这一不同规则。

论文包含重叠动作聚合及近重复状态过滤；本实验使用时间戳过期裁剪、以新动作覆盖重叠队列，并明确不实现全部聚合/过滤变体。公开客户端本身提供 `latest_only` 聚合选项。Smol70 与 LAYA 使用相同交付延迟及队列规则，只改变触发时机。

来源：[SmolVLA §3.3](https://arxiv.org/html/2506.01844v1#S3.SS3)、[官方客户端](https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/async_inference/robot_client.py)、[官方聚合选项](https://raw.githubusercontent.com/huggingface/lerobot/main/src/lerobot/async_inference/configs.py)。公开 main 的读取时间为；实际本实验实现另以脚本哈希固定。

## 原生 H=100/200 的含义

本次冻结 SmolVLA 实现按照 `config.chunk_size` 创建采样噪声、动作掩码和输出切片，动作线性投影参数不绑定 H=50。因此可不训练地尝试原生 H=100/200；须以实际输出形状和有限数值审计确认，不能用重复、插值或拼接短块充当长预测。`n_action_steps` 只影响执行队列，单改此参数不等于增加生成长度。

模型原训练长度为50，所以长输出是**零样本长度外推**。即使代码可以生成，也不代表第51至200步经过相应训练或具有有效控制质量。实验失败可以否定当前冻结检查点/配置的收益，不能泛化为长动作块训练方案均无效。

证据：[本次模型源代码快照](source_snapshot/modeling_smolvla.py)、[官方模型代码](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/smolvla/modeling_smolvla.py)、[SmolVLA训练设置](https://arxiv.org/html/2506.01844v1#S4.SS3)。

## VLASH：完整方法与本次机制迁移

VLASH训练样本为 `(image_t, state_(t+δ)) → actions_(t+δ:t+δ+H)`：固定图像，把机器人状态与动作目标一起偏移。真实异步运行在请求时利用已知旧动作推算未来执行时刻的状态；新块第0项就对应未来执行时刻。论文说明，仅在普通检查点的测试输入中替换未来状态不足以保证稳定性。

官方 LIBERO rollout 在控制时刻 t 使用前 d 步图像与**真实当前机器人状态**，直接执行预测块第0项，无需再次裁去前 d 行。它是理想状态对齐的延迟仿真，不测实际异步墙钟，也不包含未来状态估计误差。

因此本次 `VLASH-style-K5` 只代表“冻结普通 SmolVLA 的 VLASH式输入对齐、无 offset 训练”；真实当前 state 相对于陈旧 state 对照具有信息差异，必须明确。它不能代替完整 VLASH，也不能据此证明 LAYA 优于或劣于作者训练后的方法。d=1 接近现有官方评估范围；若以后使用 d=8，须标为额外压力测试。

来源：[VLASH §IV](https://arxiv.org/html/2512.01031v2#S4)、[固定提交的官方 rollout](https://github.com/mit-han-lab/vlash/blob/d2da223e2251a4a5020b667e77cf8ad57e594144/vlash/eval_libero.py)、[offset 数据构造](https://github.com/mit-han-lab/vlash/blob/d2da223e2251a4a5020b667e77cf8ad57e594144/vlash/datasets/vlash_dataset.py)。

## 官方模型独立参考

`scripts/official_reference.py` 从远端已审计的 `scripts/run_vlash_reproduction.py` 复制，仅调整输出范围和默认初态数量，并增加只读计时/权重审计。使用作者 `mit-han-lab/vlash-pi05-libero-async5` 固定检查点 `c85d5962a7b823f9f391f97d60227d6ca8bd7bc7`、官方源代码 `d2da223e2251a4a5020b667e77cf8ad57e594144`；保持 H=50、K=5、d=1、action_quant=1、种子42、10环境批量，Spatial每任务3个初态，共30回合。实际初态ID由官方wrapper记录，必须核对，不假定与另一套环境封装天然一致。

每次原始 `policy.predict_action_chunk` 调用前后执行 CUDA 同步，记录真实批量和 wall 时间。`predict_calls` 是批量预测调用次数，`predict_batch_elements` 是实际计算过的全部样本数，包含同一批内已结束但仍被官方向量循环计算的槽位。不能把批量wall直接与单环境wall比较为模型速度优势。完整冻结参数/数据类型快照及所有 state tensor 哈希在评估前后核对；不训练、不改变噪声或采样器。

本地历史官方参考为1944/2000成功；这是历史证据，不进入新30回合分母。官方模型与SmolVLA的模型规模、训练和批量不同，新参考单独报告。原表检查点对应关系未唯一验证，不能称精确复现论文全部数值。

主比较至少同时报告成功率、每100控制步调用数、总NFE、VLA实测推理成本、LAYA判断成本、实际执行块长度和丢弃尾部。逻辑延迟不等于实测推理延迟，调用减少不等于同比例墙钟加速。30个已经曝光的开发起点只提供探索性证据，不支持细小非劣界限或未见任务泛化声明。

证据：[执行合同](PLAN.md)、[官方参考脚本](scripts/official_reference.py)、历史VLASH报告（未随仓库提供；见数据说明）。
