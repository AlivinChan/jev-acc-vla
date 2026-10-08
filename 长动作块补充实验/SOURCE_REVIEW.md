# 补充实验执行前源码审查

独立只读审查。结论：**ApprovedWithLimits，可由 supervise.py 启动既定资格检查和 pilot；未发现必须先修复的调度/接口阻断。** 资格检查不通过必须保留失败并停止；本结论不预先保证真实模型可输出长块或实验有效性。

审查范围为 PLAN.md、四个 scripts 文件及三个 source_snapshot 文件；未修改生产源码、未导入模型或使用 GPU。审查时 main_experiment.py SHA256 为 `1ee93eb30d87f0fd56ea1dbf4dd9d52006cc83a80cb59ddcde64ce0e6a367f80`；official_reference.py 为 `1341d2a3006900eb8e87447141bba72429169d493623815595a01e0514e28089`。

## 已核对的机制

- 原生 H：每次同时设置 chunk_size/n_action_steps=H，噪声实际为 1×H×max_action_dim；模型掩码、流匹配输入与输出切片均读取 chunk_size。没有拼接/重复/插值。真实输出 1×H×7、有限值和同噪声复现由远端资格检查裁定。
- 冻结与记录：SmolVLA eval、requires_grad=False、no_grad，完整 state_dict 前后哈希；原检查点配置、安装源码指纹、参数 dtype 均记录。NFE=10 来自固定 num_steps=10，按预测次数累计；它不含视觉前缀的计算费用，推理 wall 同时报告。
- 70%：使用剩余/H <= 0.7，首次请求在15/30/60。Smol70 与 LAYA 的交付延迟、裁去第0行、覆盖队列完全共用；uncertain 才回退阈值，continue 持续到有限队列保护触发。
- K5：Naive 与 VLASH 新动作均在5、10、…步交付；Naive 以迟图和迟状态预测并从第1行执行；VLASH 在交付时以迟图、真实当前机器人状态预测并从第0行执行。该信息差异已在合同标出。
- 截断：step 后记录 success/terminated/truncated，遇到终止立即结束；230步硬上限；未执行尾部及 pending_at_termination 保留；环境 finally 关闭。
- IPC/清理：独立请求目录、原子发布、唯一claim、错误/超时传播；supervise 顺序启动两个模型阶段，并在失败、暂停或两小时上限时清理自有进程组。必须使用该受监督入口；单独调用主脚本的 LayaClient 构造期异常不具备同等子进程清理保证。
- 官方参考保持原采样器与向量rollout；其单独CPU审计确认包装不改输入、输出或RNG。严格首个done成功统计与官方多含一项的mask结果均记录；实际初态ID、batch费用和参数前后指纹单列。

## CPU 实测资格

使用 AST 提取实际 run_episode，注入可区分“预测调用/动作行”的 NumPy 动作、逐步图像/机器人状态及 FakeEnv/FakeGate。可复现脚本为 [verification/source_review_cpu.py](verification/source_review_cpu.py)，结果为 [verification/source_review_cpu.json](verification/source_review_cpu.json)。

| 检查 | 结果 |
|---|---|
| 三种H×四臂、230步完整调度 | 12/12通过 |
| all-uncertain LAYA vs Smol70 | 三种H逐动作及请求tick完全相同 |
| Smol70调用数，H50/100/200 | 16 / 8 / 4 |
| K5交付tick、Naive row1、VLASH row0及图/状态时间戳 | 通过 |
| all-continue有限队列保护 | 三种H通过 |
| 四臂第5步成功结束、第17步截断失败、环境关闭 | 8/8通过 |
| 实际输出形状、每次10 NFE、预测wall累计 | 通过 |

230步未成功轨迹中，Naive有47次调用而VLASH有46次：Naive在第229步已支付一次预测，但回合结束前无法交付；VLASH此时仅保存图像。该差异应计入真实计算成本并在报告说明，不应删掉Naive已付调用以制造一致调用数。

## 运行时准入与报告限制

仍须通过既定远端三长度资格、真实LAYA warmup及12回合pilot；CPU mock不证明CUDA算子、显存或长输出质量。LAYA 专项审计已只读核验固定 SDK 0.3.22 的 agent.py：usage.options 仅在选项 token span 冲突时出现，正常结果不含该项，因此本服务拒绝非空 options 正确。主实验完成须取得冻结前后哈希一致证据；中断产物只报告已完成数量，不称完整冻结资格通过。

最终报告按已定合同区分零样本长输出、特权对象状态门控、逻辑延迟与实测wall；官方π0.5参考不与SmolVLA同模型机制比较合并推导因果优势。该范围内无新增研究约束或新增审批要求。
