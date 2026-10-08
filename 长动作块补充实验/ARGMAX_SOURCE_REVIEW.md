# LAYA argmax 消融源码独立审查

结论：**ApprovedWithLimits，无必须先修复的阻断；原主实验372回合与官方30回合成功结束后，可由 supervise_argmax.py 顺序启动登记的90回合消融。** 本审查未启动模型或GPU，未修改生产源码。

审查了 ARGMAX_PLAN.md、三个新增运行脚本、analyze_argmax.py、原 main_experiment.py / laya_service.py / supervise.py 的复用接口及既有 argmax_cpu_audit.json。

- `laya_argmax_service.py` 只将原服务的 MINIMUM_PROBABILITY 设为0；原prompt、模型加载、选项校验和IPC不变。原始argmax为uncertain时仍保留uncertain，不强制二选一。
- 运行器直接复用原 Predictor、state_text、run_episode，因此真实H、噪声seed、d1队列、5步判断、有限队列保护、终止规则保持一致。90个task/state/H格完整且无重复；H顺序保留主实验对应LAYA格的相对顺序。
- ArgmaxClient核验ready.threshold=0、response.minimum_probability=0以及choice=raw_choice。资格6次VLA、预热1次门控与90回合分别记录并在结尾对账；参数冻结前后哈希/dtype相等由运行时断言检查。
- 监督器使用同一个experiment.lock，要求原supervisor completed、Smol372完成、官方30完成且冻结通过。它不会排队等待：提前启动会拒绝，因此应在原阶段成功退出后启动。原有租约/暂停/进程组清理复用，一小时上限有效；owns_output防止拒绝启动时改写已有attempt。
- 分析器按phase=argmax、method=laya定位原始产物；门控tau0、IPCs、chunk/轨迹、费用、资格计数、冻结记录及原主实验配对初态的字段均与运行器兼容。它按原始uncertain执行阈值回退检查，保留跨阶段耗时干扰和pilot后追加探索的限制；未满90回合不能标记complete。

独立CPU检查使用实际AST提取的run_episode生成三种H × raw continue/replan/uncertain，共9条230步模拟轨迹，再调用实际analyze_argmax.validate_case。9/9通过，共13,626项字段/原件检查通过；空输入正确标为partial。没有导入torch/模型/CUDA。机器记录和重放脚本分别见 [verification/argmax_source_review_cpu.json](verification/argmax_source_review_cpu.json)、[verification/argmax_source_review_cpu.py](verification/argmax_source_review_cpu.py)。既有8项CPU资格也与当前三个运行脚本SHA256一致。

审查版本：run_argmax_ablation.py `a980b2f6f53bc11ef6b10454bb9cbcac58020a1d0b4f5216919498d92d707b46`；supervise_argmax.py `b10aa6b910ed98e940ae1b7dd6dca2c47e498183d53a32c50955450ef16085b0`；analyze_argmax.py `cb0402ac0517bd0b210448d3d34bcfc3285ddbb5c0c7f78db3baa1e5a9f077f2`。

准入范围仅为登记的tau0单项探索性消融；真实CUDA资格、90回合完整性、与主实验检查点及初态一致性、终末参数指纹仍需依据运行后原件核验。该结论不预判效率或成功率收益。
