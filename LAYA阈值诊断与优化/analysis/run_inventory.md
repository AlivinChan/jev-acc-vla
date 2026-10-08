# 已归档证据清单

仅统计本优化目录；前轮SmolVLA与官方π0.5参考另存。每次生成均核验压缩原件SHA，不读取成功率作选择。

分支、离线判断、资格复跑均不能充当独立测试初态；初始参考调用包含在资格调用总数内，不再相加。

| 注册验证记录的完成状态 | 方法/H回合记录数 |
|---|---:|
| 整批完整、已分析 | 2550 |
| 分片完成并已审计，但整批未完成 | 420 |
| 作业未完成或尚未通过审计 | 199 |

后两类不纳入完整批次的主结果；记录数不等于独立初态数。中止片可能缺少结束manifest及附加调用计数，不能将缺失计数理解为零；回合计数逐行验证JSON可解析。

| 运行 | 证据用途 | 单位与数量 | 作业状态 | 审计错误 | 整批验证汇总就绪 |
|---|---|---|---|---:|---|
| budget_d1_s20_v2 | registered_validation | closed_loop_episodes=120 | completed | 0 | True |
| budget_d1_s21_v2 | registered_validation | closed_loop_episodes=120 | completed | 0 | True |
| budget_d1_s22_v2 | registered_validation | closed_loop_episodes=120 | completed | 0 | True |
| budget_d1_s23_v2 | registered_validation | closed_loop_episodes=120 | completed | 0 | True |
| budget_d1_s24_v2 | registered_validation | closed_loop_episodes=120 | completed | 0 | True |
| cadence_v1 | development | closed_loop_episodes=120 | completed | 0 | False |
| choice_order_audit_v1 | offline_prompt_diagnosis | offline_laya_decisions=2058 | completed | 0 | False |
| confirmation_v1 | development | closed_loop_episodes=160 | completed | 0 | False |
| counterfactual_calibration_v1 | counterfactual_label_collection | counterfactual_opportunities=197; counterfactual_suffix_branches=394 | completed | 0 | False |
| counterfactual_evaluation_v1 | counterfactual_label_collection | counterfactual_opportunities=178; counterfactual_suffix_branches=356 | completed | 0 | False |
| counterfactual_pilot_v1 | counterfactual_label_collection | counterfactual_opportunities=6; counterfactual_suffix_branches=12 | completed | 0 | False |
| counterfactual_train_v1 | counterfactual_label_collection | counterfactual_opportunities=183; counterfactual_suffix_branches=366 | completed | 0 | False |
| delay_d1_qualification_v1 | implementation_qualification | closed_loop_episodes=9 | completed | 0 | False |
| delay_d4_pilot_v2 | implementation_qualification | closed_loop_episodes=15 | completed | 0 | False |
| delay_d4_s30_v2 | registered_validation | closed_loop_episodes=150 | completed | 0 | True |
| delay_d4_s31_v2 | registered_validation | closed_loop_episodes=150 | completed | 0 | True |
| delay_d4_s32_v2 | registered_validation | closed_loop_episodes=150 | completed | 0 | True |
| delay_d5_pilot_v2 | implementation_qualification | closed_loop_episodes=15 | completed | 0 | False |
| delay_d5_s30_v2 | registered_validation | closed_loop_episodes=150 | completed | 0 | True |
| delay_d5_s31_v2 | registered_validation | closed_loop_episodes=150 | completed | 0 | True |
| delay_d5_s32_v2 | registered_validation | closed_loop_episodes=150 | completed | 0 | True |
| diagnosis_v1 | offline_prompt_diagnosis | offline_laya_decisions=1680 | completed | 0 | False |
| first_request_qualification_v1 | implementation_qualification | closed_loop_episodes=6 | completed | 0 | False |
| fixed_interval_v1 | development | closed_loop_episodes=160 | completed | 0 | False |
| guarded_runtime_qualification_v1 | implementation_qualification | closed_loop_episodes=21 | completed | 0 | False |
| initial_repro_plain_v1 | forensic_or_rejected_backend_qualification | closed_loop_episodes=21 | completed | 0 | False |
| initial_repro_probe_v1 | forensic_or_rejected_backend_qualification | closed_loop_episodes=21 | completed | 0 | False |
| noise_d1_pilot_v1 | implementation_qualification | closed_loop_episodes=12 | completed | 0 | False |
| noise_d4_pilot_v1 | implementation_qualification | closed_loop_episodes=21 | completed | 0 | False |
| noise_d4_s40_v1 | registered_validation | closed_loop_episodes=210 | completed | 0 | False |
| noise_d4_s41_v1 | registered_validation | closed_loop_episodes=199 | failed | None | False |
| noise_d4_s42_v1 | registered_validation | closed_loop_episodes=210 | completed | 0 | False |
| representations_v1 | development | closed_loop_episodes=80 | completed | 0 | False |
| scores_calibration_v1 | offline_score_diagnosis | offline_laya_scores=591 | completed | 0 | False |
| scores_evaluation_v1 | offline_score_diagnosis | offline_laya_scores=890 | completed | 0 | False |
| scores_train_v1 | offline_score_diagnosis | offline_laya_scores=567 | completed | 0 | False |
| scores_window_calibration_v1 | offline_score_diagnosis | offline_laya_scores=394 | completed | 0 | False |
| scores_window_train_v1 | offline_score_diagnosis | offline_laya_scores=378 | completed | 0 | False |
| single_skip_confirmation_v1 | development | closed_loop_episodes=140 | completed | 0 | False |
| single_skip_v1 | development | closed_loop_episodes=120 | completed | 0 | False |
| validation_d1_s10_v1 | excluded_old_validation_attempt | closed_loop_episodes=210 | completed | 5 | False |
| validation_d1_s10_v2 | registered_validation | closed_loop_episodes=210 | completed | 0 | True |
| validation_d1_s11_v2 | registered_validation | closed_loop_episodes=210 | completed | 0 | True |
| validation_d1_s12_v1 | excluded_old_validation_attempt | closed_loop_episodes=210 | completed | 0 | False |
| validation_d1_s12_v2 | registered_validation | closed_loop_episodes=210 | completed | 0 | True |
| validation_d1_s13_v2 | registered_validation | closed_loop_episodes=210 | completed | 0 | True |
| validation_d1_s14_v2 | registered_validation | closed_loop_episodes=210 | completed | 0 | True |
| verified_runtime_qualification_v1 | forensic_or_rejected_backend_qualification | closed_loop_episodes=21 | completed | 0 | False |

完整单位、原件指纹及资格计数见[JSON清单](run_inventory.json)。
