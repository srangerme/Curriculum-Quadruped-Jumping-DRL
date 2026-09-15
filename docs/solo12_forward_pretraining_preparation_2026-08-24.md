# Solo12 Forward 开训前准备与验证（2026-08-24）

## 结论

Forward 开训前的代码、配置和诊断出口已准备完成，当前可以进入 reward 单变量消融，但尚未启动任何 Forward 训练。

## 已完成修改

### 状态机与随机化连续性

- Forward 使用 `settled_contact_count=3`，与最终 Upward c8800 一致。
- Forward COM 随机化固定为 xyz 各 `[-0.02,0.02] m`。
- observation latency 默认覆盖 `[0,40] ms`。
- joint friction 继续继承 `[0,0.04]`。
- 补齐 Forward 共享 `check_jump()` 所需的 c8800 起跳预测高度/pitch 参数；只用于事件计算，不启用 Upward 专属 reward。

### 距离课程

- F1 初始采样 level `0-2`，课程上限 level `3`，对应最大距离 `0.30 m`。
- 增加训练参数：`--command_initial_min_level`、`--command_initial_max_level`、`--command_max_level`。
- 课程最大 level 使用 `num_levels-1` 归一化，三个正式端点精确对应：

| 阶段 | level cap | 最大 x 距离 |
|---|---:|---:|
| F1 | 3 | 0.30 m |
| F2 | 6 | 0.60 m |
| F3 | 10 | 1.00 m |

未配置新字段的其他任务保持原初始化行为和原距离分母，不受 Solo Forward 修复影响。

### Reward 与诊断

新增两项可独立开关的事件 penalty，默认 scale 均为 `0`：

1. `takeoff_pitch_cancellation`

- 计算 `J_abs - |J_signed|`，并用竖直冲量和机身长度归一化。
- 持续同向的有效 pitch 冲量不受罚；后腿先制造一个方向的 pitch、前腿再反向抵消时才增加。
- 默认自由区 `0.05`。

2. `takeoff_front_rear_timing`

- 分别计算前、后轴竖直接触力的时间质心，使用二者绝对时差。
- 不强制完全同步，默认自由区 `25 ms`。
- 若某一轴没有有效加载，以整个起跳加载时间作为时差，避免单轴发力绕过 penalty。

训练 CLI 新增：

- `--takeoff_pitch_cancellation_scale_override`
- `--takeoff_front_rear_timing_scale_override`
- `--takeoff_pitch_cancellation_free_band_override`
- `--takeoff_front_rear_timing_free_band_ms_override`
- `--task_pos_scale_override`

评估 JSON 新增：

- `takeoff_pitch_cancellation_normalized`
- `takeoff_front_rear_timing_gap_seconds`

高度 reward 默认权重没有提前修改：`task_max_height=5000`、`base_height_flight=100`。后续通过已有 override 独立比较原配置和 `1000/20`，验证后再决定正式值。

## 验证结果

### 静态与配置验证

- 4 个修改 Python 文件均通过 AST 语法解析。
- 默认 F1 `config_only` 成功生成有效配置。
- F2 level `0-3 -> cap 6`、高度 reward `5000/100 -> 1000/20`、两项新 penalty `-1000` 的组合 override 通过参数校验并正确写入有效配置。
- reward 方法注册和距离端点 `0.3/0.6/1.0 m` 断言通过。

有效配置证据：

- `outputs/quadruped-jumping-ada/logs/forward_f1_prep_default.json`
- `outputs/quadruped-jumping-ada/logs/forward_f2_reward_ablation_prep.json`

### GPU 环境步进验证

- 使用 c8800、Forward task、target x `0.2 m`、16 个 nominal episodes 做只读诊断评估。
- 首次运行发现 Forward 缺少共享起跳事件参数，已按 c8800 参数补齐。
- 修复后 16 回合完整结束，无 buffer、状态机、reward 或评估字段运行时错误。
- 新指标均产生有限值：

| 指标 | 16 回合均值 | 默认自由区 |
|---|---:|---:|
| pitch cancellation | 0.04698 | 0.05 |
| front/rear timing gap | 12.53 ms | 25 ms |

两项均位于默认自由区内，说明不会无差别惩罚 c8800 已有的 nominal Upward 起跳动作。

运行证据：

`outputs/quadruped-jumping-ada/logs/forward_prep_runtime_eval.json`

该评估只用于验证实现。c8800 未进行 Forward 训练，其落点、termination 和成功率不能作为 Forward 能力或阶段验收结论。

## 下一步训练前决策点

先从同一 c8800 起点做单变量短训：

1. R0：旧高度权重 `5000/100` 对比 `1000/20`，各 `400-600` iterations。
2. R1：固定 R0 胜者，仅比较 cancellation scale `0` 与候选负权重，各至少 `600` iterations。
3. R2：固定 R0/R1 结果，仅比较 timing scale `0` 与候选负权重，各至少 `600` iterations。

这些 checkpoint 只用于因果判断，不作为正式阶段验收。只有综合改善位置误差、起跳高度安全性、终止/稳定性、pitch cancellation 和前后轴时差，才把对应修改带入 F1 正式训练。

Upward 完整过程见：`docs/solo12_upward_training_process_2026-08-24.md`。
