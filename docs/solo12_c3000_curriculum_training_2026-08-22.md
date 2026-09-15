# Solo12 upward c3000 随机化课程训练记录（2026-08-22）

## 固定约束

- 正式起点：`Aug21_18-22-11_com002_fromzero_s1_seed23_3000/model_3000.pt`。
- Reward：旧 reward + `takeoff_pitch_angular_impulse=-200`，不启用 direct mismatch/direct pitch，不修改 PPO、observation、PD、action scale 或 8 Hz filter。
- COM：`xyz +/-0.02 m`。
- 初始 latency：`0-20 ms`。
- 课程顺序：joint friction `0.02 -> 0.03 -> 0.04`，restitution `0.2 -> 0.3 -> 0.4`，latency `20 -> 30 -> 40 ms`。
- 每级只从上一已通过 checkpoint 继续；未通过则不进入下一级。
- 每级至少训练 `500` iterations；前 `200` iterations 只用于观察趋势，不作为适应完成的证据。正式通过要求后段至少两个相邻 checkpoint 形成一致的 Pareto 改善；若 `c500` 仍在明显变化，则延长至 `800-1000` iterations。
- 每级通过后先向 `Claw` 发送文字说明，再发送 nominal 验收视频。

## Pareto 验收规则

相对父 checkpoint，候选不得出现以下任一明显回退：

- Robust height mean 下降超过 `0.01 m`。
- Takeoff pitch 增加超过 `2 deg`。
- Pre-takeoff front/rear contact mismatch 增加超过 `0.01 s`。
- Termination 增加超过 `5` 个百分点。
- Post-landing four-feet contact ratio 下降超过 `0.05`。
- Landing angular RMS 增加超过 `0.05 rad/s`。
- Torque saturation 出现明显新增。
- Nominal 非自然 timeout、明显前后腿错配、明显落地余振或二次起跳恶化。

晋级还必须在当前课程端点上产生至少一项实质改善，例如高度/P10、成功率、pitch/mismatch、termination、strict stable 或持续四脚接触改善；单项 success 或高度上升不构成通过。

## 训练进度

### Joint friction `0-0.02`：失败，不晋级

- 前段 run：`Aug22_00-50-47_curr_jf002_seed23_from3000_200`，累计 `c3000 -> c3200`。
- 后段 run：`Aug22_00-55-50_curr_jf002_seed23_from3200_300`，累计 `c3200 -> c3500`。
- Reward、COM、latency、PPO、observation、PD 和 action filter 均保持冻结；唯一变化是 joint friction 范围改为 `0-0.02`。
- 同 seed 父基线 robust：height `0.500 m`、pitch `9.37 deg`、mismatch `0.054 s`、termination `10.2%`、four-feet ratio `0.898`。
- `c3300`：height `0.518 m`，但 pitch `15.54 deg`、mismatch `0.074 s`、termination `22.7%`、four-feet ratio `0.797`，已越过 Pareto 门槛。
- `c3350-c3500` 连续表现为更高 height/success 与更差 pitch/mismatch；不是单点噪声或早期暂态。
- `c3500` robust：height `0.583 m`、success `85.9%`，但 pitch `20.04 deg`、mismatch `0.106 s`、termination `18.0%`、four-feet ratio `0.800`。
- 固定 joint friction `0.04` 时，父基线为 height `0.441 m`、success `2.3%`、pitch `8.81 deg`、mismatch `0.051 s`；`c3500` 为 height `0.499 m`、success `58.6%`，但 pitch `19.12 deg`、mismatch `0.101 s`、termination `18.8%`、four-feet ratio `0.790`。
- 结论：500 轮足以确认策略通过大 pitch 和前后腿错配补偿能量，而不是形成可部署的 joint-friction 适应。该级失败，不发送通过视频，不进入 `0-0.03`。
- 因 restitution 和 latency 课程被定义为从上一通过 checkpoint 继续，本轮不启动后续课程，避免把失败动作形态带入下一阶段并破坏单变量归因。

## Joint-friction 课程失败原因排查

### 1. 不是未收敛或训练轮数不足

- TensorBoard 覆盖完整 `c3000-c3500`，后段 `c3300-c3500` 连续沿同一方向变化。
- Learning rate 全程约 `2.5e-4 - 3.4e-4`，没有续训时异常重置；policy noise std 约 `0.96-0.99`，没有探索坍缩。
- Mean reward 从课程早期约 `28.2` 上升到后期约 `32.8`。PPO 正在稳定优化现有目标，失败来自目标与验收行为不一致。

### 2. 从 full range c3000 回退到低摩擦课程的方向不适合续训

- c3000 已经在 joint friction `0-0.04` 全范围中训练，不是尚未接触高端点的早期策略。
- 切换到 `0-0.02` 后，训练 mean max jump 在前 50 轮便由父训练末期约 `0.506 m` 跳到约 `0.574 m`；这是动力学变容易造成的即时分布变化，不是已经学会鲁棒适应。
- 后续 mean max jump 继续升至约 `0.627 m`。课程减少了高 joint friction 样本，反而解除原有的能量约束并让策略遗忘高阻力下的平衡动作。
- 因此 `0.02 -> 0.03 -> 0.04` 更适合从零课程，不适合从已经历 full range 的 c3000 向后回退。

### 3. 高度 reward 与同步/俯仰约束实际量级严重失衡

课程末 50 轮的 TensorBoard episode reward 均值：

| Reward | 实际记录量级 |
|---|---:|
| `task_max_height` | `+10.55` |
| `base_height_flight` | `+10.26` |
| `task_ori` | `+0.89` |
| `post_landing_ori` | `+1.02` |
| `post_landing_pos` | `+0.62` |
| `takeoff_pitch_angular_impulse` | `-0.0046` |
| `termination` | `-0.044` |
| `action_rate` | `-5.15` |
| `feet_contact_forces` | `-19.35` |

- 两个高度项合计约 `+20.8`，angular-impulse 惩罚只有约 `-0.005`，实际贡献相差约四千倍。
- Height reward 仍在目标 `0.65/0.70 m` 以下提供明确上升梯度；它没有鼓励无限跳高，但在当前高度区间占据主导。
- `task_ori` 是全局/阶段姿态跟踪，未直接约束首次离地瞬间的 pitch；其值从父训练末期约 `1.01` 降到 `0.89`，损失远小于高度收益。
- Direct `front_rear_contact_mismatch` 和 direct takeoff pitch reward 在正式配置中为零，因此前后腿时差本身没有直接代价。
- Termination 是稀疏事件，虽然配置 scale 为负，episode 实际贡献只有约 `-0.04`，不足以阻止策略接受更高终止率来换高度。

### 4. Angular-impulse reward 被策略规避

同 seed、256 episodes 直接比较：

| 条件 | Height | Pitch | Mismatch | Normalized angular impulse | Takeoff torque | Max vertical velocity |
|---|---:|---:|---:|---:|---:|---:|
| c3000 robust | `0.508 m` | `10.59 deg` | `0.0577 s` | `0.0776` | `1.02` | `1.38 m/s` |
| c3500 robust | `0.595 m` | `21.18 deg` | `0.1095 s` | `0.0605` | `1.77` | `1.74 m/s` |
| c3000, fixed joint friction `0.04` | `0.448 m` | `10.18 deg` | `0.0621 s` | `0.0823` | `1.54` | `1.05 m/s` |
| c3500, fixed joint friction `0.04` | `0.505 m` | `20.51 deg` | `0.1044 s` | `0.0803` | `2.53` | `1.35 m/s` |

- Robust 下 pitch 和 mismatch 约翻倍，但 angular-impulse 指标反而下降；固定高 joint friction 下该指标也基本不变。
- 当前 reward 累计并归一化首次离地前的净俯仰角冲量。前后腿错时产生的正负角冲量可在累计值中相互抵消，更强竖直冲量又会增大归一化分母，因此策略可以获得更大瞬时 pitch/mismatch，同时保持较小的 reward 指标。
- 这与此前提高 angular-impulse 权重只能降低该指标、不能降低 pitch/mismatch 的结果一致。问题不是 `-200` 单纯太小，而是代理量与目标动作解耦；继续放大权重仍可能被规避。

### 5. 次要 incentive

- `feet_contact_forces` 和 `action_rate` 惩罚量级较大，但它们不直接约束前后腿同步。错时分配蹬伸可能在获得竖直冲量的同时规避部分瞬时动作/接触代价；这是由 reward 结构推断，尚未做独立因果验证。
- Post-landing position/orientation reward 没有直接惩罚落地后的再次失去四脚接触或角速度余振，因此不能阻止以更差落地换起跳高度。

### 当前判断

- 失败主因：从 full-range c3000 回退到 easier joint-friction range，触发高度优化；高度 reward 在当前区间占优，而 angular-impulse 代理量可被错时蹬伸规避。
- 不应继续 `0-0.03/0-0.04` 课程，也不应仅提高 angular-impulse 权重。
- 在修改正式训练前，需要先决定是保持 c3000 并停止该课程，还是做最小单变量 reward/指标修复验证；不能把 restitution/latency 接到 c3500 后面。

## 9. 不可时间抵消俯仰冲量 reward 单变量验证（2026-08-22 至 2026-08-23）

### 9.1 目标与实现

为避免原 signed net pitch impulse 被前后相反力矩在时间上抵消，新增起跳前绝对俯仰接触角冲量：每个控制步把四足接触力对机身质心产生的世界系俯仰力矩投影到机身横轴，先取瞬时绝对值再对时间积分；首次离地时用 `body_length * vertical_contact_impulse` 归一化并裁剪到 `[0, 1]`。对应事件 reward 为归一化值的平方，只在首次离地结算。

涉及实现：

- `legged_robot.py`：新增累计量、首次离地归一化、episode diagnostic 和 `_reward_takeoff_pitch_angular_impulse_abs`。
- `go2_upwards_config.py`：新增 reward scale，默认 `0.0`，不改变旧训练行为。
- `train.py`：新增 `--takeoff_pitch_angular_impulse_abs_scale_override`。
- `evaluate_checkpoint.py`：新增 `takeoff_pitch_angular_impulse_abs_normalized` 诊断输出。

旧 reward、PPO、训练机制和 full randomization 均保持不变；本轮唯一训练变量是新增 reward 的权重。

### 9.2 固定策略区分能力

robust 三种子、每种子 128 episode：

- c3000：绝对俯仰冲量均值 `0.11453`。
- 已知失败 c3500：`0.12774`，增加 `11.5%`。

固定 joint friction `0.04`、256 episode：

- c3000：`0.11548`。
- 失败 c3500：`0.13208`，增加 `14.4%`。

方向在四组条件中一致。该指标区分度有限但方向正确；原 signed 指标在失败 c3500 反而下降，因此不能替代该指标。

原始结果目录：`outputs/quadruped-jumping-ada/logs/upward_retrain/evaluations/abs_pitch_impulse_validation/`。

### 9.3 权重标定与 500 轮单变量训练

冻结 parent：`Aug21_18-22-11_com002_fromzero_s1_seed23_3000/model_3000.pt`。

- control：`Aug21_19-44-54_com002_s2_seed23_from3000_500`，新增 scale `0`。
- low：`Aug22_02-29-34_absimp50_fullrand_seed23_from3000_500`，新增 scale `-50`。
- high：`Aug22_02-40-51_absimp100_fullrand_seed23_from3000_500`，新增 scale `-100`。

三组均从同一 c3000、seed 23、原 full randomization 训练 500 iteration。旧 signed reward scale 保持 `-200`。

曾用 `-5000` 做首轮标定，新增 reward 前 50 轮均值达到 `-42.37`，跳高降到约 `0.45-0.48 m`，说明权重大两个数量级；该分支 `Aug22_02-17-07_absimp5000_fullrand_seed23_from3000_500` 仅作为失败标定，不是候选，也未运行 `-10000`。

`-50` 与 `-100` 均未造成训练崩溃，末段训练平均跳高约 `0.527-0.530 m`。

### 9.4 checkpoint 筛选与正式 robust 验收

seed 341、128 episode 扫描 c3300/c3350/c3400/c3450/c3500 后：

- control 后段重新出现高俯仰和接触错位。
- low 的改善不连续。
- high 在 c3300/c3350 形成相邻候选区间；c3400 出现终止和四足落地退化，c3500 又以成功率换取低俯仰，因此不得按最后 checkpoint 选取。

对 parent、high c3300、high c3350 做 seeds 342/343/344、每种子 256 episode 的正式 robust 验收。三种子均值：

| checkpoint | height mean | success | pitch abs | mismatch | abs impulse | termination | stable | all-feet ratio | landing angular RMS | torque sat. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| parent c3000 | 0.5028 | 0.4857 | 10.150 deg | 0.0569 s | 0.1206 | 0.1328 | 0.2474 | 0.8928 | 0.3578 | 0.00512 |
| high c3300 | 0.5069 | 0.5065 | 10.709 deg | 0.0620 s | 0.0873 | 0.1250 | 0.1953 | 0.8695 | 0.3954 | 0.00399 |
| high c3350 | 0.5079 | 0.5287 | 10.583 deg | 0.0476 s | 0.0855 | 0.1016 | 0.2839 | 0.9086 | 0.3506 | 0.00451 |

结论：

- c3300 虽降低绝对冲量，但 mismatch、stable standing 和落地角速度退化，不作为最终候选。
- c3350 相对 parent：成功率 `+4.30 pp`，mismatch `-16.4%`，绝对冲量 `-29.1%`，终止率 `-3.12 pp`，stable `+3.65 pp`；高度、四足落地、落地角速度和力矩饱和均不退化。
- c3350 是本轮唯一综合候选。不得继续使用 c3400 以后 checkpoint，也不能只按新增 reward 数值选模型。

结果目录：

- `outputs/quadruped-jumping-ada/logs/upward_retrain/evaluations/absimp_ab_screen/`
- `outputs/quadruped-jumping-ada/logs/upward_retrain/evaluations/absimp_full_robust/`

### 9.5 敏感端点边界

seed 345、每项 256 episode；除被固定参数外，其余保持 robust 随机化：

- joint friction `0.04`：parent/c3350 success `2.34%/6.25%`，mismatch `0.0520/0.0427`，abs impulse `0.1310/0.0873`。候选有相对改善，但两者绝对能力都不合格，且 c3350 all-feet ratio `0.8646` 低于 parent `0.9495`；该端点未解决。
- restitution `0.4`：success `67.19%/71.09%`，termination `33.98%/18.36%`，mismatch `0.0591/0.0512`；候选明确改善。
- latency `40 ms`：success `64.84%/73.83%`，pitch `13.078/9.666 deg`，mismatch `0.0505/0.0358`；但 termination 仍约 `44%`、all-feet ratio 仍约 `0.54`，且 abs impulse 从 `0.1093` 增至 `0.1422`。候选只改善部分表现，40 ms 仍未达标且不能声称已覆盖。

结果目录：`outputs/quadruped-jumping-ada/logs/upward_retrain/evaluations/absimp_endpoint_eval/`。

### 9.6 nominal 人工验收与正式建议

nominal 视频：`artifacts/solo12_upward_retrain/videos/abs100_c3350_nominal.mp4`。

已按“先说明、后视频”发送到 Claw：

- 说明消息：`om_x100b6791c693b0a8b25f07ecc72d535`
- 视频消息：`om_x100b6791c74cc4a8b4927309b110c2d`

正式续训建议：

1. 将 high c3350 保留为当前 upward 候选，等待人工检查 nominal 视频中的落地后晃动和补跳；不要用 c3400-c3500 替换。
2. 后续继续保留全部旧 reward、训练机制和 abs impulse scale `-100`，每 50 轮做综合 checkpoint 选择，禁止只优化单一新增指标。
3. 不直接扩大所有随机化。优先按旧训练方式单变量验证并课程化 latency 到 `40 ms`；joint friction `0.04` 端点需先验证范围/课程设计，不能把当前绝对能力不足误写为已适应。
4. COM 若需调整，优先沿用旧训练的 COM 配置与课程方式；其他随机化参数只有在固定策略扫描或足够轮数短训证明因果后再调整。
5. 下一阶段验收继续综合看高度区间、成功率、起跳同步/俯仰、接触错位、终止、四足落地、落地角速度、补跳和力矩饱和，不以“稳定率”或“跳高”单项决定。
