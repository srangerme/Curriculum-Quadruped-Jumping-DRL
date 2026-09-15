# Solo12 upward c3000 不达标原因定位报告（2026-08-22）

## 结论

- 最终候选仍冻结为 `Aug21_18-22-11_com002_fromzero_s1_seed23_3000/model_3000.pt`；本轮两个短训 run 只用于原因验证，均不作为候选。
- c3000 的问题不是单一参数，也不能只解释成“模型当前能力不足”：nominal 高度只有约 `0.520 m`，确有基础能力缺口；同时固定参数扫描证明高 joint friction、latency 和 restitution 会以不同机制进一步放大失败。
- S4 high 的主要组合是：高 joint friction 压低起跳做功，40 ms latency 放大俯仰、时序和落地误差，高 restitution 增加回弹；COM 正偏置只在该组合下进一步放大 pitch/mismatch。高地面摩擦、damping、PhysX A/B/C 和单独 COM 不是主因。
- 单纯延长原训练已被旧 `c3050-c3500` 趋势否定；高 joint friction 定向短训又学成“少跳但稳定”的保守策略；高 restitution 定向短训能改善存活和高度，却通过更大的 pitch/mismatch 完成任务。两者都没有形成综合 Pareto 改善。
- 下一步应从 c3000 做逐级、单变量的随机化课程，而不是从诊断分支续训、直接上 40 ms、继续扫 reward 权重或从头重训。

## 范围与口径

- 只使用训练侧 PhysX，不使用 MuJoCo、locomotion 或实机验收。
- 固定策略扫描使用 c3000，不更新 policy。
- 随机 holdout 共 `1152` episodes：A/B/C 各 `3 seeds x 128`。
- 固定值扫描主要使用 `256` episodes、同一 seed；S4 单项/交互使用 deterministic nominal。
- 训练目标高度为 `0.65 m`，期望区间约 `0.6-0.7 m`；评估器的 task success 阈值仍是 `0.50 m`。因此“success”不等于高度目标达标。
- strict stable 同时要求落地后四脚接触、速度/姿态/腿间力矩振荡和自然 timeout；它比 task success 更严格。

## c3000 基线

| 条件 | Height | `>=0.50 m` | `>=0.62 m` | Pitch | Mismatch | Stable | Termination |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nominal | `0.520 m` | `100%` | `0%` | `2.22 deg` | `0.020 s` | `100%` | `0%` |
| Robust holdout | `0.505 m` | `51.1%` | `3.56%` | `9.78 deg` | `0.0556 s` | `26.0%` | `15.45%` |

Robust 的高度 P10 为 `0.429 m`，落地四脚接触比例约 `0.868`，落地角速度 RMS 约 `0.356 rad/s`。A/B/C 三桶的高度、pitch、stable 和 termination 差异很小，接触桶不是主导变量。

## 固定策略因果扫描

### 主要参数

| 参数 | 低/中/高结果 | 判断 |
|---|---|---|
| Joint friction | 固定 `0/0.02/0.04` 时 height `0.573/0.505/0.438 m`，success `92.2/47.7/2.7%` | 高端直接造成起跳做功能力不足，是高度失败主因 |
| Restitution | 固定 `0/0.2/0.4` 时 height `0.445/0.515/0.534 m`；高端 termination `36.7%`、四脚接触 `0.701` | 回弹提高高度，但恶化吸能、持续接触和终止 |
| Latency | 固定 `0/10/20/30/40 ms` 时 termination `9.4/16.0/31.3/24.6/37.5%`；40 ms 四脚接触 `0.623` | 主要破坏动作时序和落地，不是单纯压低高度 |
| Ground friction | `0.01` 退化；`1.5/3.0` 接近基线 | 极低端有风险，高端不是 S4 high 主因 |
| Joint damping | `0/0.005/0.01` 基本重合 | 当前范围内不是主因，不应优先强化 |
| COM | 单轴 `-0.02/0/+0.02 m` 对高度影响小；COM x 正移使 pitch 约 `8.79 -> 12.38 deg` | 单独不是主因，但能放大俯仰裕量问题 |

### 落地晃动、二次起跳和终止

- Robust baseline 的 strict stable 只有约 `24-26%`；四脚接触比例约 `0.87-0.89`，说明落地后的卸载/再次离地不是纯视觉现象。
- Restitution `0.4` 时四脚接触降至 `0.701`、落地角速度 RMS 升至 `0.429 rad/s`、termination 升至 `36.7%`。
- Latency `40 ms` 时四脚接触降至 `0.623`、落地角速度 RMS 升至 `0.402 rad/s`、termination 为 `37.5%`；其中约 `27.0%` episode 触发 landing-error termination。
- Robust baseline 的 termination 几乎全部是 post-landing position 越界；这说明终止是落地后运动的结果，不是 contact/orientation termination 本身。
- Nominal 数值验收仍给出四脚接触 `1.0`、stable `100%`，但人工视频能看到余振和小跳，说明当前 strict-stable 阈值仍不足以代替人工观感。报告只能把四脚接触下降作为二次起跳代理量，不能宣称 nominal 小跳已经解决。

## S4 high endpoint 拆解

Nominal 单项结果：

| 条件 | Height | Pitch | Mismatch | Termination |
|---|---:|---:|---:|---:|
| Baseline | `0.520 m` | `2.22 deg` | `0.020 s` | `0%` |
| Joint friction `0.04` | `0.383 m` | `3.66 deg` | `0.0025 s` | `0%` |
| Latency `40 ms` | `0.653 m` | `9.15 deg` | `0.040 s` | `100%` |
| Restitution `0.4` | `0.535 m` | `0.82 deg` | `0.0012 s` | `0%` |
| Damping/friction/单轴 COM/contact C | `0.514-0.520 m` | 无系统性崩溃 | 无系统性崩溃 | `0%` |

关键交互：

| 组合 | Height | Pitch | Mismatch | Termination |
|---|---:|---:|---:|---:|
| Joint friction + latency | `0.386 m` | `10.56 deg` | `0.020 s` | `0%` |
| Joint friction + restitution | `0.438 m` | `3.87 deg` | `0.028 s` | `0%` |
| Latency + restitution | `0.676 m` | `9.32 deg` | `0.0275 s` | `100%` |
| 三者组合 | `0.465 m` | `10.85 deg` | `0.059 s` | `100%` |
| 三者 + damping | `0.476 m` | `12.77 deg` | `0.060 s` | `100%` |
| 三者 + damping + COM `(+0.02,+0.02,+0.02)` | `0.486 m` | `15.21 deg` | `0.080 s` | `100%` |

最后一行已基本复现原 S4 high 的 `0.488 m / 16.56 deg / 0.0803 s / 100% termination`。因此 high endpoint 不是“所有 high 参数都差”，而是 joint friction 的能量损失与 latency/restitution 的时序、回弹效应耦合，COM 再放大俯仰和前后腿错配。

## 必要短训验证

### 高 joint friction 定向分支

- Run：`Aug21_23-59-40_diag_jf030040_seed23_from3000_200`
- 唯一变化：训练 joint friction 范围由 `0-0.04` 改为 `0.03-0.04`；共 `200` iterations。
- 固定 joint friction `0.04`：控制组 `c3050-c3200` height 约 `0.441-0.447 m`、success `2.7-7.0%`；定向组 height 约 `0.423-0.430 m`、success `0-0.4%`。
- 定向组同时把 pitch 降到约 `5.3-6.6 deg`、mismatch 降到 `0.022-0.031 s`、termination 降至 `1.2-3.5%`，四脚接触升至 `0.97-0.98`。
- 解释：policy 确实响应了训练分布，但选择了减少起跳做功的保守解。当前失败不只是“没有采到 high endpoint”，继续 high-only 专训方向错误。200 轮不能证明永远不可收敛，但趋势不支持继续投入。

### 高 restitution 定向分支

- Run：`Aug22_00-12-50_diag_rest030040_seed23_from3000_200`
- 唯一变化：训练 restitution 范围由 `0-0.4` 改为 `0.3-0.4`；共 `200` iterations。
- 最早 `c3050` 在固定 restitution `0.4` 下达到 height `0.543 m`、success `73.4%`、termination `13.3%`、stable `43.8%`、四脚接触 `0.875`、落地角速度 RMS `0.354 rad/s`。
- 代价是 pitch `10.94 deg`、mismatch `0.069 s`，均差于原 c3000 固定 `0.4` 的约 `9.98 deg / 0.062 s`。
- `c3100` 进一步提高到 `0.563 m / 85.5%`，但 pitch/mismatch 恶化到 `13.52 deg / 0.086 s`，不是 Pareto 改善。
- 解释：高回弹落地是可部分学习的，但当前 reward/策略会通过更激进、前后腿更不同步的起跳来换高度和存活，不能将该分支作为候选。

### Latency 已有训练证据

- 已有 `0-30 ms` run `Aug21_21-20-53_latency30_com002_s3_seed23_from3000_500`。
- `c3050/c3100` 虽略有 success 或 pitch 改善，但四脚接触、落地振荡、姿态误差或 mismatch 恶化；训练到 `500` 轮仍无 Pareto 候选。
- 因 `0-30 ms` 尚未通过，不再直接训练 `0-40 ms`。40 ms 当前既是覆盖缺口，也是已被固定扫描证明的高风险条件。

## 原因分类

### 1. 高度不足

- **基础能力缺口**：nominal 只有 `0.520 m`，离 `0.6-0.7 m` 目标有距离；这部分不能归因于随机化。
- **确定的动力学敏感性**：joint friction 高端可额外损失约 `0.14 m` nominal 高度，并在 robust 下把成功率降到约 `3%`。
- **不是简单多训不足**：原配置续训和 high-joint-friction 定向短训都没有恢复高端高度；前者在后期转向高 pitch/mismatch，后者转向少跳。

### 2. 前后腿错配和机身俯仰

- Latency 是最明确的时序放大器；COM x 正移会进一步放大 pitch。
- 当前 angular-impulse reward 能约束其直接测量量，但既有 `-300/-400`、direct mismatch 和 direct pitch 实验均被策略绕过，不能靠继续加权解决。
- Restitution 定向训练再次复现“高度/成功上升时 pitch/mismatch 变差”，说明这是策略规避路径，而非 evaluator 偶然相关。

### 3. 落地晃动和小跳

- 高 restitution 增加回弹，高 latency 让着地动作滞后，两者都显著降低持续四脚接触并提高落地角速度/终止。
- Nominal 人工可见余振未被当前 strict-stable 判失败，说明还存在验收盲区。后续每阶段必须继续发 nominal 视频人工综合判断；不能只依赖 stable 数字。

### 4. S4 high 全失败

- 直接原因是 joint friction、latency、restitution 三者耦合；damping 与 COM 主要是组合放大项。
- 这不是 PhysX A/B/C 或高地面摩擦单项导致，也不支持优先强化 damping。

## 最小必要续训建议

1. **起点保持 c3000。** 两个诊断分支均拒绝；不从它们续训，也不立即重新从零。
2. **正式 reward 保持不变。** 保留旧 reward + `takeoff_pitch_angular_impulse=-200`；不启用已失败的 direct mismatch/direct pitch，不继续做 reward 权重搜索。
3. **先做 joint-friction 课程，不做 high-only。** 从 c3000 依次使用上限 `0.02 -> 0.03 -> 0.04`，每级短训 `100-200` iterations；其他随机化、COM `xyz +/-0.02 m`、latency `0-20 ms` 全部冻结。每 `50` 轮同时验收常规 robust 和固定 `joint_friction=0/0.02/0.04`。
4. **只接受 Pareto checkpoint。** 高度/P10 和高 joint-friction 起跳必须改善，同时不得用明显更高的 pitch、mismatch、termination、落地角速度或更低的四脚接触换取；否则立即回到 c3000，不靠加长轮数等待反转。
5. **随后单独做 restitution 课程。** 仅在 joint-friction 阶段得到新 Pareto 候选后，再按 `0-0.2 -> 0-0.3 -> 0-0.4` 逐级训练。高 restitution 定向 `c3050` 可作为“可适应但同步恶化”的参考，不可作为起点。
6. **最后重试 latency 课程。** 仍按旧 upward 方式 `0-20 -> 0-30 -> 0-40 ms`；只有新候选通过 30 ms 综合验收后才进入 40 ms，不直接跳级。
7. **每级人工验收。** 先发阶段说明，再发 nominal 视频到 `Claw`；重点看落地余振、小跳和前后腿同步。数值同时检查高度分位数、pitch、mismatch、角冲量、终止、力矩饱和、strict stable、四脚接触及落地线/角速度。
8. **停止条件。** 若 joint-friction 课程仍重复“少跳”或“高 pitch/mismatch 换高度”，说明问题已不是训练轮数或单一随机化范围，需要重新评估 policy 可观测性/动作时序表达或 reward 可规避性，再决定是否从零重训。

## 产物

- 固定策略扫描：`outputs/quadruped-jumping-ada/logs/upward_retrain/evaluations/c3000_diagnosis/`
- 短训对照评估：`outputs/quadruped-jumping-ada/logs/upward_retrain/evaluations/c3000_diagnosis_training/`
- High joint-friction effective config：`outputs/quadruped-jumping-ada/logs/upward_retrain/diag_jf030040_seed23_from3000_200_effective.json`
- High restitution effective config：`outputs/quadruped-jumping-ada/logs/upward_retrain/diag_rest030040_seed23_from3000_200_effective.json`
- 为构造单变量诊断，`train.py` 新增 `--joint_friction_range_override`；默认配置和正式训练行为不变。

当前状态：原因定位完成，正式候选仍为 c3000，尚未开始建议中的正式续训。
