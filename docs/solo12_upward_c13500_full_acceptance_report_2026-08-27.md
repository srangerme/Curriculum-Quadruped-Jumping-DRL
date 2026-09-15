# Solo12 Upward c13500 完整验收报告

## 1. 对象与结论

- Checkpoint：`Aug27_08-33-31_upward_height_c13200_full028034_recovery_p1_seed31_300/model_13500.pt`
- 协议：PhysX；不使用 MuJoCo 作为训练准出依据。
- **训练侧 PhysX 准出：通过。**
- **部署就绪：未通过。** MuJoCo 已出现落地后 observation/action 发散，需先完成逐 policy-step 对齐，不能由本报告外推实机稳定性。
- 妥协项：落地后腿间距偏小；40 ms + restitution 0.2 的稳定率下降；pitch/timing 尾部高于期望。

## 2. 验收协议

- 主矩阵：`A/B/Full × p0/p1 × 3 seeds × 128 episodes`，共 2304 episodes。
- 随机化：observation noise 开启，latency `[0,40] ms`，joint friction `[0,0.02]`，restitution `[0,0.2]`，其余使用既有 Full 随机化。
- A：`contact/rest/friction_offset_threshold=0.010/0/0.010 m`。
- B：`0.005/0/0.005 m`。
- Full：系统默认 PhysX 接触参数。
- 端点矩阵：`latency 0/40 ms × restitution 0/0.2 × p0/p1 × 3 seeds × 128 episodes`，共 3072 episodes；joint friction 固定 0.02。
- 初始高度：`0.28-0.34 m × Full p1 × 3 seeds × 96 episodes`，共 2016 episodes。
- 落地稳定阈值：全足接触比例 0.95、线速度 RMS 0.1 m/s、角速度 RMS 0.5 rad/s、姿态误差 RMS 0.2 rad、腿力矩 CV 0.25。

## 3. 主矩阵结果

| Case | Height mean/P10 | Success | Stable | Termination |
|---|---:|---:|---:|---:|
| A p0 / p1 | `0.632/0.578`, `0.638/0.585` | `99.5/100%` | `89.8/89.6%` | `3.4/4.4%` |
| B p0 / p1 | `0.631/0.579`, `0.637/0.584` | `99.5/100%` | `88.8/86.7%` | `4.7/5.2%` |
| Full p0 / p1 | `0.632/0.578`, `0.638/0.585` | `99.5/100%` | `90.6/90.1%` | `3.4/4.2%` |

跨 seed 最差值：success `98.4%`，stable `83.6%`，termination `6.25%`。A/B/Full 无系统性分叉，主随机化范围通过。

落地聚合值：线速度 RMS `0.069-0.071 m/s`、角速度 RMS `0.204-0.209 rad/s`、姿态误差 RMS `0.087-0.091 rad`、腿力矩 CV `0.089-0.092`、力矩饱和比例约 `0.53%`，均满足既定稳定阈值。

## 4. 起跳质量

- 前后足主要竖直力时序差：mean `16.4 ms`，P50 `16.2 ms`，P90 `26.9 ms`，max `41.6 ms`。
- takeoff pitch：mean `9.2 deg`，P50 `9.2 deg`，P90 `13.9 deg`，max `21.9 deg`。
- 起跳竖直速度：mean `1.99 m/s`；最高点聚合 mean `0.635 m`。
- pitch cancellation normalized：mean `0.070`，P90 `0.094`。

判定：均值满足时序 25 ms和 pitch 12 deg目标，但尾部超出；记为观察项，不阻塞当前 Upward 准出，进入 Forward 后不得通过扩大前后腿错时来获得距离。

## 5. 固定端点

| Latency / restitution | p0 success/stable | p1 success/stable | Height p0/p1 | Termination p0/p1 |
|---|---:|---:|---:|---:|
| 0 ms / 0 | `99.2/98.7%` | `99.5/98.7%` | `0.573/0.572 m` | `0/0.8%` |
| 0 ms / 0.2 | `100/98.7%` | `100/97.4%` | `0.608/0.606 m` | `0.8/0.8%` |
| 40 ms / 0 | `98.2/91.1%` | `99.0/90.4%` | `0.556/0.555 m` | `3.6/3.4%` |
| 40 ms / 0.2 | `100/78.1%` | `99.7/81.8%` | `0.590/0.591 m` | `9.9/9.6%` |

低延迟能力通过。40 ms 成功率通过放宽边界，但 restitution 0.2 下稳定率约 `78-82%`，且落地关节速度 RMS `1.19-1.21 rad/s`，是当前明确的落地敏感角落。

## 6. 初始站立高度

| Height | Success | Stable | Mean height | Termination |
|---|---:|---:|---:|---:|
| 0.28 | 98.6% | 88.5% | 0.630 m | 3.8% |
| 0.29 | 96.9% | 86.5% | 0.627 m | 5.2% |
| 0.30 | 96.5% | 87.2% | 0.627 m | 4.9% |
| 0.31 | 98.6% | 88.2% | 0.635 m | 5.9% |
| 0.32 | 100% | 89.6% | 0.641 m | 3.5% |
| 0.33 | 99.3% | 90.3% | 0.639 m | 3.1% |
| 0.34 | 98.3% | 89.6% | 0.636 m | 5.2% |

结论：正确物理姿态下，`0.28-0.34 m` 的全接触初态均通过。此前 `nominal p1=0%` 使用默认 `0.12 m` reset 高度并强制全接触 history，是无效协议，不能作为 contact-history 敏感证据。

## 7. 落地形态与二次起跳

- 0.32 m Full p1 三 seed：后腿 touchdown 约 `0.102 m`，settled 约 `0.075 m`，落地后继续收缩约 `26 mm`；该项由人工接受。
- p0/p1 代表性完整轨迹中，主腾空后均未出现第二段四足全离地，最终四足接触。
- 三视角 nominal 视频未观察到二次起跳或持续可见抖动。
- 限制：现有 trajectory 导出只记录单环境，二次起跳结论是轨迹加视频验收，不是 2304 episodes 的批量统计。

## 8. 部署反馈的解释

部署侧已排除 MNN 数值误差、URDF 不一致、GT 开关和 0.32 m 起始高度；故障发生在落地后，表现为 raw action 持续极端并最终打到关节限位。这证明当前模型尚未通过 MuJoCo/部署链路，但不能由无效的 `p1=0%` 测试推导为“全接触 history 本身敏感”。

现有证据只能把范围收敛到两类：

1. MuJoCo 接触冲量、刚度、滑移等造成的落地状态超出 PhysX 稳定分布；
2. 落地后的 observation/history 构造与训练侧不一致。

下一步先记录部署侧逐 policy-step 的完整 1014 维输入与 12 维 raw action，并与训练侧同阶段对齐 contact history、四元数、角速度和 action history；再用 exact observation/action replay 区分“动力学 OOD”和“观测链路错误”。在因果定位前，不建议直接追加 reward 或盲目续训。

## 9. 最终判定

- **Upward / PhysX 训练准出：PASS。**
- **初始高度 0.28-0.34 m：PASS。**
- **40 ms + restitution 0.2：放宽通过，保留风险。**
- **后腿收拢：人工妥协。**
- **MuJoCo/部署：FAIL，待逐步对齐定位。**

原始结果目录：`outputs/quadruped-jumping-ada/logs/c13500_full_acceptance/`。
