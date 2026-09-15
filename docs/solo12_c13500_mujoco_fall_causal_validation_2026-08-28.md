# Solo12 c13500 MuJoCo 倒地因果验证

## 固定条件与判定方式

- 模型：upward c13500，MNN 数值验证已通过。
- 部署条件：`use_gt_data=true`，起始站高 `0.32 m`，目标 `(0, 0, 0)`。
- 主判定不是最终成功/失败，而是完整真实时间轴：下探、离地、腾空、首次触地、落地后 1 s 和后续稳定段。
- 比较量：1014 维观测分块、12 维 raw action、关节角/角速度、机身 pitch/角速度、接触时序、法向冲量、接触占空和动作饱和时刻。

## 关键结果

训练 nominal 的离地/落地为 `0.340/0.840 s`，腾空 `0.500 s`。原始 MuJoCo elliptic 配置的完整轨迹通常在 `0.340-0.360 s` 离地、`0.900-0.920 s` 落地，腾空多 `40-60 ms`。

稳定与发散轨迹的决定性分歧发生在落地接触尾部：

| 轨迹 | 落地后 0.2-1.0 s 接触占空 | 动作峰值 | 动作 RMSE（对训练） |
|---|---:|---:|---:|
| elliptic 稳定样本 | 0.921 | 15.79 | 1.26 |
| elliptic 发散样本 | 0.043-0.144 | 100 | 67.9-76.4 |
| pyramidal 稳定样本 | 0.906-0.976 | 14.04-15.36 | 0.69-0.95 |

发散链路是：落地后持续接触丢失，随后 contact/action history 偏离训练轨迹，raw action 在落地后约 `0.18-0.56 s` 打到 `100`。因此动作饱和是接触轨迹分歧的下游放大，不是最初原因。

## 单变量与组合验证

| 修改 | 完整轨迹结论 |
|---|---|
| 摩擦 `0.7 -> 1.0` | 未解决；落地后接触占空 0.071，动作打到 100，水平位移增大 |
| `condim 6 -> 3` | 未解决；接触占空 0.044，动作打到 100 |
| `solref 5 ms -> 2 ms` | 只延后发散；腾空增至 0.620 s，最终动作仍到 100 |
| `impratio 100 -> 1`（保持 elliptic） | 不稳定，只有 1/3 条无动作饱和，不能解释 pyramidal 的全部改善 |
| `cone elliptic -> pyramidal` | 显著改善，3 条中 2 条完整稳定；轨迹误差、落地接触和位移同时改善 |
| `pyramidal + condim=3` | 比 pyramidal 单独修改更差，不采用 |
| 仅固定 policy reset 相位，保持 elliptic | 3 条中 2 条稳定，仍有 1 条发散 |
| `pyramidal + 固定 policy reset 相位` | 3/3 条落地恢复轨迹无动作饱和；动作 RMSE 1.01-1.36，关节角 RMSE 0.199-0.205 rad，腾空 0.520-0.580 s |

组合验证第三条在落地后 1 s/2 s 高度仍为 `0.311/0.314 m`，之后在控制阶段结束附近触发 `RollFail`；这不是原始的落地后策略动作发散，需作为独立的状态切换问题处理。

## 结论

主要物理差异是 MuJoCo 当前 `elliptic` 摩擦锥产生的接触/冲量轨迹与训练侧 PhysX 不一致；主要时序差异是部署侧 policy 线程 `Reset()` 清历史但不重置 20 ms 唤醒相位，导致 Jump 入口存在 `0-20 ms` 相位漂移。二者共同把落地 contact history 推出 c13500 的稳定分布，随后 action history 正反馈并发散。

不能把原因简化为摩擦系数、`condim`、`solref` 或 `impratio` 单项。`pyramidal` 是有效物理方向，固定 reset 相位是有效时序方向，但本次只做因果验证，不保留部署改动。

## 证据路径

- 汇总指标：`/home/sranger/codes/sranger/robotcontrol-trains/artifacts/solo12_mujoco_c13500_diagnosis/single_variable_full_trajectory_metrics.json`
- pyramidal 复现：`/home/sranger/codes/sranger/robotcontrol-trains/artifacts/solo12_mujoco_c13500_diagnosis/repeat_full_trajectory_metrics.json`
- fixed-phase 组合：`/home/sranger/codes/sranger/robotcontrol-trains/artifacts/solo12_mujoco_c13500_diagnosis/pyramid_phase_reset_full_trajectory_metrics.json`
- 首次分歧明细：`/home/sranger/codes/sranger/robotcontrol-trains/artifacts/solo12_mujoco_c13500_diagnosis/pyramidal_first_divergence.txt`
- 全部原始轨迹：`/home/sranger/codes/sranger/robotcontrol-trains/artifacts/solo12_mujoco_c13500_diagnosis/traces/`
