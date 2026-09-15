# Solo12 Upward 最终有效训练记录（2026-08-27）

本文是当前规范记录，只保留最终 `c12600` 实际继承且经过验收的训练过程。旧文档中的失败分支、临时 reward 和未采用候选不属于复训路线。

## 1. 目标与设计

- 目标动作：从四足接触站立主动下探起跳，最高点主要落在 `0.60-0.70 m`，起跳 pitch 可控，落地后稳定站立且不二次四足离地。
- 高度与姿态联合：`takeoff_vz_pitch_quality=1000` 在首次真实离地时联合评价弹道预测高度和 pitch，避免只追求跳高。
- 抑制俯仰抵消：保留 `takeoff_pitch_angular_impulse=-200`、`takeoff_pitch_angular_impulse_abs=-100`，约束前后腿用相反大冲量抵消姿态。
- 课程顺序：先学 p0/p1 初始历史，再逐级扩大 latency，最后扩大 restitution。原因是同时开放这些维度会使 PPO 利用特定延迟相位或地面回弹形成捷径。
- 随机化边界：Solo12 joint friction 最终采用 `[0,0.02]`；`0.04` 不作为准出要求。S4优先保留低延迟能力，40 ms作为允许退化的边界能力。
- 验收只使用 PhysX统计和人工视频，不使用MuJoCo作为准出依据。

固定控制配置：reset height `0.32 m`、settle `1 step`、`Kp/Kd=16/0.5`、`action_scale=0.25`、20 ms policy period、8 Hz EMA action filter、action clip `100`、COM xyz各 `[-0.02,0.02] m`。最终训练覆盖 joint friction `[0,0.02]`、restitution `[0,0.2]`、observation latency `[0,40] ms`；其余质量、PD、电机和接触随机化保持既有full-randomization配置。

## 2. 最终有效checkpoint链

| 阶段 | 实际继承链与run | 设计及有效结论 |
|---|---|---|
| S1 从0形成动作 | `c0→c2000`：`Aug24_12-25-12_upward_vzpitch_reset032_fromzero_s1_seed31_2000`；`c2000→c2400`：`Aug24_13-10-27_upward_vzpitch_reset032_s1_seed31_from2000_1000`；`c2400→c2600`：`Aug24_14-07-37_upward_reset032_s1_heightsigma001_seed31_from2400_300`；`c2600→c2800`：`Aug24_14-17-28_upward_reset032_s1_heightsigma002_seed31_from2600_200`；`c2800→c3100`：`Aug24_14-25-32_upward_reset032_s1_heightsigma002_seed31_from2800_500` | seed31 `c3100`形成足够高度且较小pitch/主受力时序差；predicted-height sigma最终取`0.02`。独立seed比较后选seed31。 |
| S2 初态混合与稳定 | `c3100→c3900`：`Aug24_18-28-25_upward_reset032_s2_seed31_contactmix050_from3100_800`；`c3900→c4500`：`Aug24_18-51-21_upward_reset032_s2_seed31_contactmix050_from3900_1000`中的`c4500`；`c4500→c4900`：`Aug25_08-25-29_upward_reset032_ab_postori6_postdofvelmean050_seed31_from4500_400` | p0/p1按50/50混合，使策略同时覆盖初始无接触历史与全接触历史；`post_landing_ori=6`、`post_landing_dof_vel=-0.5`、grace `0.8 s`后，`c4900`落地姿态和站立关节抖动通过人工核验。 |
| S3 p0/p1课程 | `c4900→c5500→c6100→c6700`：`Aug26_12-18-32...l0_p050...`、`Aug26_12-35-23...h1_p075...`、`Aug26_12-51-46...h2_p050...` | latency固定0，先增加p1暴露再恢复混合；验证策略能够从全接触观测主动下探起跳，同时保留p0能力。 |
| S3 latency课程 | `c6700→c7000→c7600→c8200`：`Aug26_13-53-07...l1a...lat010...`、`Aug26_14-27-40...l1a2...`、`Aug26_14-53-25...l1b...`；`c8200→c8500`：`Aug26_15-15-35...l2a...lat020...`；`c8500→c9100`：`Aug26_16-24-33...l3...lat030...` | latency依次扩大到10/20/30 ms，并通过零延迟与最大延迟端点采样保护低延迟能力；每级至少适应300-600轮后才选checkpoint。 |
| S3 restitution课程 | `c9100→c9700→c10100`：`Aug26_16-50-18...r0_rest000...`、`Aug26_17-08-44...r0b...`；`c10100→c10700`：`Aug26_17-33-33...r1_rest000_010...`；`c10700→c11300`：`Aug26_17-50-44...r2_rest000_020...` | 在完整latency范围下，restitution从fixed 0逐步扩到`[0,0.1]`和`[0,0.2]`，避免策略只依赖地面回弹。`c11300`通过S3。 |
| S4 latency 40 ms | `c11300→c12050`：`Aug27_02-04-29_upward_reset032_s4_latency40_rest020_jf002_seed31_from11300_800`中的`c12050`；`c12050→c12600`：`Aug27_02-50-26_upward_reset032_s4_p0recover_lat40_rest020_jf002_seed31_from12050_600`中的`c12600` | 第一段扩到40 ms并适应750轮；第二段以p0 recovery训练550轮，恢复低延迟/p0能力。`c12600`是低延迟与40 ms边界能力的最佳折中。 |

## 3. 验收项目与结论

### 统计验收

- 协议：A/B/Full × p0/p1 × 3 seeds × 128 episodes，`eval_mode=robust`、observation noise开启；latency `[0,40] ms`、joint friction `[0,0.02]`、restitution `[0,0.2]`。
- A：`contact_offset/rest_offset/friction_offset_threshold=0.010/0/0.010 m`；B：`0.005/0/0.005 m`；Full：默认PhysX参数。
- 指标：高度均值/P10、success、strict stable、termination、takeoff pitch、前后足主要竖直力时间质心差。累计轻接触形成的contact tail仅记录，不作为硬门槛。

| Case | Height mean/P10 | Success | Stable | Termination | Main timing | Pitch |
|---|---:|---:|---:|---:|---:|---:|
| A p0 / p1 | `0.589/0.530`、`0.599/0.541` | `93.5/98.2%` | `87.2/91.9%` | `6.5/4.4%` | `15.4/14.4 ms` | `5.5/5.3 deg` |
| B p0 / p1 | `0.589/0.525`、`0.597/0.541` | `92.7/97.7%` | `85.9/89.8%` | `6.5/4.4%` | `13.5/13.2 ms` | `5.4/5.2 deg` |
| Full p0 / p1 | `0.589/0.531`、`0.599/0.541` | `93.5/98.2%` | `87.0/92.7%` | `6.5/4.4%` | `15.3/14.4 ms` | `5.5/5.3 deg` |

结论：A/B/Full一致；主要受力时序均低于25 ms，说明高度、身平和前后腿主要输出同步可以同时达到。

### 端点与人工验收

- 0 ms/rest=0：p0/p1 success `81.4/89.8%`，termination `2.5/0.4%`；低延迟能力保留。
- 40 ms/rest=0：success `72.1/76.4%`；40 ms/rest=0.2：success `89.6/93.9%`、termination `9.6/11.9%`。40 ms明确作为放宽的边界项，不与低延迟要求同级。
- grounded p1 nominal：从四足接触站立主动下探，主腾空`0.34-0.82 s`；落地后至3 s无二次四足离地，未见持续关节抖动或屁股抬高。
- 最终人工接受：不再要求C桶`rest_offset=-0.002 m`硬角落；contact tail降级为观察项；40 ms允许成功率和高restitution落地指标下降。

## 4. 最终产物

- Checkpoint：`/home/sranger/codes/sranger/robotcontrol-trains/outputs/quadruped-jumping-ada/logs/test_solo12_v3_1/Aug27_02-50-26_upward_reset032_s4_p0recover_lat40_rest020_jf002_seed31_from12050_600/model_12600.pt`
- MNN：`/home/sranger/codes/sranger/robotcontrol-trains/outputs/model-conversion/solo12_upward_reset032_s4_c12600/solo12_upward_reset032_s4_c12600.mnn`
- Nominal视频：`/home/sranger/codes/sranger/robotcontrol-trains/artifacts/solo12_upward/videos/c12600_s4_final_reset032_p1_nominal.mp4`
- Grounded case视频目录：`/home/sranger/codes/sranger/robotcontrol-trains/artifacts/solo12_upward/videos/c12600_cases_grounded_p1/`

最终结论：`c12600`按“低延迟主工作域优先、40 ms边界放宽”的标准通过Upward准出。按本记录复训的目标是统计等价，不保证PPO/PhysX下逐参数或逐checkpoint完全一致。
