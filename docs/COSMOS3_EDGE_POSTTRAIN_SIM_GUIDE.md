# Cosmos3 Edge domain projector 后训练模型的 RoboLab 仿真指南

核对日期：2026-11-16。适用模型为 MOZ1 数据上后训练的 Cosmos3 Edge Policy，`domain_name=moz1`、`domain_id=21`、20 维 Cartesian action。目标是把真实模型预测接到现有 QX 仿真链路，再开展任务评测。

当前可以直接请求后训练模型、保存完整预测，再通过已有 `--motion-smoke` 执行双臂和躯干 IK。正式任务成功率还需要完整的 QX 任务注册、正确的 state/action frame 和夹爪控制；现有运动诊断没有这些完整能力。历史 action 回放顺畅是有用的执行器基线，但不能单独证明模型策略能完成任务。

## 1. 确认使用的是哪次后训练

与“只训练 domain 相关参数，其余冻结”相符的是 Cosmos 仓库中的：

```text
cosmos_framework/configs/toml_config/action_policy_moz1_edge_domain_projector_train.toml
cosmos_framework/scripts/run_moz1_edge_gpu7_domain_projector_train.sh
```

该 TOML 设置 `keys_to_select=["action2llm", "llm2action"]`，其余网络参数由 optimizer 筛选逻辑冻结。`domain_id` 本身是整数索引，不是训练出的浮点参数；真正更新的是两个 `DomainAwareLinear` 中由 MOZ1 的 21 号索引选中的权重与 bias。配置设置 `weight_decay=0.0`，在全为 MOZ1 样本、从新 optimizer 状态启动的前提下，避免其他 domain 行因权重衰减而变化。配置表达了预期训练范围，实际运行仍应核对当次 resolved config 和权重差异。

该配置默认 run 名为：

```text
action_policy_moz1_edge_domain_projector_train
```

现有 `run_moz1_robolab_server.sh` 默认却指向 `action_policy_moz1_edge_gpu7_train`。因此本次必须显式设置 `MOZ1_SERVER_CKPT`，不能直接沿用启动器默认值。实际训练若覆盖过 `job.name` 或输出根目录，应以那次运行的完整 checkpoint 路径为准；本机尚未取得这一路径。

正式对比固定到 `checkpoints/iter_XXXXXXXXX` 目录。run 根目录可用于初次查找最新完整保存，但后续新增 checkpoint 会使重启后的模型变化。服务 metadata 中的 `checkpoint` 才是本进程选中的实际 iteration。使用训练保存的完整 DCP；这份训练配置不能理解为只有一个可单独加载的 domain adapter 文件。

## 2. Action 是否为增量，是否归一化

### 2.1 当前代码的确定行为

| 环节 | 实际行为 |
|---|---|
| MOZ1 数据读取 | 按固定顺序拼接 Parquet 的五组 command 字段，直接作为 label；没有减 state、逐帧差分或 SE(3) 相对变换 |
| 训练模型输入 | 第一行是当前 state，后面 16 行是未来 command；state 和 action 分别使用各自的 q01/q99 做 quantile 归一化，然后补齐到 `max_action_dim` |
| 采样过程 | denoiser 输出的是 velocity；采样器积分得到模型空间的 action，不能直接把单次网络 velocity 当作控制指令 |
| 模型生成后处理 | `ActionProcessor` 去掉补齐通道，再用 `StateActionNormalizer` 逆归一化第一行 state 和后面 action |
| MOZ1 WebSocket 服务 | 从已经后处理的 `[17,20]` 中去掉 state 行，返回 `[16,20]`；没有加回 state |
| RoboLab MOZ1 客户端 | 校验、缓存和逐步输出 action；没有再次逆归一化，也没有 DROID 的末通道翻转/二值化 |
| QX 运动诊断 | 每个 chunk 重新 anchored，再经限幅 DLS IK 得到关节目标；这是执行端轨迹处理，与训练是否做 DeltaActions 是两件事 |

**本次接线规则：服务返回的 action 已逆 Cosmos 训练归一化，不再逆归一化；当前训练读取链路没有生成 delta，不要给输出直接加 state，也不要逐帧累加。** 如果服务器训练时使用过不同版本的 adapter，则需先核对那一版本。

根据后续数据实查反馈，Parquet command 保存原始物理量，Cartesian cmd 为绝对目标，xyz 单位 m、旋转为 rotvec/rad，夹爪为 m、0 闭合/0.12 全开。这是历史用户反馈，当前开发机没有重新读取真实数据。在这一数据约定、且训练代码与当前版本一致的条件下，服务输出应按**绝对 Cartesian command**理解；20 维不是 20 个关节角。

| Python 切片或索引 | 含义 | 预期单位 |
|---|---|---|
| `0:3` / `3:6` | 左臂目标位置 / rotvec | m / rad |
| `6` | 左夹爪命令 | m |
| `7:10` / `10:13` | 右臂目标位置 / rotvec | m / rad |
| `13` | 右夹爪命令 | m |
| `14:17` / `17:20` | 躯干目标位置 / rotvec | m / rad |

`dataset_stored_20d` 是接口对“数据集存储表示”的名字，本身不保证物理单位或参考系。adapter 文档中的 provisional relative/Euler 描述尚未完全随数据语义更新；`_build_action_chunk` 的真实操作是原样拼接，不能只据注释认定输出是 delta 或欧拉角。

### 2.2 归一化计算与历史统计量问题

当前 `resolve_action_normalization("quantile", stats)` 的变换为：

```text
offset = (q99 + q01) / 2
scale  = max(q99 - q01, 1e-8) / 2
model_value = (stored_value - offset) / scale
stored_value = model_value * scale + offset
```

state 使用 `norm_stats.state`，action 使用 `norm_stats.actions`。当前 MOZ1 调用没有开启 forward clamp，也没有输出端强制裁到 `[-1,1]`；归一化后的值可以超出此范围。不要以输出是否落在 `[-1,1]` 判断是否已经逆归一化。

历史实查还报告 `norm_stats.actions` 来自 OpenPI `DeltaActions`，三组 pose 按窗口 state 逐分量作差，夹爪保持绝对，mask 为 `make_bool_mask(6,-1,6,-1,6)`。但当前 Cosmos adapter 没有对应的减 state 操作。这个问题涉及**label 与统计量的分布匹配**，不会由统计文件自动把绝对 label 变成 delta。

对于现有 checkpoint，先保持训练用的同一份 stats：用完全相同的 affine 变换，理论上仍可逆回 label 的原表示；实际训练若存在较大数值范围或使用过裁剪，质量可能受影响，需要开环对照确认。不要擅自换为新算的 absolute stats，也不要仅凭 delta stats 的来源给模型输出加 state。本次代码核对不能追溯证明服务器当次使用的 stats 文件和代码版本。

## 3. 在模型服务器启动指定后训练权重

以下 `/path/to/...` 都要替换为服务器实际路径。模型服务和 RoboLab 分别使用各自已有环境；保留 Isaac Sim 5.0 / Isaac Lab 2.2 或 5.1 / 2.3 的原配对，不混装。

在包含 `cosmos_framework/` 的 Cosmos 仓库根目录执行：

```bash
cd /path/to/cosmos-framework

export MOZ1_ROOT=/path/to/training_dataset
export EDGE_HF_CHECKPOINT=/path/to/Cosmos3-Edge-Policy-DROID
export BASE_CHECKPOINT_PATH=/path/to/Cosmos3-Edge-Policy-DROID-DCP
export WAN_VAE_PATH=/path/to/Wan2.2_VAE.pth
export MOZ1_STATS_PATH="$MOZ1_ROOT/norm_stats.json"
export MOZ1_SERVER_CKPT=/path/to/domain_projector_run/checkpoints/iter_XXXXXXXXX
export MOZ1_SERVER_GPU=2
export MOZ1_SERVER_PORT=8000

bash cosmos_framework/scripts/run_moz1_robolab_server.sh --check-only
```

GPU 编号按空闲卡调整。`--check-only` 只预检 DCP 元数据/分片与 stats，不加载 CUDA 模型，也不证明权重能成功加载。确认输出 `checkpoint` 指向正确 run/iteration、`domain_name=moz1`、`action_dim=20`、`action_chunk_size=16`，保存 `stats_sha256`，然后启动：

```bash
bash cosmos_framework/scripts/run_moz1_robolab_server.sh \
  --seed 0 --deterministic-seed --num-steps 30 --guidance 1.0 --shift 3.0
```

等待 `MOZ1 ready`。默认非 EMA、不开 torch.compile。首次对照固定每次请求 seed 以减少采样变化；之后测试其他生成 seed 时单独记录。此 seed 与仿真 seed、数据 split seed 各自独立。

`BASE_CHECKPOINT_PATH` 用于实验配置引用；实际推理的后训练权重由 `MOZ1_SERVER_CKPT` 选择。使用专用 `action_policy_server_robolab_moz`，不要切换到返回 DROID 8 维 action 的通用服务。

## 4. 先把真实模型 action 保存下来

### 4.1 使用已有 sample

在 RoboLab 的已有环境执行，输出目录使用尚不存在的新目录：

```bash
cd /home/admin/RoboLab-fork
export QX_PYTHON=/path/to/existing/isaac_env/bin/python
export MOZ1_REMOTE_HOST=POLICY_SERVER_IP

bash scripts/run_cosmos3_moz.sh \
  --remote-host "$MOZ1_REMOTE_HOST" --remote-port 8000 \
  --probe-sample /path/to/local_or_shared/sample.json \
  --probe-output output/domain_projector_probe_001
```

sample 只在模型服务器上时，用 SSH 别名读取：

```bash
bash scripts/run_cosmos3_moz.sh \
  --remote-host "$MOZ1_REMOTE_HOST" --remote-port 8000 \
  --probe-ssh cosmos-server \
  --probe-sample /path/on/server/sample.json \
  --probe-output output/domain_projector_probe_ssh_001
```

`cosmos-server` 替换为已有 SSH 配置别名或 `USER@HOST`；SSH 文件读取与 8000 端口的 WebSocket 推理独立。JSON 引用的图像、state、stats 都在读取 sample 的机器上解析。此模式不会启动 Isaac Sim，至少需 NumPy、Pillow、websockets、msgpack 及项目的轻量客户端依赖。

输出：

- `action.json`：实际模型返回的 16×20 数值，已逆训练归一化，没有经过 IK/anchored。
- `policy_metadata.json`：实际 checkpoint、stats hash、采样设置、sample 来源、生成 seed；`executed_in_simulation=false`。

先核对 metadata，确保请求没有连接到旧训练服务。保留 action 全部数值，不只保存最大/最小值。若对比 sample 同目录的 `target_action.json`，应逐通道检查绝对 command 误差，位置、旋转、夹爪分别统计；rotvec 的几何误差使用 SO(3) 相对旋转角，不能把跨 ±π 的分量差直接当作姿态误差。

### 4.2 没有 sample 时生成验证窗口

在 Cosmos 原训练环境、数据可读的机器执行。先激活该环境并沿用上节环境变量。`MOZ1_SERVER_CKPT` 此处应为明确的 iteration 目录：

```bash
python -m cosmos_framework.scripts.evaluate_moz1_open_loop prepare \
  --dataset-root "$MOZ1_ROOT" \
  --checkpoint-path "$MOZ1_SERVER_CKPT" \
  --out-dir /path/to/new/domain_projector_samples \
  --max-samples 8 --seed 42
```

从输出 `inputs/*/sample.json` 选一个完整路径用于 probe。prepare 使用按 episode 划分的验证集，split seed=42、val ratio=0.01、16 步窗口；命令的 seed=42 控制抽样，未执行模型推理。manifest 保留 episode/window 身份，供后续复查。

旧 `run_moz1_open_loop_eval.sh` 含固定服务器路径和旧 run 默认值，不适合未经检查直接复用到这次 projector run；以上 prepare 命令可以独立使用。

## 5. 把模型输出送进现有仿真执行链路

先用已通过 probe 的 sample，确认真实模型到关节执行链路。在 RoboLab 环境运行：

```bash
export QX_ROBOT_USD=/path/to/QX_and_picker.usd

bash scripts/run_cosmos3_moz.sh \
  --motion-smoke \
  --remote-host "$MOZ1_REMOTE_HOST" --remote-port 8000 \
  --probe-ssh cosmos-server \
  --probe-sample /path/on/server/sample.json \
  --motion-chunks 1 --motion-hold-seconds 5 \
  --motion-output output/domain_projector_motion_001 \
  --motion-exit-on-complete
```

sample 在本地时删除 `--probe-ssh`，换成本地路径。该命令重新请求模型，并执行这次实际返回的 action；不会读取上节保存的 `action.json`。如需两个输出逐值一致，需固定输入、权重、stats 和生成 seed，并核对实际输出，不能只依赖文件名。

执行路径是：

```text
sample 图像/state/prompt
→ 后训练模型真实生成 [16,20]
→ 保存 chunk_000.json
→ 每个 chunk 用首个预测建立 anchor
→ 左臂、右臂、躯干逐步构造目标
→ PhysX Jacobian + bounded DLS IK
→ 关节位置目标 → PD/物理仿真 → 实测关节反馈
```

当前限制及其对结果的影响：

- sample 模式重复使用固定观测；增加 `--motion-chunks` 会再次对同一样本请求模型，不会沿数据 episode 读取下一帧，也不是仿真观测闭环。
- 模型 smoke 没有历史 recording 回放的首帧实测关节恢复。anchored 会把预测轨迹重定位到当前仿真姿态，因此它验证的是动作变化的执行能力。
- 每个 chunk 的首预测用于锚定，其原绝对偏置不会作为首步 Cartesian 位移执行。默认平移半径 0.10 m、单 chunk 旋转限幅 0.25 rad、每关节更新限幅 0.02 rad；轨迹被限幅后可能偏离原预测。
- 两路夹爪命令 `6/13` 被保持，不执行开合；机器人重力、自碰撞关闭，底盘固定。这种条件不能用于抓取成功率结论。
- 模型 smoke 当前执行每次返回的全部 16 步，`--execute-horizon` 不控制这个诊断循环；该参数用于正式评测客户端的 chunk 消费。
- 控制为 1/120 s 物理步长、每次 action 4 个物理步，即仿真时间 30 Hz。模型请求耗时另计，不能据此称真实墙钟闭环达到 30 Hz。

`chunk_000.json` 保存输入 state、prompt、完整预测、生成 seed 和请求耗时。`control.jsonl` 保存逐步目标/实测关节；顺序以 `motion_summary.json` 的 runtime joint names/IDs 为准。`motion_observed` 只表示有模型轨迹变化并观测到关节响应，不等于任务完成，也不保证三组肢体全部响应。

运动不对时先检查 `chunk_*.json` 是否合理，再检查 IK target 与实测关节是否跟随；初始化停住时查看 `motion_summary.json` 的 `current_phase`、`startup.jsonl`、存在时的 `startup_error.txt`/`startup_stacks.txt`。

## 6. 实时闭环前先统一 frame 和夹爪

上次用户核对结果指出：至少左臂 state 是 `left_flange` 相对于 `torso_flange`；躯干状态的位置与 `waist03` 相差 0.15 m，旋转匹配。当前实时诊断仍取 USD 首关节参考系中的 `left07/right07`，躯干取 `waist03` 相对 root，夹爪 state 填 0。不能直接把它当作训练 state。

先在共同姿态和共同基准系下核对完整固定变换，确认右臂、action command 与 state 是否共用同一 frame 约定；不要只沿某轴直接加 0.15 m。若定义 `T_A_B` 为把 B 坐标变到 A 的变换，且 action 的左臂目标确在 torso_flange 系，则：

```text
T_world_left_flange_target
    = T_world_torso_flange @ T_torso_flange_left_flange_command

T_world_left07_target
    = T_world_left_flange_target @ inverse(T_left07_left_flange)
```

最后一式适用于仍在 `left07` 点求 IK 的情形；也可统一改用法兰点位姿及对应 Jacobian。目标、当前位姿和 Jacobian 必须处于一致参考系并对应同一末端点。躯干同时动作时，要明确双臂相对目标如何随躯干变化。

正式执行绝对 command 时，anchored 不是 IK 的必经步骤。应在 frame 一致后使用原目标，并记录任何限幅导致的偏差。若将来确认训练 label 确为 delta，再按训练所用的基准 state 和具体 delta 定义重建目标；逐分量 rotvec 差与 SE(3) 相对旋转不能混用。

还需把夹爪 m 单位开度映射到实际左右 drive/mimic 关节，核对开闭方向、限位和接触配置，并从仿真回读真实夹爪 state；不能用填 0 的诊断状态评估抓取策略。三路相机的 RGB、朝向、视场、布局也要与训练一致。

## 7. 各类任务评测的接入和运行方式

当前 `registration_moz.py` 只有注册辅助函数，需要外部提供 QX `robot_cfg`、`actions_cfg`、`image_obs_cfg`、`proprio_obs_cfg` 和 `contact_gripper`；仓库尚无现成的完整 QX MOZ1 benchmark 注册。仅设置 USD 或启动 `--motion-smoke` 不会得到任务成功率。

实现注册模块时，应满足以下可验证接口：

| 接口 | 要求 |
|---|---|
| 模块契约 | 导出与 `registration_moz.MOZ1_POLICY_CONTRACT` 一致的对象 |
| 观测 | 三路训练视角 RGB uint8；20 维 `moz1_state` 为当前实测值，字段、单位和 frame 与训练一致 |
| 控制 | 消费服务返回的 20 维 command，包括双臂、躯干、夹爪；正确 frame/TCP 转换与 IK，不重复逆归一化 |
| 时序与随机性 | `sim.dt * decimation = 1/30`；使用 `args.seed`，每 episode 清空客户端 chunk 状态 |
| 任务配置 | 使用 RoboLab 原任务成功条件和子任务检测；对 QX 核对桌面可达性、初始姿态、接触与 reset |
| 证据 | 保存 metadata、运行配置、episode 结果、轨迹和视频；明确控制限幅/IK residual 的记录方式 |

下列是**注册实现完成之后**的运行模板；`your_qx_module:register_envs` 是接口占位符，当前不能原样运行：

```bash
export MOZ1_REGISTRATION=your_qx_module:register_envs
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"

# 单任务先检查完整闭环。
"$QX_PYTHON" -m policies.cosmos3.run_moz \
  --registration "$MOZ1_REGISTRATION" \
  --robot-usd "$QX_ROBOT_USD" \
  --remote-host "$MOZ1_REMOTE_HOST" --remote-port 8000 \
  --task BananaInBowlTask --num-envs 1 --num-runs 1 \
  --execute-horizon 4 --seed 0 --headless --livestream 0

# 在同一注册范围内逐任务评测；省略 task/tag 才选择所有已注册任务。
"$QX_PYTHON" -m policies.cosmos3.run_moz \
  --registration "$MOZ1_REGISTRATION" \
  --robot-usd "$QX_ROBOT_USD" \
  --remote-host "$MOZ1_REMOTE_HOST" --remote-port 8000 \
  --num-envs 1 --num-runs 20 --execute-horizon 4 \
  --seed 0 --headless --livestream 0
```

全任务用模块入口，因为 `scripts/run_cosmos3_moz.sh` 自带 `--task BananaInBowlTask` 默认值；直接给该脚本追加 `--tag` 不能当作可靠的全任务选择。任务范围还受 `--task-dirs` 和注册实现影响。

先固定单环境便于排查，完成后再增加并行度。`execute-horizon=4` 是首次闭环的建议设置，每 4 个 action 请求一次新 chunk，服务仍生成 16 步；与 16 步执行应作为不同配置比较，同时记录推理等待对仿真的影响。不能假定较短 horizon 必然提升效果。

成功率对比至少保留：模型/checkpoint iteration、代码版本、stats hash、Isaac Sim/Lab 版本、机器人资产和控制器、相机配置、任务/指令、仿真 seed、生成 seed、horizon、episode 数和初始随机化。base 与后训练模型使用配对任务/初始条件；原 DROID base 的 MOZ1 21 号投影未必经过 MOZ1 训练，对照结果应按此解释。

结果可用仓库现有工具分析，替换实际输出目录：

```bash
"$QX_PYTHON" analysis/read_results.py output/ACTUAL_EVAL_DIR --show-episodes
"$QX_PYTHON" analysis/read_results.py output/ACTUAL_EVAL_DIR --by-attributes
"$QX_PYTHON" analysis/read_results.py output/ACTUAL_EVAL_DIR --by-difficulty
"$QX_PYTHON" analysis/read_results.py output/ACTUAL_EVAL_DIR --by-scene
"$QX_PYTHON" analysis/read_results.py output/ACTUAL_EVAL_DIR --output-csv summary.csv
```

逐任务报告成功数/总数、成功率及区间、子任务得分和失败类型，再看分类汇总；不同任务 episode 数不等时，区分按 episode 汇总与按任务平均。20 次是运行示例，不是已证明足够的统计样本量。详见 [统计口径](statistical_significance.md)、[结果分析](analysis.md) 和 [输出结构](data.md)。

## 8. 源码依据与验证范围

Cosmos 源码根为 `/home/admin/cosmos-framework`，下表 Cosmos 路径相对该根；RoboLab 路径相对本仓库。

| 结论 | 源码位置 |
|---|---|
| projector allowlist、无 decay | `cosmos_framework/configs/toml_config/action_policy_moz1_edge_domain_projector_train.toml:27` |
| 冻结未选参数 | `cosmos_framework/utils/generator/optimizer.py:165` |
| MOZ1 domain=21、raw dim=20 | `cosmos_framework/data/generator/action/domain_utils.py:26` |
| domain 参数按索引选择 | `cosmos_framework/model/generator/mot/domain_aware_linear.py:49` |
| label 原样拼接、首行 state | `cosmos_framework/data/generator/action/datasets/moz1_lerobot_dataset.py:298`、`:343` |
| quantile affine 与逆变换 | `cosmos_framework/data/generator/action/action_processing.py:44`、`:212`、`:382` |
| 服务归一化输入、取已后处理输出 | `cosmos_framework/scripts/inference_tools/action_policy_server_robolab_moz.py:151`、`:179` |
| 默认选中的旧 run | `cosmos_framework/scripts/run_moz1_robolab_server.sh:29` |
| probe 落盘 action 与 metadata | `policies/cosmos3/run_moz.py:144` |
| 模型 action 到 IK 执行 | `policies/cosmos3/motion_qx.py:335`、`:746` |
| 完整任务注册所需配置 | `policies/cosmos3/registration_moz.py:22` |

历史数据语义来源见 `logs/2026-10-23.md` 及其引用的 10-21 附件；最新 flange/frame 纠正见 `logs/2026-11-10.md`。后续 frame 证据优先于旧 README 的肩部系概述。

本次核对的是本地源码和历史反馈；没有在开发机执行真实模型推理、加载 QX USD 或计算任务成功率。开发机系统 Python 缺少模型/仿真依赖，也未找到实际训练 checkpoint、stats 和 sample；指南中的服务器命令需要实际文件和既有运行环境。
