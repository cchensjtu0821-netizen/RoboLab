# Cosmos3 MOZ1 Edge 与 RoboLab

这一入口用于 `run_moz1_edge_gpu7_train.sh` 对应的 MOZ1 后训练模型。旧的 `run.py` / `client.py` 保持 DROID 用法；MOZ1 使用 `run_moz.py` / `client_moz.py`，两者不能混用。

模型服务、仿真控制、WebRTC 可视化仍是原来的部署方式。模型通过 WebSocket 接收相机/状态/指令、返回动作；WebRTC 展示同一个 RoboLab 进程的仿真视口。策略相机与 WebRTC 画面分别采集。

## 1. 在 Cosmos 服务器启动模型

进入 **cosmos-framework 仓库根目录**（包含 `.venv` 和 `cosmos_framework/` 的目录），运行：

```bash
bash cosmos_framework/scripts/run_moz1_robolab_server.sh
```

启动器激活现有 uv `.venv`、清空 `LD_LIBRARY_PATH`，默认使用物理 GPU 2、端口 8000，按原训练启动器的服务器目录结构定位数据、Edge processor、Wan VAE 和训练输出。它默认查找：

```text
$IMAGINAIRE_OUTPUT_ROOT/cosmos3_action/action_sft/action_policy_moz1_edge_gpu7_train
```

自动选择该 run 的 `checkpoints/latest_checkpoint.txt` 指向的完整 DCP；指针缺失或指向未完成保存时，按 iteration 数值选择最近的完整保存。显式传 iteration 时，未完成保存会报错。它检查 `model/.metadata` 和非空 `.distcp` 分片；这只是文件完整性预检，实际权重可加载性由 GPU 加载确认。选择在启动时固定，不会在请求中途切换模型。

覆盖路径/GPU/端口的示例：

```bash
MOZ1_SERVER_CKPT=/path/to/run_or_checkpoints_or_iter \
MOZ1_ROOT=/path/to/training_dataset \
MOZ1_SERVER_GPU=2 MOZ1_SERVER_PORT=8000 \
bash cosmos_framework/scripts/run_moz1_robolab_server.sh
```

原来的 `BASE_CHECKPOINT_PATH`、`EDGE_HF_CHECKPOINT`、`WAN_VAE_PATH`、`IMAGINAIRE_OUTPUT_ROOT` 均可通过环境变量覆盖；数据路径变化时也可单独设置 `MOZ1_STATS_PATH`。`MOZ1_SERVER_CKPT` 选择实际推理权重，`BASE_CHECKPOINT_PATH` 只满足实验配置的路径引用，不会替代选中的后训练 DCP。

只检查 checkpoint 与统计量、不加载 CUDA：

```bash
bash cosmos_framework/scripts/run_moz1_robolab_server.sh --check-only
```

已有正确环境时也可直接运行 Python：

```bash
python -m cosmos_framework.scripts.action_policy_server_robolab_moz \
  --checkpoint-path /path/to/run_or_iter \
  --stats-path /path/to/norm_stats.json --port 8000
```

直接 Python 方式需要提前设置训练实验使用的路径环境变量。服务默认使用 `action_policy_moz1_edge` 模块配置、常规权重（非 EMA）、不启用 torch.compile，与现有 MOZ1 开环推理的非 EMA 路径一致。采样默认 `num_steps=30, guidance=1.0, shift=3.0, seed=0`；可通过 `--num-steps` 等参数覆盖。`--deterministic-seed` 为每次请求固定相同种子；默认从 seed 初始化的 RNG 逐次取种子。`--use-torch-compile`、`--decode-video` 可选，首次联调建议保持默认。

## 2. 先验证真实模型的输入输出

不需要 Isaac Sim 或 QX 控制器，可以使用已有 MOZ1 开环评估产生的 `inputs/<sample>/sample.json`。支持通过 SSH 读取服务器上的样本，也支持本地或共享文件系统路径；JSON 引用的图像、状态和 stats 必须在所选读取机器上可读。相对路径按 JSON 所在目录解析。

样本保留在 Cosmos 服务器时，可以通过 SSH 直接读取，**无需复制样本目录到 RoboLab 机器**。在 RoboLab 仓库执行（`user` 替换为服务器 SSH 登录用户名，也可使用 `~/.ssh/config` 中的主机别名）：

```bash
bash scripts/run_cosmos3_moz.sh \
  --remote-host 10.160.124.98 --remote-port 8000 \
  --probe-ssh user@10.160.124.98 \
  --probe-sample /srv/data2/c00932551/cosmos_outputs/test/moz1_open_loop_eval/iter_000010440_20261013_212322/inputs/moz1_val_00000409/sample.json \
  --probe-output output/moz1_probe_10440_ssh_001
```

需要 RoboLab 机器有 `ssh` 命令、能够登录 Cosmos 服务器，远端有 `python3`（只用标准库）。SSH 使用已有密钥、交互密码或 SSH 配置；自定义端口/跳板机可在 SSH 配置别名中设置。SSH 文件读取与 WebSocket 推理是两个连接：`--probe-ssh` 指定文件所在机器，`--remote-host` 指定模型服务机器。

客户端通过一次 SSH 调用在远端解析 JSON 的绝对/相对引用路径，将状态与图像传入本地内存，计算远端 stats 原始字节的 SHA-256 并与模型握手核对，随后通过 WebSocket 请求推理。不会在本地落盘 sample/image/state/stats，也不修改远端文件；只在本地保存推理结果和来源信息。可用 `--stats-path` 指定本地 stats 覆盖样本引用，此时校验本地文件的哈希。此功能仅用于 probe，不负责 QX 仿真控制。

样本已在本地或共享文件系统时，在 RoboLab 仓库执行：

```bash
MOZ1_REMOTE_HOST=<policy-server-ip> \
bash scripts/run_cosmos3_moz.sh \
  --probe-sample /path/to/sample.json \
  --probe-output output/moz1_probe_001
```

启动器使用 RoboLab `.venv`（其次 `.venv-51`），也可用 `QX_PYTHON=/path/to/python` 指定已有环境。probe 模式不导入或启动 Isaac Sim，只需要 NumPy、Pillow、websockets、msgpack。

probe 核对服务器契约与样本 stats 的 SHA-256，返回有限的 `[16,20]` 动作后保存 `action.json` 和 `policy_metadata.json`，**不会执行机器人动作**。输出目录必须尚不存在。预测动作仍是数据集存储空间；不会自动变成米、弧度或关节角。probe 可验证在线模型链路，不能证明任务成功或控制正确。

## 3. 先让 QX 双臂动起来

新增 `--motion-smoke`，不再需要填写示例 `my_qx_registration`。它在香蕉入碗场景中创建真实 Isaac Lab Articulation、固定底盘、通过 PhysX Jacobian / DLS IK 驱动左右各 7 个关节，并在同一进程提供 WebRTC。先保持 Cosmos 模型服务运行，再在 RoboLab 的 Isaac Sim / Isaac Lab 环境执行。停止独立的旧预览进程，避免两个进程争用串流端口。

**优先用之前已成功的 sample 做第一次运动验证**：沿用原来的 `--probe-ssh` 和 `--probe-sample` 参数，加上 `--motion-smoke`。此组合会启动物理仿真并执行模型轨迹，不再是只写 action.json 的 probe。

```bash
# 如果当前已激活原 Isaac Lab 环境，可直接使用它；也可显式指定其 python。
export QX_PYTHON="$(command -v python)"
export QX_ROBOT_USD=/实际路径/QX_and_picker.usd
policy_host=POLICY_SERVER_IP  # 改成之前 probe 成功的模型服务器 IP

NO_PROXY="$policy_host,localhost,127.0.0.1" \
no_proxy="$policy_host,localhost,127.0.0.1" \
bash scripts/run_cosmos3_moz.sh \
  --motion-smoke --remote-host "$policy_host" --remote-port 8000 \
  --probe-ssh "user@$policy_host" \
  --probe-sample /实际远程路径/sample.json \
  --motion-chunks 6 --motion-hold-seconds 30
```

将占位符换成原来成功 probe 的值。样本图像/state 在 Cosmos 服务器通过 SSH 读取，模型仍通过 WebSocket 真实推理，每次返回 `[16,20]` 后执行 16 个控制步。该模式重复使用样本输入，**是开环模型到执行器联调，不是实时观测闭环**；不需要先解决仿真相机或 SDK TCP 标定。

SSH 读样本通常使用 **22** 端口，`--remote-port 8000` 只配置模型 WebSocket；两者独立。流程先读取样本，再建立模型连接，最后发起推理。密码提示出现时需在当前终端输入 SSH 密码并回车（密码不回显）。`Connection closed ... port 22` 表示样本 SSH 连接被关闭，应先单独验证 `ssh user@服务器` 能登录、远程 `python3` 和样本路径可读，不能据此认定8000模型服务断开。

运动入口会明确打印等待阶段：`SSH sample read/authentication`、`Cosmos3 WebSocket connection/metadata`、`Cosmos3 inference chunk ...`，并显示已等待秒数；`motion_summary.json` 的 `current_phase` 同时记录当前阶段。SSH 等待阶段还没有发送模型推理请求。等待期间继续渲染并保持机器人姿态。

不带 sample 参数则使用仿真实时图像与临时状态反馈：

```bash
bash scripts/run_cosmos3_moz.sh \
  --motion-smoke --remote-host "$policy_host" --remote-port 8000 \
  --motion-chunks 6 --motion-hold-seconds 30
```

实时模式使用 USD 中现有双腕 Camera，并新增一台 high 相机；姿态输入取 `left07/right07` 法兰相对 USD 首关节父坐标变换后的位姿，使用 rotvec。**尚未标定到 SDK 的 pose_base/TCP，夹爪状态临时填 0，high 视角也未对齐训练视角**，因此优先采用 sample 模式确认关节驱动，实时模式的模型效果不作为评测依据。相机路径或运行时 DOF 不匹配会明确报错，不回退到 DROID。

两种模式采用同一诊断控制映射：每个 chunk 的首帧预测对齐当前实际法兰姿态，后续帧的位置变化、SO(3) 旋转变化驱动双臂；它不是生产 SDK 绝对 TCP 控制，也不是把 20 维当成关节角。躯干、轮子、头和夹爪保持初始关节目标，不执行对应预测通道。机器人重力、自碰撞在当前 stage 内关闭，底盘固定，不修改源 USD 或现有任务评测定义。

默认物理步长 1/120 秒，4 个物理步执行一次控制；每个关节目标相对当前角度最多改变 0.02 rad，法兰目标距启动位置最多 0.10 m，单 chunk 旋转变化最多 0.25 rad，并按实际关节限位截断。`--motion-gain` 只缩放预测平移，`--motion-radius` 和 `--motion-joint-step` 可调整上述限幅。模型请求在工作线程中执行，等待推理时仍步进仿真并保持关节目标。30 Hz 是仿真控制配置，不是模型吞吐测量。

开启 WebRTC 时，运动结束并完成 `--motion-hold-seconds` 的有限保持后，默认保存结果、暂停物理仿真、继续刷新 Kit/UI/串流，直到按 Ctrl+C 或关闭应用。终端显示 `Motion finished; physics paused, WebRTC/viewport stays open` 后仍可观察场景，不会再自动开始下一轮模型动作。可显式使用 `--motion-keep-open`；自动化无人值守运行用 `--motion-exit-on-complete` 恢复完成后退出。关闭串流时默认仍自动退出。检查视口期间不增加已保存的控制步数或改变运动结果。

先点击客户端里的 3D 视口让它获得输入焦点，再使用视口导航调整相机；确保只连接一个客户端、查看默认透视相机而非固定的策略相机。底盘固定和关节PD保持会阻止直接拖动机器人本体，这与相机导航不同。控制代码只在启动时设置一次相机，不会每帧重置视角。若推理/运动期间就无法导航，或新版本保持刷新后仍出现 `nvstPushStreamData timeout`，需另查 WebRTC 客户端输入、连接状态和串流日志，不能仅凭这一条错误确定根因。此前版本在输出 `QX motion result` 后立即关闭应用，可能表现为客户端停在最后一帧；保持视口修复不代表已完成真实串流验证。

终端会依次显示 `QX articulation initialized`、`Received [16,20]`、`Measured arm joint displacement`。结果保存在 `output/cosmos3_moz_motion/<时间戳>/`（可用 `--motion-output` 指定不存在的目录）：

- `chunk_*.json`：实际输入 state/prompt、原始 `[16,20]` 预测、生成 seed、请求耗时。
- `control.jsonl`：每个控制步的实际关节角和目标角，单位 rad，顺序见 summary。
- `motion_summary.json`：模型 metadata、软件版本、输入来源、控制配置、实际关节映射、最大观测/目标位移、完成 chunk 数和运行状态。

只有预测轨迹有变化、控制目标发生变化且实际读回的臂关节位移超过 `1e-4` rad 才报告 `motion_observed`；没有动作会报错，不通过预设摆臂伪造模型运动。该状态仅是联调运动证据，不证明任务成功、模型质量或完整 20 维映射正确。开发环境仅完成 CPU 检查，真实 USD/PhysX/GPU/串流还需在用户服务器验证。

控制 API 对齐 [Isaac Lab 2.2 的 IK 示例](https://github.com/isaac-sim/IsaacLab/blob/v2.2.0/scripts/tutorials/05_controllers/run_diff_ik.py)和[固定底盘的 schema 实现](https://github.com/isaac-sim/IsaacLab/blob/v2.2.0/source/isaaclab/isaaclab/sim/schemas/schemas.py)。保持原有 Isaac Sim 5.0 / Isaac Lab 2.2 环境，不要求升级或重新训练。

## 4. 完整 QX 任务注册与 WebRTC

当前 `scripts/run_qx_banana.sh` 的 USD 加载和 WebRTC 显示已由用户确认成功。该预览脚本仍不执行控制；上一节的 `--motion-smoke` 提供独立运动联调。完整任务评测仍需要下述注册，不能直接拿 USD 路径替换 DROID。

闭环入口已经支持自有机器人注册：

```bash
MOZ1_REMOTE_HOST=<policy-server-ip> \
MOZ1_REGISTRATION=my_qx_registration:register_envs \
bash scripts/run_cosmos3_moz.sh \
  --robot-usd /path/to/QX_and_picker.usd \
  --adapter-config /path/to/qx_control_config.json
```

**`my_qx_registration` 是需要根据真实 QX 关节/末端和数据约定实现的本地模块，本仓库尚无这个机器人配置，所以上述是接好注册后的调用格式，不能原样执行。** `--robot-usd` 和 `--adapter-config` 转交给注册函数；仅提供这两个参数不会自动创建关节映射或 IK 控制器。模块必须在 `PYTHONPATH` 上。

启动器默认 `BananaInBowlTask`、1 个环境、1 次运行、headless、`--livestream 2`。追加 `--task`、`--num-runs` 等可覆盖默认值；`MOZ1_LIVESTREAM=0` 可关闭串流。不要另开一个预览脚本来观察另一进程的场景。

注册模块要求：

- 暴露 `register_envs(args)`，其中可以读取 `args.robot_usd`、`args.adapter_config`、`args.task`、`args.task_dirs`、`args.seed`。
- 声明下文契约，确认此注册确实处理 MOZ1 的数值约定。可导入 `policies.cosmos3.registration_moz.MOZ1_POLICY_CONTRACT`。
- `image_obs` 暴露三个批量 RGB uint8 张量：`cam_high`、`cam_left_wrist`、`cam_right_wrist`，形状 `[N,H,W,3]`；高位相机 H、W 为偶数。
- `proprio_obs.moz1_state` 为 `[N,20]`，是数据集中同样表示的当前绝对状态，尚未应用 Cosmos 的 quantile 映射。也可分为训练中的五个字段名，见 `client_moz.STATE_FIELDS`。
- 动作管理器接收同样顺序的 20 维**数据集存储空间**动作，负责生产端逆变换、坐标系和旋转转换、左右夹爪控制、双臂和躯干的 IK/驱动。左右夹爪在索引 6、13，最后一维是躯干姿态的一部分，不能套用旧客户端的最后一维夹爪二值化。
- 注册控制周期必须为 1/30 秒，使用请求的 `args.seed`。`registration_moz.register_moz1_envs` 提供任务/观测/相机/接触传感器的公共接线，物理 dt=1/120、decimation=4；机器人本体和动作/观测配置必须由实际 QX 配置提供。

```python
# my_qx_registration.py 的接线形态；Qx* 配置需据真实资产实现。
from policies.cosmos3.registration_moz import MOZ1_POLICY_CONTRACT, register_moz1_envs

def register_envs(args):
    from my_qx_configs import build_qx_configs

    configs = build_qx_configs(args.robot_usd, args.adapter_config)
    return register_moz1_envs(args, **configs)
```

公共 helper 接收 `robot_cfg`、`actions_cfg`、`image_obs_cfg`、`proprio_obs_cfg`、`contact_gripper` 和可选 `scene_camera_cfg`。后三种 obs/action 参数是已实例化配置，robot_cfg 是配置类；机器人挂载的相机放到 robot_cfg，非挂载相机放到 scene_camera_cfg。机器人配置还需遵守 `docs/robots.md` 的接触体、夹具和 `ee_recorder_bodies` 要求。

在 QX 控制注册可用前，仿真模式会明确报缺少 `--registration`；不会静默改用 DROID。当前仍缺少实际 USD 与运行时 Articulation 的对应映射、驱动配置和数据生产端坐标系/单位/相对位姿定义，不能声称已完成 QX 闭环。

### QX URDF 参考与实际 USD 核对

2026-10-20 用户补充了 URDF 关节、连杆、限位和版本差异。完整数值保存在 [用户资料归档](../../logs/attachments/2026-10-20/qx_moz1_user_reference.txt)。这份资料由用户整理，Agent 尚未读取所列原始 URDF/USD 文件，也不能据它认定实际 USD 的 DOF 名称或顺序。

候选映射已明确：左臂 `LeftArm-0..6` / `left07`、右臂 `RightArm-0..6` / `right07`、腰腿 `LegWaist-0..5` / `waist03`；flange 取决于版本，双 Picker 还有各自 narrow/wide/loop 关节。用户提供的整机关节顺序共 26 项（含轮子和头部，不含夹爪），不能直接当成模型的 20-D Cartesian 动作顺序。

实际使用哪个模型版本必须核对资产；不能混合两版限位或只根据限位正负决定翻转关节。夹爪内部多个关节如何由一个开合量驱动，仍需确认主动/被动、耦合和数值标定。相机外壳或名为 D405 的 prim 不足以确认已经配置好策略可读取的相机传感器。

已在原预览入口加入静态 USD 清单导出：

```bash
bash scripts/run_qx_banana.sh --headless --inspect-robot
```

沿用原来的 `QX_ROBOT_USD`、`QX_PYTHON` 和场景资源查找方式，默认写入 `output/qx_robot_inventory.json` 后退出。可通过 `--robot-report /path/to/report.json` 指定输出。该选项不下发控制命令，不进入策略评测。

报告包含 `/World/robot` 下实际组合出的 articulation root、刚体、关节路径和类型、body0/body1 关系、原始 joint/drive 属性以及 Camera prim；同时记录源 USD 和组合 stage 的单位/朝上轴元数据。新增 `joints[].raw_usd_relationships` 保存全部关节关系，包括 `physxMimicJoint:rotX:referenceJoint` 的目标和空目标列表。旧版报告仅有 body0/body1，不能据此认定原 USD 没有配置 mimic 引用。缺少元数据时保留 null，不猜测单位；非有限数值以 `nonfinite_float` 字段显式保存。报告不转换原始 USD 数值的单位。

这只是 USD 静态结构检查，不能替代 PhysX 初始化后的 DOF 名称、顺序、固定连杆合并情况或控制验证。若出现多个 articulation root，须据实际模型决定本体与夹爪如何分别注册，不能假定整机只有一个 Articulation。

若 RoboLab 代码与已安装环境位于两个 checkout，可显式复用原环境。例如从当前代码仓库执行：

```bash
export QX_PYTHON=/srv/workspace/c00932551/RoboLab/.venv/bin/python
bash scripts/run_qx_banana.sh --headless --inspect-robot
```

该命令会启动 Isaac Sim、加载实际资产、写出报告后退出，不用连接 Cosmos 模型服务。机器人 USD 仍按原预览脚本自动查找，必要时设置实际 `QX_ROBOT_USD`。先核对报告中的 articulation roots、左右臂和躯干关节/末端路径、夹爪 drive/mimic 属性及 Camera prim；再核对实际运行时 DOF 和策略相机/控制器，不能仅凭静态报告直接下发模型动作。

**2026-10-20 用户提供的实际 USD 报告。** 原始字段与数值已归档至 [结构报告](../../logs/attachments/2026-10-20/qx_robot_inventory_user.json)，Agent 静态计算另存 [结构统计](../../logs/data/2026-10-20/qx_inventory_analysis.json)。这是用户服务器导出后的反馈，不是 Agent 加载 USD 或运行 PhysX 的结果。报告的源资产为 `/srv/workspace/c00932551/QX_robot/QX_robot/QX_USD/QX_and_picker.usd`，源和组合 stage 均为 metersPerUnit=1.0、Z up。

以下路径均相对 `/World/robot/`：

| 用途 | 实际 USD 路径/名称 |
|---|---|
| articulation root | `QX_USD/base_link` |
| 左臂 / 末端候选 | `QX_USD/joints/LeftArm_0..6` / `QX_USD/left07` |
| 右臂 / 末端候选 | `QX_USD/joints/RightArm_0..6` / `QX_USD/right07` |
| 腰腿 / 躯干候选 | `QX_USD/joints/LegWaist_0..5` / `QX_USD/waist03` |
| 左夹爪 drive 候选 | `Picker_USD_left/joints/hand_narrow1_joint` |
| 右夹爪 drive 候选 | `Picker_USD_right/joints/hand_narrow1_joint` |
| 左腕相机 | `Picker_USD_left/base_link/D405_left/camera_link/waist_camera` |
| 右腕相机 | `Picker_USD_right/base_link/D405_right/camera_link/waist_camera` |

报告包含 46 个刚体、45 个 joint（42 个转动、3 个固定）和 2 个 Camera prim。静态 body0/body1 图从所列根节点连通所有 46 个刚体，但这不是运行时 DOF 数量或初始化验证。两夹爪各一个关节带 angular drive schema，其余各 7 个带 mimic schema；两侧有同名关节和刚体，运行时须核对完整路径与实际索引，不可只按 `hand_narrow1_joint` 或 `base_link` 短名区分左右。夹爪候选 drive 的原始限位为 `[-46.0, 0.0]`，开闭方向、数据数值映射和 TCP 仍未标定。

USD revolute joint 的 lower/upperLimit 单位是度（见 [OpenUSD 定义](https://openusd.org/dev/api/class_usd_physics_revolute_joint.html)），后续控制器若使用弧度需要转换；报告本身不换算。`LegWaist_2`、`LeftArm_0`、`RightArm_0` 的关键限位与用户主版 URDF 参考相符，不能据此证明所有模型版本细节相同。机器人子树只列出左右腕两路相机，策略的 `cam_high` 还需检查场景中的其他相机或新增并对齐训练视角；两路 Camera prim 尚不等于已配置好 Isaac Lab RGB 观测。

旧报告缺少 mimic referenceJoint 关系是导出工具遗漏，不是已确认的资产错误。可更新代码后保留旧报告、另导出完整关系：

```bash
bash scripts/run_qx_banana.sh --headless --inspect-robot \
  --robot-report output/qx_robot_inventory_v2.json
```

2026-10-21 用户补充 SDK/FK 报告后，原始命令按绝对位姿、m/rotvec-rad、双臂各自基座系/躯干底盘系解释，夹爪 m、0 闭/0.12 全开；原文与精确数值见当日日志附件。完整 SDK Tf0/MDH/T7e 与实际 USD 的对应仍未落实。另须核对当次 Cosmos 训练是否将绝对命令与 OpenPI DeltaActions 的统计量配用；不能仅因统计来源就给现有输出加 state 或直接替换旧 checkpoint 的 stats。运动联调不修改已有训练或归一化，也不能消除这个待核对项。

## 固定输入输出契约

| 项目 | 值 |
|---|---|
| protocol / domain | `moz1_policy_v1` / `moz1` |
| state/action 顺序 | 左臂 pose 6、左夹爪 1、右臂 pose 6、右夹爪 1、躯干 pose 6 |
| 状态输入 | `[20]`，当前绝对状态，数据集存储空间 |
| 模型 action 输入 | 1 行状态＋16 行未来动作，补齐到模型 max_action_dim |
| 动作输出 | `[16,20]`，已去状态行，已逆 Cosmos 训练归一化 |
| 图像布局 | 上方 high、下方 left wrist/right wrist |
| 时间/分辨率 | 30 FPS、17 帧、resolution tier `256` |
| 归一化 | 与训练一致的 `norm_stats.state/actions.q01/q99` |

服务复用 `inference.action.build_action_batch` 的开环处理路径：状态只归一化一次，输出由模型的 `ActionProcessingRecord` 反归一化一次，再移除状态行。不会将数据生产端可能已归一化的值误称为物理量。客户端默认执行 16 步后重新请求；`--execute-horizon 1..16` 可以更频繁重规划，服务仍返回完整 16 步。握手拒绝 DROID/错误维度、时间设置、字段顺序；重连也拒绝 checkpoint、stats 或采样设置发生变化。

完整注册契约为：

```python
MOZ1_POLICY_CONTRACT = {
    "protocol": "moz1_policy_v1",
    "state_space": "dataset_stored_20d",
    "action_space": "dataset_stored_20d",
    "field_order": ["left_pose_6", "left_gripper_1", "right_pose_6", "right_gripper_1", "torso_pose_6"],
    "control_hz": 30,
}
```

## 本地 CPU 检查

在包含 NumPy、Pillow、pytest、websockets、msgpack 的环境中，从 RoboLab 根目录运行：

```bash
PYTHONPATH=.:/path/to/cosmos-framework \
python -m pytest -q policies/cosmos3/moz_integration_test.py
```

测试位于 `policies/`，不会加载 `tests/conftest.py` 启动 Isaac Sim。未设置 Cosmos 源码路径时会跳过服务端检查；未安装 torch 时会跳过图像拼接和模型调用适配的 CPU 测试。测试覆盖协议、分片预检/最新 checkpoint、相机布局、状态条件、动作裁剪、跨 chunk 刷新以及真实本地 WebSocket 的 probe；模型生成使用替身，不能替代 CUDA 模型加载和真实仿真验证。
