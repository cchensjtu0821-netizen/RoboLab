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

不需要 Isaac Sim 或 QX 控制器，可以使用已有 MOZ1 开环评估产生的 `inputs/<sample>/sample.json`。该 JSON 引用的 `conditioning.png`、`state.json`、`norm_stats.json` 必须在客户端机器可读；复制时一起保留它们，修正 JSON 中仅在原服务器有效的绝对路径。相对路径按 JSON 所在目录解析。

在 RoboLab 仓库执行：

```bash
MOZ1_REMOTE_HOST=<policy-server-ip> \
bash scripts/run_cosmos3_moz.sh \
  --probe-sample /path/to/sample.json \
  --probe-output output/moz1_probe_001
```

启动器使用 RoboLab `.venv`（其次 `.venv-51`），也可用 `QX_PYTHON=/path/to/python` 指定已有环境。probe 模式不导入或启动 Isaac Sim，只需要 NumPy、Pillow、websockets、msgpack。

probe 核对服务器契约与本地 stats 的 SHA-256，返回有限的 `[16,20]` 动作后保存 `action.json` 和 `policy_metadata.json`，**不会执行机器人动作**。输出目录必须尚不存在。预测动作仍是数据集存储空间；不会自动变成米、弧度或关节角。probe 可验证在线模型链路，不能证明任务成功或控制正确。

## 3. QX 仿真与 WebRTC

当前 `scripts/run_qx_banana.sh` 的 USD 加载和 WebRTC 显示已由用户确认成功。它没有 QX 的 Isaac Lab 控制注册，因此这里不能直接拿 USD 路径替换 DROID 后执行模型动作。

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

在 QX 控制注册可用前，仿真模式会明确报缺少 `--registration`；不会静默改用 DROID。当前缺少真实 USD 内容、关节/末端/驱动映射和数据生产端坐标系/单位/相对位姿定义，不能声称已完成 QX 闭环。

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
