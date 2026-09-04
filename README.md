# Optical Mask Diffusion

这是一个“目标光场 → 一次前向生成 512×512 掩码 → 线性有限孔径 ASF/FFT 传播”的研究代码仓库，用于逆光刻/计算全息类光学掩码生成实验。

## 仓库范围

仓库只保留当前有效的训练、网络、光学传播、数据读取和专家掩码优化代码。训练数据、模型 checkpoint、训练日志和集群临时文件不纳入 Git，因为它们体积很大且可能包含机器路径或实验隐私。

重要说明：当前仓库中没有实际批量生成 `real_circuit_manhattan_expert_30k_diverse_correct_v1` 的主程序。因此，不能把仓库中的某个 Python 文件误认为是这 30K 数据集的完整生成器。数据集元数据明确记录的生成器是 `multifidelity-linear-asf-v1`，但该批量生成脚本位于本仓库之外、尚未被提供。`expert_mask_quality_lab_20260829/direct_linear_gumbel_experiment.py` 只是单样本/小规模的线性 ASF + Gumbel 优化实验，不能独立重建 30K 数据集。若要让专家完整审查“数据生成 → 训练”的全部链路，还需要补充当时实际运行的批量生成主脚本。

## 主要文件

### 模型与训练

- `训练扩散专家掩码模型.py`：实际训练实现、数据集类、损失函数、验证和 checkpoint 保存。
- `训练扩散专家掩码模型_修正角谱版.py`：当前训练入口，调用上面的 `main()`。
- `扩散专家掩码模型.py`：扩散 U-Net、Fourier 特征和模型构建函数。
- `resume_mfhq120k_score_focus_40gpu.slurm`：5 节点 × 8 卡的通用分布式启动器。
- `resume_manhattan30k_score_only_40gpu.slurm`：当前 30K 数据集 score 微调启动脚本。
- `runs/.../train_config.json`（不默认提交）：某次运行实际保存的完整参数快照。

### 光学与数据

- `光学FFT前向传播_小分辨率.py`：有限支撑线性 ASF/FFT 前向传播和 `FFTPlanCache`。
- `小分辨率轨迹数据集.py`：NPZ 数据读取工具（部分旧实验使用）。
- `optical_visual_quality.py`：光场质量、连续性和辅助指标函数。
- `expert_mask_quality_lab_20260829/direct_linear_gumbel_experiment.py`：专家掩码的直接线性 ASF/Gumbel 优化实验。
- `real_circuit_manhattan_expert_30k_diverse_correct_v1/metadata.json`：数据集格式、分辨率和物理口径说明；实际 NPZ 数据被 `.gitignore` 排除。
- `real_circuit_manhattan_expert_30k_diverse_correct_v1/generation_config.json`：该数据集的生成参数记录，不是生成程序本身。

## 30K 数据集的已知生成口径

从数据集的 `generation_config.json` 和 `metadata.json` 可以确定：

```text
generator version       multifidelity-linear-asf-v1
source data directory   target_source_manhattan_diverse_30k_v1
mask / target           512×512 / 192×192
proxy propagation       inter=6
reference propagation   inter=10
continuous steps        250
binary steps            250
candidate count         16
quality attempts        3
direct polish           200 steps on failure
acceptance              fixed binary mask evaluated by reference inter=10 ASF
minimum NCC             0.92
```

因此，数据生成阶段的物理流程是“连续代理优化（inter=6）→ 硬 Gumbel 候选搜索 → 最终 inter=10 线性 ASF 验收”，而当前模型训练阶段使用的是 `训练扩散专家掩码模型.py`，训练/验证均为 `inter=10`。两者不是同一个程序。

## 当前有效训练口径

当前 score 微调实验的关键设置如下：

```text
mask resolution       512×512
target resolution     192×192
training FFT inter    10
validation FFT inter  10
train stage           direct
prediction type       x0
initial learning rate 2.5e-6
total epochs          200
LR scheduler          score_plateau
plateau patience      2 epochs
plateau factor        0.5
plateau threshold     0.001
minimum learning rate 3e-7
best metric           cosine_score
```

非二值项保留：

```text
denoise_weight        0.025
bce_weight            0.10
mask_mean_weight      0.10
physical_weight       1.0
physical_cosine       1.20
physical_l1           0.25
highpass_cosine       0.30
highpass_l1           0.15
gradient_cosine       0.20
```

本轮争议性的二值打印/Dice损失全部关闭：`binary_print_loss_weight` 及其 Dice、FP、FN、背景尾部、连续性和 endpoint 子项均为 `0`。训练日志中的 `score` 是光场与目标光场的余弦相似度实现，不是 NCC 代码中的另一套指标名称；二者在非负光场归一化场景下数值形式相同。

## 数据格式

数据集目录需要包含：

```text
dataset/
├── metadata.json
├── train/*.npz
├── val/*.npz
└── test/*.npz
```

每个 NPZ shard 至少应提供训练代码所需的 `target`、`expert_mask`、`optics` 等数组。`metadata.json` 应声明 `mask_resolution=512`、`target_resolution=192`，以及与训练一致的线性 ASF 前向模型。

## 训练示例

先准备与脚本中一致的 Conda/PyTorch/ROCm 环境，并确认数据集与 checkpoint 路径。当前 40 卡微调：

```bash
cd /path/to/optical-mask-diffusion
sbatch resume_manhattan30k_score_only_40gpu.slurm
```

脚本默认从：

```text
runs/manhattan30k_field_print_linear_asf10_40gpu_lr5e-6_v2/checkpoints/latest.pt
```

恢复，并将新 checkpoint 写入另一个输出目录；请按实际部署环境修改 `DATASET_DIR`、`BASE_CHECKPOINT` 和 `OUT_DIR`，不要覆盖基础模型。

## 评估

评估脚本不是训练必需依赖。若需要复现实验，可使用仓库中的 `eval_*.py` 和对应 Slurm 脚本；评估任务只读取 checkpoint，不应写入训练输出目录。

## 可复现性与限制

1. 光学结果依赖 PyTorch、ROCm/FFT 后端、设备精度（当前训练使用 BF16）和 `inter_num`。
2. 训练与验证必须使用同一物理口径；当前均为 `inter=10`。
3. checkpoint 和原始数据不随代码发布，专家需要代码审查时应使用与 `metadata.json` 匹配的数据或脱敏小样本。
4. 中文文件名是项目现有接口的一部分，请勿在不修改所有 import/启动脚本的情况下单独改名。
