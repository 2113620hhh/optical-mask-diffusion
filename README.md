# Optical Mask Diffusion

这是用于逆光刻/计算光学实验的研究代码仓库。仓库只保留当前有效的专家掩码生成、模型网络、模型训练和必要依赖代码。

## 1. 专家掩码生成（当前生产链路）

当前生成训练数据的主程序是 `生成多保真高质量专家掩码数据集.py`，不是训练脚本，也不是评估脚本。

启动脚本是 `generate_lithosim_asf10_400_40gpu.slurm`。它在 5 个节点上各启动 1 个 torchrun launcher，每个 launcher 启动 8 个进程，总共 40 个 rank；rank 进程通过 `run_lithosim_generator_rank.sh` 调用主程序，并使用 `cuda:$LOCAL_RANK`。

主程序的实际 Python 调用链是：

`生成多保真高质量专家掩码数据集.py`
→ `expert_mask_quality_lab_20260829/direct_linear_gumbel_experiment.py`（DirectLinearASFPlan、ncc、normalize_mean）
→ `expert_mask_quality_lab_20260829/fast_multifidelity_hybrid.py`（exponential_value、remember_candidate）
→ `光学FFT前向传播_小分辨率.py`（SmallFFTForwardPlan 和光学评分）
→ `生成多精度正确专家掩码数据集.py`（源 shard 读取、完整性校验、原子写入、旋转输出）。

因此，完整的数据生成链路由上述 5 个 Python/ shell 文件共同组成。`generation_config.json` 只记录参数，不是生成程序。

### 核心主程序：`生成多保真高质量专家掩码数据集.py`

这个文件负责真正完成“逐个 target 优化专家掩码并写出训练 shard”的全部生产逻辑，作用不是简单封装调用。主要步骤如下：

1. 读取输入目录中 `train`、`val`、`test` 的原始 NPZ shard，并按 `native_shard_index % world_size` 分配给 40 个 rank。
2. 为每个 GPU 建立一个 proxy 光学传播计划和一个 reference 线性 ASF 传播计划。
3. 对每个 target 创建 `512×512` 的随机 latent，经过 sigmoid 得到连续概率掩码，用 Adam 最大化 proxy 光场与 target 的 NCC。
4. 将连续概率转换为二分类 logits，使用 hard Gumbel-Softmax 继续优化二值掩码；每隔固定步数保存随机候选和确定性硬阈值候选。
5. 将候选掩码全部送入 reference ASF `inter=10` 前向传播，以最终固定二值掩码 NCC 重新排序，选择得分最高者。
6. 把选中的 `expert_mask`、`expert_pred`、`score`、高频/梯度 score、`mask_mean` 等字段写入输出 NPZ；完成后删除对应 partial 文件。
7. 每 4 个样本原子保存一次 partial 状态；任务中断后会跳过完整 shard，并从 partial 状态继续，不改写输入数据。

该文件的 `worker()` 函数负责分布式样本生成，`optimize_sample()` 负责单样本优化，`optimize_attempt()` 负责连续阶段、Gumbel 阶段和 reference 评价，`finalize()` 负责检查全部 shard 并生成最终 metadata。任何生成失败都会在 `failures/` 写入错误记录后终止对应 rank，不会删除已经完成的数据。

当前生产参数：mask/target 为 512×512/192×192；proxy ASF inter=10；reference ASF inter=10；连续优化 200 步；二值 Gumbel 优化 200 步；两个学习率均为 0.1；tau 从 1.0 退火到 0.08；候选数 32；min-ncc=0；field-threshold=0.57；foreground-threshold=0.05。

生成器读取 `lithosim_all_targets_192_npz`，写入 `lithosim_all_targets_192_npz_asf10_expert_steps400_40gpu`。完整 shard 会跳过，`partials/` 用于断点续跑，输入数据集只读。

## 2. 模型网络与训练（当前有效链路）

训练入口是 `训练扩散专家掩码模型_修正角谱版.py`，它调用 `训练扩散专家掩码模型.py`。

训练实现的依赖链：`训练扩散专家掩码模型.py` → `扩散专家掩码模型.py`（ExpertMaskDiffusionUNet）→ `光学FFT前向传播_小分辨率.py`（线性 ASF/FFT）→ `小分辨率轨迹数据集.py`（NPZ 读取工具）→ `optical_visual_quality.py`（质量指标）。

40 卡续训通过 `resume_manhattan30k_score_only_40gpu.slurm` 调用 `resume_mfhq120k_score_focus_40gpu.slurm`，再由 torchrun 启动上述训练入口。

训练程序只读取已生成的 `train/val/test/*.npz`，不会调用专家掩码生成器，也不会重新生成数据。

## 3. 当前训练参数

完整快照是 `runs/manhattan30k_score_only_linear_asf10_lr2.5e-6_epoch200/train_config.json`。当前关键设置为：mask/target=512×512/192×192，训练/验证 inter=10/10，初始学习率=2.5e-6，总 epoch=200，`score_plateau`（patience=2、factor=0.5、minimum lr=3e-7），best metric=`cosine_score`。denoise/BCE/mask-mean 权重为 0.025/0.10/0.10，physical cosine 权重为 1.20，二值打印/Dice 损失全部为 0。

## 4. 仓库文件

专家掩码生成：`生成多保真高质量专家掩码数据集.py`、`生成多精度正确专家掩码数据集.py`、`run_lithosim_generator_rank.sh`、`generate_lithosim_asf10_400_40gpu.slurm`、`expert_mask_quality_lab_20260829/direct_linear_gumbel_experiment.py`、`expert_mask_quality_lab_20260829/fast_multifidelity_hybrid.py`。

模型训练：`训练扩散专家掩码模型.py`、`训练扩散专家掩码模型_修正角谱版.py`、`扩散专家掩码模型.py`、`光学FFT前向传播_小分辨率.py`、`小分辨率轨迹数据集.py`、`optical_visual_quality.py`。

训练启动和参数：`resume_manhattan30k_score_only_40gpu.slurm`、`resume_mfhq120k_score_focus_40gpu.slurm`、`runs/.../train_config.json`。

数据 NPZ、checkpoint、日志和临时缓存由 `.gitignore` 排除；数据格式和生成参数快照保留在 `lithosim_all_targets_192_npz/metadata.json` 与 `lithosim_all_targets_192_npz_asf10_expert_steps400_40gpu/generation_config.json`。

## 5. 环境

Python 3.10、PyTorch 2.5.1、ROCm/DTK 25.04.2、numpy、scipy、Pillow。光学结果依赖 FFT 后端、设备精度和 `inter_num`；生成、训练和验证当前均使用 ASF `inter=10`。
