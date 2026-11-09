**全参数两阶段实验实现已完成主要训练链路，CPU 模型/分布式/数据契约及完整训练入口测试通过；完整 Qwen3-VL-2B 的 GPU 显存、吞吐和多卡长训尚未验证。** 不能把这些 CPU 检查称为正式 GPU 实验通过。历史 code/model.py、train.py 保留为旧 LoRA pilot，不用于新实验。

## 当前训练规则

- 两阶段均不创建 LoRA；Qwen 视觉编码器、视觉到语言连接、语言层、输入 embedding、Query 全部设为可训练。参数使用 FP32，计算采用 bf16 autocast。阶段一 LM 输出头接收代码交叉熵；阶段二不调用 LM 输出头，它虽未被人为冻结但不会得到不存在的文本损失梯度。阶段一动作专家没有监督，明确不加入更新；阶段二启用。
- 阶段一 Query dropout=0.2，按微 batch 完全移除 Query；该分支只计算普通代码交叉熵。
- 阶段二 Query 和动作损失永远保留；RGB 像素、位姿两项损失各独立以 p=0.5 丢弃，按微 batch 采样并在所有 rank 使用相同门控。约四分之一只训练动作、四分之一动作+像素、四分之一动作+位姿、四分之一三项全部。不做 1/(1-p) 放大，保留时权重各为 0.1，期望权重为各 0.05。
- 验证时 Query、RGB 像素、位姿全部启用；缺标签不当作 dropout。记录各损失实际被启用的次数与条件均值，避免把跳过损失记成改善。
- 动作专家为八层、512 隐藏维 Transformer，共享主干，DROID 与 LIBERO 分别使用原生状态/动作投影。DROID 为 8 维原生关节目标与夹爪，LIBERO 为 7 维 OSC 控制目标；独立域统计和 padding mask。RoboTwin/GenieSim 尚无已核验控制目标，不进入阶段二。
- 完整数值轨迹保留，但首版正式训练读取每 episode 至多 8 个包含当前/未来 RGB 对的缓存窗口，并非每一帧都有未来 RGB 监督。增加动态时间锚点需要先补标签，不能悄悄把无标签时间点当成有监督样本。

## 主实验配置

`configs/full_ft/main.json`：8 GPU，micro batch=1，累积 32，全局 batch=256。建议使用 8×H100 80GB；此资源选择尚需真实显存/吞吐预检。若使用其他卡数，保持等效 batch=256 并改配置，不直接改变优化规模。

| 参数 | 阶段一 | 阶段二 |
|---|---:|---:|
| 优化器步数 | 20,000 | 30,000 |
| 样本窗口抽取次数 | 5,120,000 | 7,680,000 |
| VLM 峰值学习率 | 1e-5 | 5e-6 |
| Query/新模块峰值学习率 | 1e-4 | 1e-4 |
| warmup | 1,000 步 | 1,500 步 |
| 辅助损失 | RGB 像素 1.0，位姿 0.25 | RGB 像素 0.1，位姿 0.1，分别 50% dropout |
| 离线验证 | 每 1,000 步，固定分来源子集最多约 512 窗口 | 同左，额外动作采样误差 |
| 完整检查点 | 每 2,000 步及结束；保留最近 3 份 | 同左 |

AdamW weight decay=.01，梯度裁剪 1.0，cosine 衰减至峰值的 10%。20k/30k 是首轮预算建议，不是保证收敛的科学结论。约 80/10/10 的 episode 划分按已冻结规范目录执行；各源先按 episode 数平方根选择来源，再任务均匀、episode 均匀、窗口均匀抽样。日志记录真实覆盖数，等效遍数不冒充每样本恰好遍历相同次数。

主实验读取全部合格数据，不截断成 100k。阶段一目标至少 100k 独立 episode，实际可能达到 150k–200k 以上，取决于 RoboTwin 包内容、去重和排除结果。DROID 94,340 + GenieSim 1,378 + LIBERO 6,500 = 102,218 是源目录/官方元数据规模，不是已完成数；RoboTwin 标称 126,500 不是已核验数量。阶段二当前可核验控制数据的源规模上限是 DROID+LIBERO=100,840，实际合格数量另算。不能声称所有 200k episode 都提供控制动作监督。

对于总量 100k、每条 8 窗口、80% 训练，阶段一 20k 步是约 8 个等效窗口遍数；150k 是约 5.33，200k 是约 4。阶段二若有约 100k 合格 episode，30k 步约 12 遍。实际量用 `python -m full_ft.size_plan --config ... --index .../index.json` 计算。

开发验证配置 `configs/full_ft/development.json` 两阶段各 2,000 步，建议先用至少约 10k 多源独立 episode 验证收敛趋势、RGB 重建基线、原生动作接口。它不是第三阶段。GPU smoke 可用 `--diagnostic --smoke-steps 8`，micro batch/累积会降为 1，硬限制最多 100 步；绝不能把诊断索引用于正式主实验。

## 文件入口

- `code/train_full.py`：多卡 DDP 两阶段训练、独立 dropout、验证、完整检查点与恢复。
- `code/full_ft/model.py`：全参数 Qwen＋连续 Query＋RGB 像素/位姿头＋分机器人动作投影。
- `code/full_ft/data.py`：不可变索引、原生维度和归一化、哈希校验、可恢复分层抽样。
- `code/build_training_index.py`：从最终规范目录构建正式索引；`--diagnostic` 只生成拒绝正式长训的诊断索引。
- `code/full_ft/runtime.py`：保存完整模型、优化器和每 rank RNG/覆盖统计，重载验证后才写 READY。
- `code/launch_full_training.sh`：独立 wall/idle watchdog 包裹 torchrun；不会申请集群资源，只在已分配资源中运行。

## 执行示例

在本任务远端根目录执行；先由最终数据发布流程提供 `data/full_v3/catalog/status.json` 的真实就绪状态，不允许手工翻转 preview 标志。

```bash
.venv/bin/python code/build_training_index.py \
  --catalog data/full_v3/catalog --output data/full_v3/training_index_v1

bash code/launch_full_training.sh configs/full_ft/main.json 1 \
  /opt/atomvla/training/full_ft_stage1

bash code/launch_full_training.sh configs/full_ft/main.json 2 \
  /opt/atomvla/training/full_ft_stage2 \
  --init /opt/atomvla/training/full_ft_stage1/checkpoints/step_00020000
```

恢复同一阶段使用 `--resume` 指向该阶段已验真的完整检查点；配置、代码、数据索引和卡数必须完全一致。跨阶段初始化要求同一架构/原生动作注册表和同一正式数据索引；不接受历史 LoRA checkpoint，也不接受诊断 checkpoint 进入正式阶段二。

第一次运行先做 GPU smoke、显存与吞吐测量，再提交正式预算。每次提交必须登记 run_state.json 的 job ID/进程、配置和 TensorBoard 链接。本次没有提交新模型训练 GPU 作业。

## 监控与异常处理

正式事件写入 `tensorboard/full_ft/formal/<run>/rank_N`，smoke 写入独立 `smoke/`。GPU watchdog 写到资源目录并链接到主 TensorBoard。每 rank 记录样本 ID、来源抽取量、Query/辅助 dropout 比例、启用损失均值/计数、梯度、学习率、显存、时间、episode 覆盖。验证包含RGB 像素 MAE 与零像素基线、位置误差、旋转角误差和动作采样误差。

两天单次 wall 上限、30 分钟无进度退出；DataLoader 180 秒、分布式通信 900 秒超时。非有限损失/梯度、保存失败立即失败退出。外层 timeout 负责兜底释放自己进程组，不无限重试。只保存完整且重载一致的 checkpoint 才可续训；最近三份以外只清理本次运行自己的完整 checkpoint。

当前离线验证属于固定子集，不能当作完整测试集或闭环任务成功率。闭环 LIBERO/RoboTwin 和真机评估尚未接入本训练脚本；RGB 像素重建尚未完成效果基线，正式预算前必须诊断。现有程序标签仍是运动派生标签，不冒充已核验语义原子。

## 已执行验证

1. 小型随机初始化的真实 Qwen3-VL（含视觉塔）：无 LoRA，全 VLM 可训练；仅动作损失真实更新视觉权重，语言层和 Query 梯度非零；阶段二 LM 头调用 0；四种辅助损失组合通过。
2. 阶段一改变目标 token 不改变 Query；无 Query 分支只走代码损失。
3. 完整模型/优化器/RNG 恢复后，下一步损失和所有参数逐值一致。
4. 两 CPU rank Gloo：混合机器人头、梯度累积、阶段一 Query dropout、阶段二辅助 dropout 切换后，所有模型张量跨 rank 完全一致。该测试不替代 NCCL/GPU 检查。
5. 实际 DROID/LIBERO/RoboTwin 数据读取、原生控制 mask、哈希和采样续跑检查通过。旧快照只有已复核 RGB 对的一部分，不能反映最新全量可用数，也不会用于正式训练。

结果见 `evidence/ft_cpu_checks.json`、`ft_ddp_checks.json`、`ft_data_checks.json` 和真实处理器检查记录。测试数据/模型均明确区分为诊断用途。


补充：完整 train_full.py CPU 诊断已用实际处理器与源记录跑通两阶段各 2 步，从第 1 步恢复后下一步 loss、样本顺序与全部参数逐值一致。模型是小型随机 Qwen，不是完整预训练 2B。证据为 evidence/ft_integration_checks.json。CPU 模式只接受有界诊断，不允许正式长训。数据工作进程采用 spawn；检查点按需 mmap 读取并预检磁盘空间；每次保存保留完整优化器和当前 rank 的 GPU RNG，避免初始化其他 GPU 的随机状态。


## Benchmark 微调衔接

已支持配置 `run_kind=benchmark_finetune`、`benchmark=libero_pro`，仍通过 `train_full.py --stage 2 --init <完整主阶段二checkpoint>` 进入。必须从已跑满预算的公共主阶段二权重初始化；优化器、样本计数重新开始。另一个 benchmark 的 checkpoint、阶段一或未完成阶段二都不能充当起点。全参数和两项 50% 辅助损失 dropout 保留。

`code/build_benchmark_index.py` 从已发布源目录派生原始 LIBERO 四套件子索引，保留原划分和每域 TRAIN 统计。正式模式不接受 preview；诊断模式输出单独目录。实际源 domain 为 `libero_libero_goal` 等。`code/prepare_benchmark_config.py` 从该最终索引计算初始预算（4卡/batch64，10个等效窗口遍数，2k–20k步）。两脚本均不提交GPU。

新增 CPU 集成检查已通过，证据为 evidence/ft_benchmark_integration_checks.json，覆盖实际数据的子索引加载、阶段二到微调初始化和精确恢复。仍不是完整 2B GPU 或闭环 benchmark 效果验证。RoboTwin registry 和原生闭环评测器尚未完成，入口保持阻止；不得将其训练 mask 改为有效来绕过数据核验。
