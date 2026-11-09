# RoboTwin 事件数据生产工具

此目录实现自动候选标注与已核验标注的分离存储。候选不是已验证的训练集。正式训练默认只读取 gold 子集。

- `batch.py`：510 个官方固定版本包，复用缓存，4 路有界并发，进度与 TensorBoard。
- `worker.py`：校验原始 ZIP 哈希，安全读取 HDF5/JSON；不执行任务代码，不反序列化 pickle。
- `task_schema.py`：静态解析任务代码，用 scene_info 对象 ID 绑定字符串参数。
- `annotate.py`：夹爪和 EEF 事件推断，生成 silver 候选。复杂、多对象、按压等任务不误标为 pick。
- `convert_archive.py`：逐 episode 原子提交，保存程序、单终点位姿、变长多相机关键帧、证据状态、来源坐标、seed 分组。动作控制未验证，不能训练动作头。
- `loader.py`：默认 gold；silver 必须显式 opt-in。变长关键帧 mask，当前帧与未来标签分离。
- `audit.py`：独立读取、哈希检查、episode 组划分检查，发布已核验 gold 子集。

数据约定：RoboTwin 原生世界坐标、米、xyzw，每臂一个终点 pose。保留机器人本体标签。尚未转换成机器人 base 坐标；不能与 base-frame 标签无条件混训。时刻是源记录 row，不伪称秒。相机在同一 get_obs 记录中同步；不同相机角色保留。官方 measured qpos / next-state 不冒充控制 command。

每个样本 `inputs` 只有当前图像、当前末端位姿、夹爪和任务指令。`targets` 包含程序、终点 pose、keyframes。未来事件角色是监督标签，不可偷看真实未来决定推理阶段。每原子在起点及中间事件时刻形成多个训练锚点，不把锚点数当原子数量。

验证 sidecar：`reviews/<episode_id>.json` 必须绑定 source_sha256、member、coordinate_frame，annotations 中 quality=gold 并包含 verification 的 accepted/reviewer/evidence_ref/reviewed_at。写入侧只校验字段存在；外部核验者必须实际核对事件、对象、终点和完成条件。已有版本保持不可变，新 review 应重新转换到新版本目录。

测试：`python -m unittest discover -s <production目录> -v`。独立验收：`python audit.py --root <输出目录>`。只有 audit 无错误且 gold 记录非零时生成 GOLD_SUBSET_READY.json；此标识不代表全词表、全数据源或控制器可训练。

## 第一阶段自动标注发布

`release.py --root <已转换目录> --output <新的发布目录>` 逐个检查源产物哈希、事件因果性、字符串参数和末端位姿，解码训练图片，排除 place 已包含的重复 move 标签和相同内容的 episode，输出 train/validation/test.jsonl。文件只引用不可变 sample.tar，不复制图片。任务种子跨机器人版本保持同一划分。

自动标签可显式用于弱监督第一阶段，**没有**升级成 gold。读取方式：

```python
from loader import load_release, collate
records = load_release('train.jsonl', accepted_policy='stage1_weak_events_v1')
sample = next(records)
batch = collate([sample])
weights = sample['loss_weights']
```

训练器必须应用 `loss_weights`：初始配置 program=0.5、future_rgb=0.5、terminal_pose=1、action=0。这些权重是初始建议，不是已验证的最优配置；仅提供读取器，没有声称接入并运行正式模型训练。不同机器人坐标/末端定义需分本体训练或显式校准。

批处理结束或达到 48 小时时限后，自动为成功转换的子集发布新清单；未下载、复杂事件未标注和异常源不包含在该清单中。r3 修复了下载进度文件删除/替换时导致主循环退出的问题。恢复批处理只复用成功提交的包和 episode，不覆盖旧产物。
