# AtomVLA endpoint pilot

本目录实现已对齐示范 → 实测终点参数 → 连续物理执行的首轮对照，以及 EEF 平移方向标注。没有训练模型。范围见本仓库 docs/STATUS.md。

## 文件

| 文件 | 用途 |
|---|---|
| `geometry.py` | N+1 状态与 `[start,end)` 区间标注、净方向/段内短时方向、XYZ 程序导出 |
| `libero_pilot.py` | 新脚本示范，原始回放、XYZ、XYZ＋姿态的配对执行 |
| `robotwin_pilot.py` | 原生 play_once 示范插桩，原始关节控制回放、cuRobo 重规划对照 |
| `audit_results.py` | 仿真完成后独立核验保存数据，导出 CSV、误差、摘要与校验值 |
| `derive_robotwin_tcp.py` | 从已记录 EEF 位姿按原生 0.12 m 局部工具偏置推导夹爪中心，另导出 TCP 方向 |
| `build_artifacts.py` | LIBERO 对比视频、段表与路径图；建议先于最终 audit 运行 |
| `build_robotwin_artifacts.py` | RoboTwin 对比视频、双臂记录的活动臂表与路径图 |
| `monitor.py` | 不依赖 Torch 的 TensorBoard 写入器 |
| `status_to_tensorboard.py` | 独立读取进度文件，避免监控打断仿真 |
| `final_summary_to_tensorboard.py` | 从完成结果生成独立 final run，区分执行率和场景处理率 |
| `collect_provenance.py` | 环境包版本、关键源码副本与哈希 |

## 接口与时序

`annotate(positions, segments, frame=..., arm_names=..., dt=..., horizon=...)` 接收米制实测位置 `[N+1,arms,3]`；单臂可用 `[N+1,3]`。原子必须完整划分 `[0,N)`，每段至少一个动作。段末参数来自实测 `positions[end]`，不是最后一个动作向量。

`endpoint_program()` 只输出 `primitive` 与 `target_xyz_m`。RoboTwin 双臂分组程序额外有 `arm`。四元数、示范持续时间、中间路径和方向标签不进入 XYZ 载荷。夹爪开合由动词决定。

`geometry.py` 中的启发式分段函数未用于本轮实测。当前边界均来自函数调用插桩，`semantic_verified:false`；不要把它当成已验证的七类人工标注。

方向采用世界坐标的平移差；无有效方向时使用 `null` 和掩码。净位移小于阈值不能自动认定整段静止，需同时看最大位移、路径长度和 `motion_type`。短时监督可用未来帧构造，但未来实测状态不能作为在线模型输入。

## 运行条件与命令

安装各自官方 LIBERO/RoboTwin 环境和资产后，查看 `libero_pilot.py --help` 或 `robotwin_pilot.py --help`。使用本机路径和新的输出目录。几何测试可独立运行：

```sh
python -m unittest discover -s code/atomvla-endpoints -p 'test_*.py'
```

独立监控入口是 `status_to_tensorboard.py`；TensorBoard 读取结果目录下事件文件。这些脚本是终点执行 pilot，不是已训练模型的 benchmark runner。

## 已知范围限制

XYZ 基线对每段保持段首姿态；LIBERO 的不同移动动词共用通用闭环控制，并非学到的原子策略。RoboTwin 原生 `move` 没有人工映射到七类语义原子。原生夹爪段的 XYZ 是统一标注字段，RoboTwin 执行器不消费它来移动手臂。

`valid_reference` 的源码判断要求原始回放轨迹长度一致；本轮所有完成场景均一致。若未来长度不一致，当前脚本会在最终有效性判定处遇到 `None` 数值比较问题，应先处理缺失差值，不能把异常样例直接纳入成功率。

没有公共语料全部 episode 的转换、预测终点评估、接触因果诊断、学习姿态补全、多任务泛化和训练集/测试集实验。源环境缺少 Git 元信息，已提供相关源码哈希和包版本，但这不是完整依赖/资产镜像。
