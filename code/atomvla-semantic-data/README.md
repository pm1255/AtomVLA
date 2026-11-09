# AtomVLA 语义事件数据接口（参考实现）

这是接收已核验标注的转换与校验模块，不是从原始视频自动识别程序的完整生产流水线，也不兼容旧 full_v3 训练器。

`semantic_records.py` 提供 `make_record(episode, annotation, current_row, horizon)` 和变长关键帧引用的 `pack_keyframes`。输入字段与时间约定见函数说明；可运行的合成输入见测试文件。只实现 pick、move、place 三个起始 schema；其他原子必须先补充、核验 schema，不能默认用单个终点代替。

- 文本只输出程序名称与字符串槽；终点 position/quaternion 单独保存，一臂一个 pose。
- pick：pregrasp、closed、lifted；move：arrived；place：preplace、supported、released。
- 要求语义、引用绑定、完成事件和事件时刻均具有核验状态及证据来源。模块验证标注的结构和来源是否齐全，不会独立证明标注内容真实。
- action[j] 必须表示 state[j] 到 state[j+1]。完成状态是 c，则只能监督 j<c 的动作。
- 所有 pose 已转换为机器人 base 坐标系（米、xyzw）。相机按显式时间容差对齐，当前输入不能使用未来相机帧。
- 未来事件角色、时刻、图像、终点 pose 均放在 targets/metadata；部署端必须用当前/历史观测估计阶段，不能把真实未来事件用于推理。
- `pack_keyframes` 只整理引用，不读取图像，不构造 MoT 注意力；padding 的 temporal mask 需要训练 loader 确实执行。

验证：在工作区运行 `.venvs/atomvla-endpoints/bin/python -m unittest discover -s code/atomvla-semantic-data -v`。8 项测试只覆盖合成数据契约，不代表真实数据语义精度。

尚缺：原始数据适配、事件检测与人工复核工具、完整词表、图像解码和分片写入、训练 loader、单终点 Query head、变长 RGB/action MoT、因果阶段估计，以及实际样本与闭环评估。未启动数据重建或训练；下一次正式提交必须接入并验证 TensorBoard。
