# Step 1：整机流程修正记录

依据《智能分拣代码审查与优化方案》第一阶段“先修复整机流程”实施。

## 本步骤已处理

- R01：PICK_TARGET 后严格执行 `PICK_ACK -> RECOGNITION_READY -> SIDE`，ACK 不再直接启动侧面识别。
- R02：每个分拣周期显式 reset 顶部/侧面 pipeline；READY 时再次清空侧面锁存，避免上一物体结果泄漏。
- R03：`ObjectInfo` 改用 `qr_present: bool`，分箱键保留二维码存在性，不再固定 `qr=""`。
- R04：保留 MCU_ERROR 和协议一致性错误的故障保护；PICK_ACK / READY / SIDE / SORT_ACK / SORT_DONE 等动作等待超时只触发周期重发，真正的串口掉线由心跳 watchdog 进入可恢复 LINK_OFFLINE；错误箱号 DONE 仍直接故障停机。
- R05：StorageManager 改为 `reserve -> commit/cancel`；只有匹配当前箱号的 SORT_DONE 才增加库存。
- R07：TopCamera 增加 close/release 统一接口；main 使用 try/finally，并逐资源独立释放。
- R08（第一部分）：STM32 接收移到独立线程；视觉推理结束、发送动作前重新检查 urgent STOP/ERROR 标志，避免慢推理后继续下发新动作。
- 额外防护：`process_top/process_side` 增加状态门控，防止调试调用造成重复 PICK/SORT 下发。

## 新增离线回归

`scripts/test_runtime_flow_step1.py`

覆盖：

1. ACK 前后状态顺序；
2. READY 前禁止侧面最终识别；
3. DONE 前库存不增加；
4. 二维码存在性参与分箱键；
5. 错误箱号 DONE 不提交；
6. 新周期不复用上一物体锁存；
7. UART 发送失败不留下库存；
8. 同批 ACK+READY / ACK+DONE 顺序消费；
9. 推理期间出现 urgent STOP/ERROR 后不再发送动作；
10. ACK 等待超时只记录 `[WAIT][RETRY]` 并重发；模拟串口掉线时进入 LINK_OFFLINE，恢复后重新 STOP 同步。

## 下一步仍需处理

- R06：把真实机械 X/Y 工作区、禁入区、夹爪尺寸写入统一配置，并在发送前做最终工作区复核。不能用猜测值。
- R08（后续）：在 Jetson 实机上测量视觉任务 P50/P95；若单次任务过长，再把视觉任务本身改为受控 worker/有界任务，而不是让主控制线程长期阻塞。
- C01~C06：统一配置入口、相机设备映射、相对路径、脚本 ROOT、生产/调试共用配置。
- 顶部/侧面标定工具和 TensorRT 优化按原审查文档后续顺序继续。

## 离线验证命令

```bash
python -m compileall -q runtime communication vision/storage vision/pipeline main.py
python -m scripts.test_sorting_protocol
python -m scripts.test_storage_manager
python -m scripts.test_runtime_flow_step1
python -m scripts.test_runtime_mock
python -m scripts.test_sorting_state_machine
```

注意：以上是无相机、无 STM32、无真实机械的离线验证，不替代 Jetson 整机验收。
