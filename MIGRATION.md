# Go2 Quad2Hand 迁移记录

## 目标与来源

将 `go2_rl/legged_gym/envs/go2` 的四足/倒立任务接入当前 Isaac Lab 扩展，并由 `scripts/rsl_rl/train.py` 训练。旧代码实际依赖 Isaac Gym (`isaacgym`、`gymtorch`)，不是 Isaac Sim 原生环境。

## 步骤

- [x] 清点旧任务配置、环境、奖励、观测、PPO 和估计器代码。
- [x] 将首次迁移的 `DirectRLEnv` 实现重构为 Isaac Lab 原生 `ManagerBasedRLEnv`，注册 `Go2-Quad2Hand-v0`，删除旧 Direct 环境文件。
- [x] 配置 Scene、Command、Action、Observation、Reward、Termination、Event 各 manager；自定义项集中于 `mdp/quad2hand.py`。
- [x] 迁移两种步态的速度命令、相位、接触、55 维单帧观测及 4 帧历史、奖励、重置和终止条件。
- [x] 将 PPO 参数和 3 维基座速度估计器接入模板的 RSL-RL 5 训练入口；检查点一并保存估计器、优化器和学习率状态。
- [x] 加入地面摩擦、基座质量和质心、关节摩擦/惯量/阻尼、PD 增益、外推及初始状态随机化。
- [x] 在 Isaac Sim 5.1 中完成 manager based 环境的 64 环境、2 次 PPO 更新短训练（3072 步），确认估计器损失和 278/570 维网络输入。
- [x] 加载 manager based 新检查点并完成 1 环境、20 步有界回放；JIT 导出速度估计器和策略。
- [ ] 正式规模的长期训练与运动效果调参。

## 对齐与差异

旧控制周期为 `0.005 × 4 = 0.02 s`；新环境相同。单帧 actor 观测为 55 维，4 帧历史后为 275 维；速度估计器输出 3 维，实际 actor 输入为 278 维；critic 为 570 维。步态模式和接触缓存由 CommandTerm 管理；历史由 ObservationTerm 管理，顺序为当前帧、从新到旧的 4 帧。旧训练器的速度估计器由独立 runner 训练；当前通过 `Go2EstimatorRunner` 和 `Go2EstimatorVecEnvWrapper` 与 RSL-RL 5 集成。估计器使用 critic 中的真实基座速度作为监督标签，每 24 个环境步更新一次。新旧 runner 的检查点格式不同，旧检查点不能直接加载。

Isaac Lab 自带 Go2 USD 取代旧 URDF/DAE，其关节顺序、碰撞体、质量和电机模型可能不同。动作与关节通过名称映射；奖励、接触和关节重置按名称索引。旧任务的执行器模型和 PhysX 实现与当前 Isaac Lab 不完全相同，随机化分布虽按旧数值设置，训练曲线仍需实测调校。旧配置中 `randomize_torque`、`randomize_motor_zero_deviation` 为关闭状态，当前未启用。

回放导出两个模型文件：`exported/policy.pt` 接收 278 维输入，`exported/velocity_estimator.pt` 接收 275 维输入。部署时先对 275 维观测运行估计器，再拼接其 3 维输出送入策略。

## 运行

使用 Isaac Lab 的 Python 启动器或已配置的 Isaac Sim 环境：

```bash
python -m pip install -e source/go2_locomotion
python scripts/rsl_rl/train.py --task Go2-Quad2Hand-v0 --headless --num_envs 64 --max_iterations 10
python scripts/rsl_rl/play.py --task Go2-Quad2Hand-v0 --num_envs 1 --max_steps 200
```

先用 64 环境完成短训练，再提高到默认 4096 环境进行正式训练。
