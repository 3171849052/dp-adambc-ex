# 实验 2：DP-AdamBC clamp / adaptivity 机制诊断

## 实验目的

本实验只针对 `bert-base-cased`、GLUE QNLI 和 standalone `DPAdamBC`，诊断
`gamma_prime=3e-9` 时的 clamp 比例、未 clamp 坐标的真实 update 能量，以及
`gamma_prime` 的作用究竟更接近 Adam adaptive preconditioning，还是
`1/sqrt(gamma_prime)` 带来的 effective learning-rate 改变。

`DiagnosticDPAdamBC` 通过 `super().step()` 使用现有 `DPAdamBC` 的唯一真实
参数更新路径。它只在父类更新后维护 clean second moment、标量归约和临时
quantile buffer；clean state 绝不参与真实参数更新。

## 指标公式

Opacus 传给底层优化器的 `parameter.grad` 是 clipped + noised mean gradient，记为
`y_t`。若 `parameter.summed_grad` 可用，clean diagnostic 使用
`x_t = summed_grad / expected_batch_size`。噪声方差为：

```text
phi = (noise_multiplier * max_grad_norm / expected_batch_size)^2
```

```text
v_hat = v_t / (1 - beta2^step)
r = v_hat - phi
q = clamp(r, min=gamma_prime)
```

clean second moment 只用于诊断：

```text
v_clean = beta2 * v_clean + (1-beta2) * x_t^2
v_clean_hat = v_clean / (1-beta2^step)
bc_clean_nrmse = ||r - v_clean_hat|| / (||v_clean_hat|| + 1e-30)
```

`negative_fraction`、`clamp_fraction` 和 `active_fraction` 分别统计全模型元素
上 `r <= 0`、`r <= gamma_prime` 和 `r > gamma_prime`，且
`active_fraction + clamp_fraction = 1`。核心指标每个 logical step 都记录；
`q_over_gamma` 的 mean/p50/p90/p99 则稀疏写入 `q_quantiles.csv`。默认每 20 个
真实 logical steps 精确计算一次，并在最终 logical step 强制计算一次。每次计算
仍基于该 step 所有 trainable parameter elements 拼接后的 CPU float64 tensor，
不是逐 tensor percentile 的平均。

floor-reference 指标使用：

```text
u_bc = m_hat / sqrt(q)
u_floor = m_hat / sqrt(gamma_prime)
update_cosine_to_floor = dot(u_bc, u_floor) / (||u_bc||*||u_floor|| + 1e-30)
update_norm_ratio_to_floor = ||u_bc|| / (||u_floor|| + 1e-30)
active_update_energy_fraction = sum_{r > gamma_prime}(u_bc^2) / (sum(u_bc^2) + 1e-30)
```

所有 dot/norm/平方和先跨全模型以 float64 归约，再计算 ratio。CSV 写入前检查
所有指标有限；完整的 clean gradient、`m_hat`、`r`、`q` 不会写入磁盘。只有被选中
的 quantile step 才创建当前 step 的 CPU float64 `q/gamma` 临时 buffer；最终 step
若未命中 interval，则从当前 optimizer state 额外精确计算一次。

隐私注意：`x = parameter.summed_grad / expected_batch_size` 是加入 DP noise
之前的 clipped gradient 信息。因此 `x2_mean` 和 `bc_clean_nrmse` 属于
research-only clean-gradient diagnostics，不是 DP post-processing。若训练数据
是真实私有数据，不应在不额外做隐私处理的情况下公开；QNLI 是公开 benchmark，
因此本实验仅用于机制研究。其余基于 privatized optimizer state / DP gradients
的 fraction、quantile、floor-update 和 active-energy 指标才属于 DP-safe
post-processing diagnostics。

## 文件结构与输出

```text
exp2/
  README.md
  diagnostic_optimizer.py
  csv_writer.py
  run_exp2.py
  run_all.sh
  configs/                 # 5 个完整 run
  tests/                   # synthetic/unit tests
  results/                 # CSV、metadata 与 smoke 结果
```

每个 run 至少写入 `bc_diagnostics.csv`、`q_quantiles.csv` 和
`validation_metrics.csv`；validation 字段为
`epoch,global_step,val_loss,val_accuracy,epsilon_spent`。正式训练自然完成全部
10 个 epochs（`max_steps=null`），每个 epoch 结束做一次 validation，最后一个
epoch 的 row 即 final validation。中间 physical batch 和空 Poisson batch 都不会
写 diagnostic row。

## 固定配置与 gamma sweep

所有配置固定 `seed=0`、`epochs=10`、`max_steps=null`、`epsilon=7`、
`delta=1e-5`、`logical_batch_size=256`、`max_physical_batch_size=256`、
`max_length=128`、`max_grad_norm=0.1`、GDP、Ghost Clipping、Poisson sampling
和 mean loss reduction。

```text
A baseline: gamma_prime=3e-9,  lr=3e-3
B fixed LR: gamma_prime=3e-8,  lr=3e-3
C fixed LR: gamma_prime=3e-10, lr=3e-3
D compensated: gamma_prime=3e-8,  lr=9.4868329805e-3
E compensated: gamma_prime=3e-10, lr=9.4868329805e-4
```

## 测试与真实 smoke test

单元测试（不含真实 integration test）：

```bash
conda activate curve
python -m pytest exp2/tests -q -m 'not integration'
```

普通真实 smoke 会加载 QNLI、构建 BERT，随后才加载 Opacus/SciPy/privacy，并实际
执行 Opacus Ghost + `DPAdamBC` + diagnostic CSV。它只运行 3 个 logical steps：

```bash
conda activate curve
python exp2/run_exp2.py \
  --config exp2/configs/bc_g3e-9_lr3e-3.yaml \
  --output-dir exp2/results/smoke \
  --device cpu \
  --max-steps 3 \
  --max-train-samples 1024 \
  --max-eval-samples 256
```

强制 BatchMemoryManager split 的真实 integration smoke 使用相同数据和步数，
但增加 `--max-physical-batch-size 64`：

```bash
conda activate curve
python exp2/run_exp2.py \
  --config exp2/configs/bc_g3e-9_lr3e-3.yaml \
  --output-dir exp2/results/smoke_bmm64 \
  --device cpu \
  --max-steps 3 \
  --max-train-samples 1024 \
  --max-eval-samples 256 \
  --max-physical-batch-size 64
```

`bc_diagnostics.csv` 只包含 10 个核心字段：

```text
global_step,phi,x2_mean,bc_clean_nrmse,negative_fraction,clamp_fraction,active_fraction,update_cosine_to_floor,update_norm_ratio_to_floor,active_update_energy_fraction
```

`q_quantiles.csv` 只在每 20 个真实 logical steps 和最终 logical step 写一行：

```text
global_step,q_over_gamma_mean,q_over_gamma_p50,q_over_gamma_p90,q_over_gamma_p99
```

实际验证结果：单元测试 `14 passed, 1 deselected`；普通 smoke exit code 0，生成
3 个 diagnostic rows、1 个 final quantile row 和 1 个 final validation row，
`epsilon_spent=1.5887756650888445`。BMM split integration 为 `1 passed`，
实际产生 13 个 physical batches，但仍只生成 global steps 1、2、3 各一行；所有
CSV 数值有限，没有 duplicate logical rows，且
`clamp_fraction + active_fraction = 1`。

普通 smoke 的 CSV 示例：

```text
# exp2/results/smoke/bc_diagnostics.csv
global_step,phi,x2_mean,bc_clean_nrmse,negative_fraction,clamp_fraction,active_fraction,update_cosine_to_floor,update_norm_ratio_to_floor,active_update_energy_fraction
1,2.4028486222960055e-07,3.692989034643907e-12,2131.6013725106955,0.6826101086952842,0.6855981287504873,0.3144018712495127,0.6162017805272697,0.4665333881249921,0.07331972383599335
```

```text
# exp2/results/smoke/q_quantiles.csv
global_step,q_over_gamma_mean,q_over_gamma_p50,q_over_gamma_p90,q_over_gamma_p99
3,25.32398219402495,1.0000000087255028,86.84817961087295,223.02572385039375

# exp2/results/smoke/validation_metrics.csv
epoch,global_step,val_loss,val_accuracy,epsilon_spent
1,3,13.464903831481934,0.484375,1.5887756650888445
```

## 完整实验启动命令（实现/验证阶段未执行）

```bash
conda activate curve

# baseline / A
python exp2/run_exp2.py --config exp2/configs/bc_g3e-9_lr3e-3.yaml --output-dir exp2/results/g3e-9_lr3e-3

# fixed LR / B
python exp2/run_exp2.py --config exp2/configs/bc_g3e-8_lr3e-3.yaml --output-dir exp2/results/g3e-8_lr3e-3

# fixed LR / C
python exp2/run_exp2.py --config exp2/configs/bc_g3e-10_lr3e-3.yaml --output-dir exp2/results/g3e-10_lr3e-3

# compensated / D
python exp2/run_exp2.py --config exp2/configs/bc_g3e-8_lr9.4868e-3.yaml --output-dir exp2/results/g3e-8_lr9.4868e-3

# compensated / E
python exp2/run_exp2.py --config exp2/configs/bc_g3e-10_lr9.4868e-4.yaml --output-dir exp2/results/g3e-10_lr9.4868e-4
```

等价地，可在激活 `curve` 后执行 `./exp2/run_all.sh`。该脚本只包含上面五个
自然完成 10 epochs 的正式命令，验证阶段没有执行它们。
