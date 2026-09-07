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
`active_fraction + clamp_fraction = 1`。`q_over_gamma` 的 mean/p50/p90/p99
在所有 trainable parameter elements 拼接后的 CPU float64 tensor 上直接计算，
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
所有指标有限；完整的 clean gradient、`m_hat`、`r`、`q` 不会写入磁盘。临时
quantile buffer 只保留当前 logical step 的 CPU float64 `q/gamma` 元素。

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

每个 run 至少写入 `bc_diagnostics.csv` 和 `validation_metrics.csv`；后者字段为
`global_step,val_loss,val_accuracy`。完整 run 在 logical steps 410、820、1230
做验证；这些是精确 logical step，而非 epoch boundary。中间 physical batch 和
空 Poisson batch 都不会写 diagnostic row。

## 固定配置与 gamma sweep

所有配置固定 `seed=0`、`epochs=10`、`max_steps=1230`、`epsilon=7`、
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

```bash
conda activate curve
python -m pytest exp2/tests -q
```

真实 smoke 会加载 QNLI、构建 BERT，随后才加载 Opacus/SciPy/privacy，并实际
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

smoke 不到默认 validation targets，因此 `validation_metrics.csv` 只有表头；
`bc_diagnostics.csv` 应包含 3 行。表头为：

```text
global_step,phi,x2_mean,bc_clean_nrmse,negative_fraction,clamp_fraction,active_fraction,q_over_gamma_mean,q_over_gamma_p50,q_over_gamma_p90,q_over_gamma_p99,update_cosine_to_floor,update_norm_ratio_to_floor,active_update_energy_fraction
```

实际验证结果：`python -m pytest exp2/tests -q` 为 `11 passed in 4.29s`；真实
smoke exit code 为 0，走通 QNLI、BERT、Opacus Ghost、standalone DPAdamBC 和
diagnostic CSV，没有 SIGSEGV。smoke 生成 3 个 logical-step rows（1、2、3），
所有字段均有限，且每行 `clamp_fraction + active_fraction = 1`。实际 CSV 前几行：

```text
global_step,phi,x2_mean,bc_clean_nrmse,negative_fraction,clamp_fraction,active_fraction,q_over_gamma_mean,q_over_gamma_p50,q_over_gamma_p90,q_over_gamma_p99,update_cosine_to_floor,update_norm_ratio_to_floor,active_update_energy_fraction
1,2.4028486222960055e-07,3.693014543961873e-12,2131.584992515027,0.6826101086952842,0.6855981287504873,0.3144018712495127,39.445933138755684,1.0,136.6135506941646,451.3225045836111,0.6162017803858539,0.46653338794106575,0.07331972343891976
2,2.4028486222960055e-07,4.668890923599596e-12,1519.1481513333154,0.6320071270814773,0.6366192352554075,0.3633807647445925,30.09740357215645,1.0,104.36952303886457,288.91720451914216,0.6840490816445963,0.5393819253081841,0.07588966450809781
3,2.4028486222960055e-07,4.0620628186537316e-12,1406.3793052772894,0.608353487407946,0.6140921577885006,0.38590784221149943,25.32398219813319,1.0,86.84817771609232,223.02572385039375,0.7186696907317853,0.5760516922256118,0.07909099573025825
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
1230-step 命令，验证阶段没有执行它们。
