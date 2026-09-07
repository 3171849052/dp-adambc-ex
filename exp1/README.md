# 实验1：FPC-DPAdam P0 机制诊断

本实验只针对 `bert-base-cased + QNLI + FPCDPAdam`，固定配置见
[`config_p0.yaml`](config_p0.yaml)。所有实验新增代码、测试与结果均在本目录。
真实参数更新由现有 `bert_qnli.optim.FPCDPAdam` 执行；[`DiagnosticFPCDPAdam`](diagnostic_optimizer.py)
只在父类更新前累计 predictor/x 统计，并在父类更新后读取未经 clamp 的
`v_fpc_hat`，维护仅用于诊断的 clean/shadow second moment。

其中：

- `x = parameter.summed_grad / expected_batch_size`，`y = parameter.grad`；
- shadow BC 使用 `beta2 * v_bc + (1-beta2) * y^2`，再减去 `phi`；
- clean/shadow state 不参与真实参数更新，也不写入磁盘；
- 所有模型元素先以 float64 累计 numerator/denominator，再计算 CSV ratio；
- `BatchMemoryManager` 的 `_is_last_step_skipped` 是唯一 logical-step CSV gate，
  所以中间 physical batch 不写行；
- NRMSE 在 clamp 前计算，update cosine/norm ratio 使用 clamp 后的
  `q_fpc/q_bc`；
- 诊断实现委托父类完成 FPC state 和参数更新，不复制或替换真实更新公式。

CSV 文件为 [`results/p0_diagnostics.csv`](results/p0_diagnostics.csv)，字段为：

```text
global_step,predictor_x_gap_mse,x2_mean,phi,predictor_gap_over_x2,predictor_gap_over_phi,fpc_expected_bias_ratio,v_clean_nrmse_fpc,v_clean_nrmse_bc,v_nrmse_ratio_fpc_over_bc,clamp_fraction_fpc,clamp_fraction_bc,clamp_fraction_delta,update_cosine_fpc_bc,update_norm_ratio_fpc_over_bc
```

## 验证记录

以下命令均在仓库根目录 `/media/data/data/lt/dp-adamwbc-ex`、`curve` Conda
环境中执行：

```bash
conda activate curve
python -m pytest exp1/tests -q
```

结果：`8 passed`。测试覆盖 lambda=1 与 DP-AdamBC 校正等价、clean/shadow
不改变 FPC 参数更新、跨参数 tensor 的全局 float64 累计、clamp 前 NRMSE、
logical-step gate、空 batch 不写行、CSV 有限值约束，以及 runner 的重型 import
延迟到 QNLI/BERT 初始化之后。

3-step synthetic mechanism smoke（不下载数据、不改变固定训练配置的诊断公式）：

```bash
conda activate curve
python exp1/run_exp1.py --synthetic-smoke --max-steps 3
```

结果：通过；`exp1/results/p0_diagnostics.csv` 有完整 15 列和 3 行 logical-step
数据，所有数值均为有限值。

另外尝试了真实 QNLI/Opacus 的 3-step 短 smoke（限制 train=1024、eval=256、CPU）：

```bash
conda activate curve
python -X faulthandler -u exp1/run_exp1.py --config exp1/config_p0.yaml --device cpu --max-steps 3 --max-train-samples 1024 --max-eval-samples 256 --output exp1/results/p0_diagnostics.real_smoke.csv
```

结果：通过；完成 3 个真实 logical steps，生成
`exp1/results/p0_diagnostics.real_smoke.csv`，包含完整 15 列和 3 行数据，
所有指标均为有限值。修复点是模仿 `scripts/train.py`：在 QNLI/BERT 完成后才
导入 Opacus/SciPy/`bert_qnli.privacy`，并在 Opacus 初始化前调用 `model.train()`。
完整的 `1230` logical-step 实验未运行。

## CSV 示例

synthetic smoke 生成的前 3 行如下：

```text
global_step,predictor_x_gap_mse,x2_mean,phi,predictor_gap_over_x2,predictor_gap_over_phi,fpc_expected_bias_ratio,v_clean_nrmse_fpc,v_clean_nrmse_bc,v_nrmse_ratio_fpc_over_bc,clamp_fraction_fpc,clamp_fraction_bc,clamp_fraction_delta,update_cosine_fpc_bc,update_norm_ratio_fpc_over_bc
1,4.291534423828125e-06,4.291534423828125e-06,1.5258789062500003e-07,1.0,28.124999999999993,-0.5,47.97371791020286,96.81912707118707,0.4954983520449403,0.0,0.0,0.0,1.0,1.4142135623730951
2,0.0003060493469238281,4.291534423828125e-06,1.5258789062500003e-07,71.31466666666667,2005.7249999999997,-35.657333333333334,74.60712213033567,99.34371863582346,0.7509998936503697,0.0,0.0,0.0,0.9999077382228063,1.145903223088279
3,0.0003341505415221661,4.291534423828125e-06,1.5258789062500003e-07,77.86271960603263,2189.8889889196676,-38.93135980301631,89.00873329809166,106.5469308388152,0.8353946246724325,0.0,0.0,0.0,0.999990443902437,1.0925535062559377
```

## 启动完整实验

唯一推荐命令如下。它从仓库根目录激活 `curve`，使用固定 P0 配置、
`epochs=10`、`max_steps=1230`，并将诊断 CSV 写入 `exp1/results/`：

```bash
cd /media/data/data/lt/dp-adamwbc-ex && source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate curve && python exp1/run_exp1.py --config exp1/config_p0.yaml --max-steps 1230 --output exp1/results/p0_diagnostics.csv
```
