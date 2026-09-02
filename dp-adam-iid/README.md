# dp-adam-iid

一个简洁、可复现的 PyTorch + Opacus 差分隐私 QNLI 实验。模型是
`FacebookAI/roberta-base`，数据使用 GLUE/QNLI，优化器是纯
`torch.optim.Adam`。

本项目不包含 JAX、`jax_privacy`、BandInvMF、Toeplitz、相关噪声或相关实现，
也不修改 `curve` 环境中的第三方包。

## 环境与运行

在仓库根目录使用现有的 `curve` conda 环境启动：

```bash
./run.sh
```

也可以显式指定配置：

```bash
./run.sh config/qnli_roberta_base.yaml
```

`run.sh` 只负责激活环境并启动 Python；配置解析、运行目录创建、训练和结果
写入都由 Python 完成。

当前环境已经提供项目依赖。若需要将项目安装为 editable package，可在不解析
依赖的情况下执行：

```bash
python -m pip install -e . --no-deps
```

## 训练机制

每个 logical batch 内，Opacus Ghost Clipping 先对每个样本求梯度并计算整个
模型梯度的 global L2 norm，然后执行 per-sample global flat clipping。裁剪后
聚合一个 logical batch 的梯度；在该 logical optimizer step 中加入一次 IID
Gaussian noise；再把 private gradient 交给 Adam。因此每个 logical step 只有
一次真正的 Adam 参数更新。

- `logical_batch_size`：隐私会计和优化器更新所对应的 batch 大小；默认 1024。
- `max_physical_batch_size`：显存中一次 forward/backward 的上限；默认 8，
  由 `BatchMemoryManager` 拆分 logical batch，支持 RTX 3080。
  如需较小的 logical batch，可把配置中的 `logical_batch_size` 改为 512，
  并独立调整这个 physical 上限。
- Ghost Clipping：Opacus 官方的 per-sample gradient clipping backend，不是
  项目自行实现。
- GDP accountant：使用 `PrivacyEngine(accountant="gdp")`。
- IID Gaussian noise：每个 logical optimizer step 由 Opacus 向聚合裁剪梯度
  添加一次独立同分布 Gaussian noise。
- fixed learning rate：Adam 的 learning rate 全程固定为 `1e-4`，无 warmup、
  scheduler、decay 或 weight decay。

训练使用 Poisson sampling，并通过
`make_private_with_epsilon()` 根据 `epsilon`、`delta`、`epochs` 和采样率自动
求 `noise_multiplier`。默认配置为：`epsilon=3.0`、`delta=1e-5`、
`max_grad_norm=1.0`、`epochs=3`、`max_length=128`。训练过程中打印 epoch、step、
loss、accuracy、当前 epsilon 和 noise multiplier。stdout 和 stderr 会同时显示在
终端并写入当前运行目录的 `train.log`。

## 输出目录与文件

每次运行都会在 `outputs/` 下创建唯一目录，目录名为：

```text
YYYYMMDD-HHMMSS_{algorithm}_eps{epsilon}_d{delta}_ep{epochs}_lb{logical_batch_size}_lr{learning_rate}_C{max_grad_norm}_s{seed}
```

例如：

```text
outputs/20260902-180500_dpadam_eps3_d1e-5_ep3_lb1024_lr1e-4_C1_s0/
```

数值采用统一格式（例如 `3.0` 写成 `3`、`1.0` 写成 `1`）；不使用配置 hash。
如果同一秒创建目录发生冲突，时间戳秒数会递增。配置中的算法通过顶层
`algorithm` 字段声明，目录管理逻辑与算法无关。

每个运行目录包含：

- `config.yaml`：原始 YAML 配置快照；
- `resolved_config.yaml`：完整配置、实际 device、数据集大小、sample rate、
  noise multiplier 等运行时派生参数；
- `metrics.csv`：训练和验证指标；
- `summary.json`：最终结果和关键参数；
- `train.log`：完整终端日志。

当前 `curve` 中的 Opacus 1.6.0 在 GDP 校准二分搜索的一个中间 sigma 上存在
数值 bracket 边界问题；项目只临时扩大 Opacus 已有 GDP 根求解区间，公式、
Ghost Clipping、噪声生成和 accountant 仍全部来自 Opacus 官方实现。

第一版使用 FP32，不使用 AMP、LoRA、adaptive/per-layer clipping 或 Hugging Face
Trainer。

## Smoke test

下面的配置仍使用真实 RoBERTa 和 QNLI，但只取少量样本和两个 logical steps，
会覆盖数据加载、forward/backward、Ghost Clipping、GDP、IID DP step、评估和
结果文件写入：

```bash
./run.sh config/qnli_roberta_base_smoke.yaml
```

单元测试：

```bash
pytest -q
```
