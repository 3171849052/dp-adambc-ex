# dp-adambc-ex

本仓库提供两套 PyTorch + Opacus 的 GLUE/QNLI 差分隐私实验：

- `roberta_qnli`：`FacebookAI/roberta-base` 的既有 prompt/MLM yes-no verbalizer 实验。
- `bert_qnli`：`bert-base-cased` 标准 sequence-pair classification。冻结 encoder
  0-10，仅训练并按官方 DP-AdamBC `train_from_scratch` 逻辑重新初始化 encoder
  11、pooler 和 classifier。

两套实验都保留本项目的 IID Gaussian DP 机制：`delta=1e-5`、GDP accountant、
Opacus Ghost Clipping、global flat clipping、Poisson sampling，以及
`make_private_with_epsilon()`。训练用 logical batch 和
`BatchMemoryManager` 拆分 physical batch；每个 logical batch 只产生一次噪声和
一次真正的 Adam 更新。支持的算法仅为 `dpadam`、`dpadambc` 和
`fpcdpadam`。

## 环境

建议使用 Python 3.11 的独立 Conda 环境 `adamex`。CUDA 12.8 版 PyTorch 从官方
wheel 源安装，其余依赖使用清华 PyPI 镜像：

```bash
conda create -n adamex python=3.11
conda activate adamex
python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## 运行

统一入口根据 YAML 中的 `model.name` 自动选择实现，不需要模型专用脚本：

```bash
./run.sh config/qnli_roberta_base.yaml
./run.sh config/qnli_roberta_base_dpadambc.yaml
./run.sh config/qnli_roberta_base_fpcdpadam.yaml

./run.sh config/qnli_bert_base.yaml
./run.sh config/qnli_bert_base_dpadambc.yaml
./run.sh config/qnli_bert_base_fpcdpadam.yaml
./run.sh config/qnli_bert_base_smoke.yaml
```

`runtime.gpu` 选择物理 GPU；启动器将它映射为训练进程内的 `cuda:0`，并在
tmux 中后台运行。每次运行在 `outputs/` 下写入配置快照、resolved config、
`metrics.csv`、`summary.json` 和 `train.log`，不保存 checkpoint。

## 测试

```bash
conda activate adamex
pytest -q
./run.sh config/qnli_bert_base_smoke.yaml
```

BERT 的 `max_length`、batch、学习率、隐私预算及 FPC 参数全部从对应 YAML
读取。DP-AdamBC 继续使用本项目的
`phi = (noise_multiplier * max_grad_norm / expected_batch_size)^2` 与
`v_hat - phi` 修正，不复制官方仓库旧版 AdamCorr 或其 privacy accounting。
