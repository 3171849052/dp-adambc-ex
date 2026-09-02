"""QNLI loading and tokenization."""

from __future__ import annotations

from dataclasses import dataclass

from datasets import DatasetDict, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, PreTrainedTokenizerBase

from .config import Config


@dataclass
class QNLIData:
    tokenizer: PreTrainedTokenizerBase
    train_loader: DataLoader
    eval_loader: DataLoader
    train_size: int
    eval_size: int


def _limit_dataset(dataset, limit: int | None):
    if limit is None:
        return dataset
    return dataset.select(range(min(limit, len(dataset))))


def load_qnli(config: Config) -> QNLIData:
    """Download/cache GLUE QNLI and return tokenized PyTorch data loaders."""

    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    # datasets>=4 no longer resolves the legacy unnamespaced ``glue`` dataset
    # script. ``nyu-mll/glue`` is the current Hub repository containing the same
    # GLUE subsets; the public experiment configuration intentionally remains
    # expressed as dataset_name=glue, dataset_config=qnli.
    dataset_id = "nyu-mll/glue" if config.data.dataset_name == "glue" else config.data.dataset_name
    raw: DatasetDict = load_dataset(dataset_id, config.data.dataset_config)

    def tokenize_batch(batch):
        return tokenizer(
            batch["question"],
            batch["sentence"],
            truncation=True,
            max_length=config.data.max_length,
        )

    # Restrict before tokenization so the smoke configuration really processes
    # only its small subset while retaining the same QNLI source and schema.
    train = _limit_dataset(raw[config.data.train_split], config.data.max_train_samples)
    valid = _limit_dataset(raw[config.data.eval_split], config.data.max_eval_samples)
    train = train.map(tokenize_batch, batched=True, desc="Tokenizing QNLI train")
    valid = valid.map(tokenize_batch, batched=True, desc="Tokenizing QNLI validation")

    columns_to_remove = ["question", "sentence", "idx"]
    for dataset_name in ("train", "valid"):
        dataset = train if dataset_name == "train" else valid
        remove = [column for column in columns_to_remove if column in dataset.column_names]
        if remove:
            dataset = dataset.remove_columns(remove)
        if "label" in dataset.column_names:
            dataset = dataset.rename_column("label", "labels")
        if dataset_name == "train":
            train = dataset
        else:
            valid = dataset

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )
    loader_kwargs = {
        "collate_fn": collator,
        "num_workers": config.runtime.num_workers,
        "pin_memory": config.runtime.pin_memory,
    }
    train_loader = DataLoader(
        train,
        batch_size=config.data.logical_batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    eval_loader = DataLoader(
        valid,
        batch_size=config.data.eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return QNLIData(
        tokenizer=tokenizer,
        train_loader=train_loader,
        eval_loader=eval_loader,
        train_size=len(train),
        eval_size=len(valid),
    )
