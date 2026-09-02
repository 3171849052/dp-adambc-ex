"""QNLI loading and tokenization."""

from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding, PreTrainedTokenizerBase

from .config import Config


QNLI_VERBALIZER = {
    0: " yes",
    1: " no",
}


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


def qnli_verbalizer_token_ids(tokenizer: PreTrainedTokenizerBase) -> tuple[int, int]:
    """Return BEDB's QNLI label-token ids in Hugging Face label order."""

    token_ids: list[int] = []
    for label in (0, 1):
        label_word = QNLI_VERBALIZER[label]
        encoded = tokenizer.encode(label_word, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"QNLI verbalizer {label_word!r} must encode to exactly one token; "
                f"got {encoded}"
            )
        token_ids.append(int(encoded[0]))
    return token_ids[0], token_ids[1]


def build_qnli_prompt(question: str, sentence: str, mask_token: str) -> str:
    """Construct the QNLI-specific BEDB text-infilling prompt."""

    question = question.strip()
    if question.endswith("?"):
        question = question[:-1].rstrip()
    return f"{question}? {mask_token}, {sentence.strip()}"


def tokenize_qnli_batch(
    tokenizer: PreTrainedTokenizerBase,
    questions: list[str],
    sentences: list[str],
    *,
    max_length: int,
):
    """Tokenize QNLI prompts and record the sole mask position."""

    if tokenizer.mask_token is None or tokenizer.mask_token_id is None:
        raise ValueError("The QNLI prompt model requires a tokenizer with a mask token")
    prompts = [
        build_qnli_prompt(question, sentence, tokenizer.mask_token)
        for question, sentence in zip(questions, sentences)
    ]
    encoded = tokenizer(
        prompts,
        truncation=True,
        max_length=max_length,
    )
    mask_positions = []
    for input_ids in encoded["input_ids"]:
        positions = [
            index
            for index, token_id in enumerate(input_ids)
            if token_id == tokenizer.mask_token_id
        ]
        if len(positions) != 1:
            raise ValueError(
                "Every tokenized QNLI prompt must contain exactly one <mask> token; "
                f"got {len(positions)}"
            )
        mask_positions.append(positions[0])
    encoded["mask_pos"] = mask_positions
    return encoded


def load_qnli(config: Config) -> QNLIData:
    """Download/cache GLUE QNLI and return tokenized PyTorch data loaders."""

    from datasets import DatasetDict, load_dataset

    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    qnli_verbalizer_token_ids(tokenizer)
    # datasets>=4 no longer resolves the legacy unnamespaced ``glue`` dataset
    # script. ``nyu-mll/glue`` is the current Hub repository containing the same
    # GLUE subsets; the public experiment configuration intentionally remains
    # expressed as dataset_name=glue, dataset_config=qnli.
    dataset_id = "nyu-mll/glue" if config.data.dataset_name == "glue" else config.data.dataset_name
    raw: DatasetDict = load_dataset(dataset_id, config.data.dataset_config)

    def tokenize_batch(batch):
        return tokenize_qnli_batch(
            tokenizer,
            batch["question"],
            batch["sentence"],
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
