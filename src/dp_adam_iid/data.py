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


def _qnli_prompt_parts(
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    sentence: str,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Encode the four textual spans used by BEDB's QNLI template."""

    if (
        tokenizer.cls_token_id is None
        or tokenizer.mask_token_id is None
        or tokenizer.sep_token_id is None
    ):
        raise ValueError(
            "The QNLI prompt model requires cls, mask, and sep token ids"
        )

    question_ids = tokenizer.encode(
        question[:-1],
        add_special_tokens=False,
    )
    question_mark_ids = tokenizer.encode("?", add_special_tokens=False)
    comma_ids = tokenizer.encode(",", add_special_tokens=False)
    sentence_lower = sentence[:1].lower() + sentence[1:]
    sentence_ids = tokenizer.encode(
        " " + sentence_lower,
        add_special_tokens=False,
    )
    return question_ids, question_mark_ids, comma_ids, sentence_ids


def build_qnli_input_ids(
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    sentence: str,
) -> list[int]:
    """Construct BEDB's QNLI template directly as token IDs.

    The template is ``*cls**sent-_0*?*mask*,*+sentl_1**sep+*``.  Each
    textual fragment is encoded independently so this remains equivalent to
    BEDB's template parser, including the literal leading space before
    ``+sentl_1``.
    """

    question_ids, question_mark_ids, comma_ids, sentence_ids = (
        _qnli_prompt_parts(tokenizer, question, sentence)
    )

    return (
        [tokenizer.cls_token_id]
        + question_ids
        + question_mark_ids
        + [tokenizer.mask_token_id]
        + comma_ids
        + sentence_ids
        + [tokenizer.sep_token_id]
    )


def _truncate_qnli_input_ids(
    tokenizer: PreTrainedTokenizerBase,
    question_ids: list[int],
    question_mark_ids: list[int],
    comma_ids: list[int],
    sentence_ids: list[int],
    *,
    max_length: int,
) -> list[int]:
    """Truncate only QNLI text fragments while retaining template structure."""

    # The fixed template pieces are everything except the question and
    # sentence spans.  They must never be removed by truncation.
    fixed_length = (
        1
        + len(question_mark_ids)
        + 1  # mask
        + len(comma_ids)
        + 1  # sep
    )
    available_text_length = max_length - fixed_length
    if available_text_length < 0:
        raise ValueError(
            "max_length is too small to retain the QNLI prompt structure"
        )

    # Preserve the question first; use any remaining budget for the sentence.
    # Only these two spans may be shortened.
    retained_question_length = min(len(question_ids), available_text_length)
    retained_question = question_ids[:retained_question_length]
    remaining_text_length = available_text_length - retained_question_length
    retained_sentence = sentence_ids[:remaining_text_length]

    return (
        [tokenizer.cls_token_id]
        + retained_question
        + question_mark_ids
        + [tokenizer.mask_token_id]
        + comma_ids
        + retained_sentence
        + [tokenizer.sep_token_id]
    )


def tokenize_qnli_batch(
    tokenizer: PreTrainedTokenizerBase,
    questions: list[str],
    sentences: list[str],
    *,
    max_length: int,
):
    """Tokenize QNLI prompts and record the sole mask position."""

    if (
        tokenizer.cls_token_id is None
        or tokenizer.mask_token_id is None
        or tokenizer.sep_token_id is None
    ):
        raise ValueError(
            "The QNLI prompt model requires a tokenizer with cls, mask, and sep tokens"
        )

    batch_input_ids = []
    for question, sentence in zip(questions, sentences):
        question_ids, question_mark_ids, comma_ids, sentence_ids = (
            _qnli_prompt_parts(tokenizer, question, sentence)
        )
        input_ids = _truncate_qnli_input_ids(
            tokenizer,
            question_ids,
            question_mark_ids,
            comma_ids,
            sentence_ids,
            max_length=max_length,
        )
        batch_input_ids.append(input_ids)

    mask_positions = []
    for input_ids in batch_input_ids:
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
    return {
        "input_ids": batch_input_ids,
        "attention_mask": [[1] * len(input_ids) for input_ids in batch_input_ids],
        "mask_pos": mask_positions,
    }


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
