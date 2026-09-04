import re

import pytest
import torch
from transformers import RobertaConfig, RobertaForMaskedLM

from roberta_qnli.data import (
    QNLI_VERBALIZER,
    build_qnli_input_ids,
    qnli_verbalizer_token_ids,
    tokenize_qnli_batch,
)
from roberta_qnli.model import RobertaPromptForQNLI


class LocalPromptTokenizer:
    cls_token_id = 0
    mask_token_id = 1
    sep_token_id = 2

    def __init__(self):
        self._token_ids = {}
        self._next_id = 3

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        tokens = re.findall(r"[A-Za-z]+|[^A-Za-z\s]", text)
        ids = []
        for token in tokens:
            if token not in self._token_ids:
                self._token_ids[token] = self._next_id
                self._next_id += 1
            ids.append(self._token_ids[token])
        return ids


@pytest.fixture
def tokenizer():
    return LocalPromptTokenizer()


def test_qnli_bedb_verbalizer_uses_single_roberta_tokens_in_label_order(tokenizer):

    assert QNLI_VERBALIZER[0] == " yes"
    assert QNLI_VERBALIZER[1] == " no"
    yes_token_id, no_token_id = qnli_verbalizer_token_ids(tokenizer)
    assert [yes_token_id] == tokenizer.encode(" yes", add_special_tokens=False)
    assert [no_token_id] == tokenizer.encode(" no", add_special_tokens=False)


def test_qnli_prompt_matches_bedb_token_ids_exactly(tokenizer):

    actual_input_ids = build_qnli_input_ids(
        tokenizer,
        "Who wrote the book?",
        "Ada wrote the book.",
    )
    expected_input_ids = (
        [tokenizer.cls_token_id]
        + tokenizer.encode(
            "Who wrote the book",
            add_special_tokens=False,
        )
        + tokenizer.encode(
            "?",
            add_special_tokens=False,
        )
        + [tokenizer.mask_token_id]
        + tokenizer.encode(
            ",",
            add_special_tokens=False,
        )
        + tokenizer.encode(
            " ada wrote the book.",
            add_special_tokens=False,
        )
        + [tokenizer.sep_token_id]
    )

    assert actual_input_ids == expected_input_ids
    encoded = tokenize_qnli_batch(
        tokenizer,
        ["Who wrote the book?"],
        ["Ada wrote the book."],
        max_length=64,
    )
    assert encoded["input_ids"][0] == expected_input_ids


def test_qnli_template_always_drops_last_question_char_and_lowercases_sentence(tokenizer):

    actual_input_ids = build_qnli_input_ids(
        tokenizer,
        "Where is Chengdu",
        "Chengdu is in Sichuan.",
    )
    expected_input_ids = (
        [tokenizer.cls_token_id]
        + tokenizer.encode("Where is Chengd", add_special_tokens=False)
        + tokenizer.encode("?", add_special_tokens=False)
        + [tokenizer.mask_token_id]
        + tokenizer.encode(",", add_special_tokens=False)
        + tokenizer.encode(
            " chengdu is in Sichuan.",
            add_special_tokens=False,
        )
        + [tokenizer.sep_token_id]
    )

    assert actual_input_ids == expected_input_ids


def test_qnli_sentence_limits_are_applied_before_template_truncation(tokenizer):
    question = "Who wrote the very long book about differential privacy?"
    sentence = "Ada wrote the very long book about differential privacy."
    first_sent_limit = 3
    other_sent_limit = 4

    encoded = tokenize_qnli_batch(
        tokenizer,
        [question],
        [sentence],
        max_length=128,
        first_sent_limit=first_sent_limit,
        other_sent_limit=other_sent_limit,
        truncate_head=True,
    )
    expected_input_ids = (
        [tokenizer.cls_token_id]
        + tokenizer.encode(question[:-1], add_special_tokens=False)[
            :first_sent_limit
        ]
        + tokenizer.encode("?", add_special_tokens=False)
        + [tokenizer.mask_token_id]
        + tokenizer.encode(",", add_special_tokens=False)
        + tokenizer.encode(" ada wrote the very long book about differential privacy.", add_special_tokens=False)[
            :other_sent_limit
        ]
        + [tokenizer.sep_token_id]
    )

    assert encoded["input_ids"][0] == expected_input_ids


def test_truncate_head_matches_bedb_after_sentence_limits(tokenizer):
    question = "Who wrote the very long book about differential privacy?"
    sentence = "Ada wrote the very long book about differential privacy."
    first_sent_limit = 8
    other_sent_limit = 8
    max_length = 12

    question_ids = tokenizer.encode(question[:-1], add_special_tokens=False)[
        :first_sent_limit
    ]
    sentence_ids = tokenizer.encode(
        " ada wrote the very long book about differential privacy.",
        add_special_tokens=False,
    )[:other_sent_limit]
    full_prompt_after_sent_limits = (
        [tokenizer.cls_token_id]
        + question_ids
        + tokenizer.encode("?", add_special_tokens=False)
        + [tokenizer.mask_token_id]
        + tokenizer.encode(",", add_special_tokens=False)
        + sentence_ids
        + [tokenizer.sep_token_id]
    )

    encoded = tokenize_qnli_batch(
        tokenizer,
        [question],
        [sentence],
        max_length=max_length,
        first_sent_limit=first_sent_limit,
        other_sent_limit=other_sent_limit,
        truncate_head=True,
    )

    assert len(full_prompt_after_sent_limits) > max_length
    assert encoded["input_ids"][0] == full_prompt_after_sent_limits[-max_length:]
    input_ids = encoded["input_ids"][0]
    assert input_ids.count(tokenizer.mask_token_id) == 1
    assert encoded["mask_pos"][0] == input_ids.index(tokenizer.mask_token_id)


def test_truncation_reports_when_bedb_removes_the_mask(tokenizer):

    with pytest.raises(ValueError, match="mask"):
        tokenize_qnli_batch(
            tokenizer,
            ["Who wrote the very long book about differential privacy?"],
            ["Ada wrote the very long book about differential privacy."],
            max_length=5,
            first_sent_limit=8,
            other_sent_limit=8,
            truncate_head=False,
        )


def test_prompt_has_one_mask_and_records_its_actual_position(tokenizer):
    questions = ["Who wrote the book?", "Where is Chengdu"]
    sentences = ["Ada wrote the book.", "Chengdu is in Sichuan."]

    encoded = tokenize_qnli_batch(
        tokenizer, questions, sentences, max_length=64
    )

    for input_ids, mask_pos in zip(encoded["input_ids"], encoded["mask_pos"]):
        assert input_ids.count(tokenizer.mask_token_id) == 1
        assert input_ids[mask_pos] == tokenizer.mask_token_id
        assert mask_pos == input_ids.index(tokenizer.mask_token_id)


def test_truncation_preserves_qnli_template_structure(tokenizer):
    encoded = tokenize_qnli_batch(
        tokenizer,
        ["Who wrote the very long book about differential privacy?"],
        ["Ada wrote the very long book about differential privacy."],
        max_length=12,
        first_sent_limit=8,
        other_sent_limit=8,
        truncate_head=True,
    )

    input_ids = encoded["input_ids"][0]
    assert len(input_ids) == 12
    assert input_ids[-1] == tokenizer.sep_token_id
    assert input_ids.count(tokenizer.mask_token_id) == 1
    assert encoded["mask_pos"][0] == input_ids.index(tokenizer.mask_token_id)


def test_prompt_model_returns_yes_no_logits_without_classification_head():
    masked_lm = RobertaForMaskedLM(
        RobertaConfig(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            max_position_embeddings=32,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
        )
    )
    model = RobertaPromptForQNLI(masked_lm, yes_token_id=5, no_token_id=6)
    model.eval()
    input_ids = torch.tensor([[0, 8, 4, 9, 2], [0, 10, 4, 11, 2]])
    attention_mask = torch.ones_like(input_ids)
    mask_pos = torch.tensor([2, 2])

    with torch.no_grad():
        hidden = model.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state
        vocab_logits = model.lm_head(hidden[torch.arange(2), mask_pos])
        class_logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mask_pos=mask_pos,
        )

    assert class_logits.shape == (2, 2)
    torch.testing.assert_close(class_logits[:, 0], vocab_logits[:, 5])
    torch.testing.assert_close(class_logits[:, 1], vocab_logits[:, 6])
    assert model.lm_head is masked_lm.lm_head
    assert (
        model.lm_head.decoder.weight
        is model.roberta.embeddings.word_embeddings.weight
    )
    assert all(parameter.requires_grad for parameter in model.parameters())
    assert not hasattr(model, "classifier")
    assert all("classifier" not in name for name, _ in model.named_modules())
