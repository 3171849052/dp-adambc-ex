from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest
import torch
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig, BertForSequenceClassification

import bert_qnli.data as bert_data
from bert_qnli.config import Config, load_config
from bert_qnli.data import tokenize_qnli_batch
from bert_qnli.model import prepare_bert_for_qnli
from bert_qnli.optim import DPAdamBC, FPCDPAdam
from bert_qnli.privacy import cleanup_private_hooks, make_private_training
from bert_qnli.trainer import train_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _tiny_bert(*, layers: int = 1) -> BertForSequenceClassification:
    return BertForSequenceClassification(
        BertConfig(
            vocab_size=64,
            hidden_size=8,
            num_hidden_layers=layers,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=32,
            type_vocab_size=2,
            pad_token_id=0,
            num_labels=2,
        )
    )


def test_bert_qnli_uses_standard_question_sentence_pair_tokenization():
    class RecordingPairTokenizer:
        def __init__(self):
            self.call = None

        def __call__(self, questions, sentences, **kwargs):
            self.call = (questions, sentences, kwargs)
            return {
                "input_ids": [[101, 10, 102, 20, 102]],
                "attention_mask": [[1, 1, 1, 1, 1]],
                "token_type_ids": [[0, 0, 0, 1, 1]],
            }

    tokenizer = RecordingPairTokenizer()
    questions = ["What?"]
    sentences = ["Answer."]
    encoded = tokenize_qnli_batch(
        tokenizer,
        questions,
        sentences,
        max_length=16,
    )

    assert tokenizer.call == (
        questions,
        sentences,
        {"max_length": 16, "truncation": True, "padding": False},
    )
    assert encoded["token_type_ids"][0] == [0, 0, 0, 1, 1]
    assert "mask_pos" not in encoded


def test_bert_pipeline_has_no_prompt_mask_or_verbalizer_state():
    model = _tiny_bert()
    assert isinstance(model, BertForSequenceClassification)
    assert not hasattr(bert_data, "QNLI_VERBALIZER")
    assert not hasattr(bert_data, "qnli_verbalizer_token_ids")
    assert not hasattr(model, "mask_pos")
    assert not hasattr(model, "yes_token_id")
    assert not hasattr(model, "no_token_id")
    assert hasattr(model, "classifier")
    assert not hasattr(model, "lm_head")


def test_bert_freezes_0_through_10_and_reinitializes_only_trainable_blocks():
    model = _tiny_bert(layers=12)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.25)

    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.startswith("bert.embeddings.")
        or any(name.startswith(f"bert.encoder.layer.{index}.") for index in range(11))
    }

    torch.manual_seed(0)
    prepare_bert_for_qnli(model)

    for index in range(11):
        assert all(
            not parameter.requires_grad
            for parameter in model.bert.encoder.layer[index].parameters()
        )
    assert all(parameter.requires_grad for parameter in model.bert.encoder.layer[11].parameters())
    assert all(parameter.requires_grad for parameter in model.bert.pooler.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
    assert all(not parameter.requires_grad for parameter in model.bert.embeddings.parameters())

    current = dict(model.named_parameters())
    for name, expected in frozen_before.items():
        torch.testing.assert_close(current[name], expected)

    assert not torch.allclose(
        model.bert.encoder.layer[11].attention.self.query.weight,
        torch.full_like(model.bert.encoder.layer[11].attention.self.query.weight, 0.25),
    )
    assert not torch.allclose(
        model.bert.pooler.dense.weight,
        torch.full_like(model.bert.pooler.dense.weight, 0.25),
    )
    assert not torch.allclose(
        model.classifier.weight,
        torch.full_like(model.classifier.weight, 0.25),
    )
    torch.testing.assert_close(model.classifier.bias, torch.zeros_like(model.classifier.bias))


def test_bert_yaml_and_config_do_not_contain_roberta_prompt_fields():
    for name in (
        "qnli_bert_base.yaml",
        "qnli_bert_base_dpadambc.yaml",
        "qnli_bert_base_fpcdpadam.yaml",
        "qnli_bert_base_smoke.yaml",
    ):
        config = load_config(PROJECT_ROOT / "config" / name)
        assert config.model.name == "bert-base-cased"
        data_fields = asdict(config.data)
        assert "first_sent_limit" not in data_fields
        assert "other_sent_limit" not in data_fields
        assert "truncate_head" not in data_fields


class TinyDataset(Dataset):
    def __init__(self, size: int = 32):
        generator = torch.Generator().manual_seed(0)
        self.input_ids = torch.randint(1, 63, (size, 8), generator=generator)
        self.attention_mask = torch.ones_like(self.input_ids)
        self.token_type_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]]).repeat(size, 1)
        self.labels = torch.arange(size) % 2

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "token_type_ids": self.token_type_ids[index],
            "labels": self.labels[index],
        }


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(64, 8)
        self.classifier = nn.Linear(8, 2)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        del attention_mask, token_type_ids
        return self.classifier(self.embedding(input_ids).mean(dim=1))


def _private_config(algorithm: str) -> Config:
    optimizer = {
        "dpadam": "adam",
        "dpadambc": "dpadambc",
        "fpcdpadam": "fpcdpadam",
    }[algorithm]
    return Config.from_dict(
        {
            "algorithm": algorithm,
            "seed": 0,
            "model": {"name": "bert-base-cased", "num_labels": 2},
            "data": {
                "logical_batch_size": 8,
                "max_physical_batch_size": 2,
                "eval_batch_size": 8,
            },
            "training": {
                "epochs": 1,
                "max_steps": 2,
                "learning_rate": 1.0e-4,
                "optimizer": optimizer,
                "gamma_prime": 1.0e-4,
                "fpc_lambda": 0.5,
            },
            "privacy": {
                "epsilon": 3.0,
                "delta": 1.0e-5,
                "accountant": "gdp",
                "noise_type": "iid_gaussian",
                "max_grad_norm": 1.0,
                "clipping": "flat",
                "grad_sample_mode": "ghost",
                "poisson_sampling": True,
                "loss_reduction": "mean",
                "wrap_model": False,
            },
            "runtime": {"device": "cpu", "num_workers": 0, "pin_memory": False},
        }
    )


@pytest.mark.parametrize(
    ("algorithm", "optimizer_type"),
    [
        ("dpadam", torch.optim.Adam),
        ("dpadambc", DPAdamBC),
        ("fpcdpadam", FPCDPAdam),
    ],
)
def test_all_bert_optimizers_use_gdp_ghost_poisson_and_one_update_per_logical_batch(
    algorithm, optimizer_type
):
    torch.manual_seed(0)
    config = _private_config(algorithm)
    model = TinyClassifier()
    loader = DataLoader(
        TinyDataset(),
        batch_size=config.data.logical_batch_size,
        shuffle=False,
    )
    model.train()
    private = make_private_training(model, loader, config)
    underlying = private.optimizer.original_optimizer
    assert isinstance(underlying, optimizer_type)
    assert private.privacy_engine.accountant.mechanism() == "gdp"
    assert private.data_loader.sample_rate == pytest.approx(8 / 32)
    assert private.optimizer.noise_multiplier > 0
    assert "FastGradientClipping" in type(private.optimizer).__name__

    real_updates = 0
    original_step = underlying.step

    def counted_step(*args, **kwargs):
        nonlocal real_updates
        real_updates += 1
        return original_step(*args, **kwargs)

    underlying.step = counted_step
    physical_batches = 0
    logical_steps = 0
    try:
        with BatchMemoryManager(
            data_loader=private.data_loader,
            max_physical_batch_size=config.data.max_physical_batch_size,
            optimizer=private.optimizer,
        ) as memory_loader:
            for batch in memory_loader:
                physical_batches += 1
                if not batch["labels"].numel():
                    private.optimizer.zero_grad()
                    private.optimizer.step()
                    private.optimizer.zero_grad()
                    continue
                labels = batch.pop("labels")
                outputs = model(**batch)
                loss = private.criterion(outputs, labels)
                loss.backward()
                private.optimizer.step()
                private.optimizer.zero_grad()
                if not private.optimizer._is_last_step_skipped:
                    logical_steps += 1
    finally:
        cleanup_private_hooks(private.hooks)

    assert logical_steps == len(private.data_loader)
    assert physical_batches > logical_steps
    assert real_updates == logical_steps


def test_bert_smoke_completes_two_real_logical_dp_steps():
    torch.manual_seed(0)
    config = _private_config("dpadam")
    config.data.max_physical_batch_size = 2
    loader = DataLoader(
        TinyDataset(size=16),
        batch_size=config.data.logical_batch_size,
        shuffle=False,
    )
    model = prepare_bert_for_qnli(_tiny_bert(layers=1))

    result = train_model(
        config,
        model,
        loader,
        loader,
        torch.device("cpu"),
    )

    assert result.global_step == 2
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    assert trainable_parameters
    assert all(
        parameter.requires_grad
        for parameter in model.bert.encoder.layer[0].parameters()
    )
