"""Prompt-based RoBERTa QNLI model construction."""

from __future__ import annotations

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase, RobertaForMaskedLM

from .config import Config
from .data import qnli_verbalizer_token_ids


class _TiedEmbeddingForGhost(nn.Embedding):
    """Route RoBERTa's tied embedding through Opacus' tied-parameter path."""


class _TiedDecoderForGhost(nn.Linear):
    """Route RoBERTa's tied MLM decoder through Opacus' tied-parameter path."""


class RobertaPromptForQNLI(nn.Module):
    """Use RoBERTa's pretrained MLM head as a yes/no QNLI classifier."""

    def __init__(
        self,
        masked_lm: RobertaForMaskedLM,
        *,
        yes_token_id: int,
        no_token_id: int,
    ) -> None:
        super().__init__()
        # Hugging Face ties the MLM decoder weight to the input embedding.
        # Opacus' norm-sampler Ghost path rejects tied parameters, while its
        # functorch fallback in the same ``ghost`` mode explicitly accumulates
        # tied uses before computing the per-sample norm. Subclass just these
        # two standard layers so Opacus selects that tied-parameter path; their
        # pretrained parameters and forward implementations stay unchanged.
        masked_lm.roberta.embeddings.word_embeddings.__class__ = (
            _TiedEmbeddingForGhost
        )
        masked_lm.lm_head.decoder.__class__ = _TiedDecoderForGhost
        # Transformers also registers the same decoder bias at two names. Keep
        # its canonical decoder registration so stateless per-sample calls see
        # each parameter exactly once.
        if (
            masked_lm.lm_head._parameters.get("bias")
            is masked_lm.lm_head.decoder.bias
        ):
            del masked_lm.lm_head._parameters["bias"]
        self.roberta = masked_lm.roberta
        self.lm_head = masked_lm.lm_head
        self.yes_token_id = yes_token_id
        self.no_token_id = no_token_id

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mask_pos: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        ).last_hidden_state
        batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
        mask_hidden = hidden[batch_indices, mask_pos]
        vocab_logits = self.lm_head(mask_hidden)
        return torch.stack(
            (
                vocab_logits[:, self.yes_token_id],
                vocab_logits[:, self.no_token_id],
            ),
            dim=-1,
        )


def build_model(config: Config, tokenizer: PreTrainedTokenizerBase):
    yes_token_id, no_token_id = qnli_verbalizer_token_ids(tokenizer)
    masked_lm = RobertaForMaskedLM.from_pretrained(config.model.name)
    return RobertaPromptForQNLI(
        masked_lm,
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
    )
