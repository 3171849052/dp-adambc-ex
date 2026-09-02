import torch
from transformers import AutoTokenizer, RobertaConfig, RobertaForMaskedLM

from dp_adam_iid.data import (
    QNLI_VERBALIZER,
    build_qnli_prompt,
    qnli_verbalizer_token_ids,
    tokenize_qnli_batch,
)
from dp_adam_iid.model import RobertaPromptForQNLI


def test_qnli_bedb_verbalizer_uses_single_roberta_tokens_in_label_order():
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")

    assert QNLI_VERBALIZER[0] == " yes"
    assert QNLI_VERBALIZER[1] == " no"
    yes_token_id, no_token_id = qnli_verbalizer_token_ids(tokenizer)
    assert [yes_token_id] == tokenizer.encode(" yes", add_special_tokens=False)
    assert [no_token_id] == tokenizer.encode(" no", add_special_tokens=False)


def test_prompt_has_one_mask_and_records_its_actual_position():
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
    questions = ["Who wrote the book?", "Where is Chengdu"]
    sentences = ["Ada wrote the book.", "Chengdu is in Sichuan."]

    assert build_qnli_prompt(questions[0], sentences[0], tokenizer.mask_token) == (
        "Who wrote the book? <mask>, Ada wrote the book."
    )
    encoded = tokenize_qnli_batch(
        tokenizer, questions, sentences, max_length=64
    )

    for input_ids, mask_pos in zip(encoded["input_ids"], encoded["mask_pos"]):
        assert input_ids.count(tokenizer.mask_token_id) == 1
        assert input_ids[mask_pos] == tokenizer.mask_token_id


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
