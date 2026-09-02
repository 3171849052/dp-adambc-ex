from types import SimpleNamespace

import torch
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch import nn
from torch.utils.data import DataLoader, Dataset


class TinyDataset(Dataset):
    def __init__(self):
        self.input_ids = torch.tensor(
            [[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10]]
        )
        self.labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {"input_ids": self.input_ids[index], "labels": self.labels[index]}


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 4)
        self.classifier = nn.Linear(4, 2)

    def forward(self, input_ids):
        return SimpleNamespace(logits=self.classifier(self.embedding(input_ids).mean(dim=1)))


def test_opacus_gdp_ghost_uses_adam_and_one_logical_step():
    torch.manual_seed(0)
    model = TinyClassifier()
    loader = DataLoader(TinyDataset(), batch_size=4, shuffle=False)
    original_optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
    criterion = nn.CrossEntropyLoss()
    engine = PrivacyEngine(accountant="gdp")
    hooks, optimizer, criterion, private_loader = engine.make_private_with_epsilon(
        module=model,
        optimizer=original_optimizer,
        criterion=criterion,
        data_loader=loader,
        target_epsilon=3.0,
        target_delta=1.0e-5,
        epochs=1,
        max_grad_norm=1.0,
        clipping="flat",
        grad_sample_mode="ghost",
        poisson_sampling=True,
        wrap_model=False,
    )
    assert isinstance(optimizer.original_optimizer, torch.optim.Adam)
    assert optimizer.noise_multiplier > 0

    logical_steps = 0
    with BatchMemoryManager(
        data_loader=private_loader,
        max_physical_batch_size=2,
        optimizer=optimizer,
    ) as memory_loader:
        for batch in memory_loader:
            if not batch["labels"].numel():
                optimizer.zero_grad()
                optimizer.step()
                optimizer.zero_grad()
                continue
            outputs = model(input_ids=batch["input_ids"])
            loss = criterion(outputs.logits, batch["labels"])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if not optimizer._is_last_step_skipped:
                logical_steps += 1

    assert logical_steps == len(private_loader)
    assert engine.get_epsilon(1.0e-5) > 0
    hooks.cleanup()
