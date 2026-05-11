from __future__ import annotations

from typing import Any


def build_early_stopping_trainer_class() -> type:
    from procnet.trainer.DocEE_proxy_node_trainer import DocEETrainer

    class EarlyStoppingDocEETrainer(DocEETrainer):
        def __init__(self, *args: Any, patience: int, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.patience = patience

        def train(self) -> dict[str, Any]:
            return self.train_with_early_stopping()

        def train_with_early_stopping(self) -> dict[str, Any]:
            best_dev_f1 = -1.0
            best_epoch = None
            best_raw_results: dict[str, Any] = {}
            epochs_without_improvement = 0
            stopped = False
            final_epoch = 0
            for epoch in range(1, self.config.max_epochs + 1):
                final_epoch = epoch
                self.train_batch_template(self.model_fn, dataloader=self.train_loader, epoch=epoch)
                dev_score, dev_raw = self.eval_batch_template(
                    self.model_fn,
                    score_fn=self.score_fn,
                    dataloader=self.dev_loader,
                    epoch=epoch,
                )
                dev_f1 = _extract_micro_f1(dev_score)
                if dev_f1 > best_dev_f1:
                    best_dev_f1 = dev_f1
                    best_epoch = epoch
                    test_score, test_raw = self.eval_batch_template(
                        self.model_fn,
                        score_fn=self.score_fn,
                        dataloader=self.test_loader,
                        epoch=epoch,
                    )
                    del test_score
                    best_raw_results = {"dev": dev_raw, "test": test_raw}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= self.patience:
                        stopped = True
                        break
            if not best_raw_results:
                _, dev_raw = self.eval_batch_template(
                    self.model_fn,
                    score_fn=self.score_fn,
                    dataloader=self.dev_loader,
                    epoch="final",
                )
                _, test_raw = self.eval_batch_template(
                    self.model_fn,
                    score_fn=self.score_fn,
                    dataloader=self.test_loader,
                    epoch="final",
                )
                best_raw_results = {"dev": dev_raw, "test": test_raw}
            return {
                "raw_results": best_raw_results,
                "early_stopping": {
                    "best_epoch": best_epoch,
                    "best_dev_f1": best_dev_f1,
                    "final_epoch": final_epoch,
                    "stopped_by_early_stopping": stopped,
                    "patience": self.patience,
                    "max_epochs": self.config.max_epochs,
                },
            }

    return EarlyStoppingDocEETrainer


def _extract_micro_f1(score: dict[str, Any]) -> float:
    return float(score.get("event", {}).get("all_event", {}).get("micro_f1", 0.0))
