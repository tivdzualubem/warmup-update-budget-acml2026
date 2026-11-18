"""
Trainer for pretrained sequence classification models.
"""

import os
import copy
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


class PretrainedTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device="cuda",
        learning_rate=2e-5,
        num_epochs=3,
        warmup_ratio=0.1,
        checkpoint_dir=None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.checkpoint_dir = checkpoint_dir

        if self.checkpoint_dir is not None:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(warmup_ratio * total_steps)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        self.best_val_loss = float("inf")
        self.best_model_state = None
        self.patience_counter = 0

    def _checkpoint_path(self, name):
        return os.path.join(self.checkpoint_dir, name)

    def save_checkpoint(self, epoch, is_best=False):
        if self.checkpoint_dir is None:
            return

        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_model_state": self.best_model_state,
            "patience_counter": self.patience_counter
        }

        torch.save(state, self._checkpoint_path("latest.pt"))
        if is_best:
            torch.save(state, self._checkpoint_path("best.pt"))

    def load_latest_checkpoint(self):
        if self.checkpoint_dir is None:
            return 0

        latest_path = self._checkpoint_path("latest.pt")
        if not os.path.exists(latest_path):
            return 0

        ckpt = torch.load(latest_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.scheduler.load_state_dict(ckpt["scheduler_state"])
        self.best_val_loss = ckpt["best_val_loss"]
        self.best_model_state = ckpt["best_model_state"]
        self.patience_counter = ckpt["patience_counter"]

        start_epoch = ckpt["epoch"] + 1
        print(f"Resuming from checkpoint: epoch {start_epoch}/{self.num_epochs}")
        return start_epoch

    def load_best_checkpoint(self):
        if self.checkpoint_dir is None:
            return False

        best_path = self._checkpoint_path("best.pt")
        if not os.path.exists(best_path):
            return False

        ckpt = torch.load(best_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.best_val_loss = ckpt["best_val_loss"]
        self.best_model_state = ckpt["best_model_state"]
        return True

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in self.train_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            self.optimizer.zero_grad()
            outputs = self.model(**batch)
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == batch["labels"]).sum().item()
            total += batch["labels"].size(0)

        return total_loss / len(self.train_loader), correct / total

    def evaluate(self, loader, return_predictions=False):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_probs = []
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)

                total_loss += loss.item()
                correct += (preds == batch["labels"]).sum().item()
                total += batch["labels"].size(0)

                if return_predictions:
                    all_preds.extend(preds.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                    all_logits.extend(logits.cpu().numpy())
                    all_labels.extend(batch["labels"].cpu().numpy())

        avg_loss = total_loss / len(loader)
        accuracy = correct / total

        if return_predictions:
            return avg_loss, accuracy, all_preds, all_probs, all_logits, all_labels
        return avg_loss, accuracy

    def train(self, early_stopping_patience=2, resume=True):
        start_epoch = self.load_latest_checkpoint() if resume else 0

        for epoch in range(start_epoch, self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.evaluate(self.val_loader)

            current_lr = self.optimizer.param_groups[0]["lr"]

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | LR: {current_lr:.2e}")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            is_best = False
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.patience_counter = 0
                is_best = True
                print("New best model saved")
            else:
                self.patience_counter += 1
                print(f"  Patience: {self.patience_counter}/{early_stopping_patience}")

            self.save_checkpoint(epoch, is_best=is_best)

            if self.patience_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        restored = self.load_best_checkpoint()
        if restored:
            print(f"\n{'='*60}")
            print(f"Restoring best model (val_loss={self.best_val_loss:.4f})")
            print('='*60)
        elif self.best_model_state is not None:
            print(f"\n{'='*60}")
            print(f"Restoring best model (val_loss={self.best_val_loss:.4f})")
            print('='*60)
            self.model.load_state_dict(self.best_model_state)
