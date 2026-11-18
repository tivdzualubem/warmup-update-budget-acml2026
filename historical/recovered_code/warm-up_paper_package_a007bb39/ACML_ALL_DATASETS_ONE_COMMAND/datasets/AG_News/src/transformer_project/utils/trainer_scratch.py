import os
import copy
import math
import torch
import torch.nn as nn


class ScratchTrainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device="cuda",
        learning_rate=1e-4,
        scheduler_mode="cosine",
        warmup_steps=8000,
        total_steps=50000,
        legacy_initial_lr=1.0,
        checkpoint_dir=None
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.learning_rate = learning_rate
        self.scheduler_mode = scheduler_mode
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.legacy_initial_lr = legacy_initial_lr
        self.current_step = 0
        self.checkpoint_dir = checkpoint_dir
        self.scheduler_trace = []

        if self.checkpoint_dir is not None:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        init_lr = legacy_initial_lr if scheduler_mode in ["legacy_buggy_warmup", "legacy_warmup_fixed_order"] else learning_rate

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=init_lr,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        self.criterion = nn.CrossEntropyLoss()

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
            "current_step": self.current_step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "best_model_state": self.best_model_state,
            "patience_counter": self.patience_counter,
            "scheduler_trace": self.scheduler_trace,
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
        self.best_val_loss = ckpt["best_val_loss"]
        self.best_model_state = ckpt["best_model_state"]
        self.patience_counter = ckpt["patience_counter"]
        self.current_step = ckpt["current_step"]
        self.scheduler_trace = ckpt.get("scheduler_trace", [])

        start_epoch = ckpt["epoch"] + 1
        print(f"Resuming from checkpoint: epoch {start_epoch}")
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

    def _get_cosine_lr(self):
        if self.current_step < self.warmup_steps:
            scale = self.current_step / max(1, self.warmup_steps)
        else:
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.learning_rate * scale

    def _get_legacy_warmup_lr(self):
        step = max(1, self.current_step)
        d_model = self.model.d_model
        return (d_model ** -0.5) * min(step ** -0.5, step * (self.warmup_steps ** -1.5))

    def _lr_for_current_mode(self):
        if self.scheduler_mode == "constant":
            return self.learning_rate
        if self.scheduler_mode == "cosine":
            return self._get_cosine_lr()
        if self.scheduler_mode in ["legacy_buggy_warmup", "legacy_warmup_fixed_order"]:
            return self._get_legacy_warmup_lr()
        raise ValueError(f"Unknown scheduler_mode: {self.scheduler_mode}")

    def _set_lr(self, lr_value):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr_value

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in self.train_loader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(input_ids)
            loss = self.criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            lr_before = float(self.optimizer.param_groups[0]["lr"])

            if self.scheduler_mode == "legacy_buggy_warmup":
                lr_used = lr_before
                self.optimizer.step()
                self.current_step += 1
                lr_after = float(self._lr_for_current_mode())
                self._set_lr(lr_after)
            else:
                self.current_step += 1
                lr_used = float(self._lr_for_current_mode())
                self._set_lr(lr_used)
                self.optimizer.step()
                lr_after = float(self.optimizer.param_groups[0]["lr"])

            self.scheduler_trace.append({
                "step": int(self.current_step),
                "lr_before_step": float(lr_before),
                "lr_used": float(lr_used),
                "lr_after_step": float(lr_after),
                "scheduler_mode": self.scheduler_mode,
            })

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

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
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(input_ids)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

                if return_predictions:
                    all_preds.extend(preds.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())
                    all_logits.extend(outputs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader)
        accuracy = correct / total

        if return_predictions:
            return avg_loss, accuracy, all_preds, all_probs, all_logits, all_labels
        return avg_loss, accuracy

    def train(self, num_epochs=30, early_stopping_patience=5, resume=True):
        start_epoch = self.load_latest_checkpoint() if resume else 0

        for epoch in range(start_epoch, num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

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
