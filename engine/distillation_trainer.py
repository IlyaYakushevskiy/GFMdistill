from .trainer import Trainer
import torch

class DistillationTrainer(Trainer): ##OR LinearClassificationTrainer to inherit?
    def __init__(self, teacher_model: torch.nn.Module, *args, **kwargs):
        """
        Initializes the DistillationTrainer.

        Args:
            teacher_model (torch.nn.Module): The pre-trained, larger teacher model.
            *args, **kwargs: All other arguments for the base Trainer class.
        """
        super().__init__(*args, **kwargs)
        
    
        self.teacher_model = teacher_model
        self.teacher_model.to(self.device)
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def train_one_epoch(self, epoch: int) -> None:
        """
        Override the training loop to include the teacher model's forward pass.
        """
        self.model.train() # the student model

        end_time = time.time()
        for batch_idx, data in enumerate(self.train_loader):
            image, target = data["image"], data["target"]
            image = {modality: value.to(self.device) for modality, value in image.items()}
            target = target.to(self.device)

            self.training_stats["data_time"].update(time.time() - end_time)

            # Get teacher logits
            with torch.no_grad():
                teacher_logits = self.teacher_model(image)

            with torch.autocast("cuda", enabled=self.enable_mixed_precision, dtype=self.precision):
                student_logits = self.model(image)
                loss = self.criterion(student_logits, target, teacher_logits)

            self.optimizer.zero_grad()

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Rank {self.rank} got infinite/NaN loss at batch {batch_idx} of epoch {epoch}!"
                )

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.training_stats['loss'].update(loss.item())

            with torch.no_grad():
                self.compute_logging_metrics(logits, target)
            if (batch_idx + 1) % self.log_interval == 0:
                self.log(batch_idx + 1, epoch)

            self.lr_scheduler.step()

            if self.use_wandb and self.rank == 0:
                self.wandb.log(
                    {
                        "train_loss": loss.item(),
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                        "epoch": epoch,
                        **{
                            f"train_{k}": v.avg
                            for k, v in self.training_metrics.items()
                        },
                    },
                    step=epoch * len(self.train_loader) + batch_idx,
                )

            self.training_stats["batch_time"].update(time.time() - end_time)
            end_time = time.time()