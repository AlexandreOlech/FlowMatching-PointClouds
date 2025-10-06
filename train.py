import os
import json
import yaml
import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger 
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from data.dataset import FlowDataset
from models.basic_pointnet_flow import PointNetFlow

class FlowLitModule(LightningModule):
    def __init__(
        self,
        dataset_path: str,
        class_id: str,
        batch_size: int,
        lr: float,
        num_workers: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = PointNetFlow()
        train_path = os.path.join(dataset_path, class_id, "train")
        val_path = os.path.join(dataset_path, class_id, "val")
        self.train_dataset = FlowDataset(train_path)
        self.val_dataset = FlowDataset(val_path)
        self.batch_size = batch_size
        self.lr = lr
        self.num_workers = num_workers

    def forward(self, x, t): return self.model(x,t)
    
    def _loss(self, z):
        B = z.size(0)
        t = torch.rand(B, device=z.device)
        eps = torch.randn_like(z)
        t_view = t.view(B, *([1]* (z.dim() - 1)))
        x = t_view * z + (1 - t_view) * eps
        pred = self(x,t)
        target = z - eps
        return F.mse_loss(pred, target)

    def training_step(self, batch, batch_idx):
        loss = self._loss(batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss = self._loss(batch)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss
    
    def configure_optimizers(self): return optim.Adam(self.parameters(), lr=self.lr)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False
        )

if __name__ == "__main__":

    with open("configs/default.yaml", "r") as f:
        cfg = yaml.safe_load(f)["train"]

    with open(cfg["categ_to_id_path"], 'r') as file:
        categ_to_id = json.load(file)
    class_id = categ_to_id[cfg["categ"]]

    num_workers = os.cpu_count() - 1 if cfg["num_workers"] == "auto" else cfg["num_workers"]

    torch.set_float32_matmul_precision(cfg["float32_matmul_precision"])
    
    lit = FlowLitModule(
        dataset_path=cfg["dataset_path"],
        class_id=class_id,
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        num_workers = num_workers
    )

    logger = TensorBoardLogger(
        save_dir=".",
        name="logs",
        default_hp_metric=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    ckpt = ModelCheckpoint(
        monitor="val/loss", 
        mode="min", 
        save_top_k=1, 
        filename="epoch{epoch:02d}",
        auto_insert_metric_name=False
    )

    trainer = Trainer(
        accelerator="auto",
        devices="auto",
        accumulate_grad_batches=cfg["accumulate_grad_batches"],
        max_epochs=cfg["max_epochs"],
        log_every_n_steps=cfg["log_every_n_steps"],
        gradient_clip_val=cfg["gradient_clip_val"],
        precision=cfg["precision"],
        enable_checkpointing=cfg["enable_checkpointing"],
        logger=logger,
        callbacks=[lr_monitor, ckpt],
        check_val_every_n_epoch=cfg["check_val_every_n_epoch"]
    )

    trainer.fit(lit)