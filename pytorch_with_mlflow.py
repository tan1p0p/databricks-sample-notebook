# Databricks notebook source
# MAGIC %md
# MAGIC # PytorchでMLFlowを使うサンプル
# MAGIC 参考: https://qiita.com/taka_yayoi/items/08a4dbea3c943a5ae2ea

# COMMAND ----------

import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
from pytorch_lightning import LightningModule, Trainer

class MNISTModel(LightningModule):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)

        # PyTorchのロガーを使って精度情報を記録
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

# COMMAND ----------

import os

import mlflow
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.datasets.mnist import MNIST

# MLflowのエンティティを全てオートロギング
mlflow.pytorch.autolog()

# モデルを初期化
mnist_model = MNISTModel()

# MNISTデータセットのDataLoaderを初期化
train_ds = MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=32)

# トレーナーを初期化
trainer = Trainer(max_epochs=2)

# モデルをトレーニング
with mlflow.start_run() as run: # run IDを取得するためにブロックを宣言
  trainer.fit(mnist_model, train_loader)
