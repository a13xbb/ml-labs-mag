
from typing import List
import torch
import torch.nn as nn
import torchmetrics
import pytorch_lightning as pl


class ResNetBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        normalization: str,
        activation: str,
        dropout_type: str,
        hidden_factor: int = 2,
        dropout_1: float = 0.1,
        dropout_2: float = 0.05,
    ):
        super().__init__()
        
        d_hidden = int(hidden_factor * input_dim)

        if normalization == "batch_norm":
            self.norm1 = nn.BatchNorm1d(d_hidden)
            self.norm2 = nn.BatchNorm1d(input_dim)
        elif normalization == "layer_norm":
            self.norm1 = nn.LayerNorm(d_hidden)
            self.norm2 = nn.LayerNorm(input_dim)
        else:
            raise ValueError("Normalization is incorrect. Possible options: 'batch_norm', 'layer_norm'.")

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError("Activation is incorrect. Possible options: 'relu', 'gelu'.")

        if dropout_type == "dropout":
            self.dropout = nn.Dropout
        elif dropout_type == "dropout1d":
            self.dropout = nn.Dropout1d
        else:
            raise ValueError("Dropout type is incorrect. Possible options: 'dropout', 'dropout1d'.")

        self.linear1 = nn.Linear(input_dim, d_hidden)
        self.linear2 = nn.Linear(d_hidden, input_dim)

        self.ff = nn.Sequential(
            self.linear1,
            self.norm1,
            self.activation,
            self.dropout(dropout_1),
            self.linear2,
            self.norm2,
            self.activation,
            self.dropout(dropout_2),
        )

    def forward(self, x: torch.Tensor, use_skip_connection: bool = True) -> torch.Tensor:
        out = self.ff(x)
        if use_skip_connection:
            out = x + out
        return out 

class ResNet(pl.LightningModule):
    def __init__(self, input_dim: int, params: dict = {}):
        super(ResNet, self).__init__()
        self.params = params
        self.input_dim = input_dim
        self.verbose = []

        self.loss = nn.BCELoss()
        self.f1 = torchmetrics.classification.BinaryF1Score(threshold=0.5)

        n_hidden = params.get("n_hidden", 2)
        layer_size = params.get("layer_size", 128)
        normalization = params.get("normalization", "layer_norm")
        activation = params.get("activation", "relu")
        dropout_type = params.get("dropout_type", "dropout")
        hidden_factor = params.get("hidden_factor", 2.0)
        dropout_1 = params.get("dropout_1", 0.1)
        dropout_2 = params.get("dropout_2", 0.05)

        resnet_block_params = [
            layer_size,
            normalization,
            activation,
            dropout_type,
            hidden_factor,
            dropout_1,
            dropout_2,
        ]

        self.blocks = nn.Sequential(nn.Linear(input_dim, layer_size))

        for i in range(n_hidden):
            self.blocks.append(ResNetBlock(*resnet_block_params))

        self.prediction = nn.Sequential(
            nn.LayerNorm(layer_size),
            nn.ReLU(),
            nn.Linear(layer_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.blocks(x)
        return self.prediction(hidden).flatten()

    def training_step(self, batch: List[torch.Tensor]):
        x, y = batch
        output = self(x)
        loss = self.loss(output, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch: List[torch.Tensor]):
        x, y = batch
        output = self(x)

        loss = self.loss(output, y)
        self.log("val_loss", loss, on_epoch=True)

        self.f1.update(output, y)
        self.log("val_f1", self.f1, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.params.get("learning_rate", 1e-3),
            weight_decay=self.params.get("weight_decay", 1e-6),
        )

        if self.params.get("use_scheduler", True):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.1,
                patience=5,
                threshold=1e-4,
                threshold_mode="abs",
            )
        else:
            scheduler = None

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_f1"},
        }
