import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorchcv.model_provider import get_model, _models 

#_models.keys()
class MSDNN_PL(pl.LightningModule):
    def __init__(
            self,
            backbone_name: str = 'efficientnet_edge_large_b',
            msdnn: bool = True,
            sigma: float = 2,
            lr: float = 1e-3,
            lr_min: float = 1e-8,
            lr_patience: int = 5,
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.msdnn = msdnn

        self.model = MSDNN(backbone_name, msdnn)

        self.sigma = sigma
        self.lr = lr
        self.lr_min = lr_min
        self.lr_patience = lr_patience

        self.labels_p = []
        self.labels_gt = []

    def forward(self, images):
        return self.model(images)

    def adaptive_loss_function(self, labels, predicted):
        # https://www.sciencedirect.com/science/article/abs/pii/S0957417419306608
        loss = torch.mean((1 + self.sigma) * torch.pow(labels - predicted, 2) / (torch.abs(labels - predicted) + self.sigma))
        return loss

    def mean_absolute_error(self, labels, predicted):
        mae = F.l1_loss(predicted, labels)
        return mae

    def training_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.reshape(1, -1).t()
        preds = self.forward(images)
        loss = self.adaptive_loss_function(labels, preds)
        mae = self.mean_absolute_error(labels, preds)
        return {"loss": loss, "mae": mae}

    def training_epoch_end(self, outs):
        if (self.current_epoch == 1):
            img = torch.rand((1, 3, 224, 224))
            self.logger.experiment.add_graph(MSDNN(self.backbone_name, self.msdnn), img)
        loss = torch.stack([out["loss"] for out in outs]).mean()
        mae = torch.stack([out["mae"] for out in outs]).mean()
        self.log("Loss/Train", loss)
        self.log("MAE/Train", mae)

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.reshape(1, -1).t()
        preds = self.forward(images)
        loss = self.adaptive_loss_function(labels, preds)
        mae = self.mean_absolute_error(labels, preds)
        return {"loss": loss, "mae": mae}

    def validation_epoch_end(self, outs):
        loss = torch.stack([out["loss"] for out in outs]).mean()
        mae = torch.stack([out["mae"] for out in outs]).mean()
        self.log("Loss/Val", loss, prog_bar=True)
        self.log("MAE/Val", mae, prog_bar=True)
        for pg in self.optimizers().param_groups:
            lr = pg['lr']
        self.log("hparams/lr", lr)

    def test_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.reshape(1, -1).t()
        preds = self.forward(images)

        self.labels_p = self.labels_p + preds.squeeze().tolist()
        self.labels_gt = self.labels_gt + labels.squeeze().tolist()

        loss = self.adaptive_loss_function(labels, preds)
        mae = self.mean_absolute_error(labels, preds)
        return {"loss": loss, "mae": mae}

    def test_epoch_end(self, outs):
        loss = torch.stack([out["loss"] for out in outs]).mean()
        mae = torch.stack([out["mae"] for out in outs]).mean()
        self.log("Loss/Test", loss)
        self.log("MAE/Test", mae)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.lr_patience, min_lr=self.lr_min)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "Loss/Val", "interval": "epoch"}

class MSDNN(nn.Module):
    def __init__(
            self,
            backbone_name: str = 'efficientnet_edge_large_b',
            msdnn: bool = True
    ):
        super().__init__()

        self.backbone_name = backbone_name
        self.msdnn = msdnn
        self.stages = []

        backbone = get_model(backbone_name, pretrained=False)
        temp = torch.Tensor(1, 3, 224, 224)

        if  hasattr(backbone.features, "init_block"):
            self.init = backbone.features.init_block
            temp = self.init(temp)

        self.stage1 = backbone.features.stage1
        temp = self.stage1(temp)
        self.stages.append(temp.shape[1])

        self.stage2 = backbone.features.stage2
        temp = self.stage2(temp)
        self.stages.append(temp.shape[1])

        self.stage3 = backbone.features.stage3
        temp = self.stage3(temp)
        self.stages.append(temp.shape[1])

        self.stage4 = backbone.features.stage4
        temp = self.stage4(temp)
        self.stages.append(temp.shape[1])

        if  hasattr(backbone.features, "stage5"):
            self.stage5 = backbone.features.stage5
            temp = self.stage5(temp)
            self.stages.append(temp.shape[1])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        if msdnn:
            self.fc = nn.Linear(sum(self.stages), 1)
        else:
            self.fc = nn.Linear(temp.shape[1], 1)

    def forward(self, images):
        if self.msdnn:
            if hasattr(self, "init"):
                images = self.init(images)
            stage1 = self.stage1(images)
            stage2 = self.stage2(stage1)
            stage3 = self.stage3(stage2)
            stage4 = self.stage4(stage3)
            if hasattr(self, "stage5"):
                stage5 = self.stage5(stage4)
                out = torch.cat((self.avgpool(stage1), self.avgpool(stage2), self.avgpool(stage3), self.avgpool(stage4), self.avgpool(stage5)), 1)
            else:
                out = torch.cat((self.avgpool(stage1), self.avgpool(stage2), self.avgpool(stage3), self.avgpool(stage4)), 1)
            out = self.flatten(out)
            out = self.fc(out)
        else:
            if hasattr(self, "init"):
                images = self.init(images)
            out = self.stage1(images)
            out = self.stage2(out)
            out = self.stage3(out)
            out = self.stage4(out)
            if hasattr(self, "stage5"):
                out = self.stage5(out)
            out = self.avgpool(out)
            out = self.flatten(out)
            out = self.fc(out)
        return out

"""
from pytorch_model_summary import summary
from time import time
# efficientnet_edge_large_b
# mobilenetv3_large_w5d4
model = MSDNN('resnet18_wd4', True).cuda()

x = torch.Tensor(1, 3, 224, 224).cuda()

l = 1000
y = model(x)

#t = time()
#for _ in range(l):
#    y = model(x)
#t = time() - t
#print(t/l)

print(summary(model, x, show_input=False, show_hierarchical=False))
"""
