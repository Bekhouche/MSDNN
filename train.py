import os
import torch
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader
from data import AgeData
from model_paper import MSDNN_PL

max_epochs = 100
batch_size = 64
num_workers = 8
backbone_name = 'efficientnet_edge_large_b'
msdnn = False
checkpoints_file = None #'weights/MORPH2_LDNN_epoch=99.ckpt'
accelerator = 'ddp'
no_multi = False
dataset = 'MORPH2'
if os.name == "nt":
    path = 'E:/Projects/GTA_CAIP_Contest'
    accelerator = 'dp'
    num_workers = 0
    face_path = 'D:\Face'
else:
    path = '/home/sbekhouche/Projects/GTA_CAIP_Contest'
    face_path = 'D:\Face'

# Dataset
train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomGrayscale(p=0.5),
            transforms.ColorJitter(brightness=0.5, contrast=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

train_data = AgeData(path=face_path, dataset=dataset, subset="Train", transform=train_transform)
val_data = AgeData(path=face_path, dataset=dataset, subset="Val", transform=val_transform)
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)



# Model
model = MSDNN_PL(backbone_name=backbone_name, msdnn=msdnn, sigma=2, lr=1e-3, lr_min=1e-8, lr_patience=5)
if checkpoints_file:
    checkpoints = torch.load(checkpoints_file)
    model.load_state_dict(checkpoints['state_dict'])

if msdnn:
    callback_checkpoint = pl.callbacks.ModelCheckpoint(
        save_top_k = -1,
        dirpath = os.path.join(path, 'checkpoints_paper'),
        filename = 'MSDNN_' + backbone_name + '_{epoch:02d}'
    )
else:
    callback_checkpoint = pl.callbacks.ModelCheckpoint(
        save_top_k = -1,
        dirpath = os.path.join(path, 'checkpoints_paper2'),
        filename = backbone_name + '_{epoch:02d}'
    )

logger = pl.loggers.TensorBoardLogger('logs', name='MSDNN', default_hp_metric=False)

trainer = pl.Trainer(
    gpus=1,
    progress_bar_refresh_rate=10,
    callbacks=[callback_checkpoint],
    logger=logger,
    max_epochs=max_epochs,
    precision=16,
    accelerator=accelerator,
    benchmark=True,
    amp_level='02'
)
trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)
#trainer.test(model, test_dataloaders=test_loader)
