# encoding: utf-8
import os
import pytorch_lightning as pl
import pandas as pd
from scipy.io import savemat
from torch.utils.data import DataLoader
from torchvision import transforms
from data import AgeData, GTAData
from model import LDNN

max_epochs = 100
batch_size = 64
num_workers = 8
backbone_name = 'efficientnet_edge_large_b'
accelerator = 'ddp'
dataset_tr = 'MORPH2'
dataset_ts = 'MORPH2'
subset = 'Test'
weights = 'LDNN_epoch=41.ckpt' # GTA_LDNN_epoch=99-v1.ckpt / MORPH2_LDNN_epoch=99-v3.ckpt / AFAD_LDNN_epoch=99-v2.ckpt / CACD_LDNN_epoch=99-v1.ckpt
if os.name == "nt":
    path = 'E:/Projects/GTA_CAIP_Contest'
    accelerator = 'dp'
    num_workers = 0
    face_path = 'D:\Face'
else:
    path = '/home/sbekhouche/Projects/GTA_CAIP_Contest/'
    face_path = 'D:\Face'

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

if dataset_ts == 'GTA':
    subset = 'Val'
    val = pd.read_pickle(os.path.join(path, "split", "val.pkl"))
    test_data = GTAData(path=path, index_list=val, transform=transform)
else:
    test_data = AgeData(path=face_path, dataset=dataset_ts, subset=subset, transform=transform)


test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
model = LDNN.load_from_checkpoint(os.path.join(path, 'checkpoints_no', weights))

trainer = pl.Trainer(weights_summary=None, gpus=1)
trainer.test(model, test_dataloaders=test_loader)
savemat("results/" + dataset_tr + "_" + dataset_ts + "_" + subset + "_NO.mat", {"labels_gt": model.labels_gt,
                                                 "labels_p": model.labels_p})

#a = train.index.tolist()
#b = val.index.tolist()
#print([i for i, j in zip(a, b) if i == j])
