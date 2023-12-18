#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys

sys.path.append("../scripts")


# In[ ]:


import os
import Models_02 as models
import Paths as paths
import Preprocessing_01 as preprocessing
import pytorch_lightning as pl
import Utils_00 as utils
from monai.utils import set_determinism
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import CSVLogger


# ## Model Training using Progressive Resizing

# In[ ]:


model = models.ResNetTransferLearningDiscriminativeLR


# In[ ]:


log_dir = paths.log_dir
print(f"all logs are stored at: {log_dir}")


# In[ ]:


model_checkpoint = ModelCheckpoint(monitor="val_MCC", mode="max", every_n_epochs=1, save_top_k=2)
set_determinism(seed=0)


# In[ ]:


# train fc-layer on small image size
trainer = pl.Trainer(
    accelerator="gpu",
    devices=-1,
    precision=16,
    max_epochs=15,
    log_every_n_steps=1,
    callbacks=[RichProgressBar(), model_checkpoint],
    num_sanity_val_steps=2,
    logger=CSVLogger(save_dir=log_dir),
)

metrics_path = f"{trainer.logger.log_dir}/metrics.csv"
checkpoint_folder = f"{trainer.logger.log_dir}/checkpoints"
print(metrics_path)
print(checkpoint_folder)

model2 = model(
    only_fc=True,
    max_lr=2e-2,
    wd=0.0001,
    first_dropout=0.1,
    lr_mult=0.9,
    alpha=0.4,
)


trainer.fit(
    model2,
    train_dataloaders=preprocessing.train_loader_small,
    val_dataloaders=preprocessing.val_loader_small,
)


# In[ ]:


# print checkpoint location and visualize metrics for the completed section of training
checkpoint = os.path.join(checkpoint_folder, os.listdir(checkpoint_folder)[0])
print(checkpoint)
utils.visualize_training_metrics(metrics_path)


# In[ ]:


# train all layers on small image size
model_checkpoint = ModelCheckpoint(monitor="val_MCC", mode="max", every_n_epochs=1, save_top_k=2)

trainer = pl.Trainer(
    accelerator="gpu",
    devices=-1,
    precision=16,
    max_epochs=10,
    log_every_n_steps=1,
    callbacks=[RichProgressBar(), model_checkpoint],
    num_sanity_val_steps=2,
    logger=CSVLogger(save_dir=log_dir),
)


metrics_path = f"{trainer.logger.log_dir}/metrics.csv"
checkpoint_folder = f"{trainer.logger.log_dir}/checkpoints"
print(metrics_path)
print(checkpoint_folder)

model2 = model.load_from_checkpoint(
    checkpoint,
    only_fc=False,
    max_lr=1e-3,
    wd=0.0001,
    first_dropout=0.1,
    lr_mult=0.9,
    alpha=0.4,
)

trainer.fit(
    model2,
    train_dataloaders=preprocessing.train_loader_small,
    val_dataloaders=preprocessing.val_loader_small,
)


# In[ ]:


checkpoint = os.path.join(checkpoint_folder, os.listdir(checkpoint_folder)[0])
print(checkpoint)
utils.visualize_training_metrics(metrics_path)


# In[ ]:


# train fc-layer on medium image size
model_checkpoint = ModelCheckpoint(monitor="val_MCC", mode="max", every_n_epochs=1, save_top_k=2)

trainer = pl.Trainer(
    accelerator="gpu",
    devices=-1,
    precision=16,
    max_epochs=15,
    log_every_n_steps=1,
    callbacks=[RichProgressBar(), model_checkpoint],
    num_sanity_val_steps=2,
    logger=CSVLogger(save_dir=log_dir),
)

metrics_path = f"{trainer.logger.log_dir}/metrics.csv"
checkpoint_folder = f"{trainer.logger.log_dir}/checkpoints"


model2 = model.load_from_checkpoint(
    checkpoint,
    only_fc=True,
    max_lr=1e-2,
    wd=0.0001,
    first_dropout=0.1,
    lr_mult=0.9,
    alpha=0.4,
)
trainer.fit(
    model2,
    train_dataloaders=preprocessing.train_loader_medium,
    val_dataloaders=preprocessing.val_loader_medium,
)


# In[ ]:


checkpoint = os.path.join(checkpoint_folder, os.listdir(checkpoint_folder)[0])
print(checkpoint)
utils.visualize_training_metrics(metrics_path)


# In[ ]:


# train all layers on medium image size
model_checkpoint = ModelCheckpoint(monitor="val_MCC", mode="max", every_n_epochs=1, save_top_k=2)

trainer = pl.Trainer(
    accelerator="gpu",
    devices=-1,
    precision=16,
    max_epochs=10,
    log_every_n_steps=1,
    callbacks=[RichProgressBar(), model_checkpoint],
    num_sanity_val_steps=2,
    logger=CSVLogger(save_dir=log_dir),
)

metrics_path = f"{trainer.logger.log_dir}/metrics.csv"
checkpoint_folder = f"{trainer.logger.log_dir}/checkpoints"

model2 = model.load_from_checkpoint(
    checkpoint,
    only_fc=False,
    max_lr=1e-3,
    wd=0.0001,
    first_dropout=0.1,
    lr_mult=0.9,
    alpha=0.4,
)
trainer.fit(
    model2,
    train_dataloaders=preprocessing.train_loader_medium,
    val_dataloaders=preprocessing.val_loader_medium,
)


# In[ ]:


checkpoint = os.path.join(checkpoint_folder, os.listdir(checkpoint_folder)[0])
print(checkpoint)
utils.visualize_training_metrics(metrics_path)


# In[ ]:


# train fc-layer on large image size
model_checkpoint = ModelCheckpoint(monitor="val_MCC", mode="max", every_n_epochs=1, save_top_k=2)

trainer = pl.Trainer(
    accelerator="gpu",
    devices=-1,
    precision=16,
    max_epochs=15,
    log_every_n_steps=1,
    callbacks=[RichProgressBar(), model_checkpoint],
    num_sanity_val_steps=2,
    logger=CSVLogger(save_dir=log_dir),
)

metrics_path = f"{trainer.logger.log_dir}/metrics.csv"
checkpoint_folder = f"{trainer.logger.log_dir}/checkpoints"

model2 = model.load_from_checkpoint(
    checkpoint,
    only_fc=True,
    max_lr=1e-2,
    wd=0.0001,
    first_dropout=0.1,
    lr_mult=0.9,
    alpha=0.4,
)
trainer.fit(
    model2,
    train_dataloaders=preprocessing.train_loader_large,
    val_dataloaders=preprocessing.val_loader_large,
)


# In[ ]:


checkpoint = os.path.join(checkpoint_folder, os.listdir(checkpoint_folder)[0])
print(checkpoint)
utils.visualize_training_metrics(metrics_path)


# In[ ]:


# train all layers on large image size
model_checkpoint = ModelCheckpoint(monitor="val_MCC", mode="max", every_n_epochs=1, save_top_k=2)

trainer = pl.Trainer(
    accelerator="gpu",
    devices=-1,
    precision=16,
    max_epochs=15,
    log_every_n_steps=1,
    callbacks=[RichProgressBar(), model_checkpoint],
    num_sanity_val_steps=2,
    logger=CSVLogger(save_dir=log_dir),
)

metrics_path = f"{trainer.logger.log_dir}/metrics.csv"
checkpoint_folder = f"{trainer.logger.log_dir}/checkpoints"

model2 = model.load_from_checkpoint(
    checkpoint,
    only_fc=False,
    max_lr=1e-3,  # 5e-4
    wd=0.0001,
    first_dropout=0.1,
    lr_mult=0.9,
    alpha=0.4,
)
trainer.fit(
    model2,
    train_dataloaders=preprocessing.train_loader_large,
    val_dataloaders=preprocessing.val_loader_large,
)


# In[ ]:


checkpoint = os.path.join(checkpoint_folder, os.listdir(checkpoint_folder)[0])
print(checkpoint)
utils.visualize_training_metrics(metrics_path)


# In[ ]:


# train fc-layer on largest image size
model_checkpoint = ModelCheckpoint(monitor="val_MCC", mode="max", every_n_epochs=1, save_top_k=2)

trainer = pl.Trainer(
    accelerator="gpu",
    devices=-1,
    precision=16,
    max_epochs=15,
    log_every_n_steps=1,
    callbacks=[RichProgressBar(), model_checkpoint],
    num_sanity_val_steps=2,
    logger=CSVLogger(save_dir=log_dir),
)

metrics_path = f"{trainer.logger.log_dir}/metrics.csv"
checkpoint_folder = f"{trainer.logger.log_dir}/checkpoints"

model2 = model.load_from_checkpoint(
    checkpoint,
    only_fc=True,
    max_lr=1e-2,
    wd=0.0001,
    first_dropout=0.1,
    lr_mult=0.9,
    alpha=0.4,
)
trainer.fit(
    model2,
    train_dataloaders=preprocessing.train_loader_largest,
    val_dataloaders=preprocessing.val_loader_largest,
)


# In[ ]:


checkpoint = os.path.join(checkpoint_folder, os.listdir(checkpoint_folder)[0])
print(checkpoint)
utils.visualize_training_metrics(metrics_path)


# In[ ]:


# train all layers on largest image size
model_checkpoint = ModelCheckpoint(monitor="val_MCC", mode="max", every_n_epochs=1, save_top_k=2)

trainer = pl.Trainer(
    accelerator="gpu",
    devices=-1,
    precision=16,
    max_epochs=25,
    log_every_n_steps=1,
    callbacks=[RichProgressBar(), model_checkpoint],
    num_sanity_val_steps=2,
    logger=CSVLogger(save_dir=log_dir),
)
metrics_path = f"{trainer.logger.log_dir}/metrics.csv"
checkpoint_folder = f"{trainer.logger.log_dir}/checkpoints"
print(checkpoint)

model2 = model.load_from_checkpoint(
    checkpoint,
    only_fc=False,
    max_lr=1e-3,
    wd=0.0001,
    first_dropout=0.1,
    lr_mult=0.9,
    alpha=0.4,
)

trainer.fit(
    model2,
    train_dataloaders=preprocessing.train_loader_largest,
    val_dataloaders=preprocessing.val_loader_largest,
)


# In[ ]:


checkpoint = os.path.join(checkpoint_folder, os.listdir(checkpoint_folder)[0])
print(checkpoint)
utils.visualize_training_metrics(metrics_path)

