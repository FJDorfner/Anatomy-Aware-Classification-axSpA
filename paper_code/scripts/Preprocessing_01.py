#!/usr/bin/env python
# coding: utf-8

# ## Preprocessing and Datahandling

# In[1]:


import sys

sys.path.append("../scripts")


# In[2]:


import os
import pandas as pd
import Utils_00 as utils
import Paths as paths
from monai.data import DataLoader, PersistentDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    LoadImage,
    SpatialPadd,
    RandAffined,
    RandFlipd,
    RandRotated,
    RandZoomd,
    Resize,
    Resized,
    ResizeWithPadOrCrop,
    ResizeWithPadOrCropd,
    Rotate90d,
    SqueezeDimd,
)
from monai.utils import set_determinism


# ## Import dataset.csv

# In[ ]:


dataset_df_old_paths = pd.read_csv(paths.dataset_path, index_col=0)

# change the dataset.csv to the correct paths
base_path_image = paths.base_path_image
base_path_label = paths.base_path_label

dataset_df = dataset_df_old_paths.copy()

for index, value in dataset_df.iterrows():
    dataset_df.loc[index, "image"] = os.path.join(base_path_image, os.path.basename(value["image"]))
    dataset_df.loc[index, "label"] = os.path.join(base_path_label, os.path.basename(value["label"]))
dataset_df.head()


# ## Set deterministic training for reproducibility

# In[ ]:


set_determinism(seed=0)


# ## Prepare training, validation and test splits
# 
# 
# 
# 

# In[ ]:


num_class = 2

# exclude broken images
excluded_images = paths.excluded_images

# only PROOF data is used for training
excluded_studies = ["DAMACT", "OptiRef", "GESPIC"]

PROOF_df = dataset_df[~dataset_df["image"].str.contains("|".join(excluded_studies))]
PROOF_df = PROOF_df[~PROOF_df["image"].str.contains("|".join(excluded_images))]


# apply the training and validation split
train_df = PROOF_df[PROOF_df["is_initial_valid_ds"] == 0]
val_df = PROOF_df[PROOF_df["is_initial_valid_ds"] == 1]
train_x = train_df["image"]
train_y = train_df["classification"]
val_x = val_df["image"]
val_y = val_df["classification"]


# ## Define Transforms

# In[ ]:


train_transforms_106x158 = Compose(
    [
        LoadImaged(
            ensure_channel_first=True,
            image_only=True,
            keys="image",
        ),
        utils.CustomCLAHE(),
        Rotate90d(keys="image", spatial_axes=(0, 1), k=3),
        SqueezeDimd(keys="image", dim=3),
        Resized(spatial_size=158, mode="bilinear", keys="image", size_mode="longest"),
        ResizeWithPadOrCropd(spatial_size=(106, 158), keys="image"),
        RandFlipd(keys=["image"], spatial_axis=1, prob=0.5),
        RandFlipd(keys=["image"], spatial_axis=0, prob=0.5),
        RandRotated(
            keys=["image"],
            range_x=0.175,
            prob=0.5,
            keep_size=True,
            padding_mode="zeros",
        ),
        RandZoomd(
            keys=["image"],
            min_zoom=0.9,
            max_zoom=1.1,
            prob=0.5,
            padding_mode="constant",
        ),
        RandAffined(keys=["image"], prob=0.5, shear_range=(0.1, 0.1), padding_mode="zeros"),
        SpatialPadd(spatial_size=(106, 158), keys="image"),
    ]
)

val_transforms_106x158 = Compose(
    [
        LoadImaged(
            ensure_channel_first=True,
            image_only=True,
            keys="image",
        ),
        utils.CustomCLAHE(),
        Rotate90d(keys="image", spatial_axes=(0, 1), k=3),
        SqueezeDimd(keys="image", dim=3),
        Resized(spatial_size=158, mode="bilinear", keys="image", size_mode="longest"),
        ResizeWithPadOrCropd(spatial_size=(106, 158), keys="image"),
    ]
)


# In[ ]:


train_transforms_208x314 = Compose(
    [
        LoadImaged(
            ensure_channel_first=True,
            image_only=True,
            keys=["image"],
        ),
        utils.CustomCLAHE(),
        Rotate90d(keys=["image"], spatial_axes=(0, 1), k=3),
        SqueezeDimd(keys=["image"], dim=3),
        Resized(spatial_size=314, mode="bilinear", keys="image", size_mode="longest"),
        ResizeWithPadOrCropd(spatial_size=(208, 314), keys="image"),
        RandFlipd(keys=["image"], spatial_axis=1, prob=0.5),
        RandFlipd(keys=["image"], spatial_axis=0, prob=0.5),
        RandRotated(
            keys=["image"],
            range_x=0.175,
            prob=0.5,
            keep_size=True,
            padding_mode="zeros",
        ),
        RandZoomd(
            keys=["image"],
            min_zoom=0.9,
            max_zoom=1.1,
            prob=0.5,
            padding_mode="constant",
        ),
        RandAffined(keys=["image"], prob=0.5, shear_range=(0.1, 0.1), padding_mode="zeros"),
        SpatialPadd(spatial_size=(208, 314), keys="image"),
    ]
)

val_transforms_208x314 = Compose(
    [
        LoadImaged(
            ensure_channel_first=True,
            image_only=True,
            keys=["image"],
        ),
        utils.CustomCLAHE(),
        Rotate90d(keys=["image"], spatial_axes=(0, 1), k=3),
        SqueezeDimd(keys=["image"], dim=3),
        Resized(spatial_size=314, mode="bilinear", keys="image", size_mode="longest"),
        ResizeWithPadOrCropd(spatial_size=(208, 314), keys="image"),
    ]
)


# In[ ]:


train_transforms_312x472 = Compose(
    [
        LoadImaged(
            ensure_channel_first=True,
            image_only=True,
            keys=["image"],
        ),
        utils.CustomCLAHE(),
        Rotate90d(keys=["image"], spatial_axes=(0, 1), k=3),
        SqueezeDimd(keys=["image"], dim=3),
        Resized(spatial_size=472, mode="area", keys="image", size_mode="longest"),
        ResizeWithPadOrCropd(spatial_size=(312, 472), keys="image"),
        RandFlipd(keys=["image"], spatial_axis=1, prob=0.5),
        RandFlipd(keys=["image"], spatial_axis=0, prob=0.5),
        RandRotated(
            keys=["image"],
            range_x=0.175,  
            prob=0.5,
            keep_size=True,
            padding_mode="zeros",
        ),
        RandZoomd(
            keys=["image"],
            min_zoom=0.9,
            max_zoom=1.1,
            prob=0.5,
            padding_mode="constant",
        ),
        RandAffined(keys=["image"], prob=0.5, shear_range=(0.1, 0.1), padding_mode="zeros"),
        SpatialPadd(spatial_size=(312, 472), keys="image"),
    ]
)

val_transforms_312x472 = Compose(
    [
        LoadImaged(
            ensure_channel_first=True,
            image_only=True,
            keys=["image"],
        ),
        utils.CustomCLAHE(),
        Rotate90d(keys=["image"], spatial_axes=(0, 1), k=3),
        SqueezeDimd(keys=["image"], dim=3),
        Resized(spatial_size=472, mode="bilinear", keys="image", size_mode="longest"),
        ResizeWithPadOrCropd(spatial_size=(312, 472), keys="image"),
    ]
)


# In[ ]:


train_transforms_416x628 = Compose(
    [
        LoadImaged(
            ensure_channel_first=True,
            image_only=True,
            keys=["image"],
        ),
        utils.CustomCLAHE(),
        Rotate90d(keys=["image"], spatial_axes=(0, 1), k=3),
        SqueezeDimd(keys=["image"], dim=3),
        Resized(spatial_size=628, mode="bilinear", keys="image", size_mode="longest"),
        ResizeWithPadOrCropd(spatial_size=(416, 628), keys="image"),
        RandFlipd(keys=["image"], spatial_axis=1, prob=0.5),
        RandFlipd(keys=["image"], spatial_axis=0, prob=0.5),
        RandRotated(
            keys=["image"],
            range_x=0.175,  # 175
            prob=0.5,
            keep_size=True,
            padding_mode="zeros",
        ),
        RandZoomd(
            keys=["image"],
            min_zoom=0.9,
            max_zoom=1.1,
            prob=0.5,
            padding_mode="constant",
        ),
        RandAffined(keys=["image"], prob=0.5, shear_range=(0.1, 0.1), padding_mode="zeros"),
        SpatialPadd(spatial_size=(416, 628), keys="image"),
    ]
)

val_transforms_416x628 = Compose(
    [
        LoadImaged(
            ensure_channel_first=True,
            image_only=True,
            keys=["image"],
        ),
        utils.CustomCLAHE(),
        Rotate90d(keys=["image"], spatial_axes=(0, 1), k=3),
        SqueezeDimd(keys=["image"], dim=3),
        Resized(spatial_size=628, mode="bilinear", keys="image", size_mode="longest"),
        ResizeWithPadOrCropd(spatial_size=(416, 628), keys="image"),
    ],
)


# In[ ]:


gradcam_transform = Compose(
    [
        LoadImage(
            ensure_channel_first=False,
            image_only=True,
            reader="NibabelReader",
            reverse_indexing=True,
        ),
        utils.CustomCLAHE(),
        Resize(spatial_size=628, mode="bilinear", size_mode="longest"),
        ResizeWithPadOrCrop(spatial_size=(416, 628)),
    ],
)


# ## Define Cache, Dataset and Dataloaders

# In[ ]:


scratch_path = paths.scratch_path


# In[ ]:


cache_small = os.path.join(scratch_path, "cache_small_masks")
if not os.path.exists(cache_small):
    os.makedirs(cache_small)

cache_medium = os.path.join(scratch_path, "cache_medium_masks")
if not os.path.exists(cache_medium):
    os.makedirs(cache_medium)

cache_large = os.path.join(scratch_path, "cache_large_masks")
if not os.path.exists(cache_large):
    os.makedirs(cache_large)

cache_largest = os.path.join(scratch_path, "cache_largest_masks")
if not os.path.exists(cache_largest):
    os.makedirs(cache_largest)


# In[ ]:


train_data = [
    {"image": image_name, "label": label_name} for image_name, label_name in zip(train_x, train_y)
]
val_data = [
    {"image": image_name, "label": label_name} for image_name, label_name in zip(val_x, val_y)
]


# In[ ]:


# Dataset and Dataloader for resoltion 106x158
train_ds_small = PersistentDataset(
    data=train_data, transform=train_transforms_106x158, cache_dir=cache_small
)
train_loader_small = DataLoader(
    train_ds_small,
    batch_size=64,
    shuffle=True,
    num_workers=48,
    pin_memory=True,
)

val_ds_small = PersistentDataset(
    data=val_data, transform=val_transforms_106x158, cache_dir=cache_small
)
val_loader_small = DataLoader(val_ds_small, batch_size=64, num_workers=48, pin_memory=True)

# Dataset and Dataloader for resoltion 208x314
train_ds_medium = PersistentDataset(
    data=train_data, transform=train_transforms_208x314, cache_dir=cache_medium
)
train_loader_medium = DataLoader(
    train_ds_medium,
    batch_size=64,
    shuffle=True,
    num_workers=48,
    pin_memory=True,
)

val_ds_medium = PersistentDataset(
    data=val_data, transform=val_transforms_208x314, cache_dir=cache_medium
)
val_loader_medium = DataLoader(val_ds_medium, batch_size=64, num_workers=48, pin_memory=True)

# Dataset and Dataloader for resoltion 312x472
train_ds_large = PersistentDataset(
    data=train_data, transform=train_transforms_312x472, cache_dir=cache_large
)
train_loader_large = DataLoader(
    train_ds_large,
    batch_size=64,
    shuffle=True,
    num_workers=48,
    pin_memory=True,
)

val_ds_large = PersistentDataset(
    data=val_data, transform=val_transforms_312x472, cache_dir=cache_large
)
val_loader_large = DataLoader(val_ds_large, batch_size=64, num_workers=48, pin_memory=True)

# Dataset and Dataloader for resoltion 416x628
train_ds_largest = PersistentDataset(
    data=train_data, transform=train_transforms_416x628, cache_dir=cache_largest
)
train_loader_largest = DataLoader(
    train_ds_largest,
    batch_size=32,
    shuffle=True,
    num_workers=48,
    pin_memory=True,
)

val_ds_largest = PersistentDataset(
    data=val_data, transform=val_transforms_416x628, cache_dir=cache_largest
)
val_loader_largest = DataLoader(val_ds_largest, batch_size=32, num_workers=48, pin_memory=True)

