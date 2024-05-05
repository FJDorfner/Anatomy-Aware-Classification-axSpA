#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys

sys.path.append("../scripts")


# In[1]:


import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn

from torch.autograd import Variable
from torchmetrics.functional.classification import (
    binary_accuracy,
    binary_auroc,
    binary_f1_score,
    binary_matthews_corrcoef,
    binary_precision,
    binary_recall,
)
from torchmetrics.utilities.data import to_onehot
from torchvision.models import (
    ResNet50_Weights,
    resnet50,
)


# ## MixUp Functions
# 
# 

# In[2]:


def mixup_data(x, y, alpha=0, verbose=False):
    """Mixes up the input data and targets according to a beta distribution.

    Args:
        x (Tensor): The input data tensor.
        y (Tensor): The target tensor.
        alpha (float, optional): The alpha parameter for the beta distribution. Default is 0.
        verbose (bool, optional): A flag to enable visualization and logging. Default is False.

    Returns:
        Tuple[Tensor, Tensor, Tensor, float]: A tuple containing the mixed input data,
        first set of mixed targets, second set of mixed targets, and lambda value for mixing.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    index = index.type_as(x).long()

    mixed_x = lam * x + (1 - lam) * torch.index_select(x, 0, index)
    y_a, y_b = y, torch.index_select(y, 0, index)

    if verbose:
        import matplotlib.pyplot as plt

        print(y_a[0], y_b[0], lam)
        plt.imshow(x[0, 0].cpu(), cmap="gray")
        plt.show()
        plt.imshow(x[index[0], 0].cpu(), cmap="gray")
        plt.show()
        plt.imshow(mixed_x[0, 0].cpu(), cmap="gray")
        plt.show()

    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Computes the mixup loss based on the original and shuffled labels.

    Args:
        criterion (function): The loss function, e.g., cross-entropy loss.
        pred (Tensor): The predictions from the network.
        y_a (Tensor): The original labels.
        y_b (Tensor): The labels of the shuffled batch.
        lam (float): The lambda value used for mixing up.

    Returns:
        Tensor: The computed mixup loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup_metric(metric, preds, y_a, y_b, lam):
    """Computes the mixup-adjusted metric based on the original and shuffled labels.

    Args:
        metric (function): The metric function to use, e.g., accuracy.
        preds (Tensor): The predictions from the network.
        y_a (Tensor): The original labels.
        y_b (Tensor): The labels of the shuffled batch.
        lam (float): The lambda value used for mixing up.

    Returns:
        float: The computed mixup-adjusted metric value.
    """

    return lam * metric(preds, y_a) + (1 - lam) * metric(preds, y_b)


# ## LightningModule 

# In[3]:


num_class = 2


# In[5]:


# ResNet50 with Integration of MixUp and Discriminative Learning rates
class ResNetTransferLearningDiscriminativeLR(pl.LightningModule):
    def __init__(
        self,
        only_fc,
        max_lr,
        wd,
        lr_mult,
        alpha,
        first_dropout,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.only_fc = only_fc
        self.max_learning_rate = max_lr
        self.weight_decay = wd
        self.lr_mult = lr_mult
        self.first_dropout = first_dropout

        self.alpha = alpha
        self.loss = torch.nn.CrossEntropyLoss(label_smoothing=0.01)
        self.metric = binary_accuracy

        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        num_input_channel = 1
        layer = backbone.conv1
        new_layer = nn.Conv2d(
            in_channels=num_input_channel,
            out_channels=layer.out_channels,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            bias=layer.bias,
        )
        new_layer.weight = nn.Parameter(layer.weight.sum(dim=1, keepdim=True))

        backbone.conv1 = new_layer

        if self.only_fc is True:
            for param in backbone.parameters():
                param.requires_grad = False

        backbone.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(self.first_dropout),
            nn.Linear(1024, 2),
        )

        self.model = backbone

        layer_names = []
        # Populate layer names from the model's named parameters
        for _idx, (name, _param) in enumerate(self.model.named_parameters()):
            layer_names.append(name)
        layer_names.reverse()
        lr = self.max_learning_rate
        lr_mult = self.lr_mult

        parameters = []
        prev_group_name = layer_names[0].split(".")[0]

        # Loop through layer names to update learning rates and collect parameters
        for _idx, name in enumerate(layer_names):
            cur_group_name = name.split(".")[0]  # Extract current group name

            # Update learning rate if group name changes
            if cur_group_name != prev_group_name:
                lr *= lr_mult
            prev_group_name = cur_group_name  # Update previous group name for next iteration

            # print(f'{idx}: lr = {lr:.6f}, {name}')

            # Store parameters and their associated learning rates
            parameters += [
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if n == name and p.requires_grad
                    ],
                    "lr": lr,
                }
            ]
            self.param = parameters
            lrs = [i["lr"] for i in self.param]
            self.lrs = lrs
            self.lrs_base = [i / 8 for i in self.lrs]

    def forward(self, x):
        out = self.model(x)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.param,
            lr=0,  # gets overridden by self.params
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.lrs,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "monitor": "val_acc",
            "strict": True,
            "name": None,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def training_step(self, batch_data, batch_idx):
        x, y = batch_data["image"], batch_data["label"]
        x, y_a, y_b, lam = mixup_data(x, y, self.alpha, verbose=False)
        x, y_a, y_b = map(Variable, (x, y_a, y_b))
        logits = self.forward(x)
        loss = mixup_criterion(self.loss, logits, y_a, y_b, lam)
        preds = torch.argmax(logits, dim=1)
        acc = mixup_metric(self.metric, preds, y_a, y_b, lam)
        self.log("train_loss", loss, sync_dist=True, prog_bar=True)
        self.log("train_acc", acc, sync_dist=True, prog_bar=True)
        return {"loss": loss, "preds": preds, "targets": y}

    def validation_step(self, batch_data, batch_idx):
        val_images, val_labels_dense = batch_data["image"], batch_data["label"]
        preds = self.forward(val_images)
        val_labels = to_onehot(val_labels_dense, num_classes=num_class)
        val_loss = torch.nn.functional.cross_entropy(preds, val_labels_dense, label_smoothing=0.01)

        val_acc = binary_accuracy(preds, val_labels)
        val_F1 = binary_f1_score(preds, val_labels)
        val_precision = binary_precision(preds, val_labels)
        val_recall = binary_recall(preds, val_labels)
        val_MCC = binary_matthews_corrcoef(preds, val_labels)

        self.log_dict(
            {
                "val_acc": val_acc,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_F1": val_F1,
                "val_MCC": val_MCC,
                "val_loss": val_loss,
            },
            sync_dist=True,
            prog_bar=True,
        )

        return {
            "val_loss": val_loss,
            "val_preds": preds,
            "val_targets": val_labels,
            "val_MCC": val_MCC,
        }

    def validation_step_end(self, outputs):
        val_acc = binary_accuracy(outputs["val_preds"], outputs["val_targets"])
        val_F1 = binary_f1_score(outputs["val_preds"], outputs["val_targets"])
        val_precision = binary_precision(outputs["val_preds"], outputs["val_targets"])
        val_recall = binary_recall(outputs["val_preds"], outputs["val_targets"])
        val_MCC = binary_matthews_corrcoef(outputs["val_preds"], outputs["val_targets"])
        auroc = binary_auroc(
            torch.Tensor(outputs["val_preds"].type(torch.float32)),
            outputs["val_targets"],
        )

        self.log_dict(
            {
                "val_acc": val_acc,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_F1": val_F1,
                "val_MCC": val_MCC,
                "val_auroc": auroc,
            },
            sync_dist=True,
            prog_bar=True,
        )
        return {
            "val_acc": val_acc,
            "val_precision": val_precision,
            "val_recall": val_recall,
            "val_F1": val_F1,
            "val_MCC": val_MCC,
            "val_auroc": auroc,
        }

    def predict_step(self, batch_data, batch_idx):
        softmax = nn.Softmax()
        inputs, targets_dense = batch_data["image"], batch_data["label"]
        preds = self.forward(inputs)
        preds = torch.tensor(softmax(preds))
        targets = to_onehot(targets_dense, num_classes=num_class)
        return {"preds": preds, "targets": targets}

