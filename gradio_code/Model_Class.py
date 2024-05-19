import pytorch_lightning as pl
import torch
import torch.nn as nn
import utils
from torchvision.models import resnet50
import torch
from monai.transforms import (
    Compose, Resize, ResizeWithPadOrCrop, 
)
from pytorch_grad_cam import GradCAM
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO

class ResNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()


        backbone = resnet50()
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


        backbone.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0),
            nn.Linear(1024, 2),
        )

        self.model = backbone

    def forward(self, x):
        out = self.model(x)
        return out


val_transforms_416x628 = Compose(
    [
        utils.CustomCLAHE(),  
        Resize(spatial_size=628, mode="bilinear", align_corners=True, size_mode="longest"),
        ResizeWithPadOrCrop(spatial_size=(416, 628)),
    ]
)

checkpoint = torch.load("classification_model.ckpt", map_location=torch.device('cpu'))
model = ResNet()
model.load_state_dict(checkpoint["state_dict"])
model.eval()


def load_and_classify_image(image_path, device):

    gpu_model = model.to(device)
    image = val_transforms_416x628(image_path)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = gpu_model(image)
        prediction = torch.nn.functional.softmax(prediction, dim=1).squeeze(0)
        return prediction.to('cpu'), image.to('cpu')
    

def make_GradCAM(image, device):

    arr = image.numpy().squeeze()
    gpu_model = model.to(device)
    image = image.to(device)
    model.eval()
    target_layers = [gpu_model.model.layer4[-1]]

    
    cam = GradCAM(model=gpu_model, target_layers=target_layers)
    targets = None
    grayscale_cam = cam(
        input_tensor=image,
        targets=targets,
        aug_smooth=False,
        eigen_smooth=True,
    )
    grayscale_cam = grayscale_cam.squeeze()

    jet = plt.colormaps.get_cmap("inferno")
    newcolors = jet(np.linspace(0, 1, 256))
    newcolors[0, :3] = 0
    new_jet = mcolors.ListedColormap(newcolors)
  
    plt.figure(figsize=(10, 10))
    plt.imshow(arr, cmap='gray')
    plt.imshow(grayscale_cam, cmap=new_jet, alpha=0.5)
    plt.axis('off')
    buffer2 = BytesIO()
    plt.savefig(buffer2, format='png', bbox_inches='tight', pad_inches=0)
    buffer2.seek(0)
    gradcam_image = np.array(Image.open(buffer2)).squeeze()
    
    return gradcam_image
