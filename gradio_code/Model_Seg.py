import torch
import numpy as np
from monai.transforms import Compose, LoadImage, EnsureChannelFirst, Lambda, Resize, NormalizeIntensity, GaussianSmooth, ScaleIntensity, AsDiscrete, KeepLargestConnectedComponent, Invert, Rotate90, SaveImage, Transform
from monai.inferers import SlidingWindowInferer
from monai.networks.nets import UNet

class RgbaToGrayscale(Transform):
    def __call__(self, x):
        # squeeze last dimension, to ensure C, H, W format
        x = x.squeeze(-1)
        # Ensure the tensor is 3D (channels, height, width)
        if x.ndim != 3:
            raise ValueError(f"Input tensor must be 3D. Shape: {x.shape}")
        
        # Check the number of channels
        if x.shape[0] == 4:  # Assuming RGBA
            rgb_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=x.device)
            # Apply weights to RGB channels, output should retain one channel dimension
            grayscale = torch.einsum('cwh,c->wh', x[:3, :, :], rgb_weights).unsqueeze(0)
        elif x.shape[0] == 3:  # Assuming RGB
            rgb_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=x.device)
            grayscale = torch.einsum('cwh,c->wh', x, rgb_weights).unsqueeze(0)
        elif x.shape[0] == 1:  # Already grayscale
            grayscale = x
        else:
            raise ValueError(f"Unsupported channel number: {x.shape[0]}")
        return grayscale

    def inverse(self, x):
        # Simply return the input as the output
        return x
    
model = UNet(
    spatial_dims=2,
    in_channels=1,
    out_channels=4,
    channels=[64, 128, 256, 512],
    strides=[2, 2, 2],
    num_res_units=3
)

checkpoint_path = 'segmentation_model.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu')  
assert model.state_dict().keys() == checkpoint['network'].keys(), "Model and checkpoint keys do not match"

model.load_state_dict(checkpoint['network'])
model.eval()

# Define transforms for preprocessing
pre_transforms = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    RgbaToGrayscale(),  # Convert RGBA to grayscale
    Resize(spatial_size=(768, 768)), 
    Lambda(func=lambda x: x.squeeze(-1)),  # Adjust if the input image has an extra unwanted dimension
    NormalizeIntensity(),
    GaussianSmooth(sigma=0.1),
    ScaleIntensity(minv=-1, maxv=1)
])



# Define transforms for postprocessing
post_transforms = Compose([
    AsDiscrete(argmax=True, to_onehot=4),
    KeepLargestConnectedComponent(),
    AsDiscrete(argmax=True),
    Invert(pre_transforms),
    #SaveImage(output_dir='./', output_postfix='seg', output_ext='.nii', resample=False)
])



def load_and_segment_image(input_image_path, device):
    image_tensor = pre_transforms(input_image_path)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    # Inference using SlidingWindowInferer
    inferer = SlidingWindowInferer(roi_size=(512, 512), sw_batch_size=16, overlap=0.75)
    with torch.no_grad():
        outputs = inferer(image_tensor, model.to(device))


    outputs = outputs.squeeze(0)

    processed_outputs = post_transforms(outputs).to('cpu')  

    output_array = processed_outputs.squeeze().detach().numpy().astype(np.uint8)


    return output_array