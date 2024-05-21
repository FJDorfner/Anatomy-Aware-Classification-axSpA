from monai.transforms import Transform, Compose, LoadImage, EnsureChannelFirst
import torch
import skimage
import torch
import SimpleITK as sitk
import numpy as np
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import SimpleITK as sitk
from matplotlib.colors import ListedColormap
import base64
import numpy as np
from cv2 import dilate
from scipy.ndimage import label
from Model_Seg import RgbaToGrayscale

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

class CustomCLAHE(Transform):
    """Implements Contrast-Limited Adaptive Histogram Equalization (CLAHE) as a custom transform, as described by Qiu et al.

    Attributes:
        p1 (float): Weighting factor, determines degree of of contour enhacement. Default is 0.6.
        p2 (None or int): Kernel size for adaptive histogram. Default is None.
        p3 (float): Clip limit for histogram equalization. Default is 0.01.

    """

    def __init__(self, p1=0.6, p2=None, p3=0.01):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3

    def __call__(self, data):
        """Apply the CLAHE algorithm to input data.

        Args:
            data (Union[dict, np.ndarray]): Input data. Could be a dictionary containing the image or the image array itself.

        Returns:
            torch.Tensor: Transformed data.
        """
        
        if isinstance(data, dict):
            im = data["image"]

        else:
            im = data
        im = im.numpy()
        

        # remove the first dimension
        im = im[0]
        im = im[None, :, :]
        #im = np.expand_dims(im, axis=0)
        im = skimage.exposure.rescale_intensity(im, in_range="image", out_range=(0, 1))
        im_noi = skimage.filters.median(im)
        im_fil = im_noi - self.p1 * skimage.filters.gaussian(im_noi, sigma=1)
        im_fil = skimage.exposure.rescale_intensity(im_fil, in_range="image", out_range=(0, 1))
        im_ce = skimage.exposure.equalize_adapthist(im_fil, kernel_size=self.p2, clip_limit=self.p3)
        if isinstance(data, dict):
            data["image"] = torch.Tensor(im_ce)
        else:
            data = torch.Tensor(im_ce)
            
        return data



def custom_colormap():

    cdict = [(0, 0, 0, 0),    # Class 0 - fully transparent (background)
             (0, 1, 0, 0.5),  # Class 1 - Green with 50% transparency 
             (1, 0, 0, 0.5),  # Class 2 - Red with 50% transparency 
             (1, 1, 0, 0.5)]  # Class 3 - Yellow with 50% transparency 
    cmap = ListedColormap(cdict)
    return cmap

def read_image(image_path):
    read_transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        RgbaToGrayscale(),  # Convert RGBA to grayscale
    ])
    try:
        original_image = read_transforms(image_path)
        original_image_np = original_image.numpy().astype(np.uint8)
        return original_image_np.squeeze()

    except Exception as e:
        try :
            original_image = sitk.ReadImage(image_path)
            original_image_np = sitk.GetArrayFromImage(original_image)
            return original_image_np.squeeze()
        except Exception as e:
            print("Failed Loading the Image: ", e)
            return None

def overlay_mask(image_path, image_mask):
    original_image_np = read_image(image_path).squeeze().astype(np.uint8)

    #adjust mask intensities for display
    image_mask_disp = image_mask
    plt.figure(figsize=(10, 10))
    plt.imshow(original_image_np, cmap='gray')

    plt.imshow(image_mask_disp, cmap=custom_colormap(), alpha=0.5)
    plt.axis('off')

    # Save the overlay to a buffer
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0)
    buffer.seek(0)
    overlay_image_np = np.array(Image.open(buffer))
    return overlay_image_np, original_image_np


def bounding_box_mask(image, label):
    """Generates a bounding box mask around a labeled region in an image

    Args:
        image (SimpleITK.Image): The input image.
        label (SimpleITK.Image): The labeled image containing the region of interest.

    Returns:
        SimpleITK.Image: An image containing the with the bounding box mask applied with the
        same spacing as the original image.

    Note:
        This function assumes that the input image and label are SimpleITK.Image objects.
        The returned bounding box mask is a binary image where pixels inside the bounding box
        are set to 1 and others are set to 0.
    """
    # get original spacing
    original_spacing = image.GetSpacing()

    # convert image and label to arrays
    image_array = sitk.GetArrayFromImage(image)
    image_array = np.squeeze(image_array)
    label_array = sitk.GetArrayFromImage(label)
    label_array = np.squeeze(label_array)

    # determine corners of the bounding box
    first_nonzero_row_index = np.nonzero(np.any(label_array != 0, axis=1))[0][0]
    last_nonzero_row_index = np.max(np.nonzero(np.any(label_array != 0, axis=1)))
    first_nonzero_column_index = np.nonzero(np.any(label_array != 0, axis=0))[0][0]
    last_nonzero_column_index = np.max(np.nonzero(np.any(label_array != 0, axis=0)))

    top_left_corner = (first_nonzero_row_index, first_nonzero_column_index)
    bottom_right_corner = (last_nonzero_row_index, last_nonzero_column_index)

    # define the bounding box as an array mask
    bounding_box_array = label_array.copy()
    bounding_box_array[
        top_left_corner[0] : bottom_right_corner[0] + 1,
        top_left_corner[1] : bottom_right_corner[1] + 1,
    ] = 1
    
    # add channel dimension
    bounding_box_array = bounding_box_array[None, ...].astype(np.uint8)

    # get Image from Array Mask and apply original spacing
    bounding_box_image = sitk.GetImageFromArray(bounding_box_array)
    bounding_box_image.SetSpacing(original_spacing)
    return bounding_box_image


def threshold_based_crop(image):
    """
    Use Otsu's threshold estimator to separate background and foreground. In medical imaging the background is
    usually air. Then crop the image using the foreground's axis aligned bounding box.
    Args:
        image (SimpleITK image): An image where the anatomy and background intensities form a
                                 bi-modal distribution
                                 (the assumption underlying Otsu's method.)
    Return:
        Cropped image based on foreground's axis aligned bounding box.
    """

    inside_value = 0
    outside_value = 255
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    # uncomment for debugging
    #sitk.WriteImage(image, "./image.png")
    label_shape_filter.Execute(sitk.OtsuThreshold(image, inside_value, outside_value))
    bounding_box = label_shape_filter.GetBoundingBox(outside_value)
    return sitk.RegionOfInterest(
        image,
        bounding_box[int(len(bounding_box) / 2) :],
        bounding_box[0 : int(len(bounding_box) / 2)],
    )

def creat_SIJ_mask(image, input_label):
    """
    Create a mask for the sacroiliac joints (SIJ) from pelvis and sascrum segmentation mask

    Args:
        image (SimpleITK.Image): x-ray image.
        input_label (SimpleITK.Image): Segmentation mask containing labels for sacrum, left- and right pelvis

    Returns:
        SimpleITK.Image: Mask of the SIJ

    """
    
    original_spacing = image.GetSpacing()
    # uncomment for debugging
    #sitk.WriteImage(input_label, "./input_label.png")
    mask_array = sitk.GetArrayFromImage(input_label).squeeze()
    
    sacrum_value = 1  
    left_pelvis_value = 2  
    right_pelvis_value = 3  
    background_value = 0

    
    sacrum_mask = (mask_array == sacrum_value)

    first_nonzero_column_index = np.nonzero(np.any(sacrum_mask != 0, axis=0))[0][0]
    last_nonzero_column_index = np.max(np.nonzero(np.any(sacrum_mask != 0, axis=0)))
    box_width=last_nonzero_column_index-first_nonzero_column_index

    dilation_extent = int(np.round(0.05 * box_width))

    dilated_sacrum_mask = dilate_mask(sacrum_mask, dilation_extent)

    intersection_left = (dilated_sacrum_mask & (mask_array == left_pelvis_value))
    if np.all(intersection_left == 0):
        print("Warning: No left intersection")
        left_pelvis_mask = (mask_array == 2)
        intersection_left = create_median_height_array(left_pelvis_mask)
        
    intersection_left = keep_largest_component(intersection_left)
    
    intersection_right = (dilated_sacrum_mask & (mask_array == right_pelvis_value))
    if np.all(intersection_right == 0):
        print("Warning: No right intersection")
        right_pelvis_mask = (mask_array == 3)
        intersection_right = create_median_height_array(right_pelvis_mask)
    intersection_right = keep_largest_component(intersection_right)
    
    intersection_mask = intersection_left +intersection_right
    intersection_mask = intersection_mask[None, ...]
                                    
    instersection_mask_im = sitk.GetImageFromArray(intersection_mask)
    instersection_mask_im.SetSpacing(original_spacing)
    return instersection_mask_im

def dilate_mask(mask, extent):
    """
    Keeps only the largest connected component in a binary segmentation mask.

    Args:
        mask (numpy.ndarray): A numpy array representing the binary segmentation mask, 
                              with 1s indicating the label and 0s indicating the background.

    Returns:
        numpy.ndarray: A modified version of the input mask, where only the largest 
                       connected component is retained, and other components are set to 0.

    """
    mask_uint8 = mask.astype(np.uint8)

    kernel = np.ones((2*extent+1, 2*extent+1), np.uint8)
    dilated_mask = dilate(mask_uint8, kernel, iterations=1)
    return dilated_mask

def mask_and_crop(image, input_label):
    """
    Performs masking and cropping operations on an image and its label.

    Args:
        image (SimpleITK.Image): The image to be processed.
        label (SimpleITK.Image): The corresponding label image.

    Returns:
        tuple: A tuple containing two SimpleITK.Image objects.
            - cropped_boxed_image: The image after applying bounding box masking and cropping.
            - mask: The binary mask corresponding to the label after cropping.

    Note:
        This function relies on other functions: bounding_box_mask() and threshold_based_crop().
    """
    input_label = creat_SIJ_mask(image,input_label)
    box_mask = bounding_box_mask(image, input_label)
    
    boxed_image = sitk.Mask(image, box_mask, maskingValue=0, outsideValue=0)
    masked_image = sitk.Mask(image, input_label, maskingValue=0, outsideValue=0)

    cropped_boxed_image = threshold_based_crop(boxed_image)
    cropped_masked_image = threshold_based_crop(masked_image)

    mask = np.squeeze(sitk.GetArrayFromImage(cropped_masked_image))
    mask = np.where(mask > 0, 1, 0)
    mask = sitk.GetImageFromArray(mask[None, ...])
    return cropped_boxed_image, mask

def create_median_height_array(mask):
    """
    Creates an array based on the median height of non-zero elements in each column of the input mask.

    Args:
        mask (numpy.ndarray): A binary mask with 1s representing the label and 0s the background.

    Returns:
        numpy.ndarray: A new binary mask array with columns filled based on the median height,
                       or None if the input mask has no non-zero columns.
                       
    Note: 
        This function is only used when there is no intersection between pelvis and sacrum, and creates an alternative
        SIJ mask, that serves as an approximate replacement.
    """
    rows, cols = mask.shape
    column_details = []

    for col in range(cols):
        column_data = mask[:, col]
        non_zero_indices = np.nonzero(column_data)[0]
        if non_zero_indices.size > 0:
            height = non_zero_indices[-1] - non_zero_indices[0] + 1
            start_idx = non_zero_indices[0]
            column_details.append((height, start_idx, col))
            
    if not column_details:
        return None  
    median_height = round(np.median([h[0] for h in column_details]))
    median_cols = [(col, start_idx) for height, start_idx, col in column_details if height == median_height]
    new_array = np.zeros_like(mask, dtype=int)
    for col, start_idx in median_cols:
        start_col = max(0, col - 5)
        end_col = min(cols, col + 5)
        new_array[start_idx:start_idx + median_height, start_col:end_col] = 1
    return new_array

def keep_largest_component(mask):
    """
    Identifies and retains the largest connected component in a binary segmentation mask.

    Args:
        mask (numpy.ndarray): A binary mask with 1s representing the label and 0s the background.

    Returns:
        numpy.ndarray: The modified mask with only the largest connected component.
    """
    # Label the connected components
    labeled_array, num_features = label(mask)

    # If no features are found, return the original mask
    if num_features <= 1:
        return mask

    # Find the largest connected component
    largest_component = np.argmax(np.bincount(labeled_array.flat)[1:]) + 1

    # Generate the mask for the largest component
    return (labeled_array == largest_component).astype(mask.dtype)