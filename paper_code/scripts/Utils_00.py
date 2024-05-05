#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import SimpleITK as sitk
import skimage
import torch
import torchmetrics
from monai.transforms import Transform
from mpl_toolkits.axes_grid1 import ImageGrid
from pytorch_grad_cam import GradCAM
from sklearn import metrics
from torchmetrics.functional.classification import binary_roc
from tqdm import tqdm


# In[3]:


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


# In[5]:


def visualize_training_metrics(metrics_path):
    """Read a CSV logging file and display it in a plot to visualize training progress.

    Args:
        metrics_path (str): Path to the CSV file containing training metrics.

    Returns:
        None: Displays the plot and prints maximum values for each metric.
    """
    df = pd.read_csv(metrics_path)
    del df["step"]
    df.set_index("epoch", inplace=True)
    sn.relplot(data=df, kind="line")
    plt.ylim(0, 1)
    plt.show()
    max_acc = df.max(axis=0)
    print(f"max values: \n{max_acc}")


# In[ ]:


def make_GradCAMs(image_paths, model, transform):
    """Create GradCAMs for a given list of image paths and a model.

    Args:
        image_paths (list of str): List of file paths to the images for which GradCAMs are to be created.
        model (torch.nn.Module): PyTorch model for which GradCAMs are generated.
        transform (torchvision.transforms.Compose): Transform pipeline for preprocessing images.

    Returns:
        list of tuple: List containing tuples of original images and their corresponding GradCAMs.

    """
    model.eval()
    target_layers = [model.model.layer4[-1]]

    visualization_list = []

    pbar = tqdm(total=len(image_paths))
    for i in image_paths:
        image_only = transform(i)
        image_only = image_only.unsqueeze(0)
        arr = image_only.numpy().squeeze()
        arr = arr[..., None]
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
        targets = None
        grayscale_cam = cam(
            input_tensor=image_only,
            targets=targets,
            aug_smooth=False,
            eigen_smooth=True,
        )
        grayscale_cam = grayscale_cam[0, :]
        visualization_list.append((arr, grayscale_cam))
        pbar.update(1)

    pbar.close()

    return visualization_list


# In[ ]:


def visualize_in_grid(image_list, num_col=4, save_path=None):
    """Visualize input GradCAMs as an image grid for inspection.

    Args:
        image_list (list of np.ndarray): List containing GradCAM images.
        num_col (int, optional): Number of columns in the grid. Defaults to 4.
        save_path (str, optional): File path to save the image grid. Defaults to None.

    Returns:
        None: Displays the image grid and optionally saves it to a file.

    """
    print(f"Displaying {len(image_list)} images in the grid:")
    fig = plt.figure(figsize=(20, 40))
    jet = plt.colormaps.get_cmap("inferno")
    newcolors = jet(np.linspace(0, 1, 256))
    newcolors[0, :3] = 0
    new_jet = mcolors.ListedColormap(newcolors)
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(
            round(len(image_list) / num_col) + 1,
            num_col,
        ),
        axes_pad=0.1,
    )
    pbar = tqdm(total=len(image_list))
    for ax, im in zip(grid, image_list):
        ax.grid(False)
        ax.axis("off")
        ax.imshow(im[0], cmap="gray")
        ax.imshow(im[1], alpha=0.5, cmap=new_jet)
        pbar.update(1)
    pbar.close()
    if save_path:
        plt.savefig(save_path, dpi=300)


# In[8]:


def classification_report(preds, targets, threshhold=0.5):
    """Generate and print a classification report.

    Args:
        preds (torch.Tensor): Tensor containing the predicted probabilities.
        targets (torch.Tensor): Tensor containing the true target values.
        threshhold (float, optional): Classification threshold for predictions. Defaults to 0.5.

    Returns:
        None: Prints the classification report to the console.

    """
    threshold = threshhold

    targets_array = targets[:,1].numpy().astype(int)
    predictions_array = np.where(preds[:,1].numpy()>threshold, 1, 0)

    print(metrics.classification_report(y_true=targets_array, y_pred=predictions_array))


# In[ ]:


def class_probs_hist(preds, targets, threshhold, save_path=None, legend_pos="best", y_ticks=10):
    """Visualize the model output probabilities for each class using a histogram.

    Args:
        preds (torch.Tensor): Tensor containing the predicted probabilities.
        targets (torch.Tensor): Tensor containing the true target values.
        threshhold (float): Cut-off value for classification.
        save_path (str, optional): File path to save the plot. Defaults to None.
        legend_pos (str or tuple, optional): The position of the legend. Defaults to "best".
        y_ticks (int): Spacing of the ticks on the y-axis

    Returns:
        None: Displays the histogram plot and optionally saves it to a file.

    """
    cut_off = threshhold
    data = pd.DataFrame({"prob_preds": preds[:, 1].numpy(), "targs": targets[:, 1].numpy()})

    plt.style.use("default")
    hist_plot = sn.histplot(
        data=data,
        x="prob_preds",
        hue="targs",
        bins=20,
        multiple="dodge",
        palette={0: "#005AB5", 1: "#DC3220"},

    )  # , edgecolor=None)
    ymax = np.max([p.get_height() for p in hist_plot.patches])
    plt.vlines(x=cut_off, color="gray", linestyle="--", ymin=0, ymax=ymax)

    plt.xlabel("Probabillities", fontsize=16)
    plt.ylabel('Count', fontsize=16)

    plt.yticks(np.arange(0, plt.ylim()[1], y_ticks))

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)

    plt.legend(labels=["cut-off value", "1", "0"], frameon=False,fontsize=12, loc=legend_pos)

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()


# In[ ]:


def Sensitivity_vs_FPR_val(preds, targets, threshhold, save_path=None, loc=None):
    """Create a plot of sensitivity versus false positive rate (FPR) based on varying threshold values.

    Args:
        preds (torch.Tensor): Tensor containing the predicted values.
        targets (torch.Tensor): Tensor containing the true target values.
        threshhold (float): Cut-off value for classification.
        save_path (str, optional): File path to save the plot. Defaults to None.
        loc (str, optional): Position of the Legend in the graph. Defaults to None.

    Returns:
        None: Displays the plot of sensitivity versus FPR and optionally saves it to a file.

    """
    cut_off = threshhold
    thresholds = np.arange(0.0, 1.01, 0.01)

    recall_list = []
    FPR_list = []
    for i in thresholds:
        recall_list.append(
            torchmetrics.functional.classification.binary_recall(preds[:,1], targets[:,1].long(), threshold=i)
        )
        FPR_list.append(
            1
            - (
                torchmetrics.functional.classification.binary_specificity(
                    preds[:,1], targets[:,1].long(), threshold=i
                )
            )
        )

    y_max = np.interp(cut_off, thresholds, recall_list)
    y_min = np.interp(cut_off, thresholds, FPR_list)

    fig, ax = plt.subplots()
    plt.style.use("default")

    ax.plot(thresholds, recall_list, linewidth=2, c="#005AB5")
    ax.plot(thresholds, FPR_list, linewidth=2, c="#DC3220")
    ax.vlines(x=cut_off, color="gray", linestyle="--", ymax=y_max, ymin=y_min)

    ax.set_ylabel("Sensitivity / False positive rate",fontsize=16)
    ax.set_xlabel("cut-off value",fontsize=16)
    
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)

    ax.legend(["Sensitivity", "False positive rate", "cut-off value"], frameon=False, loc=loc, fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


# In[ ]:


def confusion_matrix(preds, targets, threshhold=0.5, save_path=None):
    """Create and visualize a confusion matrix.

    Args:
        preds (torch.Tensor): Tensor containing the predicted values.
        targets (torch.Tensor): Tensor containing the true target values.
        threshhold (float, optional): Classification threshold for predictions. Defaults to 0.5.
        save_path (str, optional): File path to save the plot. Defaults to None.

    Returns:
        None: Displays the confusion matrix and optionally saves it to a file.

    """
    threshold = threshhold

    targets_array = targets[:,1].numpy().astype(int)
    predictions_array = np.where(preds[:,1].numpy()>threshold, 1, 0)

    confusion_matrix = metrics.confusion_matrix(y_true=targets_array, y_pred=predictions_array)
    class_labels = ["nr-AxSpA", "r-AxSpA"]

    plt.style.use("default")
    plt.figure(figsize=(8, 6))
    ax = sn.heatmap(
        confusion_matrix,
        annot=True,
        cmap="Blues",
        fmt="d",
        xticklabels=class_labels,
        yticklabels=class_labels,
        cbar=True,
        annot_kws={"size": 16}
    )
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=16)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)

    plt.title("Confusion matrix",fontsize=20)
    plt.xlabel("Prediction",fontsize=18)
    plt.ylabel("Ground truth",fontsize=18)

    if save_path:
        plt.savefig(save_path, dpi=300)

    plt.show()


# In[10]:


def roc_plot(preds, targets, save_path=None):
    """Plot the Receiver Operating Characteristic (ROC) curve and calculate the area under the curve (AUC).

    Args:
        preds (torch.Tensor): Tensor containing the predicted values.
        targets (torch.Tensor): Tensor containing the true target values.
        save_path (str, optional): File path to save the plot. Defaults to None.

    Returns:
        None: Displays the plot and optionally saves it to a file.

    """

    targets_array = targets[:,1].numpy().astype(int)
    predictions_array = preds[:,1].numpy()

    roc = metrics.roc_curve(targets_array, predictions_array)

    auc = round(metrics.roc_auc_score(targets_array, predictions_array), 3)

    p = [0, 1]
    i = [0, 1]

    fig, ax = plt.subplots()
    plt.style.use("default")
    ax.plot(roc[0], roc[1], linestyle="-", c="#DC3220", linewidth=2)
    ax.plot(p, i, linestyle="dotted", c="0.5")

    ax.set_title("Receiver operating characteristics curve", loc="center",fontsize=20)
    ax.set_xlabel("False positive rate",fontsize=18)
    ax.set_ylabel("True positive rate",fontsize=18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
    plt.text(
        0.955,
        0.05,
        f"AUC = {str(auc)}",
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=plt.gca().transAxes,
        fontsize=16
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


# In[5]:


def metric_w_CI(preds, metric, metric_name=None, threshold=0.5, return_values=False):
    """Calculate confidence intervals with bootstrapping for an input metric.

    Args:
        preds (dict): A dictionary of tensors containing the predictions and targets.
            Expected keys are "preds" and "targets".
        metric (torchmetrics.Metric): A torchmetrics metric function for which the confidence interval should be calculated.
        metric_name (str, optional): The name to be displayed next to the metric. Defaults to None.
        return_values (bool, optional): Whether to return the calculated values. Defaults to False.

    Returns:
        dict or None: If return_values is True, returns a dictionary containing the metric, lower bound, and upper bound.
        Otherwise, prints these values and returns None.

    """

    bootstrapped_metric = torchmetrics.wrappers.BootStrapper(
        metric,
        num_bootstraps=1000,
        mean=False,
        std=False,
        raw=True,
        sampling_strategy="multinomial",
    )
    if metric_name== "AUROC":
        predictions = preds["preds"][:,1]
    else:
        predictions = torch.where(preds["preds"][:,1]>threshold,1,0)
    targets = preds["targets"][:,1]
    bootstrapped_metric.update(predictions, targets)
    metric_boot_result = bootstrapped_metric.compute()
    metric_result = round(float(metric(predictions, targets)), 3)

    lower_bound = np.percentile(metric_boot_result["raw"], 2.5)
    upper_bound = np.percentile(metric_boot_result["raw"], 97.5)

    if metric_name is None:
        metric_name = str(metric)

    if return_values is True:
        return {
            metric_name: metric_result,
            "lower_bound": round(lower_bound, 3),
            "upper_bound": round(upper_bound, 3),
        }

    else:
        print(
            f"{metric_name}: {metric_result} (95% CI: {round(lower_bound,3)}, {round(upper_bound, 3)})"
        )


# In[ ]:


def size_comparison(
    first_set,
    second_set,
    labels=None,
    save_path=None,
    return_res=False,
):
    """Compares the sizes of images in two sets, plots the distribution, and optionally returns resolutions.

    Args:
        first_set (list): A list of file paths for the images in the first set.
        second_set (list): A list of file paths for the images in the second set.
        labels (list, optional): A list of two strings representing labels for the first and second sets in the plot.
            Defaults to ['first set', 'second set'].
        save_path (str, optional): The file path where the plot should be saved. If None, the plot is not saved.
            Defaults to None.
        return_res (bool, optional): A flag to indicate whether the resolutions of the images in both sets should be returned.
            Defaults to False.

    Returns:
        tuple or None: If return_res is True, returns a tuple of two lists containing the resolutions of the images in the
                       first and second sets.
                       Otherwise, displays a plot and optionally saves it, but does not return any values.
    """
    if labels is None:
        labels = ["first set", "second set"]

    resolutions_list_first_set = []
    for i in first_set[:30]:
        reader = sitk.ImageFileReader()
        reader.SetFileName(i)
        reader.ReadImageInformation()
        size = reader.GetSize()
        res = np.sqrt(size[0] * size[1])
        resolutions_list_first_set.append(res)

    resolutions_list_second_set = []
    for i in second_set[:30]:
        reader = sitk.ImageFileReader()
        reader.SetFileName(i)
        reader.ReadImageInformation()
        size = reader.GetSize()
        res = np.sqrt(size[0] * size[1])
        resolutions_list_second_set.append(res)

    size_data = pd.DataFrame(
        {
            "values": resolutions_list_first_set + resolutions_list_second_set,
            "type": ["top losses"] * len(resolutions_list_first_set)
            + ["min losses"] * len(resolutions_list_second_set),
        }
    )

    plt.style.use("default")
    sn.histplot(
        data=size_data,
        x="values",
        hue="type",
        bins=20,
        multiple="dodge",
        palette={"top losses": "#DC3220", "min losses": "#005AB5"},
    )
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.legend(labels, frameon=False)

    if return_res:
        return resolutions_list_first_set, resolutions_list_second_set

    else:
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

