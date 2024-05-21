import gradio as gr
import utils
import Model_Class 
import Model_Seg

import SimpleITK as sitk
import torch
from numpy import uint8, rot90, fliplr
from monai.transforms import Rotate90

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_base64 = utils.image_to_base64("anatomy_aware_pipeline.png")
article_html = f"<img src='data:image/png;base64,{image_base64}' alt='Anatomical pipeline illustration' style='width:100%;'>"

description_markdown = """
- This tool combines a U-Net Segmentation Model with a ResNet-50 for Classification.
- **Usage:** Just drag a pelvic x-ray into the box and hit run.
- **Process:** The input image will be segmented and cropped to the SIJ before classification.
- **Please Note:** This tool is intended for research purposes only.
- **Privacy:** This tool runs completely locally, ensuring data privacy.
"""

css = """

h1 {
    text-align: center;
    display:block;
}
.markdown-block {
    background-color: #0b0f1a; /* Light gray background */
    color: black;             /* Black text */
    padding: 10px;            /* Padding around the text */
    border-radius: 5px;       /* Rounded corners */
    box-shadow: 0 0 10px rgba(11,15,26,1);
    display: inline-flex;       /* Use inline-flex to shrink to content size */
    flex-direction: column;
    justify-content: center;    /* Vertically center content */
    align-items: center;        /* Horizontally center items within */
    margin: auto;               /* Center the block */
}

.markdown-block ul, .markdown-block ol {
    background-color: #1e2936;
    border-radius: 5px;
    padding: 10px;
    box-shadow: 0 0 10px rgba(0,0,0,0.3);
    padding-left: 20px;         /* Adjust padding for bullet alignment */
    text-align: left;           /* Ensure text within list is left-aligned */
    list-style-position: inside;/* Ensures bullets/numbers are inside the content flow */
}

footer {
    display:none !important
}
"""

def predict_image(input_image, input_file):

    if input_image is not None:
        image_path = input_image

    elif input_file is not None:
        image_path = input_file
    
    else:
        return None , None , "Please input an image before pressing run" , None , None

    image_mask = Model_Seg.load_and_segment_image(image_path, device)

    overlay_image_np, original_image_np = utils.overlay_mask(image_path, image_mask)
    overlay_image_np = rot90(overlay_image_np, k=3)
    overlay_image_np = fliplr(overlay_image_np)

    image_mask_im = sitk.GetImageFromArray(image_mask[None, :, :].astype(uint8))
    image_im = sitk.GetImageFromArray(original_image_np[None, :, :].astype(uint8))
    cropped_boxed_im, _ = utils.mask_and_crop(image_im, image_mask_im)

    cropped_boxed_array = sitk.GetArrayFromImage(cropped_boxed_im)
    cropped_boxed_tensor = torch.Tensor(cropped_boxed_array)
    rotate = Rotate90(spatial_axes=(0, 1), k=3)

    cropped_boxed_tensor = rotate(cropped_boxed_tensor)
    cropped_boxed_array_disp = cropped_boxed_tensor.numpy().squeeze().astype(uint8)
    prediction, image_transformed = Model_Class.load_and_classify_image(cropped_boxed_tensor, device)


    gradcam = Model_Class.make_GradCAM(image_transformed, device)
    
    nr_axSpA_prob = float(prediction[0].item())
    r_axSpA_prob = float(prediction[1].item())

    # Decision based on the threshold
    considered = "be considered r-axSpA" if r_axSpA_prob > 0.59 else "not be considered r-axSpA"

    explanation = f"According to the pre-determined cut-off threshold of 0.59, the image should  {considered}. This Tool is for research purposes only."

    pred_dict = {"nr-axSpA": nr_axSpA_prob, "r-axSpA": r_axSpA_prob}

    return  overlay_image_np, pred_dict, explanation, gradcam, cropped_boxed_array_disp




with gr.Blocks(css=css, title="Anatomy Aware axSpA") as iface:

    gr.Markdown("# Anatomy-Aware Image Classification for radiographic axSpA")
    gr.Markdown(description_markdown, elem_classes="markdown-block")

    with gr.Row():
        with gr.Column():

            with gr.Tab("PNG/JPG"):
                input_image = gr.Image(type='filepath', label="Upload an X-ray Image")

            with gr.Tab("NIfTI/DICOM"):
                input_file = gr.File(type='filepath', label="Upload an X-ray Image")

            with gr.Row():
                submit_button = gr.Button("Run", variant="primary")
                clear_button = gr.ClearButton()

        with gr.Column():
            overlay_image_np = gr.Image(label="Segmentation Mask")

            pred_dict = gr.Label(label="Prediction")
            explanation= gr.Textbox(label="Classification Decision")

            with gr.Accordion("Additional Information", open=False):
                gradcam = gr.Image(label="GradCAM")
                cropped_boxed_array_disp = gr.Image(label="Bounding Box")
    
    submit_button.click(predict_image, inputs = [input_image, input_file], outputs=[overlay_image_np, pred_dict, explanation, gradcam, cropped_boxed_array_disp])
    clear_button.add([input_image,overlay_image_np, pred_dict, explanation, gradcam, cropped_boxed_array_disp])
    gr.HTML(article_html)


if __name__ == "__main__":
    iface.queue()
    iface.launch(server_name='0.0.0.0', server_port=8080)
