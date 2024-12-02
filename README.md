# Anatomy-Aware-Classification-axSpA
This is the accompanying repository for the Paper **Incorporating Anatomical Awareness for Enhanced Generalizability and Progression Prediction in Deep Learning-Based Radiographic Sacroiliitis Detection**.

## Background
The assessment of radiographic sacroiliitis on pelvic radiographs plays an important role in the diagnosis and classifcation of axial Spondyloarthritis (axSpA). We were able to show that incorporating anatomical awareness into a CNN used for the detection of radiographic sacroiliitis, can improve the generalizabilty of the model and enable it to identify patients who are at a higher risk of progression to radiographic sacroiliitis. Read more about it in our preprint: https://arxiv.org/abs/2405.07369v1

## Gradio App
We have integrated the complete Anatomy-Aware pipeline into a Gradio app that can be easily run locally via Docker. Here's what running the app looks like:
![Demonstration of Classifying an x-ray image using the gradio app](/gradio_app.gif)

### Running the Gradio App Locally

1. Prerequisites: 
- Ensure you have Docker installed [Get Docker](https://docs.docker.com/get-docker/)
- There is no need for a GPU, the app can run on CPU and as little as 1.5GB of RAM.

2. Steps to Run:
- Clone the repository:
 `git clone https://github.com/FJDorfner/Anatomy-Aware-Classification-axSpA.git`
- Navigate to the Gradio directory: `cd Anatomy-Aware-Classification-axSpA/gradio_code`
- Build the Docker image: `docker build -t axspa-classifier . `    
- Start the Docker container: `docker run -p 8080:8080 axspa-classifier`
- Open your favorite web browser and go to `localhost:8080` to use the app as demonstrated above.
- Hint: Click **Additional Information** in the Interface to see a Grad-CAM as well.

## Paper Code
All code used in the experiments described in the paper can be found under _paper_code/nbs_. 
The Jupyter notebooks are organized sequentially as they were used in the study.

## Paper
```
@article{dorfner2024incorporating,
      title={Incorporating Anatomical Awareness for Enhanced Generalizability and Progression Prediction in Deep Learning-Based Radiographic Sacroiliitis Detection}, 
      author={Felix J. Dorfner and Janis L. Vahldiek and Leonhard Donle and Andrei Zhukov and Lina Xu and Hartmut Häntze and Marcus R. Makowski and Hugo J. W. L. Aerts and Fabian Proft and Valeria Rios Rodriguez and Judith Rademacher and Mikhail Protopopov and Hildrun Haibel and Torsten Diekhoff and Murat Torgutalp and Lisa C. Adams and Denis Poddubnyy and Keno K. Bressem},
      year={2024},
      eprint={2405.07369},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

