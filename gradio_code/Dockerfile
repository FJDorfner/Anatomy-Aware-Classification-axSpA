# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /app

# Install system dependencies including curl
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Download the model files from Zenodo using curl
RUN curl -o classification_model.ckpt -L https://zenodo.org/record/11116990/files/classification_model.ckpt
RUN curl -o segmentation_model.pt -L https://zenodo.org/record/11116990/files/segmentation_model.pt

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements_small.txt 
RUN pip install -r /app/requirements_small.txt && \
    pip uninstall opencv-python -y && \
    pip install opencv-contrib-python-headless==4.5.5.64

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Run app.py when the container launches
CMD ["python", "app.py"]