# Use an official Python image as the base
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /workspace

# Set environment variables to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install system dependencies and Jupyter
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    ca-certificates \
    git \
    && pip install --upgrade --no-cache-dir jupyterlab \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install m3_learning repo
RUN rm -rf /workspace/m3_learning
RUN mkdir -p /workspace && git clone --single-branch --branch Northwestern-H100-Multimodal https://github.com/zhangxinqiao314/m3_learning.git /workspace/m3_learning
# Install packages
RUN pip install --no-cache-dir -r /workspace/m3_learning/m3_learning/src/requirements.txt

# install Gaussian_Sampler packages
COPY requirements.txt /workspace/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . /workspace

# Expose the Jupyter Notebook port
EXPOSE 8888

# Define the command to run Jupyter Notebook
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# CMD bash run_docker_container.bash