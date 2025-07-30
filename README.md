# Eustachian Monument
**Image Restoring and Reconstruction**
---
**Abstract**
This project explores the possibilities of LLM-driven generation approach for geometrical understanding and spatial-wise transformation of images. We also enrich the transformed image with rare concept details, with high quality results by using [Image RAG](https://rotem-shalev.github.io/ImageRAG) and Conditional LDM generation.

## Getting Started

> **Please Note!**
>
> Make sure you are supporting CUDA. Otherwise, this code won't work.

In order to execute the code, you need to go through the following steps:

* Clone this repo:
    ```bash
    git clone https://github.com/sandangl/eustachian_monument.git
    ```
* Create a Python Virtual Environment:
    - By using venv:
        ```bash
        python -m venv eustachian_monument/venv
        ```
        and then activating the environment.
        On Linux/MacOS:
        ```bash
        source eustachian_monument/venv/bin/activate
        ```
        On Windows:
        ```cmd
        eustachian_monument/venv/bin/activate.bat
        ```
    - By using Anaconda:
        ```bash
        conda create --name eus_mon
        ```
* Install [Ollama](https://ollama.com/), which is needed for using vision models.
* You're done! Open the `image_reconstruction_pipeline.ipynb` notebook for a fast demonstration.
