# Eustachian Monument
**Image Restoring and Reconstruction**
---
**Abstract**
This project explores the possibilities of LLM-driven generation approach for geometrical understanding and spatial-wise transformation of images. We also enrich the transformed image with rare concept details, with high quality results by using [Image RAG](https://rotem-shalev.github.io/ImageRAG) and Conditional LDM generation.

## Getting Started

> **Please Note!**
>
> Make sure your machine supports CUDA for faster inference.
> To use CUDA with a compatible GPU, you need to install torch as shown [here](https://pytorch.org/get-started/locally/).
> Then add this single line inside `requirements.txt` (except for the substring 'pip3 install').
>
> However, if using CUDA is not an option, you can make inference using the CPU instead, by changing the [following line](image_reconstruction_pipeline.ipynb) into:
> ```python
> pipeline.to("cpu")
> ```


In order to execute the code, you need to go through the following steps:

* Clone this repo:
    ```bash
    git clone https://github.com/sandangl/eustachian_monument.git
    ```
* Create a Python Virtual Environment:
    We recommend using **conda** for best results.
    You can set your environment with conda by executing the script: `environment_setup.sh`.
* Install [Ollama](https://ollama.com/), which is needed for using vision models.
* You're done! Open the `image_reconstruction_pipeline.ipynb` notebook for a fast demonstration.


## System Requirements

Our solution needs *a lot* of space in order to host all the needed models, so be sure you have at least **25 GB** of free memory.


An NVIDIA GPU is not strictly needed, as inference could be made through CPU as well, yet still desirable, as it dramatically decreases image generation time.