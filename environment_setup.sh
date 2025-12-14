conda create -n eustachian_monument python=3.11
conda activate eustachian_monument
conda install -n eustachian_monument ipykernel --update-deps --force-reinstall
pip install git+https://github.com/huggingface/diffusers
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129