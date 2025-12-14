call conda create -y -n eustachian_monument_test python=3.11
call activate eustachian_monument_test
call conda install -y -n eustachian_monument_test ipykernel
pip install git+https://github.com/huggingface/diffusers
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129