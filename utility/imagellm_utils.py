from diffusers import QwenImageEditPlusPipeline
from dotenv import dotenv_values
from typing import List, Dict
from PIL import Image
import torch
from vllm_utils import VLLMUtils


class ImageLLMUtils:

    def __init__(self):
        self.mode = dotenv_values("../.env")["INFERENCE_MODE"]
        self.precision = torch.bfloat16 if self.mode == "cuda" else torch.float32
        self.vllm = VLLMUtils()

    def pipeline_init(self) -> None:
        pipeline = QwenImageEditPlusPipeline.from_pretrained(
            "Qwen/Qwen-Image-Edit-2509"
        )
        pipeline.to(self.precision)
        pipeline.to(self.mode)
        pipeline.set_progress_bar_config(disable=False)
        self.pipeline = pipeline

    def generate(
        self,
        prompt: str,
        images: List[Image.Image],
        guidance=4.0,
        neg_prompt="",
        inf_steps=50,
        gen_images=1,
        output_filename="output_edit",
    ) -> List[Image.Image]:
        polished_prompt = self.vllm.polish_prompt_en(prompt, images)
        inputs = self._make_inputs(
            images, polished_prompt, guidance, neg_prompt, inf_steps, gen_images
        )
        return self._generate_image(inputs, output_filename)

    def _generate_image(self, inputs: Dict, output_filename) -> List[Image.Image]:
        with torch.inference_mode():
            output = self.pipeline(**inputs)
            i = 0
            for output_image in output.images:
                output_image.save(f"{output_filename}_{i}.jpeg")
                i += 1
            print("Image generated (saved!)")
        return output.images

    def _make_inputs(
        self,
        images: List[Image.Image],
        prompt: str,
        guidance: float,
        neg_prompt: str,
        inf_steps: int,
        gen_images: int,
    ):
        return {
            "image": images,
            "prompt": prompt,
            "generator": torch.manual_seed(0),
            "true_cfg_scale": guidance,
            "negative_prompt": neg_prompt,
            "num_inference_steps": inf_steps,
            "num_images_per_prompt": gen_images,
        }
