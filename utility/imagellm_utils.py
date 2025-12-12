from diffusers import QwenImageEditPlusPipeline
from dotenv import dotenv_values
from typing import List, Dict
from PIL import Image
import os
import torch
from utility.vllm_utils import VLLMUtils
from utility.vector_store_utils import VectorStoreUtils


class ImageLLMUtils:
    def __init__(self):
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.env")
        self.mode = dotenv_values(env_path)["INFERENCE_MODE"]
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
        persist=True
    ) -> List[Image.Image]:
        polished_prompt = self.vllm.polish_prompt_en(prompt, images)
        inputs = self._make_inputs(
            images, polished_prompt, guidance, neg_prompt, inf_steps, gen_images
        )
        return self._generate_image(inputs, output_filename, persist=persist)

    def generate_with_rag(
        self,
        prompt: str,
        concepts: List[str],
        images: List[Image.Image],
        vectorStore: VectorStoreUtils,
        guidance=4.0,
        neg_prompt="",
        inf_steps=50,
        gen_images=1,
        output_filename="output_edit",
        persist=True,
    ) -> List[Image.Image]:
        unknown_concepts = self.vllm.knowledge_eval(concepts=concepts)
        retrieved = vectorStore.query(unknown_concepts)

        result = self.generate(
            prompt,
            images.append(retrieved),
            guidance=guidance,
            neg_prompt=neg_prompt,
            inf_steps=inf_steps,
            gen_images=gen_images,
            persist=False
        )

        missing = self.vllm.inclusion_evaluation(concepts, result)
        if len(missing["missing"]) == 0:
            return result
        else:
            retrieved = vectorStore.query(missing)
            result = self.generate(
                prompt,
                images.append(retrieved),
                guidance=guidance,
                neg_prompt=neg_prompt,
                inf_steps=inf_steps,
                gen_images=gen_images,
                output_filename=output_filename,
                persist=persist
            )
            return result

    def _generate_image(self, inputs: Dict, output_filename, persist=True) -> List[Image.Image]:
        with torch.inference_mode():
            output = self.pipeline(**inputs)
            i = 0
            if(persist):
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
