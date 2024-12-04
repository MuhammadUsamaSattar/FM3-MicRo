from transformers import (
    LlavaNextProcessor,
    LlavaNextForConditionalGeneration,
    BitsAndBytesConfig,
)
import torch
from PIL import Image
import requests
import time
from pathlib import Path
from dotenv import load_dotenv
import os


class VLM:

    def __init__(self, model_quant="fp16", device="cuda"):
        """Initializes the model and sends it to correct device"""
        load_dotenv()
        model_id = os.getenv("PATH_LLAVA")

        if model_quant == "fp16":
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                model_id,
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.float16,
                device_map=device,
            )

        else:
            if model_quant == "8b":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )

            if model_quant == "4b":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )

            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                model_id,
                low_cpu_mem_usage=True,
                quantization_config=quantization_config,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.float16,
                device_map=device,
            )

        self.processor = LlavaNextProcessor.from_pretrained(model_id)

        self.processor.patch_size = self.model.config.vision_config.patch_size
        self.processor.vision_feature_select_strategy = (
            self.model.config.vision_feature_select_strategy
        )

    def get_reward(self, img, img_parameter_type, txt, tokens):
        """Generates reward from the input image

        Args:
            img : An image that represents the current state
            txt : Prompt to pass to the model

        Returns:
            float: Reward value of the state
        """
        # prepare image and text prompt, using the appropriate prompt template
        if img_parameter_type == "url":
            url = img
            image = Image.open(requests.get(url, stream=True).raw)

        elif img_parameter_type == "image":
            image = img

        # Define a chat history and use `apply_chat_template` to get correctly formatted prompt
        # Each value in "content" has to be a list of dicts with types ("text", "image")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": txt},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
            "cuda"
        )
        # inputs = self.processor(images=img, text=prompt, return_tensors="pt").to(
        #    "cuda:0"
        # )

        # autoregressively complete prompt
        t = time.time()

        output = self.model.generate(
            **inputs,
            max_new_tokens=tokens,
        )

        dt = time.time() - t
        generated_tokens = len(output[0]) - len(inputs["input_ids"][0])

        print(
            "Token generation rate (T/s): ",
            generated_tokens / dt,
            "\n",
        )

        print(self.processor.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    vlm = VLM(model_quant="fp16", device="cuda")

    while True:
        vlm.get_reward(
            img=input("\nEnter URL: "),
            img_parameter_type="url",
            txt=input("\nEnter prompt: "),
            tokens=int(input("\nEnter tokens: ")),
        )
