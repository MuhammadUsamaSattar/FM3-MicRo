import os
import requests
import time
from typing import List

import torch
from dotenv import load_dotenv
from pathlib import Path
from PIL import Image
from transformers import (
    BitsAndBytesConfig,
    LlavaNextForConditionalGeneration,
    LlavaNextProcessor,
)


class VLM:

    def __init__(
        self,
        model_id: str,
        model_quant: str = "fp16",
        device: str = "cuda",
        verbose: bool = False,
    ):
        """Intializes the model and processor.

        Args:
            model_id (str, optional): ID of the model on hugging face repository or local path to a download model.
            model_quant (str, optional): The quantization level of the model. "fp16", "8b" and "4b" are implemented. Defaults to "fp16".
            device (str, optional): Device which runs the model. Only "cuda" is available. Defaults to "cuda".
            verbose (bool, optional): Boolen to select verbose or non-verbose mode. Defaults to False.
        """
        # Load the path from .env file
        load_dotenv()
        model_id = os.getenv(model_id)

        # Initializes self.model and self.processor parameters depending upon given model_quant
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

        # Add path_size and vision_feature_select_strategy parameter values due to a warning by transformers. This still does not fix the warning. TBD.
        self.processor.patch_size = self.model.config.vision_config.patch_size
        self.processor.vision_feature_select_strategy = (
            self.model.config.vision_feature_select_strategy
        )

        self.verbose = verbose

    def get_response(
        self,
        img: List,
        img_parameter_type: str,
        txt: str,
        tokens: int,
    ) -> str:
        """Generates a response from the input image and text.

        Args:
            img (List): Prompt image
            img_parameter_type (str): Type of image parameter. "url" and "image" are possible.
            txt (str): Prompt text
            tokens (int): Number of token to generate

        Returns:
            str: Output response
        """
        # Prepare image and text prompt, using the appropriate prompt template
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

        # Autoregressively complete prompt and output T/s
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

        output = self.processor.decode(
            output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        if self.verbose == True:
            print(output)

        return output


if __name__ == "__main__":
    vlm = VLM(
        model_id="PATH_LLAVA",
        model_quant="fp16",
        device="cuda",
        verbose=True,
    )

    while True:
        vlm.get_response(
            img=input("\nEnter URL: "),
            img_parameter_type="url",
            txt=input("\nEnter prompt: "),
            tokens=int(input("\nEnter tokens: ")),
        )
