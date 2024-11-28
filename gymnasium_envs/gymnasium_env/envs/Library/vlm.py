from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests


class VLM:

    def __init__(self):
        """Initializes the model and sends it to correct device"""

        model_id = "llava-hf/llava-v1.6-mistral-7b-hf"

        self.processor = LlavaNextProcessor.from_pretrained(model_id)
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            # load_in_4bit=True,
            # use_flash_attention_2=True,
        )
        self.model.to("cuda:0")

    def generate_reward(self, img, txt):
        """Generates reward from the input image

        Args:
            img : An image that represents the current state
            txt : Prompt to pass to the model

        Returns:
            float: Reward value of the state
        """
        # prepare image and text prompt, using the appropriate prompt template
        url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
        image = Image.open(requests.get(url, stream=True).raw)

        # Define a chat history and use `apply_chat_template` to get correctly formatted prompt
        # Each value in "content" has to be a list of dicts with types ("text", "image")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": txt},
                    {"type": "image"},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(
            "cuda:0"
        )
        # inputs = self.processor(images=img, text=prompt, return_tensors="pt").to(
        #    "cuda:0"
        # )

        # autoregressively complete prompt
        output = self.model.generate(**inputs, max_new_tokens=100)

        print(self.processor.decode(output[0], skip_special_tokens=True))
