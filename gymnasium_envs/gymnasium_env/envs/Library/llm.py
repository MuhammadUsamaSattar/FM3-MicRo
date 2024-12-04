from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from dotenv import load_dotenv
import os
import time


class LLM:

    def __init__(self, model_quant="fp16", device="cuda"):
        """Initializes the model and sends it to correct device"""
        load_dotenv()
        access_token = os.getenv("HUGGING_FACE_API_KEY")
        model_id = os.getenv("PATH_LLAMA")

        if model_quant == "fp16":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.float16,
                device_map=device,
                token=access_token,
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

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2",
                quantization_config=quantization_config,
                torch_dtype=torch.float16,
                device_map=device,
                token=access_token,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=access_token,
        )

    def get_reward(self, txt, tokens):
        """Generates reward from the input image

        Args:
            txt : Prompt to pass to the model

        Returns:
            float: Reward value of the state
        """
        inputs = self.tokenizer(txt, return_tensors="pt").to("cuda")

        t = time.time()

        output = self.model.generate(
            **inputs, max_new_tokens=tokens, pad_token_id=self.tokenizer.eos_token_id
        )

        dt = time.time() - t
        generated_tokens = len(output[0]) - len(inputs["input_ids"][0])

        print(
            "Token generation rate (T/s): ",
            generated_tokens / dt,
            "\n",
        )

        print(self.tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    llm = LLM(model_quant="fp16", device="cuda")

    while True:
        llm.get_reward(input("\nEnter prompt: "), tokens=int(input("\nEnter tokens: ")))
