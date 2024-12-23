import os
import time
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class LLM:

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
        # Load the paths from .env file
        load_dotenv()
        access_token = os.getenv("HUGGING_FACE_API_KEY")
        model_id = os.getenv(model_id)

        # Initializes self.model and self.tokenizer parameters depending upon given model_quant
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

        self.verbose = verbose

    def get_response(
        self,
        context: str,
        txt: str,
        tokens: int,
    ) -> str:
        """Generates a response from the input image and text.

        Args:
            context (str): Context for the system
            txt (str): Prompt text
            tokens (int): Number of token to generate

        Returns:
            str: Output response
        """

        # Autoregressively complete prompt and output T/s
        conversation = [
            {
                "role": "system",
                "content": context,
            },
            {
                "role": "user",
                "content": txt,
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        inputs = self.tokenizer(text=prompt, return_tensors="pt").to("cuda")

        t = time.time()
        output = self.model.generate(
            **inputs, max_new_tokens=tokens, pad_token_id=self.tokenizer.eos_token_id
        )
        dt = time.time() - t
        generated_tokens = len(output[0]) - len(inputs["input_ids"][0])

        output = self.tokenizer.decode(
            output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        if self.verbose == True:
            print("Response: ", output)

            print(
                "Token generation rate (T/s): ",
                generated_tokens / dt,
                "\n",
            )

        return output


if __name__ == "__main__":

    llm = LLM(
        model_id="TRITON_LLAMA_8B",
        model_quant="fp16",
        device="cuda",
        verbose=True,
    )

    while True:
        llm.get_response(
            input("\nEnter context: "),
            input("\nEnter prompt: "),
            tokens=int(input("\nEnter tokens: ")),
        )
