import os
import time

os.environ["TRANSFORMERS_OFFLINE"] = "1"

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
            padding_side="left",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.verbose = verbose

    def get_response(
        self,
        batches: int,
        contexts: list,
        txts: list,
        tokens: int,
    ) -> str:
        """Generates a response from the input image and text.

        Args:
            batches (int): Number of batches to process.
            contexts (list): List of contexts for the system.
            txts (list): List of prompt texts.
            tokens (int): Number of token to generate.

        Returns:
            str: Output response.
        """

        inputs = []
        for i in range(batches):
            # Autoregressively complete prompt and output T/s
            conversation = [
                {
                    "role": "system",
                    "content": contexts[i],
                },
                {
                    "role": "user",
                    "content": txts[i],
                },
            ]

            prompt = self.tokenizer.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )

            inputs.append(prompt)

        inputs = self.tokenizer(text=inputs, return_tensors="pt", padding=True).to(
            "cuda"
        )

        t = time.time()
        output = self.model.generate(
            **inputs, max_new_tokens=tokens, pad_token_id=self.tokenizer.eos_token_id
        )
        dt = time.time() - t

        generated_tokens = 0
        for i in range(len(output)):
            generated_tokens += len(output[i]) - len(inputs["input_ids"][i])

        output = self.tokenizer.batch_decode(
            output[:, inputs["input_ids"][i].shape[0] :], skip_special_tokens=True
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
        model_id="PATH_LLAMA_8B",
        model_quant="fp16",
        device="cuda",
        verbose=True,
    )

    while True:
        batches = int(input("\nEnter number of batches: "))
        tokens = int(input("\nEnter tokens: "))

        contexts = []
        prompts = []
        for i in range(batches):
            contexts.append(input("\nEnter context: "))
            prompts.append(input("\nEnter prompt: "))

        llm.get_response(
            batches,
            contexts,
            prompts,
            tokens,
        )
