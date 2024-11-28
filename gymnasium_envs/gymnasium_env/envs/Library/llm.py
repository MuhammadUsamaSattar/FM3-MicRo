from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  
import torch
from dotenv import load_dotenv
import os


class LLM:

    def __init__(self):
        """Initializes the model and sends it to correct device"""
        load_dotenv()
        access_token = os.getenv('HUGGING_FACE_API_KEY')
        model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

        quantization_config = BitsAndBytesConfig(load_in_8bit=False)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            token = access_token,
        ).to("cuda")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=access_token,
        )

    def get_reward(self, txt):
        """Generates reward from the input image

        Args:
            txt : Prompt to pass to the model

        Returns:
            float: Reward value of the state
        """
        context = "You are pirate!"
        complete_text = context + txt

        input_ids = self.tokenizer(complete_text, return_tensors="pt").to("cuda")

        output = self.model.generate(**input_ids, max_new_tokens=10)

        print(self.tokenizer.decode(output[0], skip_special_tokens=True))

if __name__ == "__main__":
    llm = LLM()
    llm.get_reward("What is your name?")
