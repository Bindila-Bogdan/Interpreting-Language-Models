from transformers import AutoTokenizer, AutoModelForCausalLM

class ModelLoader:
    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer