import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class GPT2Model:
    def __init__(self):
        model_name = 'gpt2'
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    def predict(self, input_text):
        indexed_tokens = self.tokenizer.encode(input_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        self.model.zero_grad()
        with torch.no_grad():
            outputs = self.model(tokens_tensor)
            self.predictions = outputs.logits[0, -2, :]
        return self.predictions

    def get_k_predictions(self, k, type="text"):
        top_k_predictions = torch.topk(self.predictions, k=k)
        if type == "logits":
            return top_k_predictions.values

        if type == "indices":
            return top_k_predictions.indices

        if type == "text":
            return self.tokenizer.decode(top_k_predictions.indices.tolist())

    def get_token_logit_and_index(self, word):
        target_index = self.tokenizer.convert_tokens_to_ids(" " + word)
        if target_index == 50256:
            target_index = self.tokenizer.encode(word)[0]

        prediction_index = torch.argwhere(self.predictions)[target_index].item()
        return self.predictions[target_index], prediction_index

