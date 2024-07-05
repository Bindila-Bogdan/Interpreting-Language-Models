import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class ContrastiveExplanations:
    def __init__(self):
        model_name = 'gpt2'
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    def predict(self, input_text):
        indexed_tokens = self.tokenizer.encode(input_text)
        tokens_tensor = torch.tensor([indexed_tokens])

        torch.enable_grad()
        self.model.eval()
        self.model.zero_grad()

        outputs = self.model(tokens_tensor)
        self.predictions = outputs.logits[0, -1, :]

        return outputs, tokens_tensor

    def get_k_predictions(self, k, type="logits"):
        top_k_predictions = torch.topk(self.predictions, k=k)
        if type == "logits":
            return top_k_predictions.values

        if type == "indices":
            return top_k_predictions.indices

        if type == "text":
            return self.tokenizer.decode(top_k_predictions.indices.tolist())

    def get_probability_for_prediction(self, text, prediction):
        outputs, _ = self.predict(text)
        outputs = outputs.logits[0, -1, :]
        target_index = self.tokenizer.encode(prediction)[0]
        return outputs[target_index]

    def get_token_logit_and_index(self, word):
        target_index = self.tokenizer.convert_tokens_to_ids(" " + word)
        if target_index == 50256:
            target_index = self.tokenizer.encode(word)[0]

        prediction_index = torch.argwhere(self.predictions)[target_index].item()
        return self.predictions[target_index], prediction_index

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

    def get_prediction(self):
        return self.predictions

    def get_contrastive_gradient_norm(self, text, correct, foil, explanation=True):
        normalized_gradient = []
        gradients = self.__get_g_star(text, correct, foil)
        for i in range(len(gradients)):
            gradient_norm = torch.norm(torch.tensor(gradients[i]), p=1)
            normalized_gradient.append(gradient_norm.flatten()[0])

        numpy_normalized_gradient = self.__scale_outputs(normalized_gradient)

        if explanation:
            return self.__get_contrastive_explanation(text, numpy_normalized_gradient)
        return numpy_normalized_gradient

    def get_contrastive_input_x_gradient(self, text, correct, foil, explanation=True):
        dot_product_gradients = []
        tokens = self.tokenizer.encode(text)
        gradients = self.__get_g_star(text, correct, foil)

        for i in range(len(tokens)):
            gradient = gradients[i].detach().numpy()
            word_embedding = self.model.transformer.wte.weight[tokens[i]].detach().numpy()
            dot_product_value = np.dot(gradient, word_embedding)
            dot_product_gradients.append(dot_product_value)

        normalized_probabilities = self.__scale_outputs(dot_product_gradients)

        if explanation:
            return self.__get_contrastive_explanation(text,  normalized_probabilities)
        return dot_product_gradients

    def get_input_erasure(self, text, correct, foil, explanation=True):
        input_erasure_gradients = []
        tokens = self.tokenizer.encode(text)
        correct_prediction = self.get_probability_for_prediction(text, correct)
        foil_prediction = self.get_probability_for_prediction(text, foil)

        for i in range(len(tokens)):
            token_text = self.tokenizer.decode(tokens[i])
            erased_text = text.replace(token_text, " ")
            correct_erased_prediction = self.get_probability_for_prediction(erased_text, correct)
            foil_erased_prediction = self.get_probability_for_prediction(erased_text, foil)
            input_erasure_gradients.append(
                (correct_prediction - correct_erased_prediction) -
                (foil_prediction - foil_erased_prediction)
            )

        normalized_probabilities = self.__scale_outputs(input_erasure_gradients)

        if explanation:
            return self.__get_contrastive_explanation(text,  normalized_probabilities)
        return normalized_probabilities

    def __get_g_star(self, text, correct, foil):
        gradients = []
        words = self.tokenizer.encode(text)

        outputs, input_ids = self.predict(text)
        _, index_correct = self.get_token_logit_and_index(correct)
        _, index_foil = self.get_token_logit_and_index(foil)

        # Backward pass to compute gradients
        if correct != foil:
            logit_difference = outputs.logits[0][-1][index_correct] - outputs.logits[0][-1][index_foil]
            logit_difference.backward()
        else:
            outputs.logits[0][-1][index_correct].backward()

        for i in range(len(words)):
            # Access gradient for input word embeddings
            input_word_gradients = self.model.transformer.wte.weight.grad[input_ids[0][i], :]
            gradients.append(input_word_gradients)

        return gradients

    def __get_contrastive_explanation(self, text, saliency_scores):
        tokens = self.tokenizer.encode(text)
        explanation = []

        for i in range(len(tokens)):
            explanation.append((self.tokenizer.decode(tokens[i]), saliency_scores[i]))
        return explanation

    def __scale_outputs(self, dot_product_gradients):
        numpy_outputs = np.array([x.item() for x in dot_product_gradients])
        normalized_probabilities = numpy_outputs/sum(numpy_outputs)

        return normalized_probabilities