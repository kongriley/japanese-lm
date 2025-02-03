from dotenv import load_dotenv
import os

import torch
from transformers import pipeline
from openai import OpenAI
from mlx_lm import load, generate

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
)

class App:
    """
    Base class for all apps.
    """
    def __init__(self, model, system_prompt, model_args=None):
        self.model = model
        self.system_prompt = system_prompt
        if model_args is None:
            model_args = {}
        self.model_args = model_args

        self.model_name = "model"

    def run(self, query) -> str:
        raise NotImplementedError

class ModelApp(App):
    """
    Uses a model to run a query.
    `self.model` must implement `__call__()`, which takes a formatted message list and returns a string response.
    """

    def __init__(self, model, system_prompt, model_args=None):
        super().__init__(model, system_prompt, model_args)
        self.model_name = model.name

    def run(self, query) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]
        return self.model(messages, **self.model_args)


class OpenRouterApp(App):
    """
    Uses OpenRouter API to run a query using a model name.
    """

    def __init__(self, model_name, system_prompt, model_args=None, custom_name=None):
        super().__init__(model_name, system_prompt, model_args)
        if custom_name is None:
            self.model_name = model_name
        else:
            self.model_name = custom_name

    def run(self, query) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": query},
        ]
        return (
            client.chat.completions.create(
                model=self.model, messages=messages, **self.model_args
            )
            .choices[0]
            .message.content
        )


class GemmaModel:
    def __init__(self):
        self.model = pipeline(
            "text-generation",
            model="google/gemma-2-2b-jpn-it",
            model_kwargs={"torch_dtype": torch.bfloat16},
            device="mps",
            token=hf_token,
        )
        self.model_args = {
            "return_full_text": False,
            "max_new_tokens": 256,
        }
        self.name = "gemma"

    def preprocess(self, messages):
        system_prompt = messages[0]["content"]
        query = messages[1]["content"]
        messages = [
            {"role": "user", "content": system_prompt + "\n\nQuery: " + query},
        ]
        return messages

    def __call__(self, messages, **kwargs):
        messages = self.preprocess(messages)
        return self.model(messages, return_full_text=False, **self.model_args, **kwargs)[0][
            "generated_text"
        ]


class QwenModel:
    def __init__(self):
        self.model, self.tokenizer = load(
            "Qwen/Qwen2.5-1.5B-Instruct", tokenizer_config={"eos_token": "<|im_end|>"}
        )
        self.name = "qwen"
        self.model_args = {
            "max_tokens": 256,
        }

    def __call__(self, messages, **kwargs):
        messages = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return generate(
            self.model, self.tokenizer, prompt=messages, verbose=True, **self.model_args, **kwargs
        )


if __name__ == "__main__":
    novice_system_prompt = """
    You are an assistant for a Japanese learning course, at the novice level. Respond as normal, but use words, phrases, and grammar that would be comprehensible to novices.
            •   	Focus on foundational grammar and vocabulary.
            •   	Use simple sentence structures and limited contexts (e.g., self-introduction, daily activities, shopping).
            •   	Use only hiragana, katakana, and basic kanji (~100 characters).

    Always use Japanese and do not respond in English.


    """
    max_tokens = 1024
    r1_distill_model = OpenRouterApp("deepseek/deepseek-r1-distill-qwen-32b", novice_system_prompt, {"max_tokens": max_tokens}, custom_name="r1_distill")
    # gemma_model = GemmaModel()
    # qwen_model = QwenModel()
    teacher_app = r1_distill_model

    query = """
    ３〜５文で自己紹介をしてください。
    """

    response = teacher_app.run(query)
    print(response)

    with open("novice_rubric.txt") as f:
        novice_rubric = f.read()

    novice_evaluator_prompt = f"""
    Your job is to evaluate the answer to a query regarding the Japanese language. The answer should be in novice level Japanese.\nRubric: {novice_rubric} \n\nUse the rubric to evaluate the response. Respond with a score from 1 to 5, with 5 being a response that fully meets the criteria. The final line of your response should be the score only.
    """
    evaluator_app = OpenRouterApp("gpt-4o", novice_evaluator_prompt)

    print(evaluator_app.run(f"Query: {query}\nResponse: {response}"))
