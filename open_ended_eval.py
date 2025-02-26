# %%

from app import OpenRouterApp
from eval import generate_difficulty_scores
from tqdm import tqdm
import json
import os
from time import time
from collections import defaultdict
# %%

with open("GPT-output-data/Open-ended Japanese Questions.txt", "r") as file:
    questions = file.readlines()
    questions = [' '.join(question.strip().split()[1:]) for question in questions][1:]
    questions = [question for question in questions if question != ""]

# %%

from enum import Enum

class DifficultyLevel(Enum):
    NOVICE = "novice"
    INTERMEDIATE = "intermediate"

# %%
### ARGS ###

mode = DifficultyLevel.INTERMEDIATE
model = "gemini-2.0-flash"
num_iters = 10
start_idx = 0

# %%

novice_system_prompt = """
You are an assistant for a Japanese learning course, at the novice level. Respond as normal, but use words, phrases, and grammar that would be comprehensible to novices. This corresponds to Genki Vol. 1 or JLPT N5 level.
Characteristics:
    •   Focus on foundational grammar and vocabulary.
    •   Use simple sentence structures and limited contexts (e.g., self-introduction, daily activities, shopping).
    •   Use only hiragana, katakana, and basic kanji (~100 characters).
        
Some examples of grammar patterns for novice learners:
    1. Can introduce themselves and provide basic personal information.
    2. Can express likes, dislikes, and preferences using ～が好き/嫌いです.
    3. Can ask and answer simple yes/no questions (e.g., ～ですか, ～ますか).
    4. Can use basic verbs in present and past forms (e.g., ～ます, ～ました).
    5. Can describe location using ～の前/後/中.
    6. Can use ～てくださいfor polite requests.
    7. Can talk about daily routines with simple time expressions.
    8. Can count objects using basic counters (e.g., ～個, ～枚, ～人).

Always use Japanese and do not respond in English.
"""

intermediate_system_prompt = """
You are an assistant for a Japanese learning course, at the intermediate level. Respond as normal, but use words, phrases, and grammar that would be comprehensible to intermediate learners. This corresponds to Genki Vol. 2 or JLPT N3-N4 level.
Characteristics:
    •   Expand on grammar with more complex sentence structures.
    •   You can communicate in a wider range of contexts (e.g., giving advice, discussing opinions, comparing options).
    •   You have a greater use of kanji (~300 characters) and advanced expressions.
    
Some examples of grammar patterns for intermediate learners:
    1. Can express opinions and provide reasons using ～と思いますand ～から.
    2. Can make comparisons using ～よりand ～のほうが.
    3. Can talk about past experiences using ～たことがあります.
    4. Can make hypothetical statements using ～たらor ～ば.
    5. Can use formal and polite expressions (e.g., ～ていただけますか).
    6. Can describe plans and intentions using ～つもりですor ～ようと思います.
    7. Can give and receive favors using ～てあげる/くれる/もらう.
    8. Can handle more complex requests and instructions (e.g., ～てくれませんか).
    9. Can explain causes and effects using ～のでand ～から.
    10. Can manage conversations in casual and formal settings with proper use of polite forms.

Always use Japanese and do not respond in English.
"""

system_prompt = intermediate_system_prompt if mode == DifficultyLevel.INTERMEDIATE else novice_system_prompt
models = {
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gemini-2.0-flash": "google/gemini-2.0-flash-001",
}

max_tokens = 1024
teacher_app = OpenRouterApp(models[model], system_prompt, {"max_tokens": max_tokens}, custom_name=model)

start_time = time()

os.makedirs(f"results/open-ended/{teacher_app.model_name}", exist_ok=True)
save_path = f"results/open-ended/{teacher_app.model_name}/{mode.value}_responses.jsonl"

# %%
print(mode, model, num_iters)
print(f"{(len(questions)-start_idx) * num_iters} calls")
for question in tqdm(questions[start_idx:]):
    for _ in tqdm(range(num_iters)):
        response = teacher_app.run(question)
        with open(save_path, "a") as f:
            f.write(json.dumps({"question": question, "response": response}) + "\n")

# %%
models = ["gpt-4o-mini", "gemini-2.0-flash"]
types = ["novice", "intermediate"]
eval_fns = {
    "avg": lambda scores: sum(scores) / len(scores),
    "min": lambda scores: min(scores),
}

eval_results = {}

for eval_model in models:
    for eval_type in types:
        for eval_fn_name, eval_fn in eval_fns.items():
            eval_save_path = f"results/open-ended/{eval_model}/{eval_type}_responses.jsonl"

            with open(eval_save_path, "r") as f:
                data = [json.loads(line) for line in f]
                results = defaultdict(list)

                for row in data:
                    question, response = row["question"], row["response"]
                    scores = generate_difficulty_scores(response)
                    if len(scores) > 0:
                        results[question].append(eval_fn(scores))

                eval_results[eval_model, eval_type, eval_fn_name] = sum([sum(scores) / len(scores) for scores in results.values()]) / len(results)

# Visualize results
# Create a bar plot comparing the results
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(1, len(eval_fns), figsize=(12, 5))
x = np.arange(len(models))
width = 0.35

for i, (eval_fn_name, eval_fn) in enumerate(eval_fns.items()):
    novice_scores = [eval_results[model, "novice", eval_fn_name] for model in models]
    intermediate_scores = [eval_results[model, "intermediate", eval_fn_name] for model in models]
    
    ax = axes[i]
    ax.bar(x - width/2, novice_scores, width, label='Novice')
    ax.bar(x + width/2, intermediate_scores, width, label='Intermediate')
    
    ax.set_ylabel('Score')
    ax.set_title(f'{eval_fn_name.capitalize()} Score')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45)
    ax.legend()

plt.tight_layout()
plt.show()

    
    
# %%

with open(save_path, "r") as f:
    data = [json.loads(line) for line in f]

with open(f"results/open-ended/{teacher_app.model_name}/{mode.value}_responses.txt", "w") as f:
    for row in data:
        f.write(f"Question: {row['question']}\nResponse: {row['response']}\n\n")

# %%
