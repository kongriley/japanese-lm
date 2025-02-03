# %%
import sys
sys.path.append('../..')

from app import ModelApp, GemmaModel, QwenModel, OpenRouterApp

with open('free_response_queries.txt') as f:
    queries = f.readlines()

novice_system_prompt = """
You are an assistant for a Japanese learning course, at the novice level. Respond as normal, but use words, phrases, and grammar that would be comprehensible to novices.
        •   	Focus on foundational grammar and vocabulary.
        •   	Use simple sentence structures and limited contexts (e.g., self-introduction, daily activities, shopping).
        •   	Use only hiragana, katakana, and basic kanji (~100 characters).

Always use Japanese and do not respond in English.
"""

with open('../../novice_rubric.txt') as f:
    novice_rubric = f.read()

novice_evaluator_prompt = """
Your job is to evaluate the answer to a query regarding the Japanese language. The answer should be in novice level Japanese.\nRubric: {novice_rubric} \n\nUse the rubric to evaluate the response. Respond with a score from 1 to 5, with 5 being a response that fully meets the criteria. The final line of your response should be the score only.
"""

# %%

teacher_app = OpenRouterApp("gpt-4o", novice_system_prompt)
evaluator_app = OpenRouterApp("gpt-4o", novice_evaluator_prompt)

# %% 

from tqdm import tqdm

responses = []
judgements = []
scores = []
for query in tqdm(queries):
    response = teacher_app.run(query)
    judgement = evaluator_app.run(f"Query: {query}\nResponse: {response}")
    score = judgement.split('\n')[-1]
    responses.append(response)
    judgements.append(judgement)
    scores.append(score)

# %%

import os
os.makedirs(f'../../results/free_response/{teacher_app.model_name}', exist_ok=True)

from datetime import datetime

# Get the current time and format it as a string
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

# Write results to file
with open(f'../../results/free_response/{teacher_app.model_name}/results_{current_time}.txt', 'w') as f:
    for i, (query, response, judgement, score) in enumerate(zip(queries, responses, judgements, scores)):
        f.write(f"Query {i+1}: {query}\n")
        f.write(f"Response: {response}\n") 
        f.write(f"Judgement:\n{judgement}\n")
        f.write(f"Score: {score}\n")
        f.write("\n" + "="*80 + "\n\n")

# %%
