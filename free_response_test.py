# %%
from app import ModelApp, QwenModel, OpenRouterApp

with open('free_response_queries.txt') as f:
    queries = f.readlines()

novice_system_prompt = """
You are an assistant for a Japanese learning course, at the novice level. Respond as normal, but use words, phrases, and grammar that would be comprehensible to novices.
        •   	Focus on foundational grammar and vocabulary.
        •   	Use simple sentence structures and limited contexts (e.g., self-introduction, daily activities, shopping).
        •   	Use only hiragana, katakana, and basic kanji (~100 characters).

Always use Japanese and do not respond in English.
"""

with open('novice_rubric.txt') as f:
    novice_rubric = f.read()

novice_evaluator_prompt = """
Your job is to evaluate the answer to a query regarding the Japanese language. The answer should be in novice level Japanese.\nRubric: {novice_rubric} \n\nUse the rubric to evaluate the response. Respond with a score from 1 to 5, with 5 being a response that fully meets the criteria. The final line of your response should be the score only.
"""

# %%

qwen_model = QwenModel()
teacher_app = ModelApp(qwen_model, novice_system_prompt, {"max_tokens": 256})
evaluator_app = OpenRouterApp("gpt-4o", novice_evaluator_prompt)

# %% 
responses = []
judgements = []
scores = []
for query in queries:
    response = teacher_app.run(query)
    judgement = evaluator_app.run(f"Query: {query}\nResponse: {response}")
    score = judgement.split('\n')[-1]
    responses.append(response)
    judgements.append(judgement)
    scores.append(score)

# %%

import os
os.makedirs(f'results/{teacher_app.model.name}', exist_ok=True)

# Write results to file
with open(f'results/{teacher_app.model.name}/free_response_results.txt', 'w') as f:
    for i, (query, response, judgement, score) in enumerate(zip(queries, responses, judgements, scores)):
        f.write(f"Query {i+1}: {query}\n")
        f.write(f"Response: {response}\n") 
        f.write(f"Judgement:\n{judgement}\n")
        f.write(f"Score: {score}\n")
        f.write("\n" + "="*80 + "\n\n")

# %%
