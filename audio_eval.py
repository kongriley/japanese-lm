import base64
from openai import OpenAI
import os
import dotenv

dotenv.load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=api_key
)

system_prompt = """
You are a helpful assistant that evaluates the pronunciation of a speech utterance. Rate the pronunciation as a floating point numberon a scale of 1 to 10, where 1 is the worst and 10 is the best. Then, give detailed feedback on the pronunciation.

Use the following format:

Pronunciation Score: [score]

Feedback: [feedback]
"""

ratings = []
responses = []

for file in os.listdir("audio/L6_soundfiles"):
    if not file.endswith(".wav"):
        continue

    file_path = os.path.join("audio/L6_soundfiles", file)
    with open(file_path, "rb") as audio_file:
        audio_data = audio_file.read()
        audio_str = base64.b64encode(audio_data).decode("utf-8")

    completion = client.chat.completions.create(
    model="gpt-4o-audio-preview",
    messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
        "role": "user",
        "content": [
            {
                "type": "input_audio",
                "input_audio": {
                    "data": audio_str,
                    "format": "wav"
                }
            }
        ]
        }
    ],
    )

    response = completion.choices[0].message.content
    rating = response.split("Pronunciation Score:")[1].split("Feedback:")[0].strip()
    try:
        rating = float(rating)
    except ValueError:
        print(f"Invalid rating for {file}\n{response}")
        rating = None
    ratings.append(rating)
    responses.append(response)

gt_eval_scores = {'file_1.mp4': 9.5, 'file_2.mp4': 8.7, 'file_3.mp4': 6, 'file_4.mp4': 9.2, 'file_5.mp4': 7.1, 'file_6.mp4': 9.8, 'file_7.mp4': 9.5, 'file_8.mp4': 9, 'file_9.mp4': 9, 'file_10.mp4': 7.5, 'file_11.mp4': 8.8, 'file_12.mp4': 9.7, 'file_13.mp4': 7}
gt_scores = list(gt_eval_scores.values())

os.makedirs("results/4o-audio", exist_ok=True)
with open("results/4o-audio/L6_ratings.txt", "w") as f:
    for rating in ratings:
        f.write(f"{rating}\n")

with open("results/4o-audio/L6_responses.txt", "w") as f:
    for response in responses:
        f.write(f"{response}\n")

import matplotlib.pyplot as plt

plt.scatter(ratings, gt_scores)
plt.xlabel("Pronunciation Score")
plt.ylabel("Teacher Evaluation Score")
plt.show()
