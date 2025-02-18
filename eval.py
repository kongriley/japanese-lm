import os
import csv
import json
from typing import Callable
import MeCab
wakati = MeCab.Tagger("-Owakati")

class Mode:
    def __init__(self, mode_function: Callable[[str], float]):
        self.mode_function = mode_function

    def __call__(self, input_str: str) -> float:
        return self.mode_function(input_str)


d = {}
with open("jlpt_vocab.csv") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        d[row[0]] = int(row[-1][-1])

def generate_difficulty_scores(input_str) -> list[float]:
    """
    Evaluates the difficulty of the input string.
    Returns a list of difficulty scores for each token in the input string.
    """
    tagger = MeCab.Tagger()
    node = tagger.parseToNode(input_str)
    tokens = []
    while node:
        if node.surface:  # Skip empty tokens
            features = node.feature.split(',')
            # Use the base (lemma) form if available, otherwise use the surface form
            lemma = features[7] if len(features) > 7 and features[7] != '*' else node.surface
            if lemma in d:
                tokens.append(d[lemma])
        node = node.next
    return tokens

if __name__ == "__main__":
    # print(generate_difficulty_scores("今、痛んでいる。"))
    # exit()
    files = [f for f in os.listdir("GPT-output-data") if f.endswith(".csv")]
    results = {file: [] for file in files}

    avg_mode = Mode(lambda x: sum(x) / len(x))
    min_mode = Mode(lambda x: min(x))

    for file in files:
        with open(os.path.join("GPT-output-data", file)) as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            for _, sent, _ in reader:
                result = generate_difficulty_scores(sent)
                if len(result) > 0:
                    results[file].append(min_mode(result))
    
    with open("GPT-output-data/results.jsonl", "w") as f:
        for file, scores in results.items():
            json.dump({"filename": file, "scores": scores, "average": sum(scores) / len(scores)}, f)
            print(file, sum(scores) / len(scores))
            f.write('\n')
