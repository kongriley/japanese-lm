import csv
import MeCab
wakati = MeCab.Tagger("-Owakati")

d = {}
with open("jlpt_vocab.csv") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        d[row[0]] = int(row[-1][-1])

def evaluate_input(input_str) -> float:
    """
    Evaluates the difficulty of the input string.
    Returns the average difficulty of the tokens in the input string.
    """
    tokens = wakati.parse(input_str).split()

    num_tokens = 0
    avg = 0
    for token in tokens:
        if token in d:
            num_tokens += 1
            avg += d[token]

    return avg/num_tokens

if __name__ == "__main__":
    input_str = "絵本を含めた本からの言語情報は，子供の言語発達における重要なインプットである．"
    print(evaluate_input(input_str))
