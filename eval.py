import csv
import MeCab
wakati = MeCab.Tagger("-Owakati")

input_str = "絵本を含めた本からの言語情報は，子供の言語発達における重要なインプットである．"

tokens = wakati.parse(input_str).split()

d = {}
with open("jlpt_vocab.csv") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        d[row[0]] = int(row[-1][-1])

print(tokens)
num_tokens = 0
avg = 0
for token in tokens:
    if token in d:
        print(token, d[token])
        num_tokens += 1
        avg += d[token]

print(avg/num_tokens)
