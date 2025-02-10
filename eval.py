import MeCab
wakati = MeCab.Tagger("-Owakati")
tokens = wakati.parse("時代の流れでコンピュータや携帯電話などのテクノロジーの構造がだんだん不思議になっています。").split()

# import kagglehub
import csv

# Download latest version
# path = kagglehub.dataset_download("robinpourtaud/jlpt-words-by-level")
d = {}
with open("jlpt_vocab.csv") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        d[row[0]] = int(row[-1][-1])

print(tokens)
num_tokens = 0
avg = 0
for token in tokens:
    # print(token)
    if token in d:
        print(token, d[token])
        num_tokens += 1
        avg += d[token]

print(avg/num_tokens)
