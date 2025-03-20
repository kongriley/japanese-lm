# %%
import torch
from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
import librosa
import os
import json
import Levenshtein
import pykakasi
import matplotlib.pyplot as plt
from colorama import Fore, Style
import difflib
import time

# %%

kks = pykakasi.kakasi()

def get_words(text):
    l = []
    for item in kks.convert(text):
        # Filter out non-hiragana characters
        word = [c for c in item['hira'] if '\u3040' <= c <= '\u309F']
        if word:
            l.append(''.join(word))
    return l

def load_audio(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000)
    return audio

# %%
# Set up device
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
torch_dtype = torch.bfloat16 if device.type == 'mps' else torch.float16
print(f"Using device: {device}")

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo", low_cpu_mem_usage=True, use_safetensors=True, torch_dtype=torch_dtype).to(device)

processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")

# Load the model and processor using the pipeline
whisper_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=30,
    torch_dtype=torch_dtype,
    device=device.index if device.type != 'cpu' else -1
)

# %%
# Load and process audio
gt_transcript = "日本には、苦しい時の神頼みという言葉がある。何か苦しい事や困った事があると神様、仏様、どうか助けて下さいと言って一生懸命お願いするけれど、何もない時は、神様や仏様のことはあまり考えていないという意味である。日本人の生活を見ると、色々な宗教的習慣や行事があることに気がつくだろう。まず、お正月には初詣といって、人々は神社やお寺にお参りに行く。"
gt_words = get_words(gt_transcript)
gt_text = ''.join(gt_words)

scores = {}
words = {}
texts = {}

def get_result(audio_path):
    audio_array = load_audio(audio_path)
    print(f"Generating {audio_path}...")
    result = whisper_pipeline(audio_array, generate_kwargs={"language": "japanese"})
    result_text = result['text']
    print(f"Raw result: {result_text}")
    texts[audio_path] = result_text

    result_words = get_words(result_text)
    words[audio_path] = result_words
    result_text = ''.join(result_words)

    edit_distance = Levenshtein.distance(result_text, gt_text)
    print(f"\nLevenshtein Distance: {edit_distance}")

    max_len = max(len(result_text), len(gt_text))
    normalized_distance = edit_distance / max_len if max_len > 0 else 0
    print(f"Normalized Levenshtein Distance: {normalized_distance:.4f}")
    scores[audio_path] = normalized_distance

# %%
for audio_path in os.listdir("audio/L6_soundfiles"):
    if audio_path.endswith(".mp4"):
        get_result(f"audio/L6_soundfiles/{audio_path}")

# Save scores and words to json
with open(f"audio/texts_{time.time()}.json", "w") as f:
    json.dump(texts, f)

# %%

def diff_text(result_text, gt_text):
    print("\nDiff comparison:")

    line1 = ""
    line2 = ""
    matcher = difflib.SequenceMatcher(None, gt_text, result_text)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            line1 += gt_text[i1:i2]
            line2 += result_text[j1:j2]
        elif tag == 'delete':
            line1 += Fore.RED + gt_text[i1:i2] + Style.RESET_ALL
            line2 += "　" * (i2 - i1)
        elif tag == 'insert':
            line1 += "　" * (j2 - j1)
            line2 += Fore.RED + result_text[j1:j2] + Style.RESET_ALL
        elif tag == 'replace':
            line1 += Fore.RED + gt_text[i1:i2] + Style.RESET_ALL
            padding = abs((i2 - i1) - (j2 - j1))
            if (i2 - i1) >= (j2 - j1):
                line2 += Fore.RED + result_text[j1:j2] + Style.RESET_ALL + "　" * padding
            else:
                line1 += "　" * padding
                line2 += Fore.RED + result_text[j1:j2] + Style.RESET_ALL

    # Print the two lines
    print(line1)
    print(line2)

diff_text(''.join(words['audio/L6_soundfiles/file_1.mp4']), gt_text)
# %%

gt_eval_scores = {'file_1.mp4': 9.5, 'file_2.mp4': 8.7, 'file_3.mp4': 6, 'file_4.mp4': 9.2, 'file_5.mp4': 7.1, 'file_6.mp4': 9.8, 'file_7.mp4': 9.5, 'file_8.mp4': 9, 'file_9.mp4': 9, 'file_10.mp4': 7.5, 'file_11.mp4': 8.8, 'file_12.mp4': 9.7, 'file_13.mp4': 7}

# Plot scores vs. gt_eval_scores, labeling each point with the filename
plt.scatter(list(scores.values()), list(gt_eval_scores.values()))
for audio_path in scores.keys():
    plt.text(scores[audio_path], gt_eval_scores[audio_path], audio_path, fontsize=8)
plt.xlabel("Whisper Edit Distance (normalized) (higher is more edits)")
plt.ylabel("Teacher Eval Score")
plt.show()

# %%