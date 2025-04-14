#_*_encoding:utf-8_*_
import time
import hashlib
import requests
import json
import dotenv
import os
from colorama import Fore, Style, init
from tqdm import tqdm
import matplotlib.pyplot as plt

dotenv.load_dotenv()

appKey = os.getenv("SPEECHSUPER_API_KEY")
secretKey = os.getenv("SPEECHSUPER_SECRET_KEY")

coreType = "sent.eval.jp"

def get_speech_evaluation(audioPath, refText):
    baseURL = "https://api.speechsuper.com/"
    timestamp = str(int(time.time()))
    audioType = "wav" # Change the audio type corresponding to the audio file.
    audioSampleRate = 16000
    userId = "guest"
    url =  baseURL + coreType
    connectStr = (appKey + timestamp + secretKey).encode("utf-8")
    connectSig = hashlib.sha1(connectStr).hexdigest()
    startStr = (appKey + timestamp + userId + secretKey).encode("utf-8")
    startSig = hashlib.sha1(startStr).hexdigest()

    params={
        "connect":{
            "cmd":"connect",
            "param":{
                "sdk":{
                    "version":16777472,
                    "source":9,
                    "protocol":2
                },
                "app":{
                    "applicationId":appKey,
                    "sig":connectSig,
                    "timestamp":timestamp
                }
            }
        },
        "start":{
            "cmd":"start",
            "param":{
                "app":{
                    "userId":userId,
                    "applicationId":appKey,
                    "timestamp":timestamp,
                    "sig":startSig
                },
                "audio":{
                    "audioType":audioType,
                    "channel":1,
                    "sampleBytes":2,
                    "sampleRate":audioSampleRate
                },
                "request":{
                    "coreType":coreType,
                    "refText":refText,
                    "tokenId":"tokenId"
                }

            }
        }
    }

    datas=json.dumps(params)
    data={'text':datas}
    headers={"Request-Index":"0"}
    files={"audio":open(audioPath,'rb')}
    res=requests.post(url, data=data, headers=headers, files=files)

    return res

def parse_speech_evaluation(json_string, word_by_word=False, phoneme_by_phoneme=False):
    """
    Parse Japanese speech evaluation API output and display in formatted text
    
    Args:
        json_string: The JSON string response from the API
    """
    # Initialize colorama for cross-platform colored terminal output
    init()
    
    try:
        data = json.loads(json_string) if isinstance(json_string, str) else json_string
        result = data["result"]
        
        print(f"{Fore.CYAN}OVERALL SCORES{Style.RESET_ALL}\n{'-'*50}")
        score_items = [
            ("Overall Score", result["overall"]),
            ("Pronunciation", result["pronunciation"]),
            ("Fluency", result["fluency"]),
            ("Rhythm", result["rhythm"]),
            ("Tone", result["tone"]),
            ("Integrity", result["integrity"]),
            ("Speed", result["speed"])
        ]
        
        max_label_width = max(len(item[0]) for item in score_items) + 2
        for label, score in score_items:
            color = Fore.GREEN if score >= 90 else Fore.YELLOW if score >= 75 else Fore.RED
            print(f"{label + ':': <{max_label_width}} {color}{score}{Style.RESET_ALL}/100")
        
        if word_by_word:
            print(f"\n{Fore.CYAN}WORD-BY-WORD ANALYSIS{Style.RESET_ALL}\n{'-'*50}")
            print(f"{'Word': <12} {'Pronunciation': <15} {'Overall': <15} {'Tone': <10}")
            print(f"{'-'*12} {'-'*15} {'-'*15} {'-'*10}")
            
            for word_data in result["words"]:
                word = word_data["word"]
                pron_score = word_data["scores"]["pronunciation"]
                overall = word_data["scores"]["overall"]
                tone_score = word_data["tone_stats"]["tone_score"]
                
                def colorize(score):
                    return f"{Fore.GREEN}{score}{Style.RESET_ALL}" if score >= 90 else f"{Fore.YELLOW}{score}{Style.RESET_ALL}" if score >= 75 else f"{Fore.RED}{score}{Style.RESET_ALL}"
                
                print(f"{word: <12} {colorize(pron_score): <15} {colorize(overall): <15} {colorize(tone_score): <10}")
        
        if phoneme_by_phoneme:
            print(f"\n{Fore.CYAN}DETAILED PHONEME ANALYSIS{Style.RESET_ALL}\n{'-'*50}")
            
            for i, word_data in enumerate(result["words"]):
                word = word_data["word"]
                print(f"\n{Fore.YELLOW}Word {i+1}: {word}{Style.RESET_ALL}")
                print(f"{'Phoneme': <10} {'Pronunciation': <15} {'Tone': <10} {'Time (ms)': <15}")
                print(f"{'-'*10} {'-'*15} {'-'*10} {'-'*15}")
                
                for phoneme_data in word_data["phonemes"]:
                    phoneme = phoneme_data["phoneme"]
                    pron_score = phoneme_data["pronunciation"]
                    tone = phoneme_data["tone"]
                    start, end = phoneme_data["span"]["start"], phoneme_data["span"]["end"]
                    duration = end - start
                    color = Fore.GREEN if pron_score >= 90 else Fore.YELLOW if pron_score >= 75 else Fore.RED
                    print(f"{phoneme: <10} {color}{pron_score}{Style.RESET_ALL: <15} {tone: <10} {start}-{end} ({duration}ms)")

        print(f"Rear Tone Pattern: {result['rear_tone']}")
        
    except Exception as e:
        print(f"{Fore.RED}Error parsing speech evaluation data: {str(e)}{Style.RESET_ALL}")


gt_eval_scores = {'file_1.mp4': 9.5, 'file_2.mp4': 8.7, 'file_3.mp4': 6, 'file_4.mp4': 9.2, 'file_5.mp4': 7.1, 'file_6.mp4': 9.8, 'file_7.mp4': 9.5, 'file_8.mp4': 9, 'file_9.mp4': 9, 'file_10.mp4': 7.5, 'file_11.mp4': 8.8, 'file_12.mp4': 9.7, 'file_13.mp4': 7}
gt_scores = list(gt_eval_scores.values())

scores = []

for i in tqdm(range(1, 14)):
    audioPath = f"audio/L6_soundfiles/file_{i}.wav"
    refText = "日本には、苦しい時の神頼みという葉がある。何か苦しい事や困った事があると神様、仏様、どうか助けて下さいと言って一生懸命お願いするけれど、何もない時は、神様や仏様のことはあまり考えていないという意味である。日本人の生活を見ると、色々な宗教的習慣や行事があることに気がつくだろう。まず、お正月には初詣といって、人々は神社やお寺にお参りに行く。"
    res = get_speech_evaluation(audioPath, refText)
    res_json = json.loads(res.text)
    parse_speech_evaluation(res_json, word_by_word=True, phoneme_by_phoneme=True)
    exit()
    scores.append(res_json["result"]["overall"]/10)


# Plot the correlation between gt_scores and scores
plt.scatter(scores, gt_scores)
# for i in range(len(scores)):
#     plt.text(scores[i], gt_scores[i], f"file_{i+1}.wav", fontsize=8)
plt.xlabel("SpeechSuper Overall Score")
plt.ylabel("Teacher Eval Score")
plt.show()

