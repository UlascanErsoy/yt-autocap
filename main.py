from transformers import pipeline
from pytube import YouTube
import whisper
import json
from faster_whisper import WhisperModel
from tqdm import tqdm
import argparse
import requests
import os

DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")

# Run on GPU with FP16

def seconds_to_timestamp(dt: float):
    """
    hour:minute:secs,mills
    """
    mill = dt - int(dt) 
    secs = int(dt)%60
    minute = (int(dt)//60)%60
    hour = (int(dt)//60)//60

    return f"{hour:02.0f}:{minute:02.0f}:{secs:02.0f},{mill:03.0f}"

def transcribe_audio_fast(filename: str):
    """
    """
    model_size = "large-v3"
    model = WhisperModel(
                model_size, 
                device="cuda", 
                compute_type="float16"
                )
    en,_ = model.transcribe(filename, task="translate")
    zh,_ = model.transcribe(filename, task="transcribe")
    return zh,en

def request_translation(word: str):
    """
    """
    t = requests.post(
            "https://api-free.deepl.com/v2/translate",
            headers={
                "Authorization": f"DeepL-Auth-Key {DEEPL_API_KEY}"
                },
            data={
                "text":word,
                "target_lang":"EN"
                }
            )
    return t.json()

def download_yt_audio(url: str):
    """Download the highest quality audio file
    available
    """
    yt = YouTube(url)
    audio_streams = yt.streams.filter(only_audio=True)

    print("Downloading Audio")
    audio_streams = audio_streams.order_by("abr").desc()
    audio_streams.first().download(
        output_path="./", filename=yt.video_id
    )

    return yt.video_id


def transcribe_audio(filename: str):
    """transcribe the audio file"""
    model = whisper.load_model("large-v2")

    result = model.transcribe(filename)
    return result,""


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
                    prog='Whisper Auto-captions',
                    description='Auto-caption your yt videos, translation to english')

    parser.add_argument("url")
    parser.add_argument("-t", "--translate",action='store_true')
    
    args = parser.parse_args()
    video = download_yt_audio(args.url)
    zh,en = transcribe_audio_fast(video)

    subs = [] 
    zh_text = []

    print("Transcribing...")
    for item in tqdm(zh):
        subs.append([])
        subs[-1].append(f"{item.id}")
        subs[-1].append(f"{seconds_to_timestamp(item.start)} --> {seconds_to_timestamp(item.end)}")
        subs[-1].append(f"{item.text}")
        zh_text.append(item.text)
        
    en_text = []
    if args.translate:
        print("Translating...")
        zh_appended = "\n".join(zh_text)
        en_trans = request_translation(zh_appended)["translations"][0]["text"]
        for idx, item in tqdm(enumerate(en_trans.split("\n"))):
            subs[idx].append(item.strip())
            en_text.append(item)

    open(f"{video}.txt","w").write("Chinese:\n\n" + "\n".join(zh_text) + "\n" + "-"*10 + "\nEnglish:\n\n"+ "\n".join(en_text))

    final = []
    for sub in subs:
        final.extend([*sub, ""])
    open(f"{video}.srt","w").write("\n".join(final))
