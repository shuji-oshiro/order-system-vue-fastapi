import os
import whisper
import json
import wave
from openai import OpenAI
from dotenv import load_dotenv

import sys
from vosk import Model, KaldiRecognizer

load_dotenv(dotenv_path="backend/.env")

# OpenAI APIクライアントの初期化
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Whisperモデルのロード
# ローカルモデルを使用する場合は、whisper.load_model("base")を使用
model = whisper.load_model("base")


# ローカルモデルを使用して音声ファイルを文字起こしする関数
def transcribe_audiofile_on_localmodel(file_path: str):
    result = model.transcribe(file_path, language="ja")
    reco_text = result.get("text", "")
    
    if not isinstance(reco_text, str):
        raise TypeError("Whisper result['text'] is not a string")
    
    reco_text = reco_text.strip()
    return reco_text

# # OpenAI APIを使用して音声ファイルを文字起こしする関数
def transcribe_audiofile_on_APImodel(file_path: str):
    with open(file_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    reco_text = transcript.text.strip()
    return reco_text


# VOSKモデルのロード
base_dir = os.path.dirname(__file__)
model_path = os.path.join(base_dir, "models/vosk-model-small-ja-0.22")
model_vosk = Model(model_path)
   

def transcribe_audiofile_on_Vosk(file_path: str):

    # WAVファイルを開く
    try:
        wf = wave.open(file_path, "rb")
    except FileNotFoundError:
        print(f"音声ファイル {file_path} が見つかりません。")
        sys.exit(1)

    # WAVファイルのチェック
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
        print("⚠️ WAVファイル形式が VOSK に適していません（必要: mono/16bit/16kHz）")
        sys.exit(1)

    # menu_items = ["からあげ", "ラーメン", "カレーライス", "うどん"]
    # 認識器の初期化
    recognizer = KaldiRecognizer(model_vosk, wf.getframerate())

    # 音声認識
    print("🎧 音声認識を開始...")
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if recognizer.AcceptWaveform(data):
            result = recognizer.Result()
            print("✅ 認識結果:", result)

    # 最後の部分
    final_result = recognizer.FinalResult()
    return final_result