import io
import os
import sys
import json
import wave
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import subprocess
import sounddevice as sd
from scipy.io.wavfile import write
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

wav_file = "test_sample.wav"
webm_file = "test_sample.webm"

# 検証用音声データを作成
def record_wav(duration=3, fs=16000):

    print("録音準備ができたらEnterキーを押してください...")
    input()

    print("録音開始（話してください）...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write(wav_file, fs, audio)
    print("録音終了:", wav_file)

    cmd = [
        "ffmpeg", "-y", "-i", wav_file,
        "-c:a", "libopus", "-b:a", "32k", webm_file
    ]
    subprocess.run(cmd, check=True)
    print("変換完了:", webm_file)




def test_voice_recognition_success():
    try:
        # テスト用音声ファイルを録音して変換
        record_wav()

        with open(wav_file, "rb") as f:
            files = {"file": (webm_file, f, "audio/webm")}
            #files = {"file": (wav_file, f, "audio/wav")}
            response = client.post("/voice/", files=files)
            
        data = response.json()      
        if response.status_code != 200:
            print("エラー:", response.status_code, data)
            return
        print("音声認識結果:", data.get("reco_text", ""))
        match_menus = data.get("match_menus", [])

        if match_menus:
            print("マッチしたメニュー:")
            for menu in match_menus:
                menu_obj = json.loads(json.dumps(menu["menu"], ensure_ascii=False))
                print("-", menu_obj["name"], "(スコア:", menu["score"], ")")
        else:
            print("マッチしたメニューはありません。")

    finally:
        if os.path.exists(wav_file):
            os.remove(wav_file)
        if os.path.exists(webm_file):
            os.remove(webm_file)



if __name__ == "__main__":
    test_voice_recognition_success()
    print("テストが完了しました。")