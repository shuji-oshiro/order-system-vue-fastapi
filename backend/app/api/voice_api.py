import os
import subprocess
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.app.schemas.voice_schema import VoiceResult,MatchMenu
from backend.app.service.voice_handler import transcribe_audiofile_on_localmodel
from backend.app.service.fuzzy_matcher import fuzzy_menu_match

router = APIRouter()
base_dir = os.path.dirname(__file__)
input_path = os.path.join(base_dir, "test_sample.webm")
output_path = os.path.join(base_dir, "test_sample.wav")

# webm音声からwavデータを変換する関数
async def change_voicedata(file: UploadFile):
    # 保存
    try:
        with open(input_path, "wb") as f:
            f.write(await file.read())

        # ffmpeg で .webm → .wav に変換
        cmd = [
            "ffmpeg", "-i", input_path,
            "-ar", "16000",  # Whisper用に16kHzに変換
            "-ac", "1",      # モノラル
            "-y", output_path
        ]
        subprocess.run(cmd, check=True)

        with open(output_path, "wb") as buffer:
                buffer.write(await file.read())

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"音声データ変換エラー: {str(e)}")

@router.post("/", response_model=VoiceResult)
async def handle_voice(file: UploadFile = File(...)):

    await change_voicedata(file)

    try:
        # 音声ファイルをローカルモデルで処理
        reco_text = transcribe_audiofile_on_localmodel(input_path)
        # reco_text = transcribe_audiofile_on_Vosk(input_path)

        # 音声認識結果をマッチング
        match_orders = fuzzy_menu_match(reco_text)

        if len(match_orders) == 0:
            raise HTTPException(status_code=400, detail=f"音声認識エラー: {reco_text}")

        # スコアが高い順にソート
        match_orders = sorted(match_orders, key=lambda x: x[1], reverse=True)

        return VoiceResult(
                    reco_text=reco_text,
                    match_menus=[
                        MatchMenu(menu=menu, score=score) for menu, score in match_orders
                    ]
                )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"音声処理エラー: {str(e)}")
    
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
        if os.path.exists(output_path):
            os.remove(output_path)

