from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pyngrok import ngrok
import asyncio
import wave
import uuid
import os
import uvicorn
import logging

# 🔧 외부 모듈
from whisper_stt import transcribe_batch
from diarization import run_diarization
from audio_utils import split_audio_by_speaker, save_results

# ▶️ FastAPI 앱 생성
app = FastAPI()

# ✅ CORS 설정 (React 등 프론트와 통신 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "FastAPI WebSocket 서버가 실행 중입니다."}

# 📦 오디오 처리 설정
BUFFER_TIME_SECONDS = 10
CHUNK_RATE = 16000
TEMP_DIR = "temp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("✅ 클라이언트 WebSocket 연결됨")

    audio_buffer = bytearray()
    file_id = str(uuid.uuid4())
    wav_path = os.path.join(TEMP_DIR, f"{file_id}.wav")

    try:
        while True:
            data = await websocket.receive_bytes()
            print(f"🎙️ {len(data)}바이트 오디오 수신됨")
            audio_buffer.extend(data)

            if len(audio_buffer) >= CHUNK_RATE * 2 * BUFFER_TIME_SECONDS:
                print(f"📦 {BUFFER_TIME_SECONDS}초 분량 수신 → 저장 및 분석")

                with wave.open(wav_path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(CHUNK_RATE)
                    wf.writeframes(audio_buffer)

                diarization_result = run_diarization(wav_path)
                speaker_segments = split_audio_by_speaker(wav_path, diarization_result)
                filepaths = [path for _, path in speaker_segments]
                transcriptions = transcribe_batch(filepaths)

                results = [
                    {"speaker": speaker, "text": text}
                    for (speaker, _), text in zip(speaker_segments, transcriptions)
                ]

                for res in results:
                    print(f"🗣️ [speaker {res['speaker']}] {res['text']}")

                await websocket.send_json(results)
                print("📤 결과 전송 완료")
                audio_buffer.clear()

    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        await websocket.close()

# 🟢 서버 실행 + ngrok 통합
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # ngrok 터널 열기
    public_url = ngrok.connect(8000)
    print(f"🌐 외부 접속 주소: {public_url}")
    print(f"🔊 WebSocket 주소: {public_url}/ws/audio")

    # FastAPI 서버 실행
    uvicorn.run(app, host="0.0.0.0", port=8000)
