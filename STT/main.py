import os
from pydub import AudioSegment
from dotenv import load_dotenv
from datasets import Dataset
import json
import csv

# 외부에서 가져옴
from whisper_stt import transcribe_batch
from diarization import run_diarization

# 환경 변수 로드
load_dotenv()

# 화자별 오디오 분할 함수
def split_audio_by_speaker(audio_path, diarization_result, output_dir="output_chunks"):
    os.makedirs(output_dir, exist_ok=True)
    audio = AudioSegment.from_file(audio_path)
    speaker_segments = []

    for i, segment in enumerate(diarization_result):
        start_ms = int(segment["start"] * 1000)
        end_ms = int(segment["end"] * 1000)
        chunk = audio[start_ms:end_ms]
        filename = f"{output_dir}/speaker_{segment['speaker']}_segment_{i}.wav"
        chunk.export(filename, format="wav")
        speaker_segments.append((segment["speaker"], filename))

    return speaker_segments

# 결과 저장 함수
def save_results(speaker_segments, transcriptions, output_dir="output_results"):
    os.makedirs(output_dir, exist_ok=True)

    # 3. .json 저장
    json_path = os.path.join(output_dir, "transcription3.json")
    json_data = []
    for (speaker, _), text in zip(speaker_segments, transcriptions):
        json_data.append({"speaker": speaker, "text": text})
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

# 메인 실행 함수
def main(audio_path):
    print("1. 화자 분리 중...")
    diarization_result = run_diarization(audio_path)

    print("2. 화자별 오디오 분할 중...")
    speaker_segments = split_audio_by_speaker(audio_path, diarization_result)

    print("3. STT 배치 실행 중...")
    filepaths = [path for _, path in speaker_segments]
    transcriptions = transcribe_batch(filepaths)

    for (speaker, _), text in zip(speaker_segments, transcriptions):
        print(f"[{speaker}] {text}")

    save_results(speaker_segments, transcriptions)

if __name__ == "__main__":
    audio_path = "C:/Users/asia/Desktop/vscode/backend/situation_test_000 (4).wav"
    main(audio_path)