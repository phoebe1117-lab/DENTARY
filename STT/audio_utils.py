from pydub import AudioSegment
import json

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

def save_results(speaker_segments, transcriptions, output_dir="output_results"):
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "transcription.json")
    json_data = []
    for (speaker, _), text in zip(speaker_segments, transcriptions):
        json_data.append({"speaker": speaker, "text": text})
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)