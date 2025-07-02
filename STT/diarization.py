import os
import torch
from dotenv import load_dotenv
from pyannote.audio import Pipeline

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=HF_TOKEN
)

if torch.cuda.is_available():
    diarization_pipeline.to(torch.device("cuda"))

def run_diarization(audio_path):
    diarization = diarization_pipeline(audio_path)
    results = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        results.append({
            "speaker": speaker,
            "start": round(turn.start, 2),
            "end": round(turn.end, 2)
        })
    return results
