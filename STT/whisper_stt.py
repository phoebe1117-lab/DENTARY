import torch
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import os
from datasets import Dataset
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "openai/whisper-large-v3-turbo"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    token=HF_TOKEN
).to(device)

processor = AutoProcessor.from_pretrained(model_id)

asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=0 if device == "cuda" else -1,
    return_timestamps=False
)

def transcribe_batch(filepaths):
    dataset = Dataset.from_dict({"audio": filepaths})
    results = asr_pipeline(dataset["audio"])
    return [r["text"] for r in results]
