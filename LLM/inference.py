import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from parsing import extract_fields_from_transcript
from peft import PeftModel
import ast
import json
import re

# ====== [1] 모델 경로 설정 ======
base_model_name = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
lora_model_path = "C:/Users/user/Desktop/dentary/eeve_lora/checkpoint-468"

# ====== [2] 디바이스 설정 ======
device = "cuda" if torch.cuda.is_available() else "cpu"

# ====== [3] Tokenizer 로딩 ======
tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# ====== [4] Base + LoRA 모델 로딩 ======
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="auto",
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)
model = PeftModel.from_pretrained(base_model, lora_model_path)
model.eval()

# ====== [5] 프롬프트 입력 ======
instruction = "아래 키워드를 바탕으로 환자 상태를 설명하고, 상담자 입장에서 적절한 진료 권장 대사를 작성하세요."

with open("C:/Users/user/Desktop/dentary/input/transcription_002.json", "r", encoding="utf-8") as f:
    transcript_data = json.load(f)

input_text = extract_fields_from_transcript(transcript_data)

# ====== 치료 불필요 예외 처리 ======
if input_text.strip() == "해당 환자는 치료가 필요없습니다.":
    print("\n=== 치료 불필요 케이스입니다. 모델 호출을 생략합니다. ===")
    exit()

print("=== LLM 입력 ===")
print("Instruction:")
print(instruction)
print("Input Text:")
print(input_text)

# ====== [6] Few-shot 예시 포함 프롬프트 구성 ======
prompt = f"""Instruction: {instruction}
Input: 위치: 치식번호 11, 문제: 치아 파절, 처치: 신경치료, 예상기간: 4주
Output:
{{
  "치식": "#11(#11(오른쪽위첫번째앞니))",
  "치료분류": "보존치료",
  "치료항목": "신경치료",
  "치료항목1": "초진+방사선촬영",
  "치료항목2": "신경치료",
  "치료항목3": "",
  "치료기간" : "약 4주",
  "건강보험" : "급여",
  "관련수가코드": "AA010, EB711",
  "총진료비수가": "약 11,000원",
  "본인부담금(30%/급여)": "약 3,300원",
  "비급여비용": "",
  "총부담비용": "약 14,300원",
  "메모(메시지)": ""
}}

Instruction: {instruction}
Input: 위치: 치식번호 55, 문제: 보철 손상, 처치: 보철치료(금), 예상기간: 4주
Output:
{{
  "치식": "#48(#48(오른쪽아래사랑니))",
  "치료분류": "보철치료",
  "치료항목": "보철치료(금)",
  "치료항목1": "초진+방사선촬영",
  "치료항목2": "보철치료(금)",
  "치료항목3": "",
  "치료기간" : "약 4주",
  "건강보험" : "비급여",
  "관련수가코드": "AA010, EB711",
  "총진료비수가": "약 11,000원",
  "본인부담금(30%/급여)": "약 3,300원",
  "비급여비용": "약 600,000원",
  "총부담비용": "약 655,000원",
  "메모(메시지)": "보철비용은 병원마다 상이할 수 있음"
}}

Instruction: {instruction}
Input: {input_text}
Output:
"""

# ====== [7] 토크나이징 및 생성 ======
inputs = tokenizer(prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
    )

# ====== [8] 출력 처리 ======
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
generated_text = response.split("Output:")[-1].strip()

print("\n=== 모델 원본 출력 ===")
print(generated_text)

# ====== [9] JSON 후처리 및 치식 보정 ======
TOOTH_MAP = {
    "11": "#11(오른쪽위첫번째앞니)", "12": "#12(오른쪽위두번째앞니)", "13": "#13(오른쪽위송곳니)",
    "14": "#14(오른쪽위첫째작은어금니)", "15": "#15(오른쪽위둘째작은어금니)", "16": "#16(오른쪽위첫째큰어금니)",
    "17": "#17(오른쪽위둘째큰어금니)", "18": "#18(오른쪽위사랑니)",
    "21": "#21(왼쪽위첫번째앞니)", "22": "#22(왼쪽위두번째앞니)", "23": "#23(왼쪽위송곳니)",
    "24": "#24(왼쪽위첫째작은어금니)", "25": "#25(왼쪽위둘째작은어금니)", "26": "#26(왼쪽위첫째큰어금니)",
    "27": "#27(왼쪽위둘째큰어금니)", "28": "#28(왼쪽위사랑니)",
    "31": "#31(왼쪽아래첫번째앞니)", "32": "#32(왼쪽아래두번째앞니)", "33": "#33(왼쪽아래송곳니)",
    "34": "#34(왼쪽아래첫째작은어금니)", "35": "#35(왼쪽아래둘째작은어금니)", "36": "#36(왼쪽아래첫째큰어금니)",
    "37": "#37(왼쪽아래둘째큰어금니)", "38": "#38(왼쪽아래사랑니)",
    "41": "#41(오른쪽아래첫번째앞니)", "42": "#42(오른쪽아래두번째앞니)", "43": "#43(오른쪽아래송곳니)",
    "44": "#44(오른쪽아래첫째작은어금니)", "45": "#45(오른쪽아래둘째작은어금니)", "46": "#46(오른쪽아래첫째큰어금니)",
    "47": "#47(오른쪽아래둘째큰어금니)", "48": "#48(오른쪽아래사랑니)"
}

try:
    if generated_text.startswith('"') and generated_text.endswith('"'):
        cleaned = {"응답": ast.literal_eval(generated_text)}
        print("\n=== 문자열 출력 대응 결과 ===")
        print(json.dumps(cleaned, ensure_ascii=False, indent=2))
    else:
        parsed = ast.literal_eval(generated_text)

        # ✅ [1] 여러 치식번호 추출
        tooth_nums = re.findall(r"#(\d{2})", input_text)
        corrected_teeth = [TOOTH_MAP.get(num, f"#{num}(Unknown)") for num in tooth_nums]

        # ✅ [2] 치식 필드 보정
        if corrected_teeth:
            if len(corrected_teeth) == 1:
                parsed["치식"] = corrected_teeth[0]
            else:
                parsed["치식"] = corrected_teeth

        # ✅ [3] 치료기간 → 예상기간 필드명 보정
        if "치료기간" in parsed and "예상기간" not in parsed:
            parsed["예상기간"] = parsed.pop("치료기간")
        elif "기간" in parsed and "예상기간" not in parsed:
            parsed["예상기간"] = parsed.pop("기간")

        print("\n=== JSON 파싱 및 치식 보정 성공 ===")
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
except Exception as e:
    print("\n⚠️ JSON 파싱 실패:", e)
    print("⚠️ 원본 출력 그대로 사용:")
    print(generated_text)