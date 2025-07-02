import json
import re
import os

LOCATION_MAP_PATH = os.path.join(os.path.dirname(__file__), "location_map_refined.json")
try:
    with open(LOCATION_MAP_PATH, "r", encoding="utf-8") as f:
        location_map = json.load(f)
except FileNotFoundError:
    print("⚠️ location_map_final.json 파일을 찾을 수 없습니다.")
    location_map = {}

# 치료 항목별 기본 기간 매핑
TREATMENT_DURATION_MAP = {
    "GI": "1일",
    "레진": "1일",
    "인레이": "1주",
    "신경치료": "1주~2주",
    "올세라믹": "1주~2주",
    "금": "1주~2주",
    "지르코니아": "1주~2주",
    "PFM": "1주~2주",
    "스케일링": "1일",
    "잇몸치료": "2~4주",
    "발치": "1일",
    "난발치": "2~3일",
    "매복치": "3~4일",
    "임플란트 1차수술": "5~6개월",
    "임플란트 보철치료": "2~3주",
    "라미네이트": "1~2주"
}

# 불용 조사 및 표현 정규화
def normalize_korean_text(text):
    text = text.lower()
    text = re.sub(r"[가-힣]+[이가은는를에의]", "", text)  # 조사 제거
    text = re.sub(r"첫\s*번째", "첫번째", text)
    text = re.sub(r"두\s*번째", "두번째", text)
    text = re.sub(r"첫\s*째", "첫번째", text)
    text = re.sub(r"둘\s*째", "두번째", text)
    text = text.replace("작은 어금니", "작은어금니").replace("큰 어금니", "큰어금니")
    return text.replace(" ", "")

def extract_durations(text: str):
    patterns = [
        r"\d+\s*개월\s*에서\s*\d+\s*개월",
        r"\d+\s*~\s*\d+\s*(일|주|개월)",
        r"(약\s*)?\d+\s*(일|주|개월)",
        r"\d+\s*(일|주|개월)\s*정도",
        r"\d+\s*(일|주|개월)\s*간"
    ]

    matches = []
    for pattern in patterns:
        found = re.findall(pattern, text)
        # 정규식 결과가 튜플일 수 있으므로 처리
        for f in found:
            if isinstance(f, tuple):
                matches.append("".join(f).strip())
            else:
                matches.append(f.strip())

    # 숫자가 포함된 경우만 필터링
    matches = [m for m in matches if re.search(r"\d+", m)]
    return list(set(matches))

def extract_location_from_text(text, location_map):
    norm_text = normalize_korean_text(text)
    for key, value in location_map.items():
        norm_key = normalize_korean_text(key)
        if norm_key in norm_text:
            return f"치식번호 {value}"
    match = re.search(r"(치식번호\s*[\d#]+|#\d+)", text)
    if match:
        return match.group(0).replace(" ", "")
    return None

def extract_fields_from_transcript(transcript_json):
    full_text = " ".join([item["text"].strip() for item in transcript_json])
    full_text = full_text.replace("  ", " ").strip()

    # 위치 추출
    location = None
    for desc, code in location_map.items():
        pattern = re.sub(r"\s*", r"\\s*", re.escape(desc))
        if re.search(pattern, full_text):
            location = f"#" + code
            break

    # 문제 추출
    problem_patterns = {
        r"깨(졌|짐|진|지|질|져)": "깨짐",
        r"금(가서|이나서)" : "깨짐",
        r"부러(졌|짐|진|져|질|져서)" : "파손",
        r"파손": "파손",
        r"아(파요|파서|프다|픔|팠어|팠어요|팟어요)" : "통증",
        r"통증": "통증",
        r"시(리다|려서|림|려요)" : "시림",
        r"붓기|부었|부어서": "붓기",
        r"염증": "염증",
        r"출혈|피": "출혈",
        r"잇몸": "잇몸 문제",
        r"충치": "충치"
    }
    
    problem = None
    for pattern, label in problem_patterns.items():
        if re.search(pattern, full_text):
            problem = label
            break

    # 처치 추출
    treatment_keywords = list(TREATMENT_DURATION_MAP.keys())
    treatment = next((kw for kw in treatment_keywords if kw in full_text), None)

    # 기간 추출
    duration_matches = extract_durations(full_text)
    duration = ", ".join(duration_matches) if duration_matches else None

    # 처치 기반 fallback 보정
    if not duration and treatment and treatment in TREATMENT_DURATION_MAP:
        duration = TREATMENT_DURATION_MAP[treatment]

    if not location and not problem and not treatment:
        return "해당 환자는 치료가 필요없습니다."

    location_str = f"위치: {location or '미확인'}"
    problem_str = f"문제: {problem or '미확인'}"
    treatment_str = f"처치: {treatment or '미확인'}"
    duration_str = f"예상기간: {duration or '미확인'}"

    return f"{location_str}, {problem_str}, {treatment_str}, {duration_str}"