import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("C:/Users/user/Desktop/dentary/dentary-fb288-firebase-adminsdk-fbsvc-45613b4bdc.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

# User 컬렉션 생성 및 임시 문서 1개 추가
db.collection("users").document("user001").set({
    "name" : "홍길동",
    "email" : "user@exmaple.com",
    "role" : "patient", # "patient" | "consultant" | "admin"
    "sex" : "male",
    "age" : "28",
})

# recordings 컬렉션 생성 임시 문서 1개 추가
db.collection("recordings").document("rec001").set({
    "userId" : "user001",
    "fileUrl" : "https://...",
    "createdAt": firestore.SERVER_TIMESTAMP,
    "type" : "original",
    "transcriptId" : "script001"
})

# scripts 컬렉션 생성 임시 문서 1개 추가
db.collection("scripts").document("script001").set({
    "scriptId" : "script001",
    "recordingId" : "rec001",
    "userType" : "patient",
    "summary" : "오른쪽 어금니 신경치료 및 크라운 치료 요망", # 	해당 음성 파일을 분석한 뒤, NLP를 통해 생성된 상담 요약 텍스트
    "createdAt" : firestore.SERVER_TIMESTAMP
})

db.collection("transcriptions").document("txt001").set({
    "UserId" : "user001",
    "Text" : ""
})