from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel
import os
import uuid
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import List, Optional
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from language_detector import detect_language
from kb_query import get_answer
from voice import generate_voice

load_dotenv()

# ✅ Har baar server start ho → FAISS fresh rebuild ho
print("Rebuilding FAISS index with latest data...")
from kb_setup import build_knowledge_base
build_knowledge_base()

app = FastAPI(title="Diksha - GBPIET Chatbot", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://multilingual-college-chatbot.onrender.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chat_sessions = {}
vectorstore   = None

visit_data = {
    "total_visits": 0,
    "unique_ips": set(),
    "chatbot_usage": 0,
    "daily_counts": {},
    "first_visit": None,
    "last_visit": None,
    "unique_chatbot_users": set(),
    "user_count": 0
}

REPORT_EMAIL = "bishtsuraj0311@gmail.com"


def send_visit_report():
    sender_email = os.getenv("SMTP_EMAIL")
    sender_pass  = os.getenv("SMTP_PASSWORD")
    smtp_host    = os.getenv("SMTP_HOST", "sandbox.smtp.mailtrap.io")
    smtp_port    = int(os.getenv("SMTP_PORT", "2525"))

    if not sender_email or not sender_pass:
        print("[VisitCounter] SMTP credentials missing in .env")
        return

    today = datetime.now().strftime("%Y-%m-%d")
    daily_table = "\n".join(
        f"  {d}: {c} visits"
        for d, c in sorted(visit_data["daily_counts"].items())[-7:]
    )

    body = f"""
Diksha Chatbot Visit Report:

Total Visits     : {visit_data["total_visits"]}
Chatbot Usage    : {visit_data["chatbot_usage"]}
Unique Visitors  : {len(visit_data["unique_ips"])}
New Users        : {visit_data["user_count"]}
Today Visits     : {visit_data["daily_counts"].get(today, 0)}
First Visit      : {visit_data["first_visit"]}
Last Visit       : {visit_data["last_visit"]}

Last 7 days:
{daily_table}

Report Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
— Diksha Bot | GBPIET
"""

    msg = MIMEMultipart()
    msg["From"]    = sender_email
    msg["To"]      = REPORT_EMAIL
    msg["Subject"] = f"Diksha Report — {visit_data['user_count']} Users"
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.login(sender_email, sender_pass)
            server.sendmail(sender_email, REPORT_EMAIL, msg.as_string())
        print(f"[VisitCounter] Email sent to {REPORT_EMAIL}")
    except Exception as e:
        print(f"[VisitCounter] Email failed: {e}")


def record_visit(ip: str):
    now   = datetime.now()
    today = now.strftime("%Y-%m-%d")

    visit_data["total_visits"] += 1
    visit_data["last_visit"] = now.strftime("%Y-%m-%d %H:%M:%S")
    visit_data["unique_ips"].add(ip)
    visit_data["daily_counts"][today] = visit_data["daily_counts"].get(today, 0) + 1

    if visit_data["first_visit"] is None:
        visit_data["first_visit"] = visit_data["last_visit"]

    if ip not in visit_data["unique_chatbot_users"]:
        visit_data["unique_chatbot_users"].add(ip)
        visit_data["user_count"] += 1
        print(f"[User] New chatbot user: {ip}")
        if visit_data["user_count"] % 10 == 0:
            send_visit_report()


def load_faiss_background():
    global vectorstore
    try:
        print("Loading FAISS index in background...")
        from langchain_community.embeddings import FastEmbedEmbeddings
        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        index_path = os.path.join(os.path.dirname(__file__), "faiss_index")
        vectorstore = FAISS.load_local(
            index_path, embeddings,
            allow_dangerous_deserialization=True
        )
        import kb_query
        kb_query._vs_cache = vectorstore
        print("Diksha is ready!")
    except Exception as e:
        print(f"FAISS load error: {e}")


@app.on_event("startup")
async def startup_event():
    thread = threading.Thread(target=load_faiss_background, daemon=True)
    thread.start()
    print("Server started! FAISS loading in background...")


class TTSRequest(BaseModel):
    text: str
    lang: str = "en"

class ChatRequest(BaseModel):
    question:         str
    session_id:       Optional[str] = None
    is_first_message: bool          = False
    language:         Optional[str] = None

class ChatResponse(BaseModel):
    answer:       str
    language:     str
    session_id:   str
    chatbot_name: str = "Diksha"

class HistoryResponse(BaseModel):
    session_id: str
    messages:   List[dict]


@app.get("/")
def home():
    return {"chatbot": "Diksha", "status": "running"}

@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"status": "ok"}

@app.post("/tts")
async def tts_endpoint(request: TTSRequest):
    audio = await run_in_threadpool(generate_voice, request.text, request.lang)
    return {"audio_base64": audio}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request):
    question = request.question.strip()

    if vectorstore is None:
        return ChatResponse(
            answer="Diksha is starting up, please wait 30 seconds and try again!",
            language="en",
            session_id=request.session_id or str(uuid.uuid4())
        )

    forwarded = req.headers.get("x-forwarded-for")
    ip = forwarded.split(",")[0].strip() if forwarded else req.client.host

    record_visit(ip)
    visit_data["chatbot_usage"] += 1

    if request.language and request.language in ['en', 'hi', 'ga', 'ku']:
        lang = request.language
    else:
        lang = detect_language(question)

    session_id = request.session_id or str(uuid.uuid4())

    if session_id not in chat_sessions:
        chat_sessions[session_id] = []

    history_text = ""
    if chat_sessions[session_id]:
        history_text = "Previous conversation:\n"
        for msg in chat_sessions[session_id][-6:]:
            role = "Student" if msg["role"] == "user" else "Diksha"
            history_text += f"{role}: {msg['message']}\n"

    # ✅ get_answer me ab keywords.json SABSE PEHLE check hoga
    answer = await run_in_threadpool(get_answer, question, lang, history_text)

    if not answer:
        answer = "Sorry, I don't have enough information for this query. Please try rephrasing."

    chat_sessions[session_id].append({
        "role": "user",
        "message": question,
        "language": lang,
        "timestamp": datetime.now().isoformat()
    })

    chat_sessions[session_id].append({
        "role": "diksha",
        "message": answer,
        "language": lang,
        "timestamp": datetime.now().isoformat()
    })

    return ChatResponse(
        answer=answer,
        language=lang,
        session_id=session_id
    )


@app.get("/history/{session_id}", response_model=HistoryResponse)
def get_history(session_id: str):
    return HistoryResponse(
        session_id=session_id,
        messages=chat_sessions.get(session_id, [])
    )

@app.get("/sessions")
def get_sessions():
    return {
        "total_sessions": len(chat_sessions),
        "session_ids":    list(chat_sessions.keys()),
    }

@app.get("/admin/visits")
def get_visit_stats():
    return {
        "total_visits":         visit_data["total_visits"],
        "chatbot_usage":        visit_data["chatbot_usage"],
        "unique_chatbot_users": len(visit_data["unique_chatbot_users"]),
        "unique_visitors":      len(visit_data["unique_ips"]),
        "daily_counts":         visit_data["daily_counts"],
        "first_visit":          visit_data["first_visit"],
        "last_visit":           visit_data["last_visit"],
        "faiss_loaded":         vectorstore is not None,
    }

@app.get("/admin/send-report")
def send_report_now():
    send_visit_report()
    return {"status": "Report sent", "to": REPORT_EMAIL}
