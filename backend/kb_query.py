import os
import json
import glob
import re
from langchain_community.vectorstores import FAISS
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client       = Groq(api_key=os.getenv("GROQ_API_KEY"))
_vs_cache    = None
_qa_database = []

# ══════════════════════════════════════════════════════════
# LOAD ALL QA INTO MEMORY
# ══════════════════════════════════════════════════════════
def load_qa_database():
    global _qa_database
    if _qa_database:
        return _qa_database

    data_folder = os.path.join(os.path.dirname(__file__), "data")
    for filepath in sorted(glob.glob(os.path.join(data_folder, "*.json"))):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            for item in data:
                if isinstance(item, dict) and "question" in item and "answer" in item:
                    _qa_database.append({
                        "question": item["question"].strip(),
                        "answer":   item["answer"].strip(),
                        "source":   os.path.basename(filepath)
                    })
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    print(f"[DB] Loaded {len(_qa_database)} QA pairs")
    return _qa_database


def get_vectorstore():
    """
    FIX #1: Use the SAME embedding model everywhere.
    kb_setup.py uses FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    so we must use the same model here. Never mix models.
    """
    global _vs_cache
    if _vs_cache is None:
        from langchain_community.embeddings import FastEmbedEmbeddings
        embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
        index_path = os.path.join(os.path.dirname(__file__), "faiss_index")
        _vs_cache = FAISS.load_local(
            index_path, embeddings,
            allow_dangerous_deserialization=True
        )
        print("[DB] FAISS loaded with BAAI/bge-small-en-v1.5")
    return _vs_cache


# ══════════════════════════════════════════════════════════
# HINDI → ENGLISH MAP
# ══════════════════════════════════════════════════════════
HINDI_MAP = {
    'जीबीपीआईईटी': 'gbpiet', 'जीबीपीईटी': 'gbpiet',
    'संस्थान': 'institute',   'कॉलेज': 'college',
    'पहुँचें': 'reach',        'पहुंचें': 'reach',
    'कैसे': 'how',             'रास्ता': 'route direction',
    'पता': 'address',          'कहाँ': 'where', 'कहां': 'where',
    'निदेशक': 'director',
    'विभागाध्यक्ष': 'head department hod',
    'अध्यक्ष': 'chairman',    'डीन': 'dean',
    'शिक्षक': 'faculty',       'प्राध्यापक': 'professor faculty',
    'संकाय': 'faculty',        'वार्डन': 'warden',
    'प्रवेश': 'admission',     'दाखिला': 'admission',
    'आवेदन': 'apply',          'पात्रता': 'eligibility',
    'कोर्स': 'courses',        'शाखा': 'branch',
    'फीस': 'fees',             'शुल्क': 'fees',
    'छात्रवृत्ति': 'scholarship',
    'हॉस्टल': 'hostel',        'छात्रावास': 'hostel',
    'लड़कियों': 'girls',        'लड़कों': 'boys',
    'प्रथम वर्ष': 'first year',
    'प्लेसमेंट': 'placement',  'पैकेज': 'package',
    'पुस्तकालय': 'library',    'खेल': 'sports',
    'परिवहन': 'transport',     'परिणाम': 'result',
    'परीक्षा': 'exam',         'रैगिंग': 'ragging',
    'संपर्क': 'contact',       'फोन': 'phone',
    'कितने': 'how many',        'कौन से': 'which'
}

def hi_to_en(text: str) -> str:
    t = text.lower()
    for h, e in HINDI_MAP.items():
        t = t.replace(h, ' ' + e + ' ')
    return re.sub(r'\s+', ' ', t).strip()


# ══════════════════════════════════════════════════════════
# STEP 1 — EXACT MATCH
# ══════════════════════════════════════════════════════════
def exact_match(question: str) -> str | None:
    q = question.strip().lower()
    for item in load_qa_database():
        if q == item["question"].lower():
            print(f"[EXACT] {item['question'][:60]}")
            return item["answer"]
    return None


# ══════════════════════════════════════════════════════════
# STEP 2 — KEYWORD MATCH (stricter threshold)
# ══════════════════════════════════════════════════════════
STOP = {
    'what','who','is','are','the','at','in','of','a','an','and','or',
    'for','to','how','does','do','has','have','many','which','tell',
    'me','about','please','can','you','i','my','their','kya','hai',
    'hain','ka','ki','ke','mein','se','per','ek','क्या','कौन','का',
    'की','के','में','से','है','हैं','एक','और','या','को','ने','था',
    'थी','थे','कि','जो','तो','भी','मैं','हम','आप','वे','इस','उस',
    'यह','वह','पर','बारे','कैसे','कहाँ','कहां','तक','gbpiet'
}

def get_keywords(text: str) -> set:
    words      = set(re.findall(r'[\u0900-\u097F]+|[a-zA-Z0-9]+', text.lower()))
    translated = set(re.findall(r'[a-zA-Z0-9]+', hi_to_en(text)))
    return (words | translated) - STOP

def keyword_match(question: str, threshold: int = 3) -> str | None:
    """
    FIX #2: Raised minimum threshold to 3 and minimum score to 0.4
    to avoid wrong FAQ matches with only 2 overlapping words.
    """
    q_kw = get_keywords(question)
    if not q_kw:
        return None

    best_score = 0.0
    best_ans   = None

    for item in load_qa_database():
        s_kw    = get_keywords(item["question"])
        matches = len(q_kw & s_kw)
        score   = matches / max(len(q_kw), len(s_kw), 1)

        # FIX: require both a minimum match count AND a minimum score
        if matches >= threshold and score >= 0.4 and score > best_score:
            best_score = score
            best_ans   = item["answer"]
            print(f"[KW] {score:.2f} m={matches}: {item['question'][:55]}")

    return best_ans


# ══════════════════════════════════════════════════════════
# STEP 3 — SMART FAISS ROUTING
# ══════════════════════════════════════════════════════════
def smart_query(question: str) -> str:
    q  = question.lower()
    qt = hi_to_en(q)

    if any(w in qt for w in ['admission', 'jee', 'gate', 'utuee', 'apply',
                               'eligibility', 'praves', 'daakhila']):
        if any(w in qt for w in ['btech', 'b tech', 'b.tech', 'undergraduate',
                                   'ug ', ' ug', 'engineering admission']):
            return "B.Tech Admission JEE Main UKTECH counselling branches seats GBPIET"
        elif any(w in qt for w in ['mtech', 'm tech', 'm.tech', 'postgraduate',
                                    'pg ', ' pg', 'gate admission']):
            return "M.Tech Admission GATE counselling seats eligibility GBPIET"
        elif 'mca' in qt:
            return "MCA Admission VMSB Uttarakhand Technical University 60 seats GBPIET"
        elif 'phd' in qt or 'doctorate' in qt:
            return "PhD Admission written exam interview GBPIET"
        else:
            return "B.Tech Admission JEE Main UKTECH counselling GBPIET"

    if any(w in qt for w in ['mca faculty', 'csa faculty', 'mca teacher']):
        return "CSA MCA Faculty Priti Dimri Ashish Negi Yashwant Deepak"
    if any(w in qt for w in ['cse faculty', 'cse teacher', 'cse professor']):
        return "CSE Faculty Bhumika Gupta Papendra Kumar Garima Singh"
    if any(w in qt for w in ['ece faculty', 'electronics faculty']):
        return "ECE Faculty Mahesh Agarwal Anil Gautam Rajesh Kumar"
    if any(w in qt for w in ['mechanical faculty', 'me faculty']):
        return "Mechanical Faculty KKS Mer Ashutosh Gupta"
    if any(w in qt for w in ['civil faculty']):
        return "Civil Faculty BS Khati Hira Lal Yadav"
    if any(w in qt for w in ['electrical faculty', 'ee faculty']):
        return "Electrical Faculty Sanjay Gairola Vishnu Mohan Mishra"
    if any(w in qt for w in ['biotech faculty', 'biotechnology faculty']):
        return "Biotechnology Faculty Arun Bhatt Mamta Baunthiyal"

    if any(w in qt for w in ['first year hostel', '1st year hostel', 'fresher hostel']):
        return "First Year Hostel Trishul Kailash Boys Raman Viswerwarya Girls GBPIET"
    if any(w in qt for w in ['girls hostel', 'ladies hostel']):
        return "Girls Hostel Raman Bhagirathi Viswerwarya 3 hostels 405 seats GBPIET"
    if any(w in qt for w in ['boys hostel', 'gents hostel']):
        return "Boys Hostel Neelkanth Kedar Kailash Rudra Badri Alaknanda Shivalik Trishul"
    if any(w in qt for w in ['hostel', 'warden', 'accommodation']):
        return "Hostel Warden Boys Girls GBPIET 11 hostels"

    if any(w in qt for w in ['hod', 'head department', 'head of department']):
        return "Head of Department HOD all departments GBPIET"
    if any(w in qt for w in ['director']):
        return "Director GBPIET Prof VK Banga"
    if any(w in qt for w in ['dean']):
        return "Dean Deputy Dean GBPIET Administration"
    if any(w in qt for w in ['governing council', 'board of governors']):
        return "Governing Council Board of Governors GBPIET"
    if any(w in qt for w in ['registrar']):
        return "Registrar Office GBPIET Administration"

    if any(w in qt for w in ['fees', 'fee', 'tuition', 'payment', 'scholarship']):
        if 'btech' in qt or 'b.tech' in qt:
            return "B.Tech Fee Structure GBPIET per year payment"
        elif 'mca' in qt:
            return "MCA Fee Structure GBPIET per year"
        elif 'mtech' in qt or 'm.tech' in qt:
            return "M.Tech Fee Structure GBPIET per year"
        elif 'hostel' in qt:
            return "Hostel Mess Fee GBPIET payment SBI Collect"
        return "Fee Structure GBPIET B.Tech MCA M.Tech hostel payment"

    if any(w in qt for w in ['placement', 'package', 'lpa', 'recruiter', 'job', 'campus']):
        return "Placement Record GBPIET Amazon Microsoft 92 LPA package recruiters"

    if any(w in qt for w in ['courses', 'program', 'branch', 'offered', 'available']):
        return "Courses Programs GBPIET B.Tech MCA M.Tech PhD all departments"

    if any(w in qt for w in ['contact', 'phone', 'email', 'address']):
        return "Contact GBPIET phone email address website"

    if any(w in qt for w in ['reach', 'route', 'direction', 'distance', 'location']):
        return "How to reach GBPIET Haridwar Dehradun Rishikesh Kotdwar distance route"

    if any(w in qt for w in ['result', 'exam', 'semester', 'timetable', 'calendar']):
        return "Result Exam Semester GBPIET ERP portal"

    if any(w in qt for w in ['ragging', 'anti ragging', 'helpline']):
        return "Anti Ragging Helpline GBPIET zero tolerance"

    return qt if (qt != q and len(qt) > 3) else question


def faiss_search(question: str) -> str | None:
    """
    FIX #3: Added similarity score threshold to filter out irrelevant results.
    Only use FAISS results if they are actually relevant (score >= 0.35).
    similarity_search_with_score returns (doc, score) where lower = more similar.
    """
    try:
        sq      = smart_query(question)
        results = get_vectorstore().similarity_search_with_score(sq, k=4)

        # Filter by relevance: only keep results with distance <= 1.0
        # (FAISS L2 distance: lower is more similar; > 1.0 means too far)
        RELEVANCE_THRESHOLD = 1.0
        relevant = [(doc, score) for doc, score in results if score <= RELEVANCE_THRESHOLD]

        if not relevant:
            print(f"[FAISS] No relevant results for '{sq[:50]}' (all scores > {RELEVANCE_THRESHOLD})")
            return None

        # Use top 3 relevant results
        ctx = "\n\n".join([doc.page_content for doc, _ in relevant[:3]])
        print(f"[FAISS] '{sq[:50]}' → {len(relevant)} relevant results (scores: {[round(s,2) for _,s in relevant[:3]]})")
        return ctx
    except Exception as e:
        print(f"FAISS error: {e}")
        return None


# ══════════════════════════════════════════════════════════
# GROQ LLM — improved prompts that prevent hallucination
# ══════════════════════════════════════════════════════════
def llm_answer(question: str, context: str, lang: str, history: str = "") -> str:
    """
    FIX #4: Stronger instructions to avoid answering from memory.
    Added explicit instruction: do NOT add any info not in the context.
    """

    if lang == "hi":
        prompt = f"""Aap Diksha hain — GBPIET ke liye helpful AI chatbot.

STRICT RULES (inn rules ka ullanghan BILKUL mat karein):
- HAMESHA shuddh Hindi mein jawab dein
- SIRF neeche diye gaye context ka upyog karein. Apni memory se koi bhi jankari mat dijiye.
- Agar context mein jawab NAHI milta: "माफ़ कीजिए, मुझे यह जानकारी नहीं मिल पाई। कृपया अपना सवाल दोबारा पूछें।"
- Professor/Doctor ke liye "hain" use karein (hai NAHI)
- B.Tech, MCA, M.Tech ki jankari aapas mein BILKUL mix mat karein
- Clear aur seedha jawab dein, extra baatein mat karein

{history}
Context (SIRF YAHI USE KAREIN):
{context}

Sawaal: {question}
Jawab (Hindi mein, sirf context se):"""

    elif lang == "ga":
        prompt = f"""Tum Diksha chho — GBPIET chatbot. Garhwali mein jawab do (Devanagari).
Sirf context use karo. Context mein nahi mila: "माफ करा, मी तैं त्वे सवाल समझ नि ऐ।"
Apni memory se koi bhi info mat do.
{history}
Context: {context}
Sawaal: {question}
Jawab:"""

    elif lang == "ku":
        prompt = f"""Tum Diksha chhu — GBPIET chatbot. Kumauni mein jawab do (Devanagari).
Sirf context use karo. Context mein nahi mila: "माफ करिया ! म्यर पास तस के जानकारी न्है़ंं!"
Apni memory se koi bhi info mat do.
{history}
Context: {context}
Sawaal: {question}
Jawab:"""

    else:
        prompt = f"""You are Diksha — a helpful AI assistant for GBPIET (Govind Ballabh Pant Institute of Engineering and Technology), Pauri Garhwal, Uttarakhand.

STRICT RULES (follow every rule exactly):
- Reply in English only
- Use ONLY the information in the context section below. Do NOT use any outside knowledge or memory.
- If the answer is NOT present in the context, respond with ONLY: "I'm sorry, I couldn't find that information. Could you please rephrase your query?"
- Do NOT mix up B.Tech, MCA, and M.Tech information — they are different programs
- Be respectful — use proper titles (Prof., Dr.)
- Give a direct, concise answer. Do not add unnecessary filler sentences.
- Do NOT make up names, fees, dates, or any facts not explicitly stated in the context

{history}
Context (use ONLY this — do not go beyond it):
{context}

Question: {question}
Answer:"""

    try:
        r = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are Diksha, a multilingual AI assistant for GBPIET college. "
                        "You must ONLY answer from the provided context. "
                        "If information is not in the context, say you don't know. "
                        "Never invent facts, names, or numbers."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=400,
            temperature=0.1   # FIX #5: Lowered from 0.3 → 0.1 for factual accuracy
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq error: {e}")
        return "I'm unable to process your request right now. Please try again."


# ══════════════════════════════════════════════════════════
# MAIN — Hybrid Search (Exact → Keyword → FAISS+Groq)
# ══════════════════════════════════════════════════════════
def get_answer(question: str, lang: str = "en", history: str = "") -> str:
    print(f"\n[Q/{lang}] {question}")

    # Step 1 — Exact match
    ans = exact_match(question)
    if ans:
        print("[RESULT] Exact match")
        return ans

    # Step 2 — Keyword match
    # FIX: Use threshold=3 for short questions, 4 for longer ones
    thresh = 3 if len(question.split()) <= 6 else 4
    ans    = keyword_match(question, thresh)
    if ans:
        print("[RESULT] Keyword match")
        return ans

    # Step 3 — FAISS + Groq (with relevance filtering)
    ctx = faiss_search(question)
    if ctx:
        print("[RESULT] FAISS + Groq")
        return llm_answer(question, ctx, lang, history)

    # Fallback
    print("[RESULT] Fallback")
    fb = {
        "hi": "माफ़ करें, मुझे इस विषय पर जानकारी नहीं मिल पाई। कृपया अपना सवाल दोबारा पूछें।",
        "ga": "माफ करा, मी तैं त्वे सवाल समझ नि ऐ।",
        "ku": "माफ करिया ! म्यर पास तस के जानकारी न्है़ंं!",
        "en": "I'm sorry, I don't have information on that topic. Could you please rephrase or ask something else about GBPIET?"
    }
    return fb.get(lang, fb["en"])
