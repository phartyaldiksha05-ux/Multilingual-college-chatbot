import re

# Hinglish / Roman Hindi common words
HINGLISH_WORDS = {
    'kya', 'hai', 'hain', 'ka', 'ki', 'ke', 'mein', 'se', 'kaise',
    'kaun', 'kahan', 'kitna', 'kitne', 'batao', 'bato', 'karo',
    'hoga', 'hogi', 'chahiye', 'milega', 'milegi', 'wala', 'wali',
    'aur', 'ya', 'nahi', 'nhi', 'toh', 'bhi', 'sirf', 'sab',
    'kon', 'koun', 'kab', 'kyun', 'kyunki', 'batana', 'dijiye',
    'chahte', 'chahti', 'puchna', 'poochna', 'kaisa', 'kaisi',
    'hod', 'dean', 'warden', 'hostel', 'fees', 'admission',
    'college', 'course', 'faculty', 'placement', 'director',
    'result', 'exam', 'kitab', 'library', 'campus', 'ragging',
    'scholarship', 'transport', 'contact', 'phone', 'address'
}

def detect_language(text: str) -> str:
    try:
        # Hindi Devanagari characters → Hindi
        if re.search(r'[\u0900-\u097F]', text):
            return "hi"

        # Hinglish Roman words → Hindi
        words = set(text.lower().split())
        if words & HINGLISH_WORDS:
            return "hi"

        # Baaki sab → English
        return "en"

    except:
        return "en"


if __name__ == "__main__":
    tests = [
        "What is the admission process?",
        "प्रवेश प्रक्रिया क्या है?",
        "Who is the director?",
        "फीस कितनी है?",
        "What courses are available?",
        "fees kitni hai?",
        "hod kaun hai biotech ka?",
        "hostel mein kitne rooms hain?",
    ]
    for text in tests:
        lang = detect_language(text)
        print(f"Text: {text}")
        print(f"Language: {lang}")
        print("-" * 40)
