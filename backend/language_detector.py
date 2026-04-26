from langdetect import detect

def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        if lang == "hi":
            return "hi"
        else:
            return "en"
    except:
        return "en"

if __name__ == "__main__":
    tests = [
        "What is the admission process?",
        "प्रवेश प्रक्रिया क्या है?",
        "Who is the director?",
        "फीस कितनी है?",
        "What courses are available?"
    ]

    for text in tests:
        lang = detect_language(text)
        print(f"Text: {text}")
        print(f"Language: {lang}")
        print("-" * 40)