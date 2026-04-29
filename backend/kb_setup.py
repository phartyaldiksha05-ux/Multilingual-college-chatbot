import json
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Yeh files FAISS me index NAHI hongi
SKIP_FILES = ["keywords.json"]

def detect_course_type(text: str):
    t = text.lower()
    if any(k in t for k in ["btech", "b.tech", "undergraduate", "ug", "12th"]):
        return "btech"
    if any(k in t for k in ["mca"]):
        return "mca"
    if any(k in t for k in ["mtech", "m.tech", "pg"]):
        return "mtech"
    return "general"

def build_knowledge_base():
    print("=" * 50)
    print("Starting Diksha Knowledge Base Setup...")
    print("=" * 50)

    docs          = []
    data_folder   = os.path.join(os.path.dirname(__file__), "data")
    loaded_files  = 0
    total_entries = 0

    for filename in sorted(os.listdir(data_folder)):
        if not filename.endswith(".json"):
            continue

        # ✅ keywords.json skip karo
        if filename in SKIP_FILES:
            print(f"  SKIPPED (keywords file): {filename}")
            continue

        filepath = os.path.join(data_folder, filename)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            items = data if isinstance(data, list) else [data]
            count = 0

            for item in items:
                if not isinstance(item, dict):
                    continue
                if "question" not in item or "answer" not in item:
                    continue

                q = item["question"].strip()
                a = item["answer"].strip()

                hindi_chars = sum(1 for c in q if '\u0900' <= c <= '\u097F')
                is_hindi    = hindi_chars > len(q) * 0.3
                lang_tag    = "Hindi" if is_hindi else "English"

                cat         = filename.replace("faqs_", "").replace(".json", "")
                course_type = detect_course_type(q + " " + a)

                text = f"Category: {cat}\nCourse: {course_type}\nLanguage: {lang_tag}\nQuestion: {q}\nAnswer: {a}"

                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source":   filename,
                        "category": cat,
                        "language": lang_tag,
                        "course":   course_type,
                        "question": q
                    }
                ))
                count += 1

            if count > 0:
                print(f"  {count:3d} entries  ←  {filename}")
                total_entries += count
                loaded_files  += 1

        except Exception as e:
            print(f"  ERROR {filename}: {e}")

    print(f"\n  Files:  {loaded_files}")
    print(f"  Total:  {total_entries} FAQs")

    if total_entries == 0:
        print("ERROR: No data! Check backend/data/")
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_documents(docs)
    print(f"  Chunks: {len(chunks)}")

    print("\nCreating embeddings with BAAI/bge-small-en-v1.5...")
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    index_path = os.path.join(os.path.dirname(__file__), "faiss_index")
    vectorstore.save_local(index_path)

    print("\n" + "=" * 50)
    print("FAISS Index Built Successfully!")
    print(f"Saved at: {index_path}")
    print("=" * 50)

if __name__ == "__main__":
    build_knowledge_base()
