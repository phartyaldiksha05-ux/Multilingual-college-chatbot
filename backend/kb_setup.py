import json
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


# ✅ Detect course type (VERY IMPORTANT)
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

                # ✅ Language detection
                hindi_chars = sum(1 for c in q if '\u0900' <= c <= '\u097F')
                is_hindi    = hindi_chars > len(q) * 0.3
                lang_tag    = "Hindi" if is_hindi else "English"

                # ✅ Category
                cat = filename.replace("faqs_", "").replace(".json", "")

                # ✅ Course detection (NEW 🔥)
                course_type = detect_course_type(q + " " + a)

                # ✅ Better structured text (IMPORTANT)
                text = f"""
Language: {lang_tag}
Course: {course_type}
Category: {cat}

Question: {q}
Answer: {a}
"""

                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "category": cat,
                        "language": lang_tag,
                        "course": course_type,   # 🔥 NEW
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

    # ✅ Chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "]
    )

    chunks = splitter.split_documents(docs)
    print(f"  Chunks: {len(chunks)}")

    # 🔥 MULTILINGUAL MODEL (VERY IMPORTANT FIX)
    print("\nCreating embeddings... (first time may take time)")
    embeddings = FastEmbedEmbeddings()
    print("Building FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    index_path = os.path.join(os.path.dirname(__file__), "faiss_index")
    vectorstore.save_local(index_path)

    print("\n" + "=" * 50)
    print("✅ FAISS Index Built Successfully!")
    print(f"📁 Saved at: {index_path}")
    print("=" * 50)


if __name__ == "__main__":
    build_knowledge_base()
