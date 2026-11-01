import os
from pathlib import Path
from dotenv import load_dotenv
from file_reader import extract_paragraphs
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

load_dotenv()

# Resolve path to pdfs/Penguins_ACL.pdf from project root
BASE_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = BASE_DIR / "pdfs" / "Penguins_ACL.pdf"

embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vector_store = InMemoryVectorStore(embeddings)

chunks = extract_paragraphs(str(PDF_PATH))
vector_store.add_texts(texts=chunks)

if __name__ == "__main__":
    print(f"Indexed {len(chunks)} chunks successfully.")