
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
import multiprocessing as mp
from tqdm import tqdm
import os

# ---------- Config ----------
CSV_PATH = "review_data.csv"
DB_PATH = "chroma_langchain_db"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBED_MODEL = "nomic-embed-text"
LIMIT_ROWS = 200000

# ---------- Globals ----------
# retriever = None  # DO NOT USE GLOBAL retriever

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

def create_docs(row_data):
    i, row = row_data
    full_text = f"{row['Company']} {row['Model']} {row['Year']} {row['Title']} {row['Rating']} {row['Review']}"
    chunks = text_splitter.split_text(full_text)
    docs, ids = [], []
    for j, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk,
            metadata={"rating": row["Rating"], "date": row["Date"]}
        )
        docs.append(doc)
        ids.append(f"{i}-{j}")
    return docs, ids

def get_retriever(vectorstore, k: int = 15, use_mmr: bool = True, score_threshold: float = None):
    search_type = "mmr" if use_mmr else "similarity"
    kwargs = {"k": k}
    if score_threshold:
        kwargs["score_threshold"] = score_threshold
    return vectorstore.as_retriever(search_type=search_type, search_kwargs=kwargs)

def setup_vectorstore():
    if os.path.exists(DB_PATH):
        print("[DEBUG] Vectorstore already exists. Loading it without reprocessing...")
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        vectorstore = Chroma(
            collection_name="edmund_car_reviews",
            persist_directory=DB_PATH,
            embedding_function=embeddings
        )
        return vectorstore

    # ELSE: build from scratch
    print("[DEBUG] Vectorstore not found. Creating from CSV...")
    df = pd.read_csv(CSV_PATH)

    print("[DEBUG] Cleaning data...")
    df = df.dropna(subset=["Review", "Title"])
    df["Review"] = df["Review"].astype(str).str.strip()
    df["Title"] = df["Title"].astype(str).str.strip()
    df = df[(df["Review"].str.len() > 0) & (df["Title"].str.len() > 0)]
    df = df.head(LIMIT_ROWS)

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    print("[DEBUG] Chunking & preparing documents in parallel...")
    with mp.Pool(mp.cpu_count()) as pool:
        results = list(tqdm(pool.imap(create_docs, df.iterrows()), total=len(df)))

    documents, ids = [], []
    for doc_batch, id_batch in results:
        documents.extend(doc_batch)
        ids.extend(id_batch)

    vectorstore = Chroma(
        collection_name="edmund_car_reviews",
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    print("[DEBUG] Adding documents to vectorstore in batches...")
    batch_size = 10
    total_docs = len(documents)
    for i in tqdm(range(0, total_docs, batch_size), desc="Adding to vectorstore"):
        batch_docs = documents[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        vectorstore.add_documents(documents=batch_docs, ids=batch_ids)

    print("[DEBUG] Vectorstore setup complete.")
    return vectorstore

#
# if __name__ == "__main__":
#     mp.freeze_support()
#     setup_vectorstore()
