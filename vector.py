# from langchain_ollama import OllamaEmbeddings
# from langchain_chroma import Chroma
# from langchain_core.documents import Document
# import os
# import pandas as pd
# from tqdm import tqdm
#
#
#
# print("[DEBUG] Loading review_data.csv...")
# df = pd.read_csv("review_data.csv")
#
#
# # Cleaning
# print("[DEBUG] Cleaning data...")
# df = df.dropna(subset=["Review", "Title"])
# df["Review"] = df["Review"].astype(str).str.strip()
# df["Title"] = df["Title"].astype(str).str.strip()
# df = df[(df["Review"].str.len() > 0) & (df["Title"].str.len() > 0)]
#
# df = df.head(10000)
#
# embeddings = OllamaEmbeddings(model="mxbai-embed-large")
#
# db_location = "./chroma_langchain_db"
# add_documents = not os.path.exists(db_location)
#
# if add_documents:
#     print("[DEBUG] Creating documents for embedding...")
#     documents = []
#     ids = []
#
#     for i, row in tqdm(df.iterrows(), total=len(df), desc="Creating documents"):
#         document = Document(
#         page_content=" ".join([
#             str(row["Company"]),
#             str(row["Model"]),
#             str(row["Year"]),
#             str(row["Title"]),
#             str(row["Rating"]),
#             str(row["Review"])
#         ]),
#         metadata={"rating": str(row["Rating"]), "date": row["Date"]},
#         id=str(i)
#
#         )
#
#         ids.append(str(i))
#         documents.append(document)
#
#
# vector_store = Chroma(
#     collection_name="edmund_car_reviews",
#     persist_directory=db_location,
#     embedding_function=embeddings
# )
#
# if add_documents:
#     print("[DEBUG] Adding documents to vectorstore in batches...")
#     batch_size = 10
#     total_docs = len(documents)
#     for i in tqdm(range(0, total_docs, batch_size), desc="Adding to vectorstore"):
#         batch_docs = documents[i:i + batch_size]
#         batch_ids = ids[i:i + batch_size]
#         vector_store.add_documents(documents=batch_docs, ids=batch_ids)
#     vector_store.persist()
#     # print("[DEBUG] Vectorstore persisted.")
#
#
# # def retrieve_reviews(query, min_k=2, max_k=10):
# #     """Retrieves reviews with a dynamically adjusted 'k' based on the query."""
# #     temp_retriever = vectorstore.as_retriever(search_kwargs={"k": max_k})
# #     results = temp_retriever.get_relevant_documents(query)
# #     dynamic_k = dynamic_k = max(min_k, min(max_k, len(results)))
# #     final_retriever = vectorstore.as_retriever(search_kwargs={"k": dynamic_k})
# #     return final_retriever.get_relevant_documents(query)
#
# retriever = vector_store.as_retriever(search_kwargs={"k": 10})
# print("[DEBUG] Retriever is ready.")



#
# from langchain_ollama import OllamaEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain_core.documents import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# import os
# import pandas as pd
# from tqdm import tqdm
#
# # Load data
# print("[DEBUG] Loading review_data.csv...")
# df = pd.read_csv("review_data.csv")
#
#
# # Cleaning
# print("[DEBUG] Cleaning data...")
# df = df.dropna(subset=["Review", "Title"])
# df["Review"] = df["Review"].astype(str).str.strip()
# df["Title"] = df["Title"].astype(str).str.strip()
# df = df[(df["Review"].str.len() > 0) & (df["Title"].str.len() > 0)]
#
# df = df.head(200000)
#
# # Setup embedding model
# embeddings = OllamaEmbeddings(model="nomic-embed-text")
#
# # Set vector DB path
# db_location = "./chroma_langchain_db"
# add_documents = not os.path.exists(db_location)
#
# # Text splitter (chunk long reviews for better indexing)
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,
#     chunk_overlap=100
# )
#
# # Prepare documents
# if add_documents:
#     documents = []
#     ids = []
#
#     for i, row in tqdm(df.iterrows(), total=len(df), desc="Creating documents"):
#         full_text = f"{row['Company']} {row['Model']} {row['Year']} {row['Title']} {row['Rating']} {row['Review']}"
#         chunks = text_splitter.split_text(full_text)
#
#         for j, chunk in enumerate(chunks):
#             doc = Document(
#                 page_content=chunk,
#                 metadata={"rating": row["Rating"], "date": row["Date"]}
#             )
#             documents.append(doc)
#             ids.append(f"{i}-{j}")
#
# # Initialize vectorstore (Chroma automatically loads index if it exists)
# vectorstore = Chroma(
#     collection_name="edmund_car_reviews",
#     persist_directory=db_location,
#     embedding_function=embeddings
# )
#
# # Add docs + persist index
# # if add_documents:
# #     print("[DEBUG] Adding documents to vectorstore in batches...")
# #     vectorstore.add_documents(documents=documents, ids=ids)
# #     vectorstore.persist()  # Builds and saves vector index
#
# if add_documents:
#     print("[DEBUG] Adding documents to vectorstore in batches...")
#     batch_size = 10
#     total_docs = len(documents)
#     for i in tqdm(range(0, total_docs, batch_size), desc="Adding to vectorstore"):
#         batch_docs = documents[i:i + batch_size]
#         batch_ids = ids[i:i + batch_size]
#         vectorstore.add_documents(documents=batch_docs, ids=batch_ids)
#     vectorstore.persist()
#     # print("[DEBUG] Vectorstore persisted.")
#
# # Flexible retriever with MMR & score threshold
# def get_retriever(k: int = 15, use_mmr: bool = True, score_threshold: float = None):
#     search_type = "mmr" if use_mmr else "similarity"
#     kwargs = {"k": k}
#     if score_threshold:
#         kwargs["score_threshold"] = score_threshold
#
#     return vectorstore.as_retriever(
#         search_type=search_type,
#         search_kwargs=kwargs
#     )
#
# # Default retriever
# retriever = get_retriever(k=10)  # You can pass different values if needed
# print("[DEBUG] Retriever is ready.")




#
# from langchain_ollama import OllamaEmbeddings
# from langchain_community.vectorstores import Chroma
# from langchain_core.documents import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# import pandas as pd
# import multiprocessing as mp
# from tqdm import tqdm
# import os
#
# # ---------- Config ----------
# CSV_PATH = "review_data.csv"
# DB_PATH = "./chroma_langchain_db"
# CHUNK_SIZE = 500
# CHUNK_OVERLAP = 100
# EMBED_MODEL = "nomic-embed-text"
# LIMIT_ROWS = 1000  # adjust as needed
#
# # ---------- Globals ----------
# retriever = None  # this will be initialized in main()
#
# # ---------- Text Splitter (declared at global level for multiprocessing)
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=CHUNK_SIZE,
#     chunk_overlap=CHUNK_OVERLAP
# )
#
#
# def create_docs(row_data):
#     """Function to split each row's review into chunks + documents (used in multiprocessing)."""
#     i, row = row_data
#     full_text = f"{row['Company']} {row['Model']} {row['Year']} {row['Title']} {row['Rating']} {row['Review']}"
#     chunks = text_splitter.split_text(full_text)
#
#     docs, ids = [], []
#     for j, chunk in enumerate(chunks):
#         doc = Document(
#             page_content=chunk,
#             metadata={"rating": row["Rating"], "date": row["Date"]}
#         )
#         docs.append(doc)
#         ids.append(f"{i}-{j}")
#     return docs, ids
#
#
# def get_retriever(vectorstore, k: int = 15, use_mmr: bool = True, score_threshold: float = None):
#     search_type = "mmr" if use_mmr else "similarity"
#     kwargs = {"k": k}
#     if score_threshold:
#         kwargs["score_threshold"] = score_threshold
#
#     return vectorstore.as_retriever(
#         search_type=search_type,
#         search_kwargs=kwargs
#     )
#
#
# def main():
#     global retriever
#
#     print("[DEBUG] Loading review_data.csv...")
#     df = pd.read_csv(CSV_PATH)
#
#     print("[DEBUG] Cleaning data...")
#     df = df.dropna(subset=["Review", "Title"])
#     df["Review"] = df["Review"].astype(str).str.strip()
#     df["Title"] = df["Title"].astype(str).str.strip()
#     df = df[(df["Review"].str.len() > 0) & (df["Title"].str.len() > 0)]
#     df = df.head(LIMIT_ROWS)
#
#     embeddings = OllamaEmbeddings(model=EMBED_MODEL)
#     add_documents = not os.path.exists(DB_PATH)
#
#     if add_documents:
#         print("[DEBUG] Chunking & preparing documents in parallel...")
#         with mp.Pool(mp.cpu_count()) as pool:
#             results = list(tqdm(pool.imap(create_docs, df.iterrows()), total=len(df)))
#
#         documents, ids = [], []
#         for doc_batch, id_batch in results:
#             documents.extend(doc_batch)
#             ids.extend(id_batch)
#
#     vectorstore = Chroma(
#         collection_name="edmund_car_reviews",
#         persist_directory=DB_PATH,
#         embedding_function=embeddings
#     )
#
#     # if add_documents:
#     #     print(f"[DEBUG] Adding {len(documents)} documents to vectorstore...")
#     #     vectorstore.add_documents(documents=documents, ids=ids)
#     #     vectorstore.persist()
#     #     print("[DEBUG] Vectorstore persisted.")
#
#     if add_documents:
#         print("[DEBUG] Adding documents to vectorstore in batches...")
#         batch_size = 10
#         total_docs = len(documents)
#         for i in tqdm(range(0, total_docs, batch_size), desc="Adding to vectorstore"):
#             batch_docs = documents[i:i + batch_size]
#             batch_ids = ids[i:i + batch_size]
#             vectorstore.add_documents(documents=batch_docs, ids=batch_ids)
#
#     retriever = get_retriever(vectorstore, k=10)
#     print("[DEBUG] Retriever is ready.")
#
#
# if __name__ == "__main__":
#     mp.freeze_support()  # required on Windows
#     main()



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