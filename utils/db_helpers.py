from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

from .docs_helpers import save_chunks, read_docs, split_docs

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


def create_patient_record(patient_name, pdf_path):
    docs = read_docs(pdf_path)
    print(f"Len={len(docs)} Content=", docs)

    chunk_docs = split_docs(docs)
    print("Chunk documents size=", len(chunk_docs))

    save_chunks(chunk_docs, patient_name)


def query_patient_record(patient):
    vectordb = Chroma(
        persist_directory=f"./chromadb/{patient}", embedding_function=embeddings
    )

    matching_docs = vectordb.get()

    return matching_docs["documents"]
