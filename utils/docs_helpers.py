from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

persist_directory = "./persisted"
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


def read_docs():
    loader = DirectoryLoader("./documents")
    documents = loader.load()
    return documents


def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = text_splitter.split_documents(documents)
    return docs


def save_chunks(chunks):
    vectordb = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=persist_directory
    )
    vectordb.persist()


def read_documents_and_index():
    docs = read_docs()
    print(f"Len={len(docs)} Content=", docs)

    chunk_docs = split_docs(docs)
    print("Chunk documents size=", len(chunk_docs))

    save_chunks(chunk_docs)
