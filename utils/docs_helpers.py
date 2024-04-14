from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma

persist_directory = "./chromadb/"
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


def read_docs(path):
    loader = PyPDFLoader(path)
    documents = loader.load_and_split()
    return documents


def split_docs(documents, chunk_size=1000, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = text_splitter.split_documents(documents)
    return docs


def save_chunks(chunks, patient_name):

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory + patient_name,
    )
    vectordb.persist()


def read_documents_and_index(patient_name):
    docs = read_docs()
    print(f"Len={len(docs)} Content=", docs)

    chunk_docs = split_docs(docs)
    print("Chunk documents size=", len(chunk_docs))

    save_chunks(chunk_docs, patient_name)
