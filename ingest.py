import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFMinerLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Allows duplicate loading of libomp and libiomp
os.environ['OMP_NUM_THREADS'] = '1'  # Limits the number of threads to avoid conflicts
def main():
    documents = []
    
    # Walk through the 'docs' directory and load PDF files
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(f"Loading file: {file}")
                loader = PDFMinerLoader(os.path.join(root, file))
                documents.extend(loader.load())
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create and persist Chroma vector store
    db = Chroma.from_documents(texts, embeddings, persist_directory="db")

if __name__ == "__main__":
    main()
