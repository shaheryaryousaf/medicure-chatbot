import os
import pinecone
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec

# Load environment variables from .env file
load_dotenv()

def ingest_data():
    """
    This function performs the one-time data ingestion process.
    1. Loads the PDF document.
    2. Splits the document into smaller text chunks.
    3. Initializes Pinecone, deleting an old index if it exists.
    4. Generates embeddings for each chunk using OpenAI.
    5. Upserts the embeddings into a new Pinecone index.
    """
    print("Starting data ingestion...")

    # 1. Load the PDF document
    pdf_path = "manual.pdf" # Make sure this path is correct
    if not os.path.exists(pdf_path):
        print(f"Error: The file {pdf_path} was not found.")
        return

    try:
        loader = PyPDFLoader(file_path=pdf_path)
        documents = loader.load()
        print(f"Successfully loaded {len(documents)} pages from the PDF.")
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return

    # 2. Split the document into smaller text chunks
    # Using RecursiveCharacterTextSplitter for better chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    docs = text_splitter.split_documents(documents=documents)
    print(f"Split the document into {len(docs)} chunks.")

    # 3. Initialize Pinecone and handle index creation
    # The Pinecone API key is automatically read from the PINECONE_API_KEY environment variable
    index_name = "pdf-chatbot-index"
    
    # The OpenAI API key is automatically read from the OPENAI_API_KEY environment variable
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    embedding_dimension = 1536 # Dimension for text-embedding-3-small

    print("Initializing Pinecone client...")
    pc = pinecone.Pinecone()

    # Check if the index already exists and delete it to ensure correct dimensions
    if index_name in pc.list_indexes().names():
        print(f"Index '{index_name}' already exists. Deleting it.")
        pc.delete_index(index_name)
        print("Index deleted.")

    # Create a new index with the correct dimension and spec
    print(f"Creating a new index '{index_name}' with dimension {embedding_dimension}.")
    pc.create_index(
        name=index_name,
        dimension=embedding_dimension,
        metric='cosine',
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print("Index created successfully.")


    # 4. Upsert embeddings into Pinecone
    print(f"Upserting documents into Pinecone index: {index_name}")
    try:
        PineconeVectorStore.from_documents(
            documents=docs,
            embedding=embeddings,
            index_name=index_name
        )
        print("Data ingestion complete. Your PDF is now ready for querying.")
    except Exception as e:
        print(f"Error during Pinecone operation: {e}")


if __name__ == "__main__":
    ingest_data()
