import os
import pinecone
from dotenv import load_dotenv
from langchain.vectorstores import Pinecone
from src.helper import load_pdf, text_split, openai_embedding, update_pineconedb,update_embedding

def load_environment_variables():
    """
    Loads environment variables needed for Pinecone API access.
    """
    load_dotenv()
    global PINECONE_API_KEY, PINECONE_API_ENV
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
    INDEXNAME = os.getenv("INDEXNAME")

def load_and_process_data(directory_path):
    """
    Loads PDF documents from a given directory, splits the text into chunks, and prepares embeddings.
    
    Args:
        directory_path (str): Path to the directory containing PDF documents.
        
    Returns:
        Tuple[List[str], List[str]]: A tuple containing the list of text chunks and the embeddings.
    """
    extracted_data = load_pdf(directory_path)
    text_chunks = text_split(extracted_data)
    embedding = openai_embedding()
    embeddings = update_embedding(text_chunks)
    return text_chunks, embeddings

def setup_pinecone_index(embedding_dimension):
    """
    Sets up a Pinecone index if it does not exist, creating it with the specified dimension.
    
    Args:
        embedding_dimension (int): Dimension of the embeddings to be stored in the Pinecone index.
    """
    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
    index = pc.Index(INDEXNAME)
    
    if INDEXNAME not in [index.name for index in pc.list_indexes()]:
        pinecone.create_index(INDEXNAME, dimension=embedding_dimension)
        index = pc.Index(INDEXNAME)
        return index
    else:
        return index

def main():
    """
    Main function to orchestrate loading data, processing, and updating the Pinecone database.
    """
    load_environment_variables()
    directory_path = "data/"
    text_chunks, embeddings = load_and_process_data(directory_path)
    
    index = setup_pinecone_index(embeddings.shape[1])
    update_pineconedb(text_chunks, embeddings, index)

if __name__ == "__main__":
    main()
