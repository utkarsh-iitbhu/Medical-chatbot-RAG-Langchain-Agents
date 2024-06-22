from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def load_pdf(data):
    """
    Loads PDF documents from a directory and returns them as a list of document objects.
    
    Args:
        data (str): Path to the directory containing PDFs.
        
    Returns:
        List[Document]: A list of document objects extracted from the PDFs.
    """
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

def text_split(extracted_data):
    """
    Splits the extracted text data into manageable chunks for processing.
    
    Args:
        extracted_data (List[Document]): A list of document objects to split.
        
    Returns:
        List[List[str]]: A list of lists containing the split text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

def openai_embedding():
    """
    Initializes an OpenAIEmbeddings object for embedding text data.
    
    Returns:
        OpenAIEmbeddings: An instance of OpenAIEmbeddings ready for embedding operations.
    """
    embedding = OpenAIEmbeddings()
    return embedding

def update_embedding(total_chunk):
    """
    Embeds a list of text chunks using OpenAIEmbeddings and returns the embeddings.
    
    Args:
        total_chunk (List[str]): A list of text chunks to be embedded.
        
    Returns:
        List[np.ndarray]: A list of numpy arrays representing the embeddings of the text chunks.
    """
    embedding = OpenAIEmbeddings()
    embeddings = embedding.embed_documents(total_chunk)
    return embeddings

def update_pineconedb(text_chunks, embedding, index):
    """
    Updates a Pinecone index with embeddings of text chunks and their metadata.
    
    Args:
        text_chunks (List[Document]): A list of document objects containing text chunks.
        embedding (OpenAIEmbeddings): An instance of OpenAIEmbeddings for embedding text.
        index (pinecone.Index): A Pinecone index object to update with embeddings and metadata.
    """
    total_chunk = [t.page_content for t in text_chunks]
    embeddings = embedding.embed_documents(total_chunk)
    vectors = []
    for i, chunk in enumerate(text_chunks):
        metadata = {"page_number": chunk.metadata.get("page", ""), "source": chunk.metadata.get("source", ""), "text": chunk.page_content}
        vector = {
            "id": str(i),
            "values": embeddings[i],  # Use the corresponding embedding
            "metadata": metadata
        }
        vectors.append(vector)
    
    def batch(iterable, n=1):
        l = len(iterable)
        for ndx in range(0, l, n):
            yield iterable[ndx:min(ndx + n, l)]

    batch_size = 100

    for i, batch_vectors in enumerate(batch(vectors, batch_size)):
        try:
            index.upsert(vectors=batch_vectors)
        except Exception as e:
            print(f"Error in batch {i+1}: {e}")
