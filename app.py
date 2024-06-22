import os
import pinecone
from dotenv import load_dotenv
from flask import Flask, render_template, jsonify, request
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.agents import load_tools
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
from langchain.vectorstores import Pinecone
from src.helper import load_pdf, text_split, openai_embedding, update_pineconedb, update_embedding
from src.prompt import *

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
INDEXNAME = os.getenv("INDEXNAME")

embedding = openai_embedding()

pc = pinecone.Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
indexpc = pc.Index(INDEXNAME)

docsearch = Pinecone(index=indexpc, embedding=embedding, text_key="text")

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs = {"prompt": PROMPT}

llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name='gpt-3.5-turbo')

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs,
)

memory = ConversationSummaryBufferMemory(
    llm=llm,
    memory_key='chat_history',
    return_messages=True,
    output_key='answer'
)

conv_qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=docsearch.as_retriever(search_kwargs={'k': 3}),
    memory=memory,
    get_chat_history=lambda h: h,
    return_source_documents=True
)

tool = load_tools(["serpapi"], serpapi_api_key=SERPAPI_KEY, llm=llm)

agent = initialize_agent(tool, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

def get_answer(user_input: str) -> str:
    """
    Processes the user input through the conversational retrieval chain and uses SerpAPI if necessary.
    
    Args:
        user_input (str): The user's question or statement.
        
    Returns:
        str: The processed answer from the AI or a web search result.
    """
    result = conv_qa_chain({"question": user_input})
    answer = result['answer']
    
    if "I don't" in answer or "I'm sorry" in answer:
        serp_result = agent({"input": user_input})
        return "Websearch : " + serp_result["output"]
    else:
        return answer

@app.route("/")
def index() -> str:
    """
    Renders the main page of the application.
    
    Returns:
        str: The rendered HTML template.
    """
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat() -> str:
    """
    Handles incoming chat messages, processes them, and returns the AI-generated response.
    
    Returns:
        str: The AI-generated response to the user's message.
    """
    msg = request.form["msg"]
    result = get_answer(msg)
    return str(result)

if __name__ == '__main__':
    app.run(debug=True)