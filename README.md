# Project Title: Advanced Chatbot System

## Overview

This project showcases an advanced chatbot system built with Flask, Pinecone, RAG(Retrieval-Augmented Generation) and Langchain. It leverages OpenAI embeddings for semantic understanding and integrates a retrieval-based chat system with agents and memory capabilities to handle complex conversations and follow-up questions. The system is designed to retrieve relevant information from a Pinecone vector store and utilize SerpAPI for fetching web search results when necessary.

The response generated with our model using Langchain, OpenAI, RAG

<img src="https://github.com/utkarsh-iitbhu/Medical-chatbot-RAG-Langchain-Agents/assets/84759422/c32cbcd2-74c8-4e8f-be9f-78c6fb518f69" alt="Chat Memory Example" width="400">

Able to catch the memory, as I have not specified the drugs for which disease, competent to comprehend chat history.

<img src="https://github.com/utkarsh-iitbhu/Medical-chatbot-RAG-Langchain-Agents/assets/84759422/15465ed2-fee0-467d-bd3e-136231a3d29b" alt="Web Search Example" width="400">

If the chatbot cannot find the response from the given data, our Serp-API agents come into play and give you the web search results.

<img src="https://github.com/utkarsh-iitbhu/Medical-chatbot-RAG-Langchain-Agents/assets/84759422/553e6f54-fa2c-43b4-b519-10b6964ff33a" alt="Reduced Image Size Example" width="400">

## Features

- **Conversational AI**: Harnesses the power of `RAG` alongside `Langchain` and `OpenAI's LLM` to craft nuanced and contextually relevant responses, generating coherent and context-aware responses.
- **Information Retrieval**: Utilizes Pinecone for efficient storage and retrieval of document embeddings, enabling the chatbot to fetch relevant information based on user queries.
- **Semantic Understanding**: Incorporates OpenAI embeddings to comprehend the nuances of user inputs, enhancing the accuracy of retrieved information.
- **Agent Integration**: Implements custom agents for specialized functionalities, such as leveraging SerpAPI to fetch web search results when the internal knowledge base is insufficient.
- **Memory Management**: Equipped with a memory component that allows the chatbot to remember past interactions and ask follow-up questions, improving the flow of conversations.

## Tech Stack

- Frontend: HTML, CSS, Bootstrap
- Backend: Flask
- Vector Storage: Pinecone
- Natural Language Processing: Langchain, OpenAI, RAG
- Search API: SerpAPI

## Setup Instructions

### Step 1: Create a Virtual Environment

1. Open your terminal or command prompt.
2. Navigate to the project directory.
3. Run `python -m venv venv` to create a virtual environment named `venv`.

### Step 2: Activate the Virtual Environment

- On Windows, run `.\venv\Scripts\activate`.
- On macOS/Linux, run `source venv/bin/activate`.

### Step 3: Install Dependencies

Run `pip install -r requirements.txt` to install all necessary packages.

### Step 4: Configure Environment Variables

Create a `.env` file in the root of your project and add the following lines:

```plaintext
OPENAI_API_KEY="your_openai_api_key_here"
PINECONE_API_KEY="your_pinecone_api_key_here"
PINECONE_API_ENV="your_pinecone_api_env_here"
SERPAPI_KEY="your_serpapi_key_here"
INDEXNAME="your_index_name_here"
```

Replace the placeholders with your actual API keys and environment-specific values.

### Step 5: Update User Image (Optional)

- To change the user image, replace the existing `profile.jpeg` in the `static/img` folder with your desired image.
- Ensure the image name remains `profile.jpeg` for consistency.

### Step 6: Running the Flask App

- To use the default medical chatbot, simply run `python app.py` in your terminal.
- For a customized chatbot, add or append your data in PDF format to the `data` folder.
- Run `python store_index.py` to update your vector database and store its embeddings.
- Finally, run `python app.py` again to see the customized results.

## Usage

- Access the chatbot by navigating to `http://localhost:5000/`.
- Interact with the chatbot by typing your queries in the input field and pressing Enter to receive responses.

## Contributing

Contributions are welcome. Feel free to submit pull requests or open issues for discussions.

# Thank You
