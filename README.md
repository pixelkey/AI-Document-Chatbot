# Support Chatbot

This project is a support chatbot that uses OpenAI's GPT-4o, FAISS for similarity search, and Gradio for the web interface. The chatbot retrieves relevant documents from a specified folder and generates responses based on those documents.

## Benefits of RAG with LLM

Retrieval-Augmented Generation (RAG) is a powerful technique that combines the strengths of retrieval-based and generation-based models. By integrating a retrieval mechanism with a language model, RAG can provide more accurate and contextually relevant responses. 

### Key Benefits:
1. **Enhanced Accuracy:** By retrieving relevant documents, the LLM can generate responses based on specific information rather than general knowledge.
2. **Improved Context:** The context from the retrieved documents helps the LLM produce more coherent and contextually accurate replies.
3. **Versatility:** Can be used across various applications, including support chatbots, content creation, and more.

### Use Case: Support Chatbot
This support chatbot can be integrated into your website to assist users by providing instant responses based on the information available in your documents. This can be particularly useful for answering FAQs, providing product information, and more.

## Installation

Follow these steps to set up and run the project:

### 1. Navigate to your project directory
```bash
cd support-chatbot
```

### 2. Create a virtual environment using Python 3.11.8
```bash
python3.11 -m venv venv
```

### 3. Activate the virtual environment
For macOS/Linux:
```bash
source venv/bin/activate
```
For Windows:
```bash
.\venv\Scripts\activate
```

### 4. Install required packages
Ensure you have a `requirements.txt` file in your project directory, then run:
```bash
pip install -r requirements.txt
```

### 5. Set up environment variables
Create a `.env` file in the project directory with the following content:
```
OPENAI_API_KEY=your-openai-api-key
EMBEDDING_DIM=1536
FAISS_INDEX_PATH=../embeddings/faiss_index.bin
METADATA_PATH=../embeddings/metadata.pkl
DOCSTORE_PATH=../embeddings/docstore.pkl
INGEST_PATH=../ingest
SYSTEM_PROMPT=Please only provide responses based on the information provided. If it is not available, please let me know.
SIMILARITY_THRESHOLD=0.1
TOP_SIMILARITY_RESULTS=2
```
Replace `your-openai-api-key` with your actual OpenAI API key.

### 6. Prepare the ingest folder
Make sure the `INGEST_PATH` directory specified in the `.env` file exists and contains documents with a `.txt` extension.

### 7. Run the application
```bash
python scripts/chatbot.py
```

This will launch the Gradio interface for the chatbot. Open the provided local URL in your web browser to interact with the chatbot.
You can find the local http address in the logs. 
Find this line: Running on local URL:  http://127.0.0.1:7860

## Usage
- **Input Text:** Type your query in the input text box and click "Submit" to get a response from the chatbot. The response will be based on the similarity search of the provided documents.
- **Chat History:** View the conversation history with the chatbot.
- **References:** View the references of the documents used for generating the response.
- **Clear:** Clear the chat history and references.

## Configuration Options
The `.env` file contains several configuration options:

- **OPENAI_API_KEY:** Your OpenAI API key for accessing OpenAI services.
- **EMBEDDING_DIM:** The dimension of the embeddings used by OpenAI.
- **FAISS_INDEX_PATH:** Path to the FAISS index file.
- **METADATA_PATH:** Path to the metadata file associated with the FAISS index.
- **DOCSTORE_PATH:** Path to the docstore file for storing documents.
- **INGEST_PATH:** Path to the folder containing documents to be ingested.
- **SYSTEM_PROMPT:** The system prompt used by the chatbot to generate responses.
- **SIMILARITY_THRESHOLD:** Threshold for document similarity; documents with a similarity score below this value will be ignored.
- **TOP_SIMILARITY_RESULTS:** The number of top similar results to be considered for generating responses.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.