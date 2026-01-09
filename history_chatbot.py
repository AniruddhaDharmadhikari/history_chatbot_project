import os
from dotenv import load_dotenv
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_classic.memory.buffer import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
# Ensure these are set in your .env file
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "false")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "Historical Figures Chatbot")

# --- PDF Ingestion ---
def initialize_vector_store(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}. Please ensure the file exists.")
    
    print(f"Loading {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    print("Splitting documents into chunks...")
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=30)
    docs = text_splitter.split_documents(documents)
    
    print("Generating embeddings and creating vector store...")
    # Using a lightweight HuggingFace model for embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vectorstore = FAISS.from_documents(docs, embeddings)
    return vectorstore

# --- Chatbot Logic ---
def create_chatbot_chain(vectorstore):
    # Initializes the LLM. Ensure OPENAI_API_KEY is in your .env
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    template = """You are HistoryBot, an expert on historical figures. Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        memory=memory
    )
    
    return qa_chain

# --- Initialization ---
# Update this path to where your PDF is located
PDF_FILE_PATH = "historical_figures.pdf"

try:
    vector_store = initialize_vector_store(PDF_FILE_PATH)
    qa_chain = create_chatbot_chain(vector_store)
except Exception as e:
    print(f"Error during initialization: {e}")
    # Create a dummy chain or exit if initialization fails
    qa_chain = None

# --- Gradio UI ---
def chat_response(message, history):
    if qa_chain is None:
        return "Error: Chatbot not initialized. Check if the PDF file exists and API keys are set."
    
    response = qa_chain.invoke({"query": message})
    return response["result"]

def clear_history():
    if qa_chain:
        qa_chain.memory.clear()
    return None

with gr.Blocks() as demo:
    gr.Markdown("# Historical Figures Chatbot")
    chatbot = gr.Chatbot(label="HistoryBot")
    msg = gr.Textbox(label="Ask about a historical figure:", placeholder="e.g., Who was Leonardo da Vinci?")
    
    with gr.Row():
        submit = gr.Button("Submit", variant="primary")
        clear = gr.Button("Clear History")

    def respond(message, chat_history):
        bot_message = chat_response(message, chat_history)
        chat_history.append((message, bot_message))
        return "", chat_history

    submit.click(respond, [msg, chatbot], [msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: [], None, chatbot, queue=False).then(clear_history)
    
    # Initial greeting
    chatbot.value = [[None, "Hello, I am HistoryBot, your expert on historical figures. How can I assist you today?"]]

if __name__ == "__main__":
    # Launch the app
    demo.launch()
