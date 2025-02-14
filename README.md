# RAG-Projects

**AI-Powered PDF Chatbot with Gemini & FAISS**

This project is an AI-powered chatbot that allows users to upload multiple PDFs and ask questions based on the extracted content. It uses Google Gemini (Gemini 1.5 Pro) for natural language processing and FAISS (Facebook AI Similarity Search) for efficient document retrieval, enabling precise and context-aware responses.

**🚀 Features**
✅ Upload Multiple PDFs – Extracts and processes text from multiple documents.
✅ AI-Powered Q&A – Ask questions based on the uploaded PDFs and get contextual responses.
✅ Text Chunking & Embeddings – Uses LangChain's RecursiveCharacterTextSplitter to split large text into meaningful chunks.
✅ Efficient Search with FAISS – Stores text embeddings for fast and accurate retrieval.
✅ Google Gemini Integration – Utilizes Gemini 1.5 Pro for generating responses.

**🛠️ Tech Stack**
Python 🐍
Streamlit (for an interactive UI)
PyPDF2 (to extract text from PDFs)
LangChain (for text chunking, vector storage, and prompting)
Google Gemini API (for AI-based responses)
FAISS (for similarity search and document retrieval)
dotenv (for API key management)

1️⃣ Clone the repository
git clone https://github.com/your-username/pdf-chatbot-gemini.git
cd pdf-chatbot-gemini

2️⃣ Create a virtual environment (Optional but recommended)
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate  # On Windows

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Set up the Google Gemini API key
Create a .env file and add:
gemini_key=YOUR_GEMINI_API_KEY
