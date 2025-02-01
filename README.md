<div align="center">
  
  <a href="">![LangChain](https://img.shields.io/badge/langchain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)</a>
  
</div>

# OllaRag

A powerful and user-friendly RAG (Retrieval-Augmented Generation) application that combines the capabilities of Ollama with an intuitive interface for document management and contextual chat interactions.

## 🌟 Features

- 💬 Interactive chat interface with local LLM support through Ollama
- 📚 Document management system with semantic chunking
- 🔍 RAG (Retrieval-Augmented Generation) capabilities for context-aware responses
- 📊 Performance tracking with Langfuse integration
- 🎯 Customizable chunking parameters for optimal document processing
- 🖥️ Clean and modern Streamlit-based UI

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Ollama installed and running locally
- ChromaDB for vector storage
- Langfuse account for tracking (optional)

### Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/ollrag.git
cd ollrag
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up environment variables:
   Create a `.env` file with the following (optional for Langfuse tracking):

```
LANGFUSE_PUBLIC_KEY=your_public_key
LANGFUSE_SECRET_KEY=your_secret_key
```

### Usage

1. Start the application:

```bash
streamlit run app/ollrag.py
```

2. Navigate to:
   - 💬 **Chat**: Interact with the LLM using your document context
   - 📂 **Data**: Manage your document collections and chunking settings

## 🏗️ Project Structure

- `app/`: Main application files
  - `ollrag.py`: Main application entry point
  - `chat_page.py`: Chat interface implementation
  - `data_page.py`: Document management interface
- `scripts/`: Core functionality
  - `ollama_rag.py`: RAG implementation
  - `ollama_data.py`: Document processing and chunking
- `data/`: Storage for document collections

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.
