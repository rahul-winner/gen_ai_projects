# Gen AI Projects

A collection of advanced AI agent implementations and experiments using LangChain, LangGraph, and various LLM providers. This repository showcases practical applications of RAG (Retrieval-Augmented Generation), agentic workflows, and interactive AI systems.

## 🚀 Features

### AI Agents
- **Reflection Agent**: Self-improving agent with reflection capabilities using LangGraph
- **Thinking Agent with Planner**: Advanced planning and reasoning capabilities
- **Travel Agent with Memory**: Context-aware travel planning agent with conversation memory
- **Better Thinking Agent**: Enhanced reasoning patterns for complex problem-solving

### RAG Implementations
- **Simple RAG**: Basic retrieval-augmented generation using vector stores
- **Agentic RAG**: Intelligent document retrieval with agent-based decision making
- **PDF Processing**: Multiple PDF ingestion and querying capabilities

### Interactive Interfaces
- **Streamlit App**: Chat interface for PDF-based question answering
- **Gradio Integration**: Alternative UI for agent interactions

## 📦 Tech Stack

- **LLM Providers**: OpenAI, Groq, Google Gemini, Ollama (local)
- **Frameworks**: LangChain, LangGraph
- **Vector Store**: ChromaDB
- **UI**: Streamlit, Gradio
- **Notebooks**: Jupyter Lab for experimentation

## 🛠️ Installation

### Prerequisites
- Python >= 3.13
- Poetry or pip for dependency management

### Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd gen_ai_projects
```

2. Install dependencies using pip:
```bash
pip install -e .
```

For development dependencies (Jupyter, etc.):
```bash
pip install -e ".[dev]"
```

3. Set up environment variables:
Create a `.env` file in the root directory with your API keys:
```bash
OPENAI_API_KEY=your_openai_key
GROQ_API_KEY=your_groq_key
GOOGLE_API_KEY=your_google_key
# Add other API keys as needed
```

## 📚 Project Structure

```
gen_ai_projects/
├── notebooks/                          # Jupyter notebooks for experimentation
│   ├── simple_rag_1.ipynb             # Basic RAG implementation
│   ├── simple_reflect_agent.ipynb     # Reflection agent example
│   ├── better_thinking_agent.ipynb    # Advanced reasoning agent
│   ├── thinking_agent_with_planner.ipynb
│   ├── travel_agent_planner.ipynb     # Travel planning agent
│   ├── travel_agent_planner_with_memory.ipynb
│   └── vectorstore_simple_rag.ipynb   # Vector store operations
├── simple_reflect_agent.py            # Standalone reflection agent
├── streamlit_app.py                   # PDF RAG Streamlit interface
├── main.py                            # Main entry point
├── my_pdfs/                           # PDF storage directory
├── more_pdfs/                         # Additional PDFs
└── chroma_db/                         # Persistent vector database
└── .env                               # ENV file. add your API keys in it
```

## 🎯 Usage

### Running the Streamlit App

Launch the interactive PDF question-answering interface:
```bash
streamlit run streamlit_app.py
```

This provides a chat interface where you can ask questions about your uploaded PDFs.

### Running Jupyter Notebooks

Start Jupyter Lab to explore individual examples:
```bash
jupyter lab
```

Navigate to the `notebooks/` directory and open any notebook to experiment with different agent implementations.

### Using Individual Agents

Run the reflection agent directly:
```bash
python simple_reflect_agent.py
```

## 🔧 Key Components

### State Management
Uses LangGraph's state management with `Annotated[list, operator.add]` for maintaining conversation history and agent reasoning chains.

### Tool Integration
Agents are equipped with various tools:
- Weather lookup (example implementation)
- Web search capabilities
- PDF document retrieval
- Custom tool binding

### Vector Store
ChromaDB is used for persistent vector storage, enabling semantic search across documents.

### LLM Flexibility
Support for multiple LLM providers:
- **Ollama**: For local, privacy-focused deployments
- **OpenAI**: GPT models for production use
- **Groq**: Fast inference
- **Google Gemini**: Alternative cloud provider

## 📖 Examples

### Simple RAG Query
```python
# Load documents into vector store
# Query using semantic search
# Generate response with LLM
```

### Reflection Agent
```python
# Agent thinks about the problem
# Reflects on its reasoning
# Improves its answer iteratively
```

### Travel Agent with Memory
```python
# Maintains conversation context
# Plans complex itineraries
# Remembers user preferences
```

## 🔐 Privacy & Local Deployment

This project supports **Ollama** for running LLMs locally, ensuring:
- No data leaves your machine
- Complete privacy for sensitive operations
- Suitable for enterprise R&D with IP protection

## 🤝 Contributing

This is a learning project showcasing various AI agent patterns. Feel free to:
- Experiment with different agent architectures
- Add new tools and capabilities
- Try different LLM providers
- Improve existing implementations

## 📝 Dependencies

Key dependencies include:
- `langchain` - LLM application framework
- `langgraph` - State machine for agents
- `langchain-chroma` - Vector store integration
- `streamlit` - Web interface
- `gradio` - Alternative UI
- `pypdf` - PDF processing
- `langchain-openai`, `langchain-groq`, `langchain-google-genai`, `langchain-ollama` - LLM providers

See [pyproject.toml](pyproject.toml) for complete dependency list.

## 🎓 Learning Resources

This project implements concepts from:
- LangChain documentation and tutorials
- LangGraph state management patterns
- RAG best practices
- Agentic workflow design patterns

## 📄 License

[Add your license here]

## 🔗 Additional Notes

- Vector embeddings are stored in `chroma_db/` directory
- PDF files should be placed in `my_pdfs/` or `more_pdfs/`
- Each notebook is self-contained and can be run independently
- The project uses Python 3.13+ features

---

**Note**: This is a learning and experimentation project. For production deployments, additional error handling, monitoring, and security measures should be implemented.
