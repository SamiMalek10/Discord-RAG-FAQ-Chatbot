# Discord RAG FAQ Chatbot 🤖

A Retrieval-Augmented Generation (RAG) powered Discord bot for answering FAQ questions about the PM Accelerator AI Bootcamp program.

## 🎯 Project Overview

This chatbot uses RAG to provide accurate, context-aware answers based on the AI Bootcamp knowledge base. It retrieves relevant information from documents before generating responses, ensuring answers are grounded in actual content rather than hallucinated.

## ✅ Project Status

| Phase   | Status      | Description                                              |
| ------- | ----------- | -------------------------------------------------------- |
| Phase 1 | ✅ Complete | Core RAG Components (Chunking, Embeddings, Vector Store) |
| Phase 2 | ✅ Complete | LLM Service, Memory, RAG Chain, RAG Agent                |
| Phase 3 | ✅ Complete | Discord Bot Integration                                  |

**Total Tests: 131 passing** ✅

## 🏗️ Architecture

```
User Question (Discord)
    → Query Embedding (sentence-transformers)
    → Vector Search (FAISS/MongoDB)
    → Retrieve Top-K Chunks
    → Build Prompt [Context + Question]
    → LLM Generation (Mistral/Gemini/OpenAI/Ollama)
    → Return Answer with Sources
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repository>
cd Discrord-RAG-FAQ-Chatbot-PM
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Run Demo (Test Mode)

```bash
python demo_phase3.py
# Select option 2 for test mode (no Discord required)
```

### 4. Run Discord Bot

```bash
# First, set DISCORD_BOT_TOKEN in .env
python demo_phase3.py
# Select option 1 to run the bot
# or directly : 
python run_bot.py
```

## 📁 Project Structure

```
├── src/
│   ├── chunker.py          # Document chunking
│   ├── embeddings.py       # Embedding generation
│   ├── vector_store.py     # FAISS/MongoDB vector store
│   ├── llm_service.py      # LLM providers (Ollama, OpenAI, Gemini, Mistral)
│   ├── memory.py           # Conversation memory
│   ├── rag_chain.py        # RAG pipeline orchestration
│   ├── rag_agent.py        # High-level RAG agent API
│   └── discord_bot.py      # Discord bot integration
├── tests/                  # 131 unit tests
├── config/
│   └── settings.py         # Configuration management
├── data/
│   └── knowledge_base/     # FAQ documents
├── demo_phase2.py          # Phase 2 demo (RAG components)
├── demo_phase3.py          # Phase 3 demo (Discord bot)
└── test_quality.py         # Response quality testing
```

## ⚙️ Configuration

### LLM Providers

- **Mistral** (Recommended): Fast, reliable, generous free tier
- **Gemini**: Google's AI model
- **OpenAI**: GPT-3.5/4
- **Ollama**: Local LLM (free, offline)

### Environment Variables

```env
# LLM Provider (mistral recommended)
LLM_PROVIDER=mistral
MISTRAL_API_KEY=your_key_here

# Discord Bot
DISCORD_BOT_TOKEN=your_discord_bot_token

# Embedding
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=all-MiniLM-L6-v2

# RAG Settings
TOP_K_RESULTS=10
SIMILARITY_THRESHOLD=0.25
```

## 🎮 Discord Bot Commands

| Command             | Description                          |
| ------------------- | ------------------------------------ |
| `/ask <question>` | Ask a question about the AI Bootcamp |
| `/help`           | Show help information                |
| `/status`         | Bot status and statistics            |
| `/clear`          | Clear conversation history           |
| `@bot <question>` | Mention the bot with your question   |

## 🧪 Running Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_discord_bot.py -v

# Response quality test
python test_quality.py
```

## 📊 Response Quality

The RAG pipeline has been tuned for high-quality responses:

- **Improved prompts**: Encourages detailed extraction from context
- **Deduplication**: Removes duplicate chunks in results
- **Optimized retrieval**: Top-K=10, threshold=0.25
- **Quality metrics**: 5/5 test queries passing

## 🔧 Discord Bot Setup

1. Go to [Discord Developer Portal](https://discord.com/developers/applications)
2. Create New Application
3. Add Bot → Copy Token
4. Enable **Message Content Intent**
5. OAuth2 → URL Generator:
   - Scopes: `bot`, `applications.commands`
   - Permissions: Send Messages, Embed Links, Use Slash Commands
6. Add bot to your server
7. Set `DISCORD_BOT_TOKEN` in `.env`
8. Run `python demo_phase3.py`

## 📚 Knowledge Base

The bot answers questions from these documents:

- AI Bootcamp Journey & Learning Path
- Training For AI Engineer Interns
- Intern FAQ - AI Bootcamp

## 🛠️ Technologies

- **Python 3.11**
- **discord.py** - Discord API
- **sentence-transformers** - Local embeddings
- **FAISS** - Vector search
- **Mistral AI** - LLM provider
- **LangChain** - Document chunking only

## 📈 Performance

- **Latency**: ~2s per query
- **Confidence**: 47-58% average
- **Source Attribution**: Always included
- **Memory**: Conversation context per channel

## 👥 Team

Built as part of the PM Accelerator AI Bootcamp (Weeks 1-3 Project)

**Role**: Data Scientist / PM Cohort 7 & 8 / Sami Malek - Core RAG "Intelligence" & Agent Logic
