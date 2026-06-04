# Versa AI

A conversational AI backend with a novel approach to retrieval and context management. Built to be token-efficient, provider-agnostic, and easy to run locally.

> This project is being built in public. The core architecture will become clearer as development progresses.

---

## What's here

This is the initial scaffolding commit. It establishes the foundations the rest of the system is built on.

### LLM Provider Facade

A clean abstraction layer that decouples the application from any specific LLM provider. Anthropic and OpenAI are supported from day one, with a simple interface that makes adding future providers straightforward.

- `ILLMProvider` — the base interface all providers implement
- `AnthropicProvider` — wraps the Anthropic SDK (Claude models)
- `OpenAIProvider` — wraps the OpenAI SDK (GPT models + embeddings)
- `FastEmbedProvider` — local CPU-based embeddings, no API key required

### Configuration

Environment-based config via `.env`. Copy `.env.example` to get started — choose your provider, add your key, and you're ready.

### Database Layer

PostgreSQL with pgvector for vector storage. Async connection pool, schema setup, and query helpers are all wired up.

### Session Architecture

The core of how this system manages conversations. Details to follow as the project develops.

---

## Stack

- **Python** + **FastAPI**
- **PostgreSQL** + **pgvector**
- **Anthropic SDK** / **OpenAI SDK** / **FastEmbed**

---

## Getting started

**Prerequisites:** Python 3.11+, PostgreSQL with the pgvector extension

```bash
# Clone and enter the repo
git clone https://github.com/your-username/versa-ai.git
cd versa-ai

# Install dependencies
pip install -r api/requirements.txt

# Configure environment
cp api/.env.example api/.env
# Edit api/.env — set ACTIVE_PROVIDER, your API key, and DATABASE_URL

# Run tests
pip install -r requirements-dev.txt
pytest
```

---

## Project structure

```
versa-ai/
├── api/
│   ├── core/          # Configuration and session logic
│   ├── db/            # Database client and schema
│   └── providers/     # LLM provider adapters
├── tests/             # Test suite
├── pytest.ini
├── requirements-dev.txt
└── README.md
```

---

## Status

Early development. Core infrastructure is in place. More to come.
