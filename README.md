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

### Frontend

A Next.js 15 chat UI that streams responses from the FastAPI backend via SSE.

- `web/src/app/` — App Router layout and page
- `web/src/components/` — Chat shell, message list, input, deflect card
- `web/src/hooks/useChat.ts` — Session state and SSE stream parser
- `web/src/lib/api.ts` — API URL helper (`NEXT_PUBLIC_API_URL`)

---

## Stack

**Backend**
- **Python** + **FastAPI**
- **PostgreSQL** + **pgvector**
- **Anthropic SDK** / **OpenAI SDK** / **FastEmbed**

**Frontend**
- **Next.js 15** + **React 19** + **TypeScript**
- **Tailwind CSS**
- **Jest** (via `@next/jest`)

---

## Getting started

### Backend

**Prerequisites:** Python 3.11+, PostgreSQL with the pgvector extension

```bash
# Clone and enter the repo
git clone https://github.com/your-username/versa-ai.git
cd versa-ai/backend

# (Optional) create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp api/.env.example api/.env
# Edit api/.env — set ACTIVE_PROVIDER, your API key, and DATABASE_URL

# Run tests
pip install -r requirements-dev.txt
pytest
```

### Frontend

**Prerequisites:** Node.js 18+

```bash
cd web
npm install

# Configure environment
cp .env.example .env.local
# Edit web/.env.local — set API_URL to point to your FastAPI server

npm run dev   # http://localhost:3000
npm test      # Jest unit tests
```

---

## Project structure

```
versa-ai/
├── backend/
│   ├── api/
│   │   ├── core/          # Configuration and session logic
│   │   ├── db/            # Database client and schema
│   │   └── providers/     # LLM provider adapters
│   ├── tests/             # Backend test suite
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   └── pytest.ini
├── web/
│   ├── src/
│   │   ├── app/           # Next.js App Router (layout, page, globals.css)
│   │   ├── components/    # UI components
│   │   ├── hooks/         # useChat (SSE streaming)
│   │   ├── lib/           # apiUrl helper
│   │   └── types/         # SSE event and message types
│   ├── next.config.ts     # API rewrite (proxies /api/* to FastAPI)
│   └── jest.config.ts
└── README.md
```

---

## Status

Early development. Core infrastructure is in place. More to come.
